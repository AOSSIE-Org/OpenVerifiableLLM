"""Durable cost-health observations from actual work, never training evidence.

The caller holds the journal lease. Heartbeats cannot renew progress or export
age. Transfer counters, runtime sequences and export identities survive restart.
The independent rental controller/watchdog retain all cost and deadline authority.
"""
from pathlib import Path
import re
import time

from ovl_pipeline.canonical import EvidenceError,digest,read_json,require_digest,verify_inventory,confined
from ovl_pipeline.canonical import write_json
from ovl_pipeline.schema import control,fields,integer
from rental_safety import Lifetime,boot_clock
from run_external_watchdog import validate_intent


def terminal_status(value,job):
    if value.get('state')=='EXITED':
        fields(value,'schema job_sha256 state exit_code','observed workload exit')
        integer(value['exit_code'],-255,255,'workload exit code')
        if value['schema']!='ovl.workload-job-exit.v1':raise EvidenceError('wrong terminal schema')
    elif value.get('state')=='ABANDONED':
        fields(value,'schema job_sha256 state observed_child exit_code scope','observed workload abandonment')
        if value['schema']!='ovl.workload-job-abandonment.v1' or value['exit_code']!='UNAVAILABLE':raise EvidenceError('abandonment cannot invent exit code')
        if value['observed_child'] is not None:
            c=value['observed_child'];fields(c,'pid start_ticks process_group','abandoned process identity')
            for key in ('pid','start_ticks','process_group'):integer(c[key],1,2**53-1,'abandoned process identity')
            if c['pid']!=c['process_group']:raise EvidenceError('abandoned process is not its group leader')
    else:raise EvidenceError('workload has no terminal observation')
    if value['job_sha256']!=job:raise EvidenceError('terminal observation selects another job')
    return value['state']


class Health:
    def __init__(self,journal,watchdog_intent,pod_id,*,wall=time.time,clock=boot_clock):
        from threading import RLock
        self._publication_lock=RLock();self._clock_lock=RLock()
        self.journal=journal;self.wall=wall;self.root=digest(watchdog_intent)
        self.plan=validate_intent(watchdog_intent,self.root);self.pod=pod_id
        if type(pod_id) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,96}',pod_id):raise EvidenceError('invalid adopted pod ID')
        identity={'schema':'ovl.workload-health-identity.v2','watchdog_intent_sha256':self.root,'pod_id':pod_id}
        previous=[e['body'] for e in journal.events if e['kind']=='creation-intent']
        fresh=not journal.events
        if fresh:journal.append('creation-intent',identity)
        elif previous!=[identity]:raise EvidenceError('workload health identity differs from retained journal')
        self.lifetime=Lifetime(journal,self.plan,wall=wall,clock=clock,initialize=fresh)
        self.progress=self.exported=self.plan['input']['now_epoch'];self.complete=False
        self.jobs={};self.transfers={};self.activities={};self.phase_activities={};self.exports=set()
        self._complete_verified=False
        for event in journal.events:
            body=event['body']
            if event['kind'] in ('creation-intent',):continue
            if event['kind']=='decision' and body.get('action')=='LIFETIME_CLOCK':continue
            self._apply(body)

    def _apply(self,body):
        fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','cost activity event')
        if body['schema']!='ovl.cost-activity-event.v2':raise EvidenceError('unsupported cost activity event')
        integer(body['observed_epoch'],self.plan['input']['now_epoch'],self.plan['external_terminate_epoch'],'cost event clock')
        if any(type(body[k]) is not bool for k in ('advances_progress','advances_export','completes')):
            raise EvidenceError('explicit activity decisions required')
        kind=body['kind'];d=body['detail']
        if kind=='job-start':self.jobs[d['job_sha256']]={'finished':False,'selection':d}
        elif kind=='bytes':self.transfers[d['operation_sha256']]=d
        elif kind=='activity':self.activities[d['job_sha256']]=d['observation']
        elif kind=='pilot-phases':self.phase_activities[d['job_sha256']]=d['observation']
        elif kind=='export':self.exports.add(d['export_sha256'])
        elif kind in ('job-exit','job-abandon'):self.jobs[d['job_sha256']]['finished']=True
        elif kind!='complete':raise EvidenceError('unknown cost activity event')
        if body['advances_progress']:self.progress=max(self.progress,body['observed_epoch'])
        if body['advances_export']:self.exported=max(self.exported,body['observed_epoch'])
        self.complete=self.complete or body['completes']

    def now(self):
        with self._clock_lock:
            if self.lifetime.remaining()<=0:raise EvidenceError('workload deadline/boot/clock changed; stop without renewing health')
            now=int(self.wall())
            if now<max(self.progress,self.exported):raise EvidenceError('workload clock regressed')
            return now

    def event(self,kind,detail,*,progress=False,export=False,complete=False):
        if self.complete:raise EvidenceError('completed workload cannot acquire new work')
        body={'schema':'ovl.cost-activity-event.v2','kind':kind,'observed_epoch':self.now(),'detail':detail,
              'advances_progress':progress,'advances_export':export,'completes':complete}
        self.journal.append('checkpoint' if export else 'decision',body);self._apply(body)
        return body

    def start_job(self,selection):
        fields(selection,'schema job_sha256 pod_id kind','selected workload job')
        require_digest(selection['job_sha256'])
        if selection['schema']!='ovl.selected-workload-job.v1' or selection['pod_id']!=self.pod or selection['kind'] not in ('setup','pilot','export'):
            raise EvidenceError('unsupported or foreign workload stage')
        prior=self.jobs.get(selection['job_sha256'])
        if prior:
            if prior['selection']!=selection:raise EvidenceError('job selection changed')
            return False
        if any(not j['finished'] for j in self.jobs.values()):raise EvidenceError('another workload stage is still active')
        # Starting a process alone is not observed useful progress.
        self.event('job-start',selection);return True

    def active(self,job):
        require_digest(job)
        if job not in self.jobs or self.jobs[job]['finished']:raise EvidenceError('selected job is not active')

    def bytes(self,operation,counts,*,total,direction='receive'):
        """Bounded useful byte progress: at least 1 MiB or one complete file.

        The operation digest must bind selected job/path/content, never a poll
        UUID. Metadata polls are not transfers. Declared size/direction and the
        last credited threshold survive retries and coordinator restart.
        """
        require_digest(operation);fields(counts,'bytes_sent bytes_received','actual transfer counters')
        integer(total,0,2**40,'selected operation bytes')
        if direction not in ('send','receive'):raise EvidenceError('explicit byte progress direction required')
        current=tuple(counts[k] for k in ('bytes_sent','bytes_received'))
        for count in current:integer(count,0,2**40,'transfer byte count')
        prior=self.transfers.get(operation)
        if prior and (prior['total']!=total or prior['direction']!=direction):raise EvidenceError('selected byte operation changed')
        previous=(prior['bytes_sent'],prior['bytes_received']) if prior else (0,0)
        credited=prior['credited_bytes'] if prior else 0
        # A resumed transfer must exceed its preserved high-water counts. Repeated
        # bytes or reset counters cannot make a stalled retry look like progress.
        high=tuple(max(a,b) for a,b in zip(current,previous))
        count=high[0 if direction=='send' else 1]
        if count>total:raise EvidenceError('byte progress exceeds selected operation size')
        if high==previous and prior:return False
        advances=count-credited>=1024**2 or count==total and count>credited
        self.event('bytes',{'operation_sha256':operation,'bytes_sent':high[0],'bytes_received':high[1],
                           'total':total,'direction':direction,'credited_bytes':count if advances else credited},progress=advances)
        return advances

    def activity(self,job,observation):
        self.active(job)
        if type(observation) is dict and observation.get('schema')=='ovl.audited-pilot-phases.v1':return self.pilot_phases(job,observation)
        if job in self.phase_activities:raise EvidenceError('job changed its activity protocol')
        fields(observation,'schema process_instance pid sequence kind control scope','runtime activity')
        if (observation['schema']!='ovl.runtime-activity.v1' or observation['kind']!='completed-numerical-update'
            or observation['scope']!='operator-supervision-only-not-training-verification'
            or not re.fullmatch('[0-9a-f]{32}',observation['process_instance'])):
            raise EvidenceError('unsupported runtime activity')
        integer(observation['pid'],1,2**31-1,'runtime PID');integer(observation['sequence'],1,2**53-1,'runtime sequence')
        c=dict(observation['control'])
        if 'pilot_cycle' in c:integer(c.pop('pilot_cycle'),0,2**53-1,'pilot cycle')
        control(c)
        previous=self.activities.get(job)
        if previous:
            if any(previous[k]!=observation[k] for k in ('process_instance','pid')):
                raise EvidenceError('runtime process changed within selected job')
            if observation['sequence']<previous['sequence']:raise EvidenceError('runtime sequence regressed')
            if observation['sequence']==previous['sequence']:
                if previous!=observation:raise EvidenceError('same runtime sequence changed content')
                return False
        changed=previous is None or previous['control']!=observation['control']
        # Warmup is genuine paid work and may reset global_step when fresh
        # initialization starts. This is liveness, never a monotonic coverage proof.
        self.event('activity',{'job_sha256':job,'observation':observation},progress=changed)
        return changed

    def pilot_phases(self,job,observation):
        """At most three completed audited phases; never a heartbeat or export.

        The selected trusted wrapper emits these only after child success and
        report checks. This remains peer-reported operational progress, not proof
        of arithmetic, authenticity, safe-state retention or training acceptance.
        """
        self.active(job)
        if self.jobs[job]['selection']['kind']!='pilot' or job in self.activities:
            raise EvidenceError('completed-phase activity requires a distinct pilot wrapper')
        fields(observation,'schema process_instance pid completed scope','audited pilot phase activity')
        if (observation['schema']!='ovl.audited-pilot-phases.v1'
            or observation['scope']!='operator-supervision-only-not-training-verification'
            or type(observation['process_instance']) is not str or not re.fullmatch('[0-9a-f]{32}',observation['process_instance'])):
            raise EvidenceError('unsupported audited pilot phase activity')
        integer(observation['pid'],1,2**31-1,'pilot wrapper PID')
        completed=observation['completed']
        if type(completed) is not list or not 1<=len(completed)<=3:raise EvidenceError('bounded completed pilot prefix required')
        for item,phase in zip(completed,('record','replay','resume')):
            fields(item,'phase report_sha256','completed audited pilot phase');require_digest(item['report_sha256'])
            if item['phase']!=phase:raise EvidenceError('audited pilot phases reordered')
        if len({x['report_sha256'] for x in completed})!=len(completed):raise EvidenceError('pilot phases repeat a report')
        previous=self.phase_activities.get(job)
        if previous:
            if any(previous[k]!=observation[k] for k in ('process_instance','pid')):
                raise EvidenceError('pilot wrapper process changed')
            if completed[:len(previous['completed'])]!=previous['completed']:
                raise EvidenceError('completed pilot prefix changed or regressed')
            if previous==observation:return False
        self.event('pilot-phases',{'job_sha256':job,'observation':observation},progress=True)
        return True

    def exported_files(self,job,directory,expected,*,deadline=None):
        """Verify actual complete selected file bytes, not a report's PASS field.

        The caller selects the inventory from its bounded export protocol. Its
        semantic completeness (e.g. safe state/control) is checked by that protocol.
        Re-copying the identical inventory cannot renew the export interval.
        """
        self.active(job);directory=Path(directory)
        if directory.is_symlink() or not directory.is_dir():raise EvidenceError('regular export directory required')
        actual=[]
        for p in directory.rglob('*'):
            if p.is_symlink() or not(p.is_file() or p.is_dir()):raise EvidenceError('unsafe export tree')
            if p.is_file():actual.append(p.relative_to(directory).as_posix())
        if sorted(actual)!=[f['path'] for f in expected]:raise EvidenceError('export inventory does not cover exact local file tree')
        verify_inventory(directory,expected)
        identity=digest({'job_sha256':job,'files':expected})
        if identity in self.exports:return False
        manifest={'schema':'ovl.workload-export-inventory.v1','job_sha256':job,'files':expected}
        path=self.journal.directory/'export-manifests'/f'{identity}.json'
        if path.exists():
            if read_json(path)!=manifest:raise EvidenceError('retained export manifest changed')
        else:write_json(path,manifest)
        if deadline is not None:
            integer(deadline,1,self.plan['external_terminate_epoch'],'export acceptance deadline')
            if self.now()>=deadline:raise EvidenceError('export verification exceeded original deadline')
        self.event('export',{'job_sha256':job,'export_sha256':identity,'files_sha256':digest(expected),
                            'manifest_sha256':digest(manifest),
                            'directory':str(directory.resolve()),'scope':'complete selected bytes rehashed off pod; no training verification'},export=True,progress=True)
        return True

    def job_exit(self,job,status):
        self.active(job)
        if terminal_status(status,job)!='EXITED':
            raise EvidenceError('wrong/unfinished workload exit')
        if not any(e['body'].get('kind')=='export' and e['body']['detail']['job_sha256']==job for e in self.journal.events):
            raise EvidenceError('job has no verified off-pod export')
        self.event('job-exit',{'job_sha256':job,'exit':status},progress=True)

    def abandon_job(self,job,status):
        self.active(job)
        if terminal_status(status,job)!='ABANDONED':raise EvidenceError('distinct abandonment evidence required')
        if not any(e['body'].get('kind')=='export' and e['body']['detail']['job_sha256']==job for e in self.journal.events):
            raise EvidenceError('abandoned job has no verified off-pod export')
        self.event('job-abandon',{'job_sha256':job,'exit':status},progress=True)

    def _check_final(self,directory,expected):
        if not self.jobs or any(not j['finished'] for j in self.jobs.values()):raise EvidenceError('workload jobs have not all exited')
        paths=[];directory=Path(directory)
        if directory.is_symlink() or not directory.is_dir():raise EvidenceError('regular final export directory required')
        for p in directory.rglob('*'):
            if p.is_symlink() or not(p.is_file() or p.is_dir()):raise EvidenceError('unsafe final export tree')
            if p.is_file():paths.append(p.relative_to(directory).as_posix())
        if sorted(paths)!=[f['path'] for f in expected]:raise EvidenceError('final export inventory is incomplete')
        verify_inventory(directory,expected)
        for event in self.journal.events:
            if event['body'].get('kind')!='export':continue
            d=event['body']['detail']
            manifest=read_json(confined(self.journal.directory,'export-manifests/'+d['export_sha256']+'.json'))
            if (digest(manifest)!=d['manifest_sha256'] or digest(manifest['files'])!=d['files_sha256']
                or manifest['job_sha256']!=d['job_sha256']):raise EvidenceError('preserved export manifest differs')
            prior=Path(d['directory'])
            if prior.is_symlink():raise EvidenceError('preserved export directory is now a symlink')
            verify_inventory(prior,manifest['files'])
        # Re-read the terminal records in the actual exported bytes. A standalone
        # local exit assertion without its preserved matching remote record fails.
        terminal={digest(read_json(directory/f['path'])) for f in expected if Path(f['path']).name in ('exit.json','abandoned.json')}
        exits=[e['body']['detail']['exit'] for e in self.journal.events if e['body'].get('kind') in ('job-exit','job-abandon')]
        if any(digest(v) not in terminal for v in exits):raise EvidenceError('terminal job records missing from final export')

    def finish(self,directory,expected):
        """Final rehash after observed exits; failure artifacts can also teardown.

        Terminal records must be present. All prior exports are rehashed too.
        This is cost completion, not a successful model verification gate.
        """
        directory=Path(directory);self._check_final(directory,expected)
        path=self.journal.directory/'final-export-inventory.json'
        if path.exists():
            if read_json(path)!=expected:raise EvidenceError('final export selection changed')
        else:write_json(path,expected)
        self.event('complete',{'export_inventory_sha256':digest(expected),'directory':str(directory.resolve()),
                              'scope':'observed exited workload and rehashed selected exports; no model acceptance'},progress=True,export=True,complete=True)
        self._complete_verified=True

    def write(self,path):
        if self.complete and not self._complete_verified:
            final=[e['body']['detail'] for e in self.journal.events if e['body'].get('kind')=='complete']
            if len(final)!=1:raise EvidenceError('ambiguous completed health journal')
            files=read_json(confined(self.journal.directory,'final-export-inventory.json'))
            if digest(files)!=final[0]['export_inventory_sha256']:raise EvidenceError('final export inventory changed after restart')
            self._check_final(Path(final[0]['directory']),files);self._complete_verified=True
        return self.pulse(path)

    def pulse(self,path):
        """Publish observed health without credit or expensive final revalidation.

        The owner may call this while an existing verification/transfer is busy.
        Progress/export ages and both original clocks remain unchanged. A replayed
        completion cannot be published until write() rechecks its actual bytes.
        No journal mutation or provider observation is supplied by this method.
        """
        with self._publication_lock:
            if self.journal._fd is None:raise EvidenceError('health publication requires its original journal lease')
            # Sample credited ages before the observation clock. The main owner
            # may advance them concurrently; publishing older credit is safe,
            # whereas reading a newer age after now() could look like the future.
            progress=self.progress;exported=self.exported
            complete=bool(self.complete and self._complete_verified)
            value={'schema':'ovl.rental-workload-health.v1','intent_sha256':self.root,'pod_id':self.pod,
                   'observed_epoch':self.now(),'progress_epoch':progress,'exported_checkpoint_epoch':exported,
                   'complete':complete}
            write_json(Path(path),value);return value
