"""One leased rental lifetime through qualification, anchoring and full replay.

The caller supplies pinned phase/job selections and the existing provider guards.
This coordinator creates no compute and never extends the original rental. Its
completion closes paid work after retention, not scientific/release acceptance.
"""
from dataclasses import asdict
from decimal import Decimal
from private_transport_diagnostics import capture
from pathlib import Path
from contextlib import ExitStack
import time

from ovl_pipeline.canonical import EvidenceError,digest,file_hash,inventory,read_json
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.supervision import Journal
from ovl_pipeline.production_anchoring import packet_objects,verify_packet
from ovl_pipeline.production_parents import validate_parents
from ovl_pipeline.production_replay import authenticate
from ovl_pipeline.state import read_state,unpack
from pod_job_client import save_once,worker_stop_request,stop_delivery
from run_rental_controller import validate as validate_rental,watchdog_heartbeat
from run_workload_coordinator import controller_observation
from production_run_health import ProductionRunHealth
from workload_heartbeat import Heartbeat
import production_run_parents as parents
import production_run_job as job_check
import run_sustained_pilot as development
import publish_production_registration as registration_publisher
from production_checkpoint_poll import CheckpointRetention
from production_boundary_poll import BoundaryPublisher
from run_production_stage import run_stage
from publication_activity import emit


def restore_phase_bindings(plan,expected,output,bindings,downloads):
    """Rebuild historical contracts before journal replay, without new work.

    Complete retained parent checks follow under the observation-only heartbeat.
    This is metadata selection, never qualification or a renewed progress clock.
    """
    if digest(plan)!=expected:raise EvidenceError('original phase selection differs')
    for stage in plan['stages']:
        base=output/'derived'/stage['name'];selected=base/'selection.json'
        if not selected.exists():
            if (base/'job.json').exists():raise EvidenceError('job lost its original selection')
            continue
        value=read_json(selected);job=read_json(base/'job.json');identity=value['identity']
        wanted={k:stage[k] for k in ('name','template_sha256','work_seconds','export_reserve_seconds','parent_record_root')}
        if (identity['plan_sha256']!=expected or identity['stage']!=wanted
            or identity['template_sha256']!=stage['template_sha256'] or digest(job)!=value['job_sha256']
            or job['deadline_epoch']!=value['deadline_epoch']
            or value['deadline_epoch']!=value['admitted_epoch']+stage['work_seconds']):
            raise EvidenceError('historical phase derivation differs')
        for target,name in ((bindings,'validation_binding'),(downloads,'download_binding')):
            if stage[name] is not None:
                old=target.get(value['job_sha256'])
                if old is not None and old!=stage[name]:raise EvidenceError('historical phase contract collision')
                target[value['job_sha256']]=stage[name]


class Run:
    def __init__(self,selection,expected,rental,control,worker,controller,watchdog_file,output,health_file,
                 bindings,downloads,*,sleep=time.sleep):
        optimized=selection.get('schema')=='ovl.production-run-selection.v2'
        fields(selection,'schema rental_intent_sha256 profile_sha256 worker_sha256 phases timing object_store'+(' optimization_policy' if optimized else ''),'enclosing production selection')
        if selection['schema'] not in ('ovl.production-run-selection.v1','ovl.production-run-selection.v2') or digest(selection)!=expected:
            raise EvidenceError('original enclosing run selection differs')
        self.watchdog,self.plan=validate_rental(rental,selection['rental_intent_sha256'])
        if digest(control.profile)!=selection['profile_sha256'] or file_hash(worker)!=selection['worker_sha256']:
            raise EvidenceError('enclosing run endpoint or worker differs')
        fields(selection['phases'],'qualification optimization initialization-baseline initialization-candidate' if optimized else 'qualification initialization','selected development phases')
        if optimized:
            from pilot_optimization import policy
            policy(selection['optimization_policy'])
        fields(selection['timing'],'registration_seconds record_seconds replay_seconds record_fixed_seconds replay_fixed_seconds export_seconds checkpoint_policy publication_policy','enclosing fixed phase budgets')
        for name in ('registration_seconds','export_seconds'):integer(selection['timing'][name],1,1500,name)
        for name in ('record_seconds','replay_seconds','record_fixed_seconds','replay_fixed_seconds'):
            integer(selection['timing'][name],1,7*86400,name)
        self.selection=selection;self.root=expected;self.rental=rental;self.control=control;self.worker=worker
        self.controller=controller;self.watchdog_file=watchdog_file;self.output=output;self.health_file=health_file
        self.bindings=bindings;self.downloads=downloads;self.sleep=sleep;self.stack=ExitStack()
        self.qualified=self.initial=None;self.baseline=self.candidate=None;self.authenticated=None;self.stage_results={};self.phase_jobs={}
        self.record_files=None
        self.active_stage=None
        self.store=Path(selection['object_store'])
        if not self.store.is_absolute() or self.store.is_symlink():raise EvidenceError('explicit regular absolute content store required')
        output.mkdir(parents=True,exist_ok=True);save_once(output/'selection.json',selection)

    def __enter__(self):
        try:
            registration=None
            if (self.output/'registration.json').exists():registration=read_json(self.output/'registration.json')
            journal=self.stack.enter_context(Journal(self.output/'health-journal').lease())
            self.health=ProductionRunHealth(journal,self.watchdog,self.control.profile['pod_id'],registration,self.bindings,self.downloads)
            self.heartbeat=self.stack.enter_context(Heartbeat(self.health,self.health_file))
            active=[job for job,item in self.health.jobs.items() if not item['finished'] and item['selection']['kind'] in ('production-record','full-replay')]
            if len(active)>1:raise EvidenceError('multiple active production jobs')
            if active:
                job=active[0];binding=self.bindings[job];job_file=Path(binding['job_file'])
                self.active_stage=(binding,job_file,job,job_file.parent,self.store)
            if (self.output/'failure/reason.json').exists() and not self.health.complete:
                self.abort(read_json(self.output/'failure/reason.json')['exception_class'])
                raise EvidenceError('prior failed run retained and stopped; no replacement work')
            if not self.health.complete:self.guards(starting=not any(not j['finished'] for j in self.health.jobs.values()))
            return self
        except BaseException:
            self.stack.close();raise

    def __exit__(self,kind,value,tb):
        if value is not None:capture(value,self.output/'private-transport-diagnostics',{'phase':'run-dispatch'})
        try:
            if kind is not None and not self.health.complete:
                try:self.abort(kind.__name__)
                except Exception as secondary:
                    capture(secondary,self.output/'private-transport-diagnostics',{'phase':'failure-retention'})
        finally:
            try:self.stack.__exit__(kind,value,tb)
            except Exception as secondary:
                if value is None:raise
                capture(secondary,self.output/'private-transport-diagnostics',{'phase':'run-cleanup'})
        return False

    def guards(self,*,starting=True):
        self.heartbeat.check()
        if starting and (self.controller/'stop-request.json').exists():raise EvidenceError('original controller requests stop')
        controller_observation(self.controller,self.rental,self.control.profile['pod_id'],starting=starting)
        watchdog_heartbeat(self.watchdog_file,self.watchdog,self.health.now(),self.control.profile['pod_id'])

    def phase(self,name,plan,expected,inputs,output):
        if name not in self.selection['phases'] or self.selection['phases'][name]!=expected:
            raise EvidenceError('phase differs from original run selection')
        save_once(self.output/'development-recovery'/f'{name}.json',{
            'schema':'ovl.development-recovery-selection.v1','run_selection_sha256':self.root,
            'name':name,'plan':plan,'plan_sha256':expected,'inputs':str(Path(inputs).resolve()),'output':str(Path(output).resolve())})
        initialization=name.startswith('initialization')
        if plan['schema']!=('ovl.initialization-cycle-plan.v2' if initialization else 'ovl.sustained-pilot-plan.v2'):
            raise EvidenceError('phase cannot close the enclosing rental')
        if initialization and self.qualified is None:raise EvidenceError('initialization requires checked qualification first')
        if name=='optimization' and self.baseline is None:raise EvidenceError('optimization requires checked baseline first')
        if name=='optimization':
            from pilot_optimization import screen
            planned=screen(self.baseline,self.selection['optimization_policy'])
            if planned['decision']!='TRY_ONE_BATCH_DOUBLING' or read_json(self.output/'optimization-screen.json')!=planned:
                raise EvidenceError('candidate launch lacks its measured bottleneck decision')
        if initialization and self.selection['schema']=='ovl.production-run-selection.v2':
            chosen=read_json(self.output/'optimization-decision.json')['selected']
            if name!='initialization-'+chosen:raise EvidenceError('initialization differs from selected optimization outcome')
        if plan['prior_jobs']!=[]:raise EvidenceError('enclosing phase template must leave prior-job selection to coordinator')
        previous=sorted(j for n,items in self.phase_jobs.items() if n!=name for j in items)
        selected={**plan,'prior_jobs':[] if name=='qualification' else previous}
        expected=digest(selected);plan=selected
        if (output/'selected-plan.json').exists() and read_json(output/'selected-plan.json')!=plan:
            raise EvidenceError('original resolved phase changed')
        final=output/'final/result.json'
        if not final.exists():
            self.guards(starting=not any(not j['finished'] for j in self.health.jobs.values()))
            development.run(plan,expected,self.rental,self.controller,self.watchdog_file,self.control,inputs,
                self.worker,output,self.health_file,sleep=self.sleep,run_health=self.health)
        if name=='qualification':
            result=parents.qualification(plan,expected,output,self.control.profile);self.qualified=self.baseline=result
        elif name=='optimization':
            result=parents.qualification(plan,expected,output,self.control.profile,prepared_qualification=self.baseline);self.candidate=result
        else:
            result=parents.initialization(plan,expected,output,self.control.profile,self.qualified);self.initial=result
        self.phase_jobs[name]=sorted(e['job_sha256'] for e in read_json(final)['stages'])
        save_once(self.output/(name+'-parents.json'),result)
        return result

    def optimize(self,phase):
        """At most one frozen candidate; an interruption adopts the same choice."""
        from pilot_optimization import screen,choose
        if self.baseline is None:raise EvidenceError('checked baseline required before profiling decision')
        value=self.selection['optimization_policy'];selected=screen(self.baseline,value)
        save_once(self.output/'optimization-screen.json',selected)
        if selected['decision']=='TRY_ONE_BATCH_DOUBLING':
            started=self.output/'optimization-start.json'
            if not started.exists():save_once(started,{'screen_sha256':digest(selected),'epoch':self.health.now()})
            begin=read_json(started)
            if begin['screen_sha256']!=digest(selected):raise EvidenceError('optimization start selection changed')
            candidate=self.phase('optimization',*phase)
            finished=self.output/'optimization-end.json'
            if not finished.exists():save_once(finished,{'candidate_sha256':digest(candidate),'epoch':self.health.now()})
            end=read_json(finished)
            if end['candidate_sha256']!=digest(candidate):raise EvidenceError('completed candidate selection changed')
            for stamp in (begin['epoch'],end['epoch']):integer(stamp,1,2**53-1,'original optimization clock')
            if end['epoch']<begin['epoch']:raise EvidenceError('optimization duration clock regressed')
            decision=choose(self.baseline,candidate,value,max(1,(end['epoch']-begin['epoch'])*1000))
        else:
            decision={'schema':'ovl.single-optimization-no-trial.v1','screen_sha256':digest(selected),
                      'selected':'baseline','reason':selected['reason'],'candidate_execution':'NOT_RUN'}
        save_once(self.output/'optimization-decision.json',decision)
        self.qualified=self.candidate if decision['selected']=='candidate' else self.baseline
        return decision['selected'],self.qualified

    def forecast_window(self,r):
        """Cost and phase selection must cover the actual qualified full work."""
        from ovl_pipeline.budget import forecast
        projection=forecast(r['forecast_input']);timing=self.selection['timing'];inp=self.plan['input']
        selected=r['forecast_input']
        if Decimal(selected['hourly_usd'])<Decimal(inp['hourly_upper_usd']):raise EvidenceError('registration underprices original rental')
        elapsed=max(0,self.health.now()-inp['now_epoch'])
        prior=Decimal(inp['spent_usd'])+Decimal(inp['outstanding_usd'])+Decimal(inp['reserved_remaining_usd'])
        incurred=Decimal(elapsed)*Decimal(inp['hourly_upper_usd'])/3600
        if Decimal(selected['spent_usd'])+Decimal(selected['committed_future_usd'])<prior+incurred:
            raise EvidenceError('registration omits prior reservations or elapsed rental exposure')
        # Keep the monetary forecast's slower rate for BOTH directions. Phase
        # deadlines instead use their own authenticated complete pilot timing,
        # including checkpoint save/delivery/comparison, with the same 25% margin.
        # Every production update is charged at a full-batch rate, even tails;
        # registration requires zero completed coverage and checks checkpoint density.
        from ovl_pipeline.budget import ceil_div
        numerical={}
        for direction,field in [('record','measured_ms'),('replay','replay_measured_ms')]:
            numerical[direction]=sum(ceil_div(ceil_div(p['updates']*p[field],p['measured_full_batch_updates'])*5,4000)
                                     for p in selected['phases'].values())
            if self.qualified is None:raise EvidenceError('complete qualified setup measurements required')
            reports=self.qualified['pilot_records' if direction=='record' else 'pilot_replays']
            setup=[]
            for phase in selected['phases']:
                measured=reports[phase].get('setup_including_warmup_ms')
                integer(measured,1,2**53-1,'complete pilot setup measurement')
                setup.append(ceil_div(measured*5,4000))
            fixed=timing[direction+'_fixed_seconds']
            if fixed<sum(setup):raise EvidenceError('fixed phase reserve omits measured setup with margin')
            numerical[direction]+=fixed
        from ovl_pipeline.production_chain import schedule
        publication=len(schedule(r))*timing['publication_policy']['boundary_seconds']
        if timing['record_seconds']<numerical['record']+publication or timing['replay_seconds']<numerical['replay']:
            raise EvidenceError('phase budgets omit full measured numerical work or public boundary waits')
        return projection

    def register(self,r,source,source_bundle,source_policy,prepared,source_checkout):
        if self.qualified is None or self.initial is None:raise EvidenceError('complete same-host qualification and initialization required')
        if any(not j['finished'] for j in self.health.jobs.values()):raise EvidenceError('precommitment cannot overlap numerical work')
        from ovl_pipeline.production_parents import public_initialization
        public_initial=public_initialization(self.initial)
        expected={'source':source,'source_policy':asdict(source_policy),'prepared':prepared,'initial_record':public_initial['record'],
                  'initial_verification':public_initial['verification'],'pilot_records':self.qualified['pilot_records'],
                  'pilot_replays':self.qualified['pilot_replays']}
        validate_parents(r,**expected)
        if not (self.output/'registration-deadline.json').exists():self.forecast_window(r)
        packet=self.output/'packet'
        if packet.exists():
            old,objects=packet_objects(packet)
            if old!=r or objects!={k:v for k,v in expected.items() if k!='source_policy'}:
                raise EvidenceError('retained packet differs from actual same-host parents')
        else:parents.packet(r,source,source_bundle,asdict(source_policy),prepared,self.qualified,self.initial,packet)
        save_once(self.output/'registration.json',r)
        # Registration selection precedes endorsement solely to bind finite cost
        # observations. It grants no permission for a production update.
        if self.health.registration is not None and self.health.registration!=r:raise EvidenceError('run registration changed')
        self.health.registration=r;self.health.registration_root=digest(r)
        timing=self.selection['timing'];intent=self.output/'registration-deadline.json'
        if not intent.exists():
            self.guards();now=self.health.now()
            needed=timing['registration_seconds']+timing['record_seconds']+timing['replay_seconds']+2*timing['export_seconds']
            if now+needed>self.plan['request_checkpoint_epoch']:raise EvidenceError('whole remaining production/replay does not fit original rental')
            save_once(intent,{'run_selection_sha256':self.root,'registration_sha256':digest(r),
                              'started_epoch':now,'deadline_epoch':now+timing['registration_seconds']})
        fixed=read_json(intent)
        if (fixed['run_selection_sha256']!=self.root or fixed['registration_sha256']!=digest(r)
            or fixed['deadline_epoch']!=fixed['started_epoch']+timing['registration_seconds']):raise EvidenceError('registration deadline changed')
        publication=self.output/'registration-publication';receipt=publication/'verified-registration.json'
        def progress(stage,identity):
            emit(publication/'activity',digest(r),digest(r),stage,identity,fixed['deadline_epoch'])
            self.health.registration_activity(read_json(publication/'activity'/(stage+'.json')))
            self.health.write(self.health_file);self.heartbeat.check()
        if not receipt.exists():
            if self.health.now()>=fixed['deadline_epoch']:raise EvidenceError('original registration publication expired')
            registration_publisher.publish(packet,digest(r),source_policy,source_checkout,publication,fixed['deadline_epoch'],progress=progress)
        saved=read_json(receipt)
        # Policy is independently rebuilt from the operator-selected append-only
        # request commit, not taken from a returned signature or remote packet.
        revision=read_json(publication/'request-commit/public-commit.json')['revision']
        policy=registration_publisher.expected_policy(revision,r)
        if saved['production_policy']!=asdict(policy) or saved['registration_sha256']!=digest(r):raise EvidenceError('public registration receipt identity differs')
        downloaded=job_check.downloaded_registration(publication,saved)
        bundle=downloaded['anchor-publication']/'registration.sigstore.json'
        if inventory(downloaded['packet-publication'],sorted(p.name for p in packet.iterdir()))!=inventory(packet,sorted(p.name for p in packet.iterdir())):
            raise EvidenceError('public registration packet bytes differ from selected local packet')
        check=verify_packet(packet,bundle,policy,source_policy,source_checkout=source_checkout)
        self.authenticated={'registration':r,'packet':packet,'bundle':bundle,'production_policy':policy,
                            'source_policy':source_policy,'source_checkout':source_checkout,'check':check,'publication':saved}
        return self.authenticated

    def select_job(self,template,output):
        """Freeze a full measured phase once; adoption never changes its timeout."""
        from sustained_pilot_selection import DEADLINE
        from pod_job_worker import validate_job
        if self.authenticated is None:raise EvidenceError('authenticated public registration required before production selection')
        kind=template.get('kind')
        if kind not in ('production-record','full-replay') or template.get('deadline_epoch')!=DEADLINE:
            raise EvidenceError('production deadline template required')
        output.mkdir(parents=True,exist_ok=True);selection=output/'job-selection.json';job_file=output/'job.json'
        name='record_seconds' if kind=='production-record' else 'replay_seconds';timing=self.selection['timing']
        identity={'run_selection_sha256':self.root,'registration_sha256':self.health.registration_root,
                  'template_sha256':digest(template),'work_seconds':timing[name]}
        if selection.exists():
            saved=read_json(selection)
            if saved['identity']!=identity:raise EvidenceError('original production job selection changed')
            now=saved['started_epoch'];deadline=saved['deadline_epoch']
        else:
            if job_file.exists():raise EvidenceError('production job lost its original deadline selection')
            self.guards();now=self.health.now();deadline=now+timing[name]
        needed=deadline+timing['export_seconds']
        if kind=='production-record':needed+=timing['replay_seconds']+timing['export_seconds']
        if deadline!=now+timing[name] or needed>self.plan['request_checkpoint_epoch']:
            raise EvidenceError('complete record and full replay cannot fit original remaining rental')
        job={**template,'deadline_epoch':deadline,'argv':[str(deadline) if a==DEADLINE else a for a in template['argv']]}
        validate_job(job,resolve_executable=False)
        selected={'identity':identity,'started_epoch':now,'deadline_epoch':deadline,'job_sha256':digest(job)}
        save_once(selection,selected);save_once(job_file,job);return job_file,digest(job)

    def stage(self,kind,job_file,expected,transports,numerical,output,store,*,chain=None,progress_directory=None,progress_policies=None):
        if kind not in ('production-record','full-replay') or self.authenticated is None:
            raise EvidenceError('actual authenticated public registration required before production')
        if Path(store)!=self.store or Path(job_file).parent!=output:raise EvidenceError('stage store or original job directory changed')
        a=self.authenticated;r=a['registration'];job=read_json(job_file)
        if digest(job)!=expected or job['kind']!=kind:raise EvidenceError('selected production descriptor differs')
        original=read_json(output/'job-selection.json')
        if (original['job_sha256']!=expected or original['deadline_epoch']!=job['deadline_epoch']
            or original['identity']['run_selection_sha256']!=self.root
            or original['identity']['registration_sha256']!=digest(r)):
            raise EvidenceError('production descriptor lacks its original one-rental admission')
        job_check.command(job,r,a['packet'],a['bundle'],asdict(a['production_policy']),asdict(a['source_policy']),numerical,kind)
        verify_packet(a['packet'],a['bundle'],a['production_policy'],a['source_policy'],source_checkout=a['source_checkout'])
        envelopes=None
        if kind=='full-replay':
            if 'production-record' not in self.stage_results:raise EvidenceError('complete retained record required before replay')
            if self.record_files is None:raise EvidenceError('complete record object inventory required before replay')
            from pod_versioned_export import retained_object
            # This same store is pinned for both jobs. Require all retained
            # record bytes, including recovery states, before starting replay.
            # Live replay retention separately forbids a cold transfer fallback.
            for item in self.record_files:
                retained_object(self.store/'objects'/item['sha256'],item)
            if any(v is None for v in (chain,progress_directory,progress_policies)):raise EvidenceError('complete recorded public parents required')
            authenticated,envelopes,_=authenticate(a['packet'],a['bundle'],a['production_policy'],a['source_policy'],a['source_checkout'],
                                                   chain,progress_directory,progress_policies)
            if authenticated!=r:raise EvidenceError('replay registration differs')
            # Complete off-pod state identity, not a sample or another execution.
            for envelope in envelopes:
                body=envelope['body'];metadata,tensors=read_state(chain/body['checkpoint_path'],body['checkpoint'])
                if unpack(metadata['tree'],tensors)['control']!=body['control']:raise EvidenceError('retained record state control differs')
        binding={'job_file':job_file,'worker_sha256':self.selection['worker_sha256'],'control':self.control,'transports':transports}
        if expected in self.bindings and self.bindings[expected]!=binding:raise EvidenceError('production binding changed')
        self.bindings[expected]=binding
        checkpoint=CheckpointRetention(r,expected,self.health,self.health_file,numerical,store,output/'live',
                                       self.selection['timing']['checkpoint_policy'],envelopes=envelopes)
        publisher=None
        if kind=='production-record':
            config=self.output/'registration-publication/progress-publisher-config.json'
            args={'packet':str(a['packet'].resolve()),'registration-bundle':str(a['bundle'].resolve()),
                  'source-policy':str((self.output/'source-policy.json').resolve()),
                  'production-policy':str((self.output/'production-policy.json').resolve()),
                  'source-checkout':str(a['source_checkout'].resolve()),'config':str(config.resolve())}
            save_once(self.output/'source-policy.json',asdict(a['source_policy']))
            save_once(self.output/'production-policy.json',asdict(a['production_policy']))
            publisher=BoundaryPublisher(r,expected,self.health,self.health_file,numerical,output/'publication',args,
                                        self.selection['timing']['publication_policy'],retained_store=self.store)
        if not (output/'stage-result.json').exists():self.guards(starting=not (output/'launch/launch-intent.json').exists())
        if not self.health.jobs.get(expected,{}).get('finished',False):self.active_stage=(binding,job_file,expected,output,store)
        result=run_stage(self.control,self.health,job_file,expected,self.worker,self.selection['worker_sha256'],output,
            self.health_file,self.controller/'stop-request.json',self.selection['rental_intent_sha256'],checkpoint,publisher,
            export_seconds=self.selection['timing']['export_seconds'],sleep=self.sleep)
        if result['terminal']['state']!='EXITED' or result['terminal']['exit_code']!=0:
            raise EvidenceError('production stage did not exit successfully; all outputs retained')
        numerical_result=job_check.result(r,result,self.control,transports,job_file,expected,self.selection['worker_sha256'],numerical.profile['remote_root'],kind)
        save_once(output/'checked-numerical-result.json',numerical_result)
        if kind=='full-replay':
            job_check.complete_replay(r,result,self.control,transports,job_file,expected,self.selection['worker_sha256'],numerical.profile['remote_root'],chain,envelopes,numerical_result)
        else:
            self.record_publisher=publisher
            self.record_files=numerical_result['files']
        self.stage_results[kind]=result
        if self.active_stage is not None and self.active_stage[2]==expected:self.active_stage=None
        return result

    def finish(self):
        if set(self.stage_results)!={'production-record','full-replay'} or any(not j['finished'] for j in self.health.jobs.values()):
            raise EvidenceError('both complete retained production stage exits required')
        final=self.output/'final';final.mkdir(exist_ok=True)
        for name,result in self.stage_results.items():save_once(final/(name+'.json'),result)
        self.terminal_copies(final)
        value={'schema':'ovl.production-run-retention-completion.v1','run_selection_sha256':self.root,
               'registration_sha256':self.health.registration_root,'jobs':sorted(self.health.jobs),
               'stages':{name:digest(r) for name,r in self.stage_results.items()},
               'scope':'successful job exits and complete off-pod retention; terminate rental then finish release verification',
               'production_acceptance':'NOT_RUN','independent_third_party':False}
        save_once(final/'result.json',value)
        if not self.health.complete:self.health.finish(final,inventory(final,sorted(p.relative_to(final).as_posix() for p in final.rglob('*') if p.is_file())))
        self.health.write(self.health_file);return value

    def terminal_copies(self,final):
        for event in self.health.journal.events:
            body=event['body']
            if body.get('kind') not in ('job-exit','job-abandon'):continue
            terminal=body['detail']['exit'];job=body['detail']['job_sha256']
            save_once(final/job/('exit.json' if terminal['state']=='EXITED' else 'abandoned.json'),terminal)

    def retain_failed_development(self):
        """Adopt only a previously fenced failure backup, never rerun a phase."""
        active=[job for job,item in self.health.jobs.items() if not item['finished']
                and item['selection']['kind'] not in ('production-record','full-replay')]
        if not active:return
        if len(active)!=1:raise EvidenceError('multiple unfinished development jobs')
        job_root=active[0];matches=[]
        for name,expected in self.selection['phases'].items():
            path=self.output/'development-recovery'/f'{name}.json'
            if not path.exists():continue
            value=read_json(path)
            fields(value,'schema run_selection_sha256 name plan plan_sha256 inputs output','development recovery selection')
            if (value['schema']!='ovl.development-recovery-selection.v1' or value['run_selection_sha256']!=self.root
                or value['name']!=name or value['plan_sha256']!=expected or digest(value['plan'])!=expected):
                raise EvidenceError('development recovery selection changed')
            output=Path(value['output'])
            if not(output/'selected-plan.json').exists():continue
            selected=read_json(output/'selected-plan.json')
            if {**selected,'prior_jobs':[] }!=value['plan']:raise EvidenceError('resolved development plan changed')
            _,_,stages=development.validate(selected,digest(selected),self.rental,self.control,Path(value['inputs']),self.worker)
            for stage,_ in stages:
                derived=output/'derived'/stage['name']
                if not(derived/'job.json').exists() or digest(read_json(derived/'job.json'))!=job_root:continue
                stage_output=output/'stages'/stage['name']
                if not(stage_output/'failure-retention-eligibility.json').exists():continue
                matches.append((stage,derived/'job.json',stage_output,output))
        if not matches:return  # No new eligibility or normal-phase re-entry.
        if len(matches)!=1:raise EvidenceError('ambiguous failed development retention')
        stage,job_file,stage_output,phase_output=matches[0];job=read_json(job_file)
        from pilot_retention import InitialRetention
        from pilot_checkpoint_retention import CheckpointRetention as PilotCheckpoints
        from sustained_pilot_abort import stop_and_retain
        policy=stage['retention']
        if policy is None:raise EvidenceError('failed development retention lacks selected object store')
        kind=PilotCheckpoints if policy['schema']=='ovl.pilot-checkpoint-retention.v1' else InitialRetention
        hook=kind(self.control,self.health,job,job_root,policy,phase_output/'initial-retention'/stage['name'],
                  self.health_file,phase_output/'objects')
        bounds={'maximum_bytes':stage['maximum_export_bytes'],'export_seconds':stage['export_reserve_seconds'],
                'maximum_uncached_bytes':policy['maximum_uncached_export_bytes'] if kind is PilotCheckpoints else stage['maximum_export_bytes']}
        stop_and_retain(self.control,self.health,job_file,job_root,self.worker,self.selection['worker_sha256'],
            stage_output,self.health_file,self.controller/'stop-request.json',self.selection['rental_intent_sha256'],
            sleep=self.sleep,initial_retention=hook,failure_limits=bounds)

    def abort(self,error_class):
        """Bounded stop and full retention; never a replacement launch."""
        from pod_job_client import job_supervision
        from production_retention import retain
        failure=self.output/'failure';failure.mkdir(exist_ok=True)
        save_once(failure/'reason.json',{'schema':'ovl.production-run-failure.v1','run_selection_sha256':self.root,
                  'exception_class':error_class,'scope':'stop and preserve; no production acceptance'})
        self.retain_failed_development()
        if self.active_stage is not None:
            binding,job_file,job,stage,store=self.active_stage
            if (stage/'launch/launch-intent.json').exists() and not self.health.jobs.get(job,{}).get('finished',False):
                window=failure/'stop-window.json'
                if not window.exists():
                    now=self.health.now()
                    save_once(window,{'job_sha256':job,'started_epoch':now,'deadline_epoch':min(
                        self.plan['external_terminate_epoch'],now+self.plan['input']['checkpoint_grace_seconds'])})
                fixed=read_json(window)
                if fixed['job_sha256']!=job:raise EvidenceError('failure recovery changed active job')
                limit=fixed['deadline_epoch']
                original=stage/'terminal-export-intent.json'
                if original.exists():
                    export=read_json(original);selection=read_json(stage/'selection.json')
                    fields(export,'schema selection_sha256 terminal started_epoch deadline_epoch','original terminal export')
                    if (export['schema']!='ovl.production-stage-export.v1' or export['selection_sha256']!=digest(selection)
                        or export['deadline_epoch']!=min(self.plan['external_terminate_epoch'],export['started_epoch']+self.selection['timing']['export_seconds'])):
                        raise EvidenceError('original terminal export selection changed')
                    limit=min(limit,export['deadline_epoch'])
                marker=failure/'request-stop'
                save_once(marker,{'schema':'ovl.production-dispatch-stop.v1','job_sha256':job,'reason':'dispatcher failure; preserve all outputs'})
                worker_marker=failure/'worker-stop.json';save_once(worker_marker,worker_stop_request(job))
                name='jobs/'+job+'/request-stop'
                stop_delivery(self.control,job,name,worker_marker,failure/'stop-delivery.json',min(limit,self.health.now()+30))
                while True:
                    self.health.write(self.health_file)
                    observed=job_supervision(self.control,job,self.selection['worker_sha256'],min(limit,self.health.now()+30))
                    if observed['state']=='ABANDONED' or observed['state']=='EXITED' and not observed['runner_alive'] and not observed['child_alive']:break
                    if self.health.now()>=limit-self.selection['timing']['export_seconds']:raise EvidenceError('original failure shutdown window exhausted')
                    if observed['state'] in ('SUPERVISOR_ABSENT','LAUNCH_FENCE_WITHOUT_INTENT'):
                        job_supervision(self.control,job,self.selection['worker_sha256'],min(limit,self.health.now()+30),abandon=True)
                    elif observed['state'] in ('CHILD_IDENTITY_UNKNOWN','LAUNCH_NOT_OBSERVED'):
                        raise EvidenceError('failure process identity unresolved; preserve guards')
                    self.sleep(1)
                destination=failure/'terminal'
                if not(destination/'retention.json').exists():
                    if destination.exists():destination=failure/'terminal-retry'
                    if destination.exists() and not(destination/'retention.json').exists():raise EvidenceError('preserved failure export attempts exhausted')
                    if not destination.exists():
                        def progress(operation,counts,total):self.health.bytes(operation,counts,total=total)
                        retain(self.control,binding['transports'],job_file,job,self.selection['worker_sha256'],store,destination,limit,progress=progress)
                receipt=destination/'retention.json';terminal=read_json(receipt)['terminal']
                self.health.terminal_retained(job,receipt)
                (self.health.job_exit if terminal['state']=='EXITED' else self.health.abandon_job)(job,terminal)
        if self.health.jobs and all(j['finished'] for j in self.health.jobs.values()):
            self.terminal_copies(failure)
            self.health.finish(failure,inventory(failure,sorted(p.relative_to(failure).as_posix() for p in failure.rglob('*') if p.is_file())))
            self.health.write(self.health_file)
