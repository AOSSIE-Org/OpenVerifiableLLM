"""State-aware cost health for dedicated production/replay dispatchers.

Callers must supply the independently authenticated registration and exact selected
jobs/transports. This class never authenticates publisher identity on their behalf
or authorizes an update. The numerical record/replay drivers retain those gates.
"""
from pathlib import Path
from ovl_pipeline.canonical import EvidenceError,digest,read_json,require_digest,write_json
from ovl_pipeline.schema import fields,integer
from pod_checkpoint_handoff import state_check
from production_retention import verify as verify_retention
from publication_activity import validate as validate_publication
from workload_health import Health,terminal_status


class ProductionHealth(Health):
    def __init__(self,journal,watchdog_intent,pod_id,registration,bindings,**kwargs):
        self.registration=registration;self.registration_root=digest(registration)
        self.bindings=bindings;self.publications={}
        super().__init__(journal,watchdog_intent,pod_id,**kwargs)

    def contract(self,job):
        require_digest(job)
        if job not in self.bindings:raise EvidenceError('production job lacks independently selected retention bindings')
        b=self.bindings[job];fields(b,'job_file worker_sha256 control transports','production retention binding')
        require_digest(b['worker_sha256']);value=read_json(Path(b['job_file']))
        if digest(value)!=job or value['kind'] not in ('production-record','full-replay'):raise EvidenceError('production job selection changed')
        from production_retention import roots
        selected=roots(b['control'],b['transports'],value,job)
        if b['control'].profile['pod_id']!=self.pod:raise EvidenceError('production retention selects another pod')
        return {'registration_sha256':self.registration_root,'job_sha256':job,'kind':value['kind'],
                'worker_sha256':b['worker_sha256'],
                'outputs':[{'root':root,'profile_sha256':digest(t.profile)} for t,_,root in selected]}

    def start_job(self,selection):
        if selection.get('kind') not in ('production-record','full-replay'):return super().start_job(selection)
        fields(selection,'schema job_sha256 pod_id kind','selected production job')
        job=selection['job_sha256'];contract=self.contract(job)
        if selection['schema']!='ovl.selected-workload-job.v1' or selection['pod_id']!=self.pod or selection['kind']!=contract['kind']:
            raise EvidenceError('selected production identity differs')
        selected={**selection,'retention_contract_sha256':digest(contract)}
        if job in self.jobs:
            if self.jobs[job]['selection']!=selected:raise EvidenceError('retained production contract changed')
            return False
        if any(not x['finished'] for x in self.jobs.values()):raise EvidenceError('another workload stage is active')
        self.event('job-start',selected);return True

    def _apply(self,body):
        if body.get('kind')!='publication':return super()._apply(body)
        fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','publication cost event')
        integer(body['observed_epoch'],self.plan['input']['now_epoch'],self.plan['external_terminate_epoch'],'publication event clock')
        if (body['schema']!='ovl.cost-activity-event.v2' or body['advances_progress'] is not True
            or body['advances_export'] is not False or body['completes'] is not False):raise EvidenceError('invalid publication liveness decision')
        d=body['detail'];fields(d,'job_sha256 observation','publication cost detail');value=d['observation']
        validate_publication(value,self.registration_root,value['boundary_sha256'],value['deadline_epoch'])
        key=(d['job_sha256'],value['boundary_sha256'],value['stage'])
        if key in self.publications:raise EvidenceError('duplicate durable publication credit')
        self.publications[key]=value;self.progress=max(self.progress,body['observed_epoch'])

    def publication(self,job,snapshot,deadline,value):
        self.active(job);contract=self.contract(job)
        if contract['kind']!='production-record':raise EvidenceError('replay does not publish a new recorded boundary')
        integer(deadline,1,self.plan['provider_terminate_epoch'],'selected publication deadline')
        if self.now()>=deadline:raise EvidenceError('publication deadline reached')
        validate_publication(value,self.registration_root,value['boundary_sha256'],deadline)
        key=(job,value['boundary_sha256'],value['stage'])
        prior=self.publications.get(key)
        if prior is not None:
            if prior!=value:raise EvidenceError('publication identity changed after credited stage')
            return False
        chain,body,waiting,checked=state_check(self.registration,self.registration_root,Path(snapshot))
        validate_publication(value,self.registration_root,waiting['boundary_sha256'],deadline)
        others=[v for (j,b,s),v in self.publications.items() if j==job and b==value['boundary_sha256']]
        if others and any(v['deadline_epoch']!=deadline for v in others):raise EvidenceError('publication deadline changed within boundary')
        self.event('publication',{'job_sha256':job,'observation':value},progress=True);return True

    def terminal_retained(self,job,proof):
        self.active(job);contract=self.contract(job);b=self.bindings[job]
        proof=Path(proof)
        if proof.is_symlink():raise EvidenceError('terminal retention proof symlink')
        value=read_json(proof)
        receipts=verify_retention(value,b['control'],b['transports'],b['job_file'],job,b['worker_sha256'])
        for receipt in receipts:self.exported_files(job,Path(receipt['files_directory']),receipt['files'])
        selected={'proof_path':str(proof.resolve()),'proof_sha256':digest(value),'contract_sha256':digest(contract)}
        directory=self.journal.directory/'production-retention';directory.mkdir(mode=0o700,exist_ok=True)
        path=directory/(job+'.json')
        if path.exists():
            if read_json(path)!=selected:raise EvidenceError('terminal retention proof selection changed')
        else:write_json(path,selected)
        return selected

    def retained(self,job):
        selected=read_json(self.journal.directory/'production-retention'/(job+'.json'))
        fields(selected,'proof_path proof_sha256 contract_sha256','retained production proof selection')
        if selected['contract_sha256']!=digest(self.contract(job)):raise EvidenceError('production retention contract changed')
        proof=Path(selected['proof_path'])
        if proof.is_symlink():raise EvidenceError('retention proof replaced by symlink')
        value=read_json(proof)
        if digest(value)!=selected['proof_sha256']:raise EvidenceError('retention proof changed')
        b=self.bindings[job];verify_retention(value,b['control'],b['transports'],b['job_file'],job,b['worker_sha256'])
        return selected,value

    def terminal(self,job,status,state):
        if self.jobs[job]['selection']['kind'] not in ('production-record','full-replay'):
            return (super().job_exit if state=='EXITED' else super().abandon_job)(job,status)
        self.active(job)
        if terminal_status(status,job)!=state:raise EvidenceError('incorrect production terminal mode')
        selected,value=self.retained(job)
        if value['terminal']!=status:raise EvidenceError('terminal status differs from complete retained outputs')
        self.event('job-exit' if state=='EXITED' else 'job-abandon',{'job_sha256':job,'exit':status,
                   'retention_sha256':digest(selected)},progress=True)

    def job_exit(self,job,status):return self.terminal(job,status,'EXITED')
    def abandon_job(self,job,status):return self.terminal(job,status,'ABANDONED')

    def _check_final(self,directory,expected):
        super()._check_final(directory,expected)
        for job,record in self.jobs.items():
            if record['selection']['kind'] not in ('production-record','full-replay'):continue
            selected,value=self.retained(job)
            if record['selection']['retention_contract_sha256']!=selected['contract_sha256']:raise EvidenceError('started production contract differs')
            terminal=[e['body']['detail'] for e in self.journal.events if e['body'].get('kind') in ('job-exit','job-abandon') and e['body']['detail']['job_sha256']==job]
            if len(terminal)!=1 or terminal[0].get('retention_sha256')!=digest(selected):raise EvidenceError('production terminal omitted complete-output retention')
