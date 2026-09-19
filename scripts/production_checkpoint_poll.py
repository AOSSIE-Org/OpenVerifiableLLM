"""Complete live checkpoint retention under an already selected production job.

Observations are peer metadata, not training proof. All state bytes are checked
before export health advances. Each checkpoint retains its original copy deadline;
coordinator recovery never grants a fresh budget or starts numerical work.
"""
from pathlib import Path
import uuid
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.supervision import Journal
from pod_checkpoint_handoff import observe
from pod_job_client import save_once
from pod_versioned_export import regular_directory
from production_live_retention import (record_selection,record_recovery_selection,replay_selection,
    replay_recovery_selection,retain)


class CheckpointRetention:
    def __init__(self,registration,job,health,health_file,transport,store,output,policy,*,envelopes=None):
        fields(policy,'schema export_seconds maximum_checkpoint_bytes','live checkpoint copy policy')
        if policy['schema']!='ovl.production-checkpoint-copy-policy.v1':raise EvidenceError('wrong checkpoint copy policy')
        integer(policy['export_seconds'],1,1500,'fixed checkpoint copy interval')
        integer(policy['maximum_checkpoint_bytes'],1,2**40,'complete checkpoint size bound')
        self.registration=registration;self.root=digest(registration);self.job=job;self.health=health
        self.health_file=health_file;self.transport=transport;self.store=Path(store);self.output=Path(output)
        self.policy=dict(policy);self.envelopes=envelopes
        contract=health.contract(job)
        if contract['registration_sha256']!=self.root or not any(
            x['root']==transport.profile['remote_root'] and x['profile_sha256']==digest(transport.profile)
            for x in contract['outputs']):raise EvidenceError('live checkpoint transport or registration differs')
        self.kind=contract['kind']
        if self.kind=='full-replay':
            from ovl_pipeline.production_chain import verify_chain
            verify_chain(registration,self.root,envelopes,complete=True)
        elif envelopes is not None:raise EvidenceError('recording must not import a future replay chain')
        self.output=regular_directory(self.output)
        self.identity={'job_sha256':job,'registration_sha256':self.root,
            'profile_sha256':digest(transport.profile),'policy':policy,
            'recorded_envelopes_sha256':digest(envelopes) if envelopes is not None else None}
        save_once(self.output/'selection.json',self.identity)

    def observe_selection(self,selected):
        """Persist monotonic checkpoint observations, without health credit."""
        save_once(self.output/'selection.json',self.identity)
        def rank(value):
            c=value['control'];phase={'wikipedia':0,'conversation':1}.get(c['phase'])
            if phase is None:raise EvidenceError('unknown observed checkpoint phase')
            return c['global_step'],phase
        with Journal(self.output/'checkpoint-observations').lease() as journal:
            if not journal.events:journal.append('creation-intent',self.identity)
            if journal.events[0]['kind']!='creation-intent' or journal.events[0]['body']!=self.identity:
                raise EvidenceError('checkpoint observation journal selects another job')
            previous=None
            for event in journal.events[1:]:
                if event['kind']!='checkpoint':raise EvidenceError('unsupported checkpoint observation event')
                value=event['body']
                if previous is not None and rank(value)<=rank(previous):
                    raise EvidenceError('retained checkpoint observations regressed or repeat a step')
                previous=value
            if previous is not None:
                if previous==selected:return
                if rank(selected)<=rank(previous):raise EvidenceError('peer checkpoint regressed or changed at the same step')
            journal.append('checkpoint',selected)

    def poll(self):
        self.health.active(self.job)
        contract=self.health.contract(self.job)
        if contract['registration_sha256']!=self.root or contract['kind']!=self.kind:
            raise EvidenceError('live checkpoint job contract changed')
        out=self.output/'observations'/uuid.uuid4().hex;out.mkdir(mode=0o700,parents=True)
        deadline=min(self.health.plan['external_terminate_epoch'],self.health.now()+30)
        def get(name,maximum=16*1024**2):
            return observe(self.transport,name,out/name,maximum,deadline,optional=True)
        if self.kind=='production-record':
            chain=get('chain.json')
            if chain is None:return None
            selected=record_selection(self.registration,self.root,chain)
            recoveries=get('recoveries.json')
            if recoveries is not None:
                selected=record_recovery_selection(self.registration,self.root,chain,recoveries)
        else:
            session=get('session.json');progress=get('progress.json')
            if progress is None:return None
            if session is None:raise EvidenceError('replay progress has no selected session')
            selected=replay_selection(self.registration,self.root,self.envelopes,session,progress)
            recoveries=get('recoveries.json')
            if recoveries is not None:
                fields(recoveries,'session_sha256 recoveries','observed verifier recoveries')
                if type(recoveries['recoveries']) is not list or not recoveries['recoveries']:
                    raise EvidenceError('declared verifier recoveries must be nonempty')
                step=recoveries['recoveries'][-1]['global_step']
                integer(step,1,2**53-1,'observed verifier recovery step')
                # An existing later primary makes an older recovery irrelevant.
                marker=selected['checkpoint']
                if step>selected['control']['global_step']:
                    name=f'verifier-recovery-{step:09d}/checkpoint.json'
                    (out/Path(name).parent).mkdir()
                    marker=get(name,1024**2)
                    if marker is None:raise EvidenceError('declared verifier recovery marker absent')
                selected=replay_recovery_selection(self.registration,self.root,self.envelopes,session,progress,recoveries,marker)
        self.observe_selection(selected)
        identity=digest(selected);dest=self.output/'checkpoints'/identity;dest.mkdir(mode=0o700,parents=True,exist_ok=True)
        policy_file=dest/'copy-intent.json'
        if policy_file.exists():
            intent=read_json(policy_file)
            fields(intent,'schema job_sha256 selection_sha256 policy_sha256 deadline_epoch','live copy intent')
            if (intent['schema']!='ovl.live-checkpoint-copy-intent.v1' or intent['job_sha256']!=self.job
                or intent['selection_sha256']!=identity or intent['policy_sha256']!=digest(self.policy)):
                raise EvidenceError('original live checkpoint copy selection changed')
            integer(intent['deadline_epoch'],1,self.health.plan['external_terminate_epoch'],'original copy deadline')
        else:
            intent={'schema':'ovl.live-checkpoint-copy-intent.v1','job_sha256':self.job,'selection_sha256':identity,
                'policy_sha256':digest(self.policy),'deadline_epoch':min(self.health.plan['external_terminate_epoch'],
                    self.health.now()+self.policy['export_seconds'])}
            save_once(policy_file,intent)
        result=retain(self.transport,selected,identity,self.job,self.health,self.health_file,self.store,
                      dest/'retained',intent['deadline_epoch'],self.policy['maximum_checkpoint_bytes'])
        return {'selection':selected,'selection_sha256':identity,'retention':result,'directory':str(dest.resolve())}
