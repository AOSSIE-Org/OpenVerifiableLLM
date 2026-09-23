"""Join retained production boundaries to the existing persistent publisher.

No provisioning or numerical admission. The enclosing dispatcher authenticates
the registration and selects every input. Publication and policy delivery retain
their own signature, ancestry and actual-download checks. Restart adopts the
original snapshot, service and absolute deadline; it never republishes a boundary.
"""
from dataclasses import asdict
from pathlib import Path
import uuid

from ovl_pipeline.canonical import EvidenceError,canonical,digest,inventory,read_json
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy,verify_prefix
from pod_job_client import save_once
from pod_checkpoint_handoff import observe,snapshot,state_check,deliver
from pod_versioned_export import regular_directory,classified_export,require_retryable_checkpoint
from reconcile_checkpoint_delivery import reconcile
import persistent_publication as publisher


class BoundaryPublisher:
    def __init__(self,registration,job,health,health_file,transport,output,arguments,policy,*,retained_store=None):
        fields(policy,'schema boundary_seconds snapshot_seconds delivery_seconds python','production publication policy')
        if policy['schema']!='ovl.production-boundary-publication-policy.v1':
            raise EvidenceError('unsupported production publication policy')
        for k in ('boundary_seconds','snapshot_seconds','delivery_seconds'):
            integer(policy[k],1,1500,k)
        if policy['snapshot_seconds']+policy['delivery_seconds']+60>=policy['boundary_seconds']:
            raise EvidenceError('publication lacks a bounded service interval')
        fields(arguments,'packet registration-bundle production-policy source-policy source-checkout config','selected publication parents')
        self.registration=registration;self.root=digest(registration);self.job=job
        self.health=health;self.health_file=health_file;self.transport=transport
        self.output=regular_directory(output);self.arguments=dict(arguments);self.policy=dict(policy)
        self.retained_store=None if retained_store is None else Path(retained_store)
        if self.retained_store is not None and (not self.retained_store.is_absolute()
                or any(p.is_symlink() for p in [self.retained_store,*self.retained_store.parents])):
            raise EvidenceError('explicit regular retained publication store required')
        contract=health.contract(job)
        if (contract['kind']!='production-record' or contract['registration_sha256']!=self.root
            or not any(x['root']==transport.profile['remote_root'] and x['profile_sha256']==digest(transport.profile)
                       for x in contract['outputs'])):
            raise EvidenceError('boundary publisher differs from selected production output')
        self.identity={'schema':'ovl.production-boundary-publisher.v1','job_sha256':job,
            'registration_sha256':self.root,'profile_sha256':digest(transport.profile),
            'arguments':arguments,'policy':policy}
        if self.retained_store is not None:self.identity['retained_store']=str(self.retained_store)
        save_once(self.output/'selection.json',self.identity)

    def previous(self,index,envelopes):
        """Adopt only delivered preceding policies from this publisher's history."""
        empty=self.output/'empty-prefix';empty.mkdir(exist_ok=True)
        policies=[];anchors=empty
        for i in range(index):
            state=self.output/'boundaries'/f'boundary-{i:05d}'
            proof=read_json(state/'complete.json')
            ack=read_json(self.output/'publications'/f'boundary-{i:05d}'/'ack.json')
            if (proof['ack_sha256']!=digest(ack) or proof['index']!=i
                or proof['boundary_sha256']!=digest(envelopes[i]) or proof['registration_sha256']!=self.root):
                raise EvidenceError('preceding publication history differs')
            selected=read_json(state/'external-policy.json')
            if selected!=ack['policy']:raise EvidenceError('preceding external policy differs')
            policies.append(ProgressPublisherPolicy(**selected));anchors=Path(ack['anchor_directory'])
        if index:verify_prefix(self.registration,self.root,envelopes[:index],anchors,policies,complete=False)
        return anchors,policies

    def poll(self,*,retained_selection=None):
        self.health.active(self.job)
        if read_json(self.output/'selection.json')!=self.identity:
            raise EvidenceError('publisher selection changed')
        observation=self.output/'observations'/uuid.uuid4().hex;observation.mkdir(parents=True)
        waiting=observe(self.transport,'awaiting-anchor.json',observation/'waiting.json',1024**2,
                        min(self.health.plan['external_terminate_epoch'],self.health.now()+30),optional=True)
        if waiting is None:return None
        fields(waiting,'schema registration_sha256 index boundary_sha256 checkpoint_path checkpoint','waiting production boundary')
        integer(waiting['index'],0,4095,'boundary index')
        if waiting['schema']!='ovl.awaiting-public-progress.v1' or waiting['registration_sha256']!=self.root:
            raise EvidenceError('waiting boundary belongs to another registration')
        pending=[p for p in sorted((self.output/'boundaries').glob('boundary-*/intent.json'))
                 if not(p.parent/'complete.json').exists()]
        if len(pending)>1:raise EvidenceError('multiple unfinished publication boundaries')
        if pending:
            old=read_json(pending[0])['waiting']
            if waiting['index'] not in (old['index'],old['index']+1):
                raise EvidenceError('remote boundary skipped unfinished handoff')
            if waiting['index']==old['index'] and waiting!=old:
                raise EvidenceError('waiting boundary changed at the same index')
            waiting=old
        if self.retained_store is not None and not pending:
            # A primary can appear between live-retention and publication polls.
            # Defer until that exact primary was retained; absence is not cache
            # corruption and must not trigger a second transfer or paid abort.
            if retained_selection is None:return {'result':'AWAITING_RETENTION','index':waiting['index']}
            if retained_selection.get('registration_sha256')!=self.root:
                raise EvidenceError('retained publication selection belongs to another registration')
            if (retained_selection.get('kind')!='record-primary'
                or retained_selection.get('parent_sha256')!=waiting['boundary_sha256']):
                return {'result':'AWAITING_RETENTION','index':waiting['index']}
            if (retained_selection.get('path')!=waiting['checkpoint_path']
                or retained_selection.get('checkpoint')!=waiting['checkpoint']):
                raise EvidenceError('retained primary differs from waiting publication')
        index=waiting['index'];state=self.output/'boundaries'/f'boundary-{index:05d}'
        state.mkdir(parents=True,exist_ok=True)
        intent_file=state/'intent.json'
        if intent_file.exists():
            intent=read_json(intent_file)
        else:
            now=self.health.now()
            if now>=self.health.plan['request_checkpoint_epoch']:
                raise EvidenceError('graceful stop forbids a new publication')
            intent={'schema':'ovl.production-boundary-work.v1','selection_sha256':digest(self.identity),
                'waiting':waiting,'started_epoch':now,
                'deadline_epoch':min(now+self.policy['boundary_seconds'],self.health.plan['request_checkpoint_epoch'])}
            save_once(intent_file,intent)
        fields(intent,'schema selection_sha256 waiting started_epoch deadline_epoch','boundary work intent')
        if (intent['schema']!='ovl.production-boundary-work.v1' or intent['selection_sha256']!=digest(self.identity)
            or intent['waiting']!=waiting
            or intent['deadline_epoch']!=min(intent['started_epoch']+self.policy['boundary_seconds'],self.health.plan['request_checkpoint_epoch'])):
            raise EvidenceError('original boundary identity/deadline changed')
        integer(intent['started_epoch'],self.health.plan['input']['now_epoch'],self.health.plan['request_checkpoint_epoch'],'boundary start')
        deadline=intent['deadline_epoch'];copy_deadline=min(deadline,intent['started_epoch']+self.policy['snapshot_seconds'])
        snap=state/'snapshot'
        selected_files=[{'path':'checkpoint.json','bytes':len(canonical(waiting['checkpoint'])),
                         'sha256':digest(waiting['checkpoint'])},*waiting['checkpoint']['files']]
        # One positively classified zero-payload retry; never reset exhausted
        # range/small-read budgets or retry strict failures on publisher re-entry.
        if snap.exists() and not(snap/'export.json').exists():
            require_retryable_checkpoint(self.transport,waiting['checkpoint_path'],snap,copy_deadline,selected_files)
            snap=state/'snapshot-retry'
        if not snap.exists():
            owner=self
            class ObservedSnapshot:
                def __getattr__(self,name):return getattr(owner.transport,name)
                def get(self,name,destination,item,limit,**kwargs):
                    operation=digest({'job':owner.job,'boundary':digest(waiting),'file':item})
                    def progress(counts):
                        owner.health.bytes(operation,counts,total=item['bytes']);owner.health.write(owner.health_file)
                    kwargs['progress']=progress
                    return owner.transport.get(name,destination,item,limit,**kwargs)
            observed=ObservedSnapshot()
            from pod_transfer import TransientTransportError
            try:
                classified_export(observed,waiting['checkpoint_path'],snap,copy_deadline,selected_files,
                    lambda:snapshot(observed,self.registration,self.root,snap,copy_deadline,retained_store=self.retained_store))
            except TransientTransportError:
                # Consume only the existing classified zero-payload allowance
                # within this owner. Escaping to the enclosing run would abort.
                require_retryable_checkpoint(self.transport,waiting['checkpoint_path'],snap,copy_deadline,selected_files)
                retry=state/'snapshot-retry'
                if snap==retry or retry.exists():raise
                snap=retry
                classified_export(observed,waiting['checkpoint_path'],snap,copy_deadline,selected_files,
                    lambda:snapshot(observed,self.registration,self.root,snap,copy_deadline,retained_store=self.retained_store))
        if not(snap/'export.json').is_file():raise EvidenceError('boundary snapshot retries exhausted')
        chain,body,selected,_=state_check(self.registration,self.root,snap)
        if selected!=waiting:raise EvidenceError('snapshot selected a different waiting boundary')
        self.health.exported_files(self.job,snap/body['checkpoint_path'],
            inventory(snap/body['checkpoint_path'],['checkpoint.json','state.json','state.safetensors']))
        self.health.write(self.health_file)
        anchors,policies=self.previous(index,chain['boundaries'])
        prior=state/'previous-policies.json';save_once(prior,[asdict(p) for p in policies])
        published=self.output/'publications'/f'boundary-{index:05d}'
        args={**self.arguments,'chain-directory':str(snap.resolve()),'previous-directory':str(anchors.resolve()),
              'previous-policies':str(prior.resolve()),'output':str(published.resolve())}
        spec_file=state/'publisher-selection.json'
        if spec_file.exists():spec=read_json(spec_file)
        else:
            spec=publisher.selection(self.root,waiting['boundary_sha256'],deadline-self.policy['delivery_seconds'],
                                     args,python=Path(self.policy['python']))
            save_once(spec_file,spec)
        if (spec['arguments']!=args or spec['registration_sha256']!=self.root
            or spec['boundary_sha256']!=waiting['boundary_sha256']
            or spec['deadline_epoch']!=deadline-self.policy['delivery_seconds']):
            raise EvidenceError('publisher selection changed boundary/parents/deadline')
        complete=state/'complete.json'
        if complete.exists():
            ack=read_json(published/'ack.json');result=read_json(complete)
            if result!={'schema':'ovl.production-boundary-completion.v1','registration_sha256':self.root,
                'index':index,'boundary_sha256':waiting['boundary_sha256'],'ack_sha256':digest(ack),
                'external_policy_sha256':digest(read_json(state/'external-policy.json'))}:
                raise EvidenceError('completed boundary selection changed')
            verify_prefix(self.registration,self.root,chain['boundaries'],Path(ack['anchor_directory']),
                          [*policies,ProgressPublisherPolicy(**read_json(state/'external-policy.json'))],complete=False)
            return result
        if self.health.now()>=deadline:raise EvidenceError('original boundary publication deadline expired')
        service=publisher.start_or_adopt(spec,digest(spec),state/'service')
        ack_file=published/'ack.json';service_result=state/'service/result.json'
        if not service_result.exists():
            for activity in sorted((published/'activity').glob('*.json')):
                self.health.publication(self.job,snap,spec['deadline_epoch'],read_json(activity))
            self.health.write(self.health_file)
            if service['observation']['ActiveState'] not in ('active','activating'):
                raise EvidenceError('persistent publisher exited without a checked result')
            return {'result':'PUBLISHING','index':index,'deadline_epoch':deadline}
        # Completion adoption and delivery have the original enclosing boundary
        # deadline. Do not replay expired liveness events after the worker has
        # finished: they grant no new credit and cannot renew its deadline.
        # The worker's result, acknowledgement, identities and complete public
        # prefix still undergo every check below before any policy is delivered.
        done=read_json(service_result);ack=read_json(ack_file)
        fields(done,'schema selection_sha256 ack_sha256 ack_path scope','publisher result')
        if (done['selection_sha256']!=digest(spec) or done['ack_sha256']!=digest(ack)
            or done['schema']!='ovl.publisher-service-result.v1' or done['ack_path']!=str(ack_file.resolve())):
            raise EvidenceError('publisher result differs from selected acknowledgement')
        publisher.validate(spec,digest(spec))
        if self.health.now()>=deadline:raise EvidenceError('original policy delivery deadline expired')
        # Independently retained operator policy is created from the selected
        # request commit by the publisher, never selected from ack itself.
        selected_policy=read_json(published/'operator-policy.json')
        from publish_progress_boundary import expected_policy
        from ovl_pipeline.progress_anchoring import statement
        previous_root=(verify_prefix(self.registration,self.root,chain['boundaries'][:-1],anchors,policies,
                       complete=False)['closing_statement_sha256'] if index else self.root)
        expected_statement=statement(self.registration,self.root,chain['boundaries'],ack['checkpoint_archive'],previous_root)
        revision=read_json(published/'git-request/public-commit.json')['revision']
        if selected_policy!=asdict(expected_policy(revision,expected_statement)):
            raise EvidenceError('operator policy differs from selected request and independently rebuilt statement')
        save_once(state/'external-policy.json',selected_policy)
        current=ProgressPublisherPolicy(**selected_policy)
        if ack['policy']!=selected_policy:raise EvidenceError('ack chose another policy')
        policies=[*policies,current]
        handoff=state/'delivery'
        if handoff.exists():
            check=state/('reconcile-'+uuid.uuid4().hex)
            recovered=reconcile(self.transport,self.registration,self.root,snap,ack,policies,check,deadline)
            if recovered['result']!='DELIVERED':
                handoff=state/'delivery-retry'
                if handoff.exists():raise EvidenceError('bounded policy handoff attempts exhausted')
                deliver(self.transport,self.registration,self.root,snap,ack,policies,handoff,deadline)
        else:deliver(self.transport,self.registration,self.root,snap,ack,policies,handoff,deadline)
        result={'schema':'ovl.production-boundary-completion.v1','registration_sha256':self.root,
            'index':index,'boundary_sha256':waiting['boundary_sha256'],'ack_sha256':digest(ack),
            'external_policy_sha256':digest(selected_policy)}
        save_once(complete,result);return result
