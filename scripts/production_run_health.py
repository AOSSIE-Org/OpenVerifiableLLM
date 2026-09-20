"""One unchanged rental clock across qualification, initialization and production.

The enclosing coordinator owns independently pinned job contracts and authenticates
the registration before adding production bindings. This class neither admits a
job nor finishes a rental at a phase boundary. Every prior binding must be supplied
again when the single journal is adopted, including finished development jobs.
"""
from ovl_pipeline.canonical import EvidenceError,digest
from production_health import ProductionHealth
from initialization_health import InitializationCycleHealth,InitializationHealth
from sustained_health import SustainedHealth
from pilot_health import PilotHealth
from ovl_pipeline.schema import fields,integer
from publication_activity import validate as validate_publication


class ProductionRunHealth(ProductionHealth,InitializationCycleHealth):
    def __init__(self,journal,watchdog_intent,pod_id,registration,bindings,download_bindings,**kwargs):
        self.registration=registration;self.registration_root=digest(registration)
        self.publications={};self.scans={}
        self.registration_publications={}
        # Initialize each shared Health field exactly once. All later contracts
        # are selected by their closed schema, never a peer's activity message.
        SustainedHealth.__init__(self,journal,watchdog_intent,pod_id,bindings,download_bindings,**kwargs)

    def contract(self,job):
        b=self.bindings.get(job)
        if b is None:raise EvidenceError('run job lacks its original selected contract')
        if b.get('schema')=='ovl.pilot-validation-binding.v1':return PilotHealth.contract(self,job)
        if b.get('schema')=='ovl.initialization-validation-binding.v1':return InitializationHealth.contract(self,job)
        if self.registration is None:raise EvidenceError('production contract requires authenticated registration')
        return ProductionHealth.contract(self,job)

    def _initialization(self,job):
        return self.bindings.get(job,{}).get('schema')=='ovl.initialization-validation-binding.v1'

    def _validation(self,job,value):
        if self._initialization(job):return InitializationHealth._validation(self,job,value)
        return PilotHealth._validation(self,job,value)

    def _apply(self,body):
        # The initializer must retain its original scan protocol. Ordinary pilot
        # scans bypass only that protocol discriminator, retaining every original
        # PilotHealth identity, monotonicity, selected-input and credit check.
        if body.get('kind')=='registration-publication':
            fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','registration publication cost event')
            fields(body['detail'],'observation','registration publication detail')
            value=body['detail']['observation']
            if self.registration is None or digest(self.registration)!=self.registration_root:
                raise EvidenceError('registration publication lacks its original selection')
            if any(not j['finished'] or j['selection']['kind'] in ('production-record','full-replay') for j in self.jobs.values()):
                raise EvidenceError('registration publication overlaps numerical work or follows production')
            validate_publication(value,self.registration_root,self.registration_root,value['deadline_epoch'])
            integer(value['deadline_epoch'],self.plan['input']['now_epoch']+1,self.plan['request_checkpoint_epoch'],'registration publication deadline')
            integer(body['observed_epoch'],self.plan['input']['now_epoch'],value['deadline_epoch']-1,'registration publication clock')
            if (body['schema']!='ovl.cost-activity-event.v2' or body['advances_progress'] is not True
                or body['advances_export'] is not False or body['completes'] is not False):
                raise EvidenceError('registration observation cannot grant export or completion')
            if (value['stage'] in self.registration_publications
                or any(v['deadline_epoch']!=value['deadline_epoch'] for v in self.registration_publications.values())):
                raise EvidenceError('registration publication repeats credit or changes deadline')
            self.registration_publications[value['stage']]=value
            self.progress=max(self.progress,body['observed_epoch']);return
        if body.get('kind')=='stream-validation':
            job=body.get('detail',{}).get('job_sha256')
            if self._initialization(job):raise EvidenceError('initialization requires original scan evidence')
            return PilotHealth._apply(self,body)
        if body.get('kind')=='initialization-scan' and not self._initialization(body.get('detail',{}).get('job_sha256')):
            raise EvidenceError('initializer protocol on a different selected job')
        return ProductionHealth._apply(self,body)

    def registration_activity(self,value):
        """Finite publisher transitions only; no numerical or export credit."""
        from copy import deepcopy
        value=deepcopy(value)  # Caller reuse must not mutate the durable decision in memory.
        if self.registration is None or digest(self.registration)!=self.registration_root:
            raise EvidenceError('registration publication lacks its original selection')
        if any(not j['finished'] or j['selection']['kind'] in ('production-record','full-replay') for j in self.jobs.values()):
            raise EvidenceError('registration publication overlaps numerical work or follows production')
        validate_publication(value,self.registration_root,self.registration_root,value['deadline_epoch'])
        integer(value['deadline_epoch'],self.plan['input']['now_epoch']+1,self.plan['request_checkpoint_epoch'],'registration publication deadline')
        if any(v['deadline_epoch']!=value['deadline_epoch'] for v in self.registration_publications.values()):
            raise EvidenceError('registration publication changes original deadline')
        if self.now()>=value['deadline_epoch']:raise EvidenceError('original registration publication deadline expired')
        prior=self.registration_publications.get(value['stage'])
        if prior is not None:
            if prior!=value:raise EvidenceError('registration activity identity changed')
            return False
        self.event('registration-publication',{'observation':value},progress=True);return True

    def activity(self,job,value):
        self.active(job)
        if self.jobs[job]['selection']['kind'] in ('production-record','full-replay'):
            return ProductionHealth.activity(self,job,value)
        if self._initialization(job):return InitializationHealth.activity(self,job,value)
        return SustainedHealth.activity(self,job,value)
