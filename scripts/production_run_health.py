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


class ProductionRunHealth(ProductionHealth,InitializationCycleHealth):
    def __init__(self,journal,watchdog_intent,pod_id,registration,bindings,download_bindings,**kwargs):
        self.registration=registration;self.registration_root=digest(registration)
        self.publications={};self.scans={}
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
        if body.get('kind')=='stream-validation':
            job=body.get('detail',{}).get('job_sha256')
            if self._initialization(job):raise EvidenceError('initialization requires original scan evidence')
            return PilotHealth._apply(self,body)
        if body.get('kind')=='initialization-scan' and not self._initialization(body.get('detail',{}).get('job_sha256')):
            raise EvidenceError('initializer protocol on a different selected job')
        return ProductionHealth._apply(self,body)

    def activity(self,job,value):
        self.active(job)
        if self.jobs[job]['selection']['kind'] in ('production-record','full-replay'):
            return ProductionHealth.activity(self,job,value)
        if self._initialization(job):return InitializationHealth.activity(self,job,value)
        return SustainedHealth.activity(self,job,value)
