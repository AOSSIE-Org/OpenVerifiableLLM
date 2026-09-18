"""Preproduction initialization scan liveness under the original rental limits.

The selected job remains development work, never production authorization.
Preserve raw scan observations in the journal and reuse the pilot's strict
stream/process/sequence checks. Only the initializer's single validator pass is
accepted; neither this observation nor discarded warmup proves regeneration.
"""
from ovl_pipeline.canonical import EvidenceError,require_digest
from ovl_pipeline.schema import fields,integer
from pilot_health import PilotHealth
from sustained_health import SustainedHealth


class InitializationHealth(PilotHealth):
    def contract(self,job):
        require_digest(job)
        if job not in self.bindings:raise EvidenceError('initialization lacks selected stream binding')
        value=self.bindings[job]
        fields(value,'schema stream_sha256 documents action','initialization validation binding')
        if (value['schema']!='ovl.initialization-validation-binding.v1'
                or value['action'] not in ('record','verify')):
            raise EvidenceError('unsupported initialization binding')
        require_digest(value['stream_sha256'])
        integer(value['documents'],1,2**53-1,'initialization document total')
        return value

    @staticmethod
    def normalized(value):
        fields(value,'schema process_instance pid sequence pass_index operation stream_sha256 documents completed_documents complete scope',
               'initialization scan')
        integer(value['pass_index'],1,1,'initialization scan pass')
        if (value['schema']!='ovl.runtime-production-scan.v1'
                or value['operation']!='stream-validation'):
            raise EvidenceError('initialization requires its single complete validator scan')
        return {**{k:v for k,v in value.items() if k not in ('pass_index','operation')},
                'schema':'ovl.runtime-stream-validation.v1'}

    def _validation(self,job,value):
        advances=super()._validation(job,value)
        # Starting an empty scan is not useful work. Its process/sequence still
        # becomes fixed, including when the coordinator adopts this journal.
        if job not in self.validations and value['completed_documents']==0:return False
        return advances

    def _apply(self,body):
        if body.get('kind')=='stream-validation':
            raise EvidenceError('initialization journal requires the original scan observation')
        if body.get('kind')!='initialization-scan':return super()._apply(body)
        detail=body.get('detail',{})
        fields(detail,'job_sha256 observation','initialization scan detail')
        normalized=self.normalized(detail['observation'])
        return super()._apply({**body,'kind':'stream-validation',
                              'detail':{**detail,'observation':normalized}})

    def activity(self,job,observation):
        self.active(job)
        if self.jobs[job]['selection']['kind']!='pilot':return super().activity(job,observation)
        if type(observation) is dict and observation.get('schema')=='ovl.runtime-production-scan.v1':
            normalized=self.normalized(observation)
            advances=self._validation(job,normalized)
            if self.validations.get(job)==normalized:return False
            self.event('initialization-scan',{'job_sha256':job,'observation':observation},progress=advances)
            return advances
        if type(observation) is dict and observation.get('schema')=='ovl.runtime-stream-validation.v1':
            raise EvidenceError('initialization requires the original scan protocol')
        return super().activity(job,observation)


class InitializationCycleHealth(InitializationHealth,SustainedHealth):
    """Add existing pinned download liveness to the initialization scan protocol.

    SustainedHealth owns download contracts; InitializationHealth owns numerical
    contracts and raw initializer scans. Both retain the same base cost journal.
    """
