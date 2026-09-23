"""Finite stream-hash I/O liveness for independently selected numerical jobs.

No input validation, export or numerical acceptance credit. Complete manifests
must hash to the selected stream root. Journal adoption retains byte high-water
marks and process/sequence identity across row, hash and numerical observations.
"""
import re
from ovl_pipeline import schema
from ovl_pipeline.canonical import EvidenceError,digest,require_digest
from ovl_pipeline.inventory_activity import MAX_PASSES
from ovl_pipeline.schema import fields,integer


class InventoryHealth:
    def __init__(self,owner):
        self.owner=owner;self.reads={};self.credited={}

    def transition(self,job,value):
        previous=self.reads.get(job)
        if previous and (any(value.get(k)!=previous[k] for k in ('process_instance','pid'))
                         or type(value.get('sequence')) is not int or value['sequence']<=previous['sequence']):
            raise EvidenceError('activity changed inventory process or sequence')

    def validate(self,job,value):
        h=self.owner;h.active(job)
        if job in h.activities or job in h.phase_activities:raise EvidenceError('inventory read follows numerical work')
        selected,limit=h.inventory_contract(job)
        fields(value,'schema process_instance pid sequence pass_index manifest stream_sha256 read_bytes scope','inventory read activity')
        if (value['schema']!='ovl.runtime-inventory-read.v1'
                or value['scope']!='operator-supervision-only-not-training-verification'
                or type(value['process_instance']) is not str or not re.fullmatch('[0-9a-f]{32}',value['process_instance'])):
            raise EvidenceError('unsupported inventory read activity')
        integer(value['pid'],1,2**31-1,'inventory PID');integer(value['sequence'],1,2**53-1,'inventory sequence')
        integer(value['pass_index'],1,min(limit,MAX_PASSES),'inventory pass')
        manifest=value['manifest'];schema.stream(manifest);require_digest(value['stream_sha256'])
        if digest(manifest)!=value['stream_sha256'] or (value['stream_sha256'],manifest['documents']) not in selected:
            raise EvidenceError('inventory differs from independently selected stream')
        entries=manifest['files']
        if len(entries)!=3:raise EvidenceError('inventory requires exactly three stream files')
        for entry in entries:
            fields(entry,'path bytes sha256','stream file');integer(entry['bytes'],1,32*1024**3,'stream file bytes');require_digest(entry['sha256'])
        total=sum(e['bytes'] for e in entries)
        integer(value['read_bytes'],0,total,'completed hash input bytes')
        previous=self.reads.get(job)
        if previous:
            if any(value[k]!=previous[k] for k in ('process_instance','pid')):raise EvidenceError('inventory process changed')
            if value['sequence']<previous['sequence'] or value['pass_index']<previous['pass_index']:
                raise EvidenceError('inventory sequence/pass regressed')
            if value['sequence']==previous['sequence'] and value!=previous:raise EvidenceError('inventory same sequence changed')
            if value['pass_index']==previous['pass_index']:
                if value['manifest']!=previous['manifest'] or value['read_bytes']<previous['read_bytes']:
                    raise EvidenceError('inventory bytes/root regressed')
        for observations in (getattr(h,'validations',{}),getattr(h,'scans',{})):
            other=observations.get(job)
            if other and limit==1:raise EvidenceError('single-pass inventory follows row validation')
            if (other and previous and other['sequence']>previous['sequence']
                    and value['pass_index']<=previous['pass_index']):
                raise EvidenceError('inventory pass restarted after row validation')
            if other and (any(value[k]!=other[k] for k in ('process_instance','pid')) or value['sequence']<=other['sequence']):
                raise EvidenceError('inventory changed scan process or sequence')
        old=self.credited.get(job,0) if previous and previous['pass_index']==value['pass_index'] else 0
        # New pass/sequence/zero alone supplies no useful-progress credit.
        return value['read_bytes']-old>=1024**2 or value['read_bytes']==total and value['read_bytes']>old

    def activity(self,job,value):
        advances=self.validate(job,value)
        if self.reads.get(job)==value:return False
        self.owner.event('inventory-read',{'job_sha256':job,'observation':value},progress=advances)
        return advances

    def apply(self,body):
        if body.get('kind')!='inventory-read':return False
        h=self.owner
        fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','inventory cost event')
        fields(body['detail'],'job_sha256 observation','inventory cost detail')
        job=body['detail']['job_sha256'];value=body['detail']['observation'];advances=self.validate(job,value)
        integer(body['observed_epoch'],h.plan['input']['now_epoch'],h.plan['external_terminate_epoch'],'inventory cost clock')
        if (body['schema']!='ovl.cost-activity-event.v2' or body['advances_progress'] is not advances
                or body['advances_export'] is not False or body['completes'] is not False):
            raise EvidenceError('invalid inventory cost decision')
        previous=self.reads.get(job)
        if previous is None or previous['pass_index']!=value['pass_index']:self.credited[job]=0
        if advances:self.credited[job]=value['read_bytes']
        self.reads[job]=value
        if advances:h.progress=max(h.progress,body['observed_epoch'])
        return True
