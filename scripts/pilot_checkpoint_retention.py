"""Deliver every pilot checkpoint before acknowledging further GPU updates."""
from pathlib import Path

from ovl_pipeline import pilot_delivery
from ovl_pipeline.canonical import EvidenceError,canonical,digest,read_json,sha256,verify_inventory
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.state import read_state,unpack
from pod_job_client import save_once
from pod_versioned_export import export
from run_workload_stage import remote_name


def selection(value):
    fields(value,'schema session mode output_root phase maximum_checkpoint_bytes copy_timeout_seconds maximum_uncached_export_bytes binding','checkpoint retention selection')
    if value['schema']!='ovl.pilot-checkpoint-retention.v1':raise EvidenceError('wrong pilot checkpoint retention schema')
    pilot_delivery.policy(selected_policy(value,1))
    integer(value['maximum_uncached_export_bytes'],1,2**40,'uncached terminal export bytes')
    fields(value['binding'],'schema recipe_sha256 kernel_sha256 stream_sha256 code_root','selected pilot binding')
    if value['binding']['schema']!='ovl.pilot-record-parent-binding.v1':raise EvidenceError('wrong pilot binding')
    from ovl_pipeline.canonical import require_digest
    for k in ('recipe_sha256','kernel_sha256','stream_sha256','code_root'):require_digest(value['binding'][k])
    return value


def selected_policy(value,deadline):
    return {'schema':'ovl.pilot-delivery-policy.v1','session':value['session'],'mode':value['mode'],
            'phase':value['phase'],'deadline_epoch':deadline,'copy_timeout_seconds':value['copy_timeout_seconds'],
            'maximum_checkpoint_bytes':value['maximum_checkpoint_bytes']}


def command_selection(job,value,deadline):
    """Require exact command flags, including the original worker deadline."""
    argv=job['argv']
    if argv.count('--')!=1:raise EvidenceError('explicit pilot command separator required')
    args=argv[argv.index('--')+1:]
    if not args or args[0]!=value['mode']:raise EvidenceError('delivery mode differs from command')
    expected={'--delivery-session':value['session'],'--delivery-deadline':str(deadline),
              '--delivery-timeout':str(value['copy_timeout_seconds']),
              '--delivery-maximum-bytes':str(value['maximum_checkpoint_bytes']),'--output':value['output_root']}
    for flag,want in expected.items():
        if args.count(flag)!=1 or args[args.index(flag)+1:args.index(flag)+2]!=[want]:
            raise EvidenceError('delivery command differs from selected '+flag)
    if '--resume-from' in args:raise EvidenceError('bounded delivery supports full replay only')
    if value['mode']=='replay':
        if args.count('--expected-record-sha256')!=1:raise EvidenceError('selected replay digest required')
        found=args[args.index('--expected-record-sha256')+1:args.index('--expected-record-sha256')+2]
        if len(found)!=1:raise EvidenceError('missing replay parent')
        return found[0]
    return None


class CheckpointRetention:
    def __init__(self,transport,health,job,job_sha256,value,output,health_file,store):
        selection(value)
        if digest(job)!=job_sha256 or job['kind']!='pilot' or value['output_root'] not in job['export_roots']:
            raise EvidenceError('checkpoint retention differs from pilot output')
        record=command_selection(job,value,job['deadline_epoch'])
        self.transport=transport;self.health=health;self.job=job_sha256;self.selection=value
        self.output=Path(output);self.output.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.health_file=health_file;self.store=Path(store)
        self.policy=selected_policy(value,job['deadline_epoch'])
        self.origin=pilot_delivery.origin(value['binding'],record)
        self.root=remote_name(transport,value['output_root']);self.marker=self.root+'/delivery/request.json'
        self.index=0;self.previous=digest(self.policy);self.last=None
        save_once(self.output/'selection.json',{'job_sha256':job_sha256,'selection':value,'profile_sha256':digest(transport.profile)})
        # Re-read retained complete bytes before adopting prior acknowledgements.
        while (self.output/f'checkpoint-{self.index:05d}'/'ack.json').exists():
            dest=self.output/f'checkpoint-{self.index:05d}';request=read_json(dest/'request.json')
            pilot_delivery.validate_request(request,self.policy,self.origin,self.index,self.previous)
            self._check_retained(dest,request)
            ack=read_json(dest/'ack.json');pilot_delivery.check_ack(ack,request)
            if ack['receipt_sha256']!=digest(read_json(dest/'retained.json')):raise EvidenceError('retained acknowledgement receipt changed')
            self.previous=digest(request);self.last=request;self.index+=1

    def observation_paths(self):return [self.marker]

    def _check_retained(self,dest,request):
        retained=read_json(dest/'retained.json')
        fields(retained,'schema job_sha256 request_sha256 receipt receipt_sha256','pilot retained checkpoint')
        if (retained['schema']!='ovl.pilot-retained-checkpoint.v1' or retained['job_sha256']!=self.job
            or retained['request_sha256']!=digest(request)):raise EvidenceError('wrong retained delivery identity')
        if Path(retained['receipt']) not in [(dest/name/'export.json').resolve() for name in ('snapshot-000','snapshot-001')]:
            raise EvidenceError('retained receipt outside selected checkpoint attempts')
        receipt=read_json(Path(retained['receipt']))
        if digest(receipt)!=retained['receipt_sha256']:raise EvidenceError('changed retained delivery receipt')
        expected=self._files(request)
        if (receipt['schema']!='ovl.offpod-versioned-tree-export.v1' or receipt['result']!='PASS'
            or receipt['pod_id']!=self.health.pod or receipt['profile_sha256']!=digest(self.transport.profile)
            or receipt['root']!=self.root+'/'+request['path'] or receipt['files']!=expected
            or receipt['numerical_verification']!='NOT_RUN'):raise EvidenceError('retained delivery peer or inventory differs')
        directory=Path(receipt['files_directory'])
        if any(p.is_symlink() for p in [directory,*directory.absolute().parents]):raise EvidenceError('unsafe retained state directory')
        if {p.name for p in directory.iterdir()}!={'checkpoint.json','state.json','state.safetensors'}:
            raise EvidenceError('complete retained safe state required')
        verify_inventory(directory,expected)
        if read_json(directory/'checkpoint.json')!=request['checkpoint']:raise EvidenceError('retained marker changed')
        metadata,tensors=read_state(directory,request['checkpoint'])
        if unpack(metadata['tree'],tensors)['control']!=request['control']:raise EvidenceError('retained control differs')
        return retained,directory,expected

    @staticmethod
    def _files(request):
        encoded=canonical(request['checkpoint'])
        return [{'path':'checkpoint.json','bytes':len(encoded),'sha256':sha256(encoded)},*request['checkpoint']['files']]

    def _ack(self,dest,request):
        retained,directory,files=self._check_retained(dest,request)
        self.health.exported_files(self.job,directory,files);self.health.write(self.health_file)
        ack=pilot_delivery.acknowledgement(request,digest(retained))
        save_once(dest/'ack.json',ack)
        if self.health.now()>=request['copy_deadline_epoch']:raise EvidenceError('checkpoint acknowledgement deadline expired')
        # Same bytes may be redelivered after a lost SSH response. No deadline
        # changes, new computation, or durable-export age credit is granted.
        result=self.transport.put(self.root+f'/delivery/ack-{request["index"]:05d}.json',dest/'ack.json',request['copy_deadline_epoch'])
        if result['sha256']!=digest(ack) or result['bytes_sent']!=len(canonical(ack)):raise EvidenceError('acknowledgement delivery differs')
        save_once(dest/'ack-delivery.json',{'schema':'ovl.pilot-ack-delivery.v1','request_sha256':digest(request),'ack_sha256':digest(ack)})

    def observe(self,values):
        request=values[self.marker]
        if request is None:
            if self.last is not None:raise EvidenceError('acknowledged delivery request disappeared')
            return False
        if self.last is not None and request==self.last:
            dest=self.output/f'checkpoint-{request["index"]:05d}'
            if not (dest/'ack-delivery.json').exists():self._ack(dest,request)
            return False
        pilot_delivery.validate_request(request,self.policy,self.origin,self.index,self.previous)
        if self.last is not None and request['control']['global_step']<=self.last['control']['global_step']:
            raise EvidenceError('pilot delivery regressed')
        now=self.health.now();deadline=request['copy_deadline_epoch']
        if not now<deadline<=min(self.policy['deadline_epoch'],now+self.policy['copy_timeout_seconds']):
            raise EvidenceError('invalid or expired fixed checkpoint deadline')
        dest=self.output/f'checkpoint-{self.index:05d}';dest.mkdir(mode=0o700,exist_ok=True)
        save_once(dest/'request.json',request)
        if not (dest/'retained.json').exists():
            snapshot=dest/'snapshot-000'
            if snapshot.exists() and not(snapshot/'export.json').exists():snapshot=dest/'snapshot-001'
            if snapshot.exists():
                if not(snapshot/'export.json').exists():raise EvidenceError('checkpoint transfer retries exhausted')
                receipt=read_json(snapshot/'export.json')
            else:
                def progress(operation,counts,total):
                    self.health.bytes(operation,counts,total=total);self.health.write(self.health_file)
                receipt=export(self.transport,self.root+'/'+request['path'],self.store,snapshot,deadline,
                               progress=progress,expected_files=self._files(request))
            save_once(dest/'retained.json',{'schema':'ovl.pilot-retained-checkpoint.v1','job_sha256':self.job,
                'request_sha256':digest(request),'receipt':str((snapshot/'export.json').resolve()),'receipt_sha256':digest(receipt)})
        self._ack(dest,request)
        self.previous=digest(request);self.last=request;self.index+=1
        return True
