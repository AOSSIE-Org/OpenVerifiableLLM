"""One complete initial safe-state snapshot during a sustained development job.

Final whole-output retention remains mandatory. This only keeps its original
1800-second durable-export guard supplied with actual checked off-pod state,
never a checkpoint timestamp or a peer's PASS field. Failed copies are retained.
"""
from pathlib import Path

from ovl_pipeline.canonical import EvidenceError,canonical,digest,read_json,require_digest,sha256,verify_inventory
from ovl_pipeline.schema import fields,integer,control as validate_control
from ovl_pipeline.state import read_state,unpack
from pod_job_client import save_once
from pod_versioned_export import export
from run_workload_stage import remote_name


def check_state(directory,expected,phase):
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():raise EvidenceError('regular pilot checkpoint directory required')
    if {p.name for p in directory.iterdir()}!={'checkpoint.json','state.json','state.safetensors'}:
        raise EvidenceError('initial pilot snapshot must retain the complete safe state')
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):raise EvidenceError('unsafe pilot snapshot entry')
    if read_json(directory/'checkpoint.json')!=expected:raise EvidenceError('initial pilot snapshot marker changed')
    metadata,tensors=read_state(directory,expected)
    control=unpack(metadata['tree'],tensors)['control']
    numerical=dict(control);cycle=numerical.pop('pilot_cycle',None);validate_control(numerical)
    integer(cycle,0,0,'initial pilot cycle')
    if (control['global_step']!=0 or control['phase_step']!=0 or control['cursor']!=0
        or control['phase']!=phase or control.get('pilot_cycle')!=0):
        raise EvidenceError('selected initial pilot snapshot has advanced or foreign control')
    return expected['state_root']


class InitialRetention:
    def __init__(self,transport,health,job,job_sha256,selection,output,health_file,store):
        fields(selection,'schema mode output_root phase maximum_initial_bytes','initial pilot retention selection')
        if (selection['schema']!='ovl.pilot-initial-retention.v1' or selection['mode'] not in ('record','replay','resume')
            or selection['phase'] not in ('wikipedia','conversation')):
            raise EvidenceError('invalid initial pilot retention selection')
        integer(selection['maximum_initial_bytes'],1,2**40,'initial pilot snapshot bound')
        if digest(job)!=job_sha256 or job['kind']!='pilot' or selection['output_root'] not in job['export_roots']:
            raise EvidenceError('initial retention differs from selected pilot outputs')
        self.transport=transport;self.health=health;self.job=job_sha256;self.selection=selection
        self.output=Path(output);self.output.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.health_file=health_file;self.store=Path(store)
        self._verified_marker=None
        prefix='boundary' if selection['mode']=='record' else 'verifier-boundary'
        self.name=remote_name(transport,selection['output_root'])+'/'+prefix+'-00000'
        self.marker=self.name+'/checkpoint.json'
        save_once(self.output/'selection.json',{'job_sha256':job_sha256,'selection':selection,'profile_sha256':digest(transport.profile)})

    def observation_paths(self):return [self.marker]

    def observe(self,values):
        marker=values[self.marker]
        if self._verified_marker is not None:
            if marker!=self._verified_marker:raise EvidenceError('initial pilot checkpoint changed or disappeared')
            return False
        if marker is None:return False
        fields(marker,'schema state_root files','observed initial checkpoint')
        if marker['schema']!='ovl.checkpoint.v1':raise EvidenceError('unknown initial checkpoint format')
        require_digest(marker['state_root'])
        if type(marker['files']) is not list or len(marker['files'])!=2:
            raise EvidenceError('complete bounded checkpoint inventory required')
        for f in marker['files']:
            fields(f,'path bytes sha256','initial checkpoint file');require_digest(f['sha256'])
            integer(f['bytes'],1,self.selection['maximum_initial_bytes'],'selected initial checkpoint bytes')
        if [f['path'] for f in marker['files']]!=['state.json','state.safetensors']:
            raise EvidenceError('closed sorted safe checkpoint file names required')
        encoded=canonical(marker)
        expected_files=[{'path':'checkpoint.json','bytes':len(encoded),'sha256':sha256(encoded)},*marker['files']]
        if sum(f['bytes'] for f in expected_files)>self.selection['maximum_initial_bytes']:
            raise EvidenceError('initial checkpoint exceeds selected budget before transfer')
        receipt_path=self.output/'retained.json'
        if receipt_path.exists():
            retained=read_json(receipt_path)
            fields(retained,'schema marker receipt_path receipt_sha256 state_root','initial pilot retention receipt')
            if retained['schema']!='ovl.pilot-initial-retention-receipt.v1' or retained['marker']!=marker:
                raise EvidenceError('initial pilot checkpoint changed after retention')
            receipt=read_json(Path(retained['receipt_path']))
            if digest(receipt)!=retained['receipt_sha256']:raise EvidenceError('retained pilot snapshot receipt changed')
        else:
            # At most one fresh retry; no replacement of a complete receipt.
            dest=self.output/'snapshot-000'
            if dest.exists() and not (dest/'export.json').exists():dest=self.output/'snapshot-001'
            if dest.exists():
                if not (dest/'export.json').exists():raise EvidenceError('initial pilot snapshot retries exhausted')
                receipt=read_json(dest/'export.json')
            else:
                def progress(operation,counts,total):
                    self.health.bytes(operation,counts,total=total);self.health.write(self.health_file)
                receipt=export(self.transport,self.name,self.store,dest,self.health.plan['external_terminate_epoch'],
                               progress=progress,expected_files=expected_files)
            retained={'schema':'ovl.pilot-initial-retention-receipt.v1','marker':marker,
                      'receipt_path':str((dest/'export.json').resolve()),'receipt_sha256':digest(receipt),'state_root':marker['state_root']}
        if (receipt['schema']!='ovl.offpod-versioned-tree-export.v1' or receipt['result']!='PASS'
            or receipt['pod_id']!=self.health.pod or receipt['profile_sha256']!=digest(self.transport.profile)
            or receipt['root']!=self.name or receipt['numerical_verification']!='NOT_RUN'):
            raise EvidenceError('initial pilot snapshot selects different peer/output')
        if receipt['files']!=expected_files:raise EvidenceError('retained initial snapshot file selection differs')
        if sum(f['bytes'] for f in receipt['files'])>self.selection['maximum_initial_bytes']:
            raise EvidenceError('initial pilot snapshot exceeds selected budget; copies preserved')
        directory=Path(receipt['files_directory']);verify_inventory(directory,receipt['files'])
        state_root=check_state(directory,marker,self.selection['phase'])
        if state_root!=retained['state_root']:raise EvidenceError('retained initial state changed')
        save_once(receipt_path,retained)
        credited=self.health.exported_files(self.job,directory,receipt['files'])
        self.health.write(self.health_file)
        self._verified_marker=marker
        return credited
