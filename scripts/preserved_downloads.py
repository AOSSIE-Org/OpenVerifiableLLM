"""Compare failure backups with every bound, previously observed immutable byte.

Local controller records are trusted inputs, not peer attestation. These checks
grant neither numerical acceptance nor another checkpoint acknowledgement.
"""
from pathlib import Path

from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,read_json,require_digest
from ovl_pipeline.schema import fields,integer
from pod_transfer import relative


def pin(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in [path,*path.parents]) or not path.is_file():
        raise EvidenceError('unsafe preserved download observation')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)}


def observations(store,roots,profile):
    """Freeze pre-existing partials; a missing binding is a strict refusal."""
    incoming=Path(store)/'incoming';result=[]
    selections=Path(store)/'selections'
    if selections.exists():
        if selections.is_symlink():raise EvidenceError('unsafe preserved selections')
        for path in sorted(selections.iterdir()):
            record=pin(path);selected=read_json(path)
            fields(selected,'schema profile_sha256 files','preserved tree selection')
            if (selected['schema']!='ovl.preserved-tree-selection.v1' or path.name!=digest(selected)+'.json'
                or selected['profile_sha256']!=digest(profile)):raise EvidenceError('preserved tree selection changed')
            for item in selected['files']:
                if any(item['path'].startswith(root+'/') for root in roots):
                    result.append({'expected':item,'records':[record],'parts':[]})
    if not incoming.exists():return result
    if incoming.is_symlink():raise EvidenceError('unsafe incoming store')
    for attempt in sorted(incoming.iterdir()):
        if attempt.is_symlink() or not attempt.is_dir():raise EvidenceError('unsafe incoming attempt')
        partial=attempt/'verified.partial';ranges=attempt/'verified.partial.ranges'
        completed=attempt/'verified'
        if not partial.exists() and not partial.is_symlink() and not ranges.exists() and not completed.exists():continue
        selected=pin(attempt/'download-selection.json');value=read_json(Path(selected['path']))
        fields(value,'schema profile_sha256 expected deadline_epoch','preserved download selection')
        item=value['expected'];fields(item,'path bytes sha256','preserved expected file')
        relative(item['path']);integer(item['bytes'],0,2**40,'preserved file size');require_digest(item['sha256'])
        if value['schema']!='ovl.preserved-download-selection.v1' or value['profile_sha256']!=digest(profile):
            raise EvidenceError('preserved download peer differs')
        if not any(item['path'].startswith(root+'/') for root in roots):continue
        parts=[];records=[selected]
        if completed.exists() or completed.is_symlink():parts.append({'offset':0,'file':pin(completed)})
        if partial.exists() or partial.is_symlink():parts.append({'offset':0,'file':pin(partial)})
        if ranges.exists() or ranges.is_symlink():
            if ranges.is_symlink() or not ranges.is_dir():raise EvidenceError('unsafe preserved ranges')
            seen=set()
            for receipt_file in sorted(ranges.glob('attempt-*.json')):
                record=pin(receipt_file);records.append(record);seen.add(receipt_file.name)
                receipt=read_json(receipt_file)
                if receipt.get('result')=='FAILED':
                    fields(receipt,'offset bytes_requested bytes_received saved_bytes deadline_epoch result error_type partial_sha256','failed range receipt')
                    if receipt['error_type']!='TransientTransportError':raise EvidenceError('strict preserved range failure forbids backup recovery')
                    part=pin(receipt_file.with_suffix('.partial'));seen.add(Path(part['path']).name)
                    if part['sha256']!=receipt['partial_sha256'] or part['bytes']!=receipt['saved_bytes']:
                        raise EvidenceError('preserved failed range changed')
                    parts.append({'offset':receipt['offset'],'file':part})
                elif receipt.get('result')!='TRANSFERRED_NOT_YET_WHOLE_FILE_VERIFIED':
                    raise EvidenceError('unknown preserved range outcome')
            if {p.name for p in ranges.iterdir()}!=seen:raise EvidenceError('unbound preserved range bytes')
        for part in parts:
            integer(part['offset'],0,item['bytes'],'preserved offset')
            if part['offset']+part['file']['bytes']>item['bytes']:raise EvidenceError('preserved bytes exceed selection')
        result.append({'expected':item,'records':records,'parts':parts})
    return result


def compare(saved,exports,check_deadline):
    selected={root['remote_root']+'/'+item['path']:(root,item) for root in exports for item in root['files']}
    for observation in saved:
        item=observation['expected'];name=item['path']
        if name not in selected:raise EvidenceError('preserved download disappeared from terminal export')
        root,actual=selected[name]
        if {**actual,'path':name}!=item:raise EvidenceError('terminal inventory contradicts preserved selection')
        final=confined(Path(root['directory']),actual['path'])
        for record in observation['records']:
            check_deadline()
            if pin(record['path'])!=record:raise EvidenceError('preserved observation record changed')
        for part in observation['parts']:
            record=part['file'];check_deadline()
            if pin(record['path'])!=record:raise EvidenceError('preserved partial changed')
            with Path(record['path']).open('rb') as old,final.open('rb') as new:
                new.seek(part['offset'])
                while True:
                    check_deadline();data=old.read(1024*1024)
                    if not data:break
                    if new.read(len(data))!=data:raise EvidenceError('terminal retention contradicts preserved immutable bytes')
            if pin(record['path'])!=record:raise EvidenceError('preserved partial changed during comparison')
