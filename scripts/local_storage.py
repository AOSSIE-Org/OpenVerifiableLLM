"""Byte-based local headroom; never changes evidence or numerical acceptance.

The caller pins a workload budget before creation. Runtime observations request
ordinary bounded shutdown while an explicit export allowance remains available.
Existing objects, failed partials and unrelated files are never deleted here.
"""
from pathlib import Path
import shutil
from ovl_pipeline.canonical import EvidenceError, digest, read_json, require_digest
from ovl_pipeline.schema import fields, integer

HOST_RESERVE_BYTES = 8 * 1024**3


def require_space(path, additional_bytes, *, disk_usage=None):
    integer(additional_bytes, 0, 2**50, 'incremental storage bytes')
    usage = (disk_usage or shutil.disk_usage)(path)
    if usage.free < additional_bytes + HOST_RESERVE_BYTES:
        raise EvidenceError('insufficient workload storage headroom; preserve evidence')
    return usage.free


class Budget:
    def __init__(self, path, expected):
        require_digest(expected)
        value = read_json(Path(path))
        if digest(value) != expected:
            raise EvidenceError('local storage budget pin differs')
        fields(value, 'schema directory device peak_incremental_bytes host_reserve_bytes shutdown_export_bytes observation_slack_bytes basis_sha256', 'local storage budget')
        if value['schema'] != 'ovl.local-storage-budget.v1':
            raise EvidenceError('unsupported local storage budget')
        require_digest(value['basis_sha256'])
        for key in ('device', 'peak_incremental_bytes', 'host_reserve_bytes', 'shutdown_export_bytes', 'observation_slack_bytes'):
            integer(value[key], 1, 2**50, key)
        if (value['host_reserve_bytes'] < HOST_RESERVE_BYTES
                or value['peak_incremental_bytes'] < value['shutdown_export_bytes'] + value['observation_slack_bytes']):
            raise EvidenceError('storage budget omits operational or shutdown headroom')
        if type(value['directory']) is not str:
            raise EvidenceError('explicit local storage directory required')
        self.directory = Path(value['directory'])
        if not self.directory.is_absolute() or '..' in self.directory.parts:
            raise EvidenceError('absolute local storage directory required')
        self.value = value
        self.free()

    def free(self):
        if (any(p.is_symlink() for p in [self.directory, *self.directory.parents])
                or not self.directory.is_dir() or self.directory.stat().st_dev != self.value['device']):
            raise EvidenceError('local storage filesystem identity changed')
        return shutil.disk_usage(self.directory).free

    def admit(self):
        available = self.free()
        if available < self.value['peak_incremental_bytes'] + self.value['host_reserve_bytes']:
            raise EvidenceError('complete incremental workload does not fit local storage')
        return available

    def covers(self, path):
        path = Path(path).absolute()
        if '..' in path.parts:
            raise EvidenceError('storage destination outside normalized path boundary')
        if not path.is_relative_to(self.directory):
            raise EvidenceError('storage destination outside selected filesystem root')
        for candidate in [path, *path.parents]:
            if candidate.is_symlink():raise EvidenceError('storage destination symlink')
            if candidate.exists():
                if candidate.stat().st_dev != self.value['device']:
                    raise EvidenceError('storage destination filesystem differs')
            if candidate==self.directory:break
        self.free()

    def shutdown_needed(self):
        return self.observation()['shutdown_needed']

    def observation(self):
        free = self.free()
        threshold = sum(self.value[k] for k in (
            'host_reserve_bytes', 'shutdown_export_bytes', 'observation_slack_bytes'))
        return {'free_bytes':free, 'shutdown_threshold_bytes':threshold,
                'shutdown_needed':free < threshold}
