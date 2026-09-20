"""Local preparation recovery checkpoints; not public transformation verification.

The cache is outside prepared artifacts. Resume checks every completed stage file
and its input identity. Final acceptance still requires a fresh raw reconstruction;
an operator-local cache is not an independent trust root.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import uuid
from datetime import datetime, timezone

from .canonical import EvidenceError, digest, inventory, read_json, verify_inventory, write_json
from .schema import fields

MAX_PRESERVED_STAGES = 8

def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)

def durable_mkdir(path):
    if path.is_symlink():raise EvidenceError("symlink directory")
    if not path.exists():
        durable_mkdir(path.parent)
        path.mkdir()
        sync_directory(path);sync_directory(path.parent)
    elif not path.is_dir():raise EvidenceError("expected directory")

MANIFESTS = {"corpus":"corpus.json", "tokenizer":"tokenizer-manifest.json",
             "wikipedia":"stream.json", "conversation-selection":"selection.json",
             "conversation":"stream.json", "conversation-validation":"stream.json"}


def regular_names(directory):
    if not directory.is_dir() or directory.is_symlink():
        raise EvidenceError("preparation stage must be a regular directory")
    paths=list(directory.rglob("*"))
    if any(p.is_symlink() or (not p.is_file() and not p.is_dir()) for p in paths):
        raise EvidenceError("nonregular object in preparation stage")
    return sorted(p.relative_to(directory).as_posix() for p in paths if p.is_file())


class Stages:
    def __init__(self, output, source_sha256):
        self.output=Path(output)
        self.progress=self.output.parent/(self.output.name+"-progress")
        self.context={"schema":"ovl.preparation-recovery.v1","source_commitment_sha256":source_sha256}
        self.active=False
        self.executed=[];self.adopted=[]
        self.observation={}

    @contextmanager
    def lease(self, *, resume, admission=None):
        if self.output.is_symlink() or self.progress.is_symlink():
            raise EvidenceError("symlink preparation/recovery root")
        if not resume:
            if self.output.exists() or self.progress.exists():
                raise EvidenceError("preparation output and progress must be fresh")
            durable_mkdir(self.progress)
        elif not self.progress.is_dir():
            raise EvidenceError("preparation resume requires original progress context")
        # Lock the directory inode: unlinking a conventional lock file cannot
        # admit a second cooperative writer. The owner must not replace roots.
        fd=os.open(self.progress,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as e:raise EvidenceError("preparation already has an active writer") from e
            context=self.progress/"context.json"
            if context.is_symlink():raise EvidenceError("symlink preparation recovery context")
            if resume:
                if read_json(context)!=self.context:raise EvidenceError("preparation resume source/code/recipe differs")
            else:write_json(context,self.context)
            durable_mkdir(self.output)
            if self.output.stat().st_dev!=os.fstat(fd).st_dev:
                raise EvidenceError("preparation and recovery must share a filesystem")
            allowed=set(MANIFESTS)|{"preparation.json","admission-observation.json"}
            if any(p.name not in allowed or p.is_symlink() for p in self.output.iterdir()):
                raise EvidenceError("unexpected preparation output object")
            observation_dir=self.progress/"observations"
            durable_mkdir(observation_dir)
            self.observation={"schema":"ovl.preparation-execution.v1","invocation_id":uuid.uuid4().hex,
                "source_commitment_sha256":self.context["source_commitment_sha256"],
                "resume_requested":resume,"started_at":datetime.now(timezone.utc).isoformat(),
                "pid":os.getpid(),"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "process_start_ticks":Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],
                "admission":admission,"status":"STARTED"}
            write_json(observation_dir/(digest(self.observation)+".json"),self.observation)
            self.active=True
            succeeded=False
            try:
                yield self
                succeeded=True
            finally:
                self.observation.update(status="PASS" if succeeded else "INTERRUPTED_OR_FAILED",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    stages_executed_this_run=self.executed,stages_adopted_from_local_cache=self.adopted)
                write_json(observation_dir/(digest(self.observation)+".json"),self.observation)
        finally:
            self.active=False;os.close(fd)

    def storage_observation(self, required_bytes=0):
        recovery=self.progress/"incomplete"
        directories=[]
        if recovery.exists():
            regular_names(recovery)
            directories=[p for p in recovery.iterdir() if p.is_dir()]
        total=sum((p/n).stat().st_size for p in directories for n in regular_names(p))
        usage=shutil.disk_usage(self.output)
        value={"schema":"ovl.preparation-storage.v1","observed_at":datetime.now(timezone.utc).isoformat(),
            "preserved_directories":sorted(p.name for p in directories),"preserved_bytes":total,
            "filesystem_total_bytes":usage.total,"filesystem_free_bytes":usage.free,
            "minimum_free_percent":20,"replacement_lower_bound_bytes":required_bytes,
            "maximum_preserved_stages":MAX_PRESERVED_STAGES,
            "full_stage_capacity_guaranteed":False}
        write_json(self.progress/"observations"/(digest(value)+".json"),value)
        if len(directories)>MAX_PRESERVED_STAGES or usage.free-required_bytes < (usage.total+4)//5:
            raise EvidenceError("preparation recovery storage/headroom bound exceeded; preserve evidence")
        return value

    def run(self, name, parents, producer):
        if not self.active or name not in MANIFESTS:raise EvidenceError("stage requires active preparation lease")
        expected={"context":digest(self.context),"stage":name,"parents":parents}
        path=self.output/name;receipt=self.progress/(name+".json")
        if receipt.is_symlink():raise EvidenceError("symlink preparation stage receipt")
        if receipt.exists():
            cached=read_json(receipt)
            fields(cached,"schema inputs result files","preparation stage receipt")
            if cached["schema"]!="ovl.preparation-stage.v1" or cached["inputs"]!=expected:
                raise EvidenceError("preparation stage input ancestry differs")
            names=regular_names(path)
            if names!=sorted(e["path"] for e in cached["files"]):
                raise EvidenceError("preparation stage file set differs")
            verify_inventory(path,cached["files"])
            if read_json(path/MANIFESTS[name])!=cached["result"]:
                raise EvidenceError("preparation stage manifest/result differs")
            self.adopted.append(name)
            return cached["result"]
        storage=self.storage_observation()
        if path.exists():
            names=regular_names(path)
            partial_bytes=sum((path/n).stat().st_size for n in names)
            self.storage_observation(partial_bytes)
            if len(storage["preserved_directories"])>=MAX_PRESERVED_STAGES:
                raise EvidenceError("preparation recovery count bound exceeded; preserve evidence")
            recovery=self.progress/"incomplete"
            durable_mkdir(recovery)
            destination=recovery/(name+"-"+uuid.uuid4().hex)
            # Durable intent precedes an atomic same-filesystem rename. No copy/delete fallback.
            retained={"schema":"ovl.preparation-preservation.v1","inputs":expected,
                "source_directory":name,"preserved_directory":destination.name,
                "files":inventory(path,names)}
            write_json(recovery/(destination.name+"-intent.json"),retained)
            os.rename(path,destination)
            sync_directory(self.output);sync_directory(recovery)
            write_json(recovery/(destination.name+".json"),{**retained,"status":"PRESERVED"})
            self.storage_observation(partial_bytes)
        result=producer(path)
        if read_json(path/MANIFESTS[name])!=result:
            raise EvidenceError("preparation producer result differs from saved manifest")
        names=regular_names(path)
        # Receipt publication must follow durability of every stage output.
        for name_in_stage in names:
            fd=os.open(path/name_in_stage,os.O_RDONLY|os.O_NOFOLLOW)
            try:os.fsync(fd)
            finally:os.close(fd)
        for directory in [p for p in path.rglob("*") if p.is_dir()][::-1]+[path]:
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:os.fsync(fd)
            finally:os.close(fd)
        sync_directory(self.output)
        sync_directory(self.output.parent)
        write_json(receipt,{"schema":"ovl.preparation-stage.v1","inputs":expected,
                            "result":result,"files":inventory(path,names)})
        self.executed.append(name)
        return result
