"""Executable synthetic lifecycle with durable recovery and strict local scope.

This adapter performs real preparation, model updates, replay and inference. It
does not rent compute, publish anything or admit a production training run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import uuid

from nacl.signing import SigningKey

from .canonical import (EvidenceError, atomic_write, confined, digest, file_hash,
                        inventory, read_json, verify_inventory, write_json)
from .fixture import prepare, recipe, export_model, infer, verify_fixture
from .lifecycle import Journal, exclusive, durable_mkdir, sync_directory
from .lifecycle_artifacts import file_names, snapshot, check_snapshot, durable_tree
from .state import read_state
from .training import (TrustPolicy, code_root, environment, configure, full_replay,
                       make_registration, signed, train, validate_chain,
                       validate_registration, verify_signed)


SCOPE = "local-synthetic-fixture"
STAGES = ("prepare", "initialize", "record", "replay", "export", "download", "verify", "close")


class FixtureLifecycle:
    def __init__(self, source: Path, root: Path, *, event=lambda _: None):
        self.source, self.root = Path(source).resolve(), Path(root).resolve()
        if self.root == self.source or self.root.is_relative_to(self.source) or self.source.is_relative_to(self.root):
            raise EvidenceError("source and execution directories must be disjoint")
        self.event = event
        self.objects = self.root / "objects"
        self.private = self.root / "private"
        self.recovery = self.private / "recovery"
        self.policy_path = self.private / "trust.json"
        self.raw_inventory = inventory(self.source, ["wiki.xml", "conversations.json"])

    def object(self, name):
        if name not in STAGES:
            raise EvidenceError("unknown lifecycle stage")
        return confined(self.objects, name)

    def transaction(self, name, operation, build):
        """Adopt a durable receipt, or preserve and rebuild an incomplete stage."""
        target = self.object(name)
        marker = self.private / (name + ".receipt.json")
        if marker.exists():
            record = read_json(marker)
            if set(record) != {"operation", "output"} or record["operation"] != operation:
                raise EvidenceError("stage receipt operation identity mismatch")
            receipt = record["output"]
            check_snapshot(target, receipt)
            return receipt
        # No completion marker means no acceptance; retain all incomplete bytes.
        if target.exists():
            durable_mkdir(self.recovery)
            os.rename(target, self.recovery / (name + "-" + uuid.uuid4().hex))
            sync_directory(self.recovery)
            sync_directory(self.objects)
        pending = self.private / (name + ".pending-" + uuid.uuid4().hex)
        durable_mkdir(pending)
        build(pending)
        durable_tree(pending)
        receipt = snapshot(pending)
        self.event(name + ":built")
        os.rename(pending, target)
        sync_directory(self.private)
        fd = os.open(self.objects, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.event(name + ":renamed")
        write_json(marker, {"operation": operation, "output": receipt})
        return receipt

    def policy(self):
        return TrustPolicy(**read_json(self.policy_path))

    def registration(self):
        return verify_signed(read_json(self.object("initialize") / "registration.json"), self.policy().run_public_key_hex)

    def streams(self):
        return {p: self.object("prepare") / "prepared" / p for p in ("wikipedia", "conversation")}

    def key(self):
        path = self.private / "run-key"
        if not path.exists():
            atomic_write(path, bytes(SigningKey.generate()))
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise EvidenceError("signing key must be a private regular file")
        return SigningKey(path.read_bytes())

    def prepare(self, _):
        def build(out):
            raw = out / "raw"
            raw.mkdir()
            for e in self.raw_inventory:
                shutil.copyfile(confined(self.source, e["path"]), raw / e["path"])
            verify_inventory(raw, self.raw_inventory)
            write_json(out / "source.json", {"schema": "ovl.source.v1", "scope": "synthetic", "files": self.raw_inventory})
            write_json(out / "preparation.json", prepare(raw, out / "prepared"))
        return self.transaction("prepare", _, build)

    def initialize(self, _):
        def build(out):
            prepared = read_json(self.object("prepare") / "preparation.json")
            key = self.key()
            args = (recipe(prepared["tokenizer"]["vocab_size"]), prepared["streams"], key,
                    digest(self.raw_inventory), digest(prepared))
            registration = make_registration(*args)
            if make_registration(*args) != registration:
                raise EvidenceError("initial-state regeneration differs")
            policy = asdict(TrustPolicy(digest(registration), bytes(key.verify_key).hex(), SCOPE))
            if self.policy_path.exists() and read_json(self.policy_path) != policy:
                raise EvidenceError("external trust policy differs from initialization")
            write_json(self.policy_path, policy)
            write_json(out / "registration.json", signed(registration, key))
        result = self.transaction("initialize", _, build)
        self.policy().validate(self.registration())
        return result

    def record(self, _):
        target = self.object("record")
        r, policy, streams = self.registration(), self.policy(), self.streams()
        validate_registration(r, policy, streams)
        if target.exists():
            # Process death can bypass atomic_write's finally clause. Preserve
            # only its reserved temporary namespace outside canonical artifacts.
            for temporary in sorted(target.rglob(".pending-*")):
                if temporary.is_symlink() or not temporary.is_file():
                    raise EvidenceError("invalid interrupted atomic-write entry")
                destination = self.recovery / "atomic-writes" / uuid.uuid4().hex / temporary.relative_to(target)
                durable_mkdir(destination.parent)
                os.rename(temporary, destination)
                sync_directory(destination.parent)
                sync_directory(temporary.parent)
        chain_path = target / "chain.json"
        if chain_path.exists():
            chain = read_json(chain_path)
            if set(chain) != {"schema", "complete", "boundaries"} or chain["schema"] != "ovl.chain.v1" or type(chain["complete"]) is not bool:
                raise EvidenceError("invalid record journal")
            validate_chain(r, policy, chain["boundaries"], complete=chain["complete"])
            for env in chain["boundaries"]:
                b = env["body"]
                read_state(confined(target, b["checkpoint_path"]), b["checkpoint"])
            if chain["complete"]:
                durable_tree(target)
                sync_directory(self.objects)
                return snapshot(target)
        # A real committed intermediate boundary provides a crash/resume seam.
        if not target.exists():
            chain = train(r, policy, streams, target, self.key(), stop_after=3,
                          recovery_directory=self.recovery / "record")
            self.event("record:boundary")
            if chain[-1]["body"]["kind"] == "final":
                durable_tree(target)
                sync_directory(self.objects)
                return snapshot(target)
        train(r, policy, streams, target, self.key(), resume=True,
              recovery_directory=self.recovery / "record")
        durable_tree(target)
        sync_directory(self.objects)
        if self.recovery.exists():
            durable_tree(self.recovery)
        return snapshot(target)

    def replay(self, _):
        def build(out):
            # Reconstruct every fixture transformation from raw inputs anew.
            rebuilt = prepare(self.object("prepare") / "raw", out / "prepared")
            r = self.registration()
            if digest(rebuilt) != r["preparation_root"] or rebuilt["streams"] != r["streams"]:
                raise EvidenceError("full raw reconstruction differs")
            expected = snapshot(self.object("prepare") / "prepared")
            check_snapshot(out / "prepared", expected)
            report = full_replay(r, self.policy(), {p: out / "prepared" / p for p in r["streams"]}, self.object("record"))
            write_json(out / "replay.json", report)
        return self.transaction("replay", _, build)

    def export(self, _):
        def build(out):
            p = self.object("prepare")
            for name in ("raw", "prepared"):
                shutil.copytree(p / name, out / name)
            for name in ("source.json", "preparation.json"):
                shutil.copyfile(p / name, out / name)
            shutil.copyfile(self.object("initialize") / "registration.json", out / "registration.json")
            shutil.copytree(self.object("record"), out / "training")
            replay = read_json(self.object("replay") / "replay.json")
            write_json(out / "replay.json", replay)
            r = self.registration()
            chain = read_json(out / "training/chain.json")["boundaries"]
            for phase, kind in (("base", "base"), ("chat", "final")):
                b = next(e["body"] for e in chain if e["body"]["kind"] == kind)
                root = export_model(r, b, out / "training", out / phase, out / "prepared/tokenizer", phase)
                if root != replay[phase + "_model_root"]:
                    raise EvidenceError("export differs from continuous replay")
                write_json(out / phase / "inference.json", infer(out / phase))
            release = {"schema": "ovl.fixture-release.v1", "scope": SCOPE,
                       "registration": self.policy().registration_sha256,
                       "chain_root": replay["chain_root"], "files": inventory(out, file_names(out))}
            write_json(out / "release.json", signed(release, self.key()))
        return self.transaction("export", _, build)

    def download(self, _):
        # An exact clean filesystem copy is deliberately labeled non-public.
        def build(out):
            src = self.object("export")
            for name in file_names(src):
                target = confined(out, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(confined(src, name), target)
            check_snapshot(out, snapshot(src))
        return self.transaction("download", _, build)

    def verify(self, _):
        def build(out):
            result = verify_fixture(self.object("download"), self.policy())
            result["transport_scope"] = "local-copy-not-public-download"
            write_json(out / "verification.json", result)
        return self.transaction("verify", _, build)

    def close(self, _):
        def build(out):
            verified = read_json(self.object("verify") / "verification.json")
            if verified["result"] != "PASS" or verified["production_acceptance"] != "NOT_RUN":
                raise EvidenceError("unexpected verification scope")
            write_json(out / "closure.json", {"schema": "ovl.fixture-closure.v1", "result": "PASS",
                       "scope": SCOPE, "verification_sha256": digest(verified),
                       "paid_resources_created": [], "publications": [], "production_acceptance": "NOT_RUN"})
        return self.transaction("close", _, build)

    def validate_stage(self, name, receipt, operation):
        check_snapshot(self.object(name), receipt)
        if name != "record":
            if read_json(self.private / (name + ".receipt.json")) != {"operation": operation, "output": receipt}:
                raise EvidenceError("stage receipt differs from operation journal")
        if name == "initialize":
            registration = self.registration()
            self.policy().validate(registration)
            if registration["raw_root"] != digest(self.raw_inventory):
                raise EvidenceError("registration raw ancestry differs from current lifecycle")
            if registration["preparation_root"] != digest(read_json(self.object("prepare") / "preparation.json")):
                raise EvidenceError("registration preparation ancestry differs")

    def run(self, *, through="close"):
        if through not in STAGES:
            raise EvidenceError("invalid final stage")
        with exclusive(self.private):
            if not (self.private / "journal.json").exists() and self.objects.exists() and any(self.objects.iterdir()):
                raise EvidenceError("missing journal beside prior lifecycle objects")
            durable_mkdir(self.objects)
            configure(1234)
            identity = {"scope": SCOPE, "raw": self.raw_inventory, "code_root": code_root(), "environment": environment()}
            journal = Journal(self.private, identity)
            parent = digest(identity)
            for name in STAGES:
                operation = journal.operation(name, {"parent": parent})["id"]
                result = journal.local(name, {"parent": parent}, getattr(self, name),
                                       lambda receipt, n=name, op=operation: self.validate_stage(n, receipt, op), event=self.event)
                parent = digest(result)
                if name == through:
                    return {"completed_through": name, "stage_root": parent, "scope": SCOPE,
                            "production_acceptance": "NOT_RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--through", choices=STAGES, default="close")
    args = parser.parse_args()
    task = FixtureLifecycle(args.source, args.run)
    if args.action == "run":
        result = task.run(through=args.through)
    else:
        with exclusive(task.private):
            result = verify_fixture(task.object("download"), task.policy())
            result["transport_scope"] = "local-copy-not-public-download"
    from .canonical import canonical
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
