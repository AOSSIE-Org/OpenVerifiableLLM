# Locked wheel payload audit

`runtime_audit.py` reconstructs an inventory by fully hashing every selected wheel
archive against `requirements/gpu.lock`, then streaming every member's actual bytes.
It compares installed payloads with that reconstructed inventory and rejects unknown
importable files in site-packages. Installed RECORD hashes cannot substitute for
this comparison: an attacker changing both a package file and its RECORD still fails.

The supported lock is a complete unconditional list of exact versions or HTTPS
wheel URLs, with SHA-256 hashes. The directory must contain exactly one wheel for
every locked package. Missing packages, unregistered wheels, ambiguous members,
symlinks, traversal and unsupported install layouts fail closed.

```bash
PYTHONPATH=src .ovllm-cache/gpu-venv/bin/python -m ovl_pipeline.runtime_audit \
  --lock requirements/gpu.lock --wheels COMPLETE_WHEEL_DIRECTORY \
  --verify-current --allowed-generated OPERATOR_PINNED_INSTALLER_FILES.json \
  --output FRESH_AUDIT_DIRECTORY
```

Wheel payloads are mapped according to the [PyPA wheel installation
scheme](https://packaging.python.org/en/latest/specifications/binary-distribution-format/).
The installed RECORD and narrowly named generated distribution metadata are
inventoried separately, since installers rewrite them. Explicit installer hooks
require independently selected hashes; allowing a hook is a trust decision, not
proof of its origin. The [installed-project specification](https://packaging.python.org/en/latest/specifications/recording-installed-packages/)
explains the RECORD format and installer-generated files.

Generated `__pycache__` files are enumerated but not credited as original source.
A production launcher must force a fresh empty bytecode-cache prefix before startup
and constrain its import paths, preventing stale installed bytecode or unexpected
source directories from overriding audited code. `runtime_launch` now performs the
audit in the operator's trusted verifier environment before starting the selected
target interpreter with `-s -S -P` and a newly created cache prefix. Its bootstrap
uses only standard-library imports, disables site hooks and sets the exact source,
standard-library, extension and audited package paths. `gpu.configure` requires
that launch evidence and process settings; the compatible GPU fingerprint binds
the complete wheel payload root. A local subprocess test constructs a valid stale
bytecode override, shows the ordinary importer executing it, and confirms this
bootstrap instead executes the audited source without running site hooks.

```bash
# Parent runs in a separately trusted environment; target packages are not imported
# until their complete audit succeeds. This command inspects startup, not CUDA.
PYTHONPATH=src TRUSTED_PYTHON -m ovl_pipeline.runtime_launch \
  --lock requirements/gpu.lock --wheels COMPLETE_WHEEL_DIRECTORY \
  --venv TARGET_VENV --source src --output FRESH_LAUNCH_DIRECTORY \
  --allowed-generated OPERATOR_PINNED_INSTALLER_FILES.json \
  --module ovl_pipeline.runtime_launch -- --inspect-current
```

Roots must remain owner-controlled and unchanged while running. The launcher is
not a hostile-process sandbox or protection from a compromised verifier/OS. It
does not attest the interpreter/standard library, container,
GPU hardware or executed machine code; their required provenance observations and
fresh deterministic pilots remain separate gates. No production admission follows
from a package audit alone.
