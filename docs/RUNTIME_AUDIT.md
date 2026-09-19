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
does not attest a remote container, GPU hardware or executed machine code; their
required provenance observations and
fresh deterministic pilots remain separate gates. No production admission follows
from a package audit alone.

The GPU gate now also requires a complete public interpreter-origin audit. The
operator selects an install-only Python archive by SHA-256, reconstructs its complete
inventory, and extracts it unchanged into a fresh private directory. The launcher
rehashes every installed non-bytecode payload and checks every symlink before
starting the target interpreter. It rejects additional source, shared libraries,
sourceless bytecode, changed executable modes and paths outside the selected tree.
Only generated `__pycache__/*.pyc` is excluded, under the same fresh-cache startup
requirement. Wheel-only developer launches remain usable for CPU fixtures but
cannot pass `gpu.configure`.

```bash
PYTHONPATH=src TRUSTED_PYTHON -m ovl_pipeline.python_origin extract \
  --archive PUBLIC_INSTALL_ONLY_ARCHIVE --sha256 EXTERNALLY_SELECTED_SHA256 \
  --root FRESH_PUBLIC_PYTHON --evidence FRESH_ORIGIN_EVIDENCE

# Create the target venv with FRESH_PUBLIC_PYTHON/python/bin/python3.12 and
# install the complete hash lock. Add these options to runtime_launch:
# --interpreter-archive PUBLIC_INSTALL_ONLY_ARCHIVE
# --interpreter-sha256 EXTERNALLY_SELECTED_SHA256
# --interpreter-root FRESH_PUBLIC_PYTHON
```

The tested candidate is the [20260814 python-build-standalone release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260814),
CPython3.12.14 x86_64 Linux GNU, install-only stripped archive, SHA-256
`5acfa3e9ba26b51ae161c83aff278da915b590d22373a424b2ba55b8afe91fcc`.
Its complete 34,143,739-byte download matched both the GitHub asset digest and the
retained public checksum file. The original uv-managed installation had a rewritten
sysconfig module, so the production candidate uses a separate unmodified extraction;
the running preparation environment is preserved. This establishes public binary
identity, not reproduction of the upstream CPython compiler/build. Container, OS,
driver and actual hardware observations and measured deterministic pilots remain
required. It is not remote attestation or independent third-party verification.

The runtime fingerprint now hashes all file-backed mapped shared ELF images,
including the OS libraries and loader. Before hashing, each file descriptor's
actual device/inode must match `/proc/self/maps`; missing/deleted/replaced images
fail closed. A before/after metadata check detects mutation during the read. The
CPU initialization/inference observation separately binds the mapped GNU libc,
libm, C++/GCC libraries and loader. Paths and ASLR addresses are excluded from the
compatible digest; library names, lengths and full SHA-256 digests are retained.
These are observations of backing files, not proof of relocated in-memory code,
protection against a hostile kernel, or reproduction of the container's build.
The public container digest and actual compatible CUDA pilots remain separate.
