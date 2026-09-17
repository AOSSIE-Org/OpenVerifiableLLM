The existing artifact verifier and sampled audits do not provide the full reconstruction and continuous replay required by the Wikipedia project goal. This adds a separate, fail-closed pipeline whose synthetic fixture rebuilds raw inputs, regenerates initialization, replays both training phases, verifies safe checkpoints, and checks exported model/tokenizer/configuration and inference identity.

The implementation also adds resumable checksum-verified Wikipedia acquisition, OASST branch selection with split and exclusion accounting, adversarial tests, crash recovery, and an explicit project goal/state. Production admission remains disabled until public anchoring, resource guards and GPU exactness are implemented and verified. No G01–G10 completion is claimed.

Validation: 79 tests passed locally; three legacy GPU-dependent tests skipped. Reproduced and corrected independent advisory findings, preserving failed-counterexample evidence. Legacy tests now use temporary keys and preserve the repository's existing public key. CI runs the combined regression suite.

Public evidence staging: https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence

The complete raw Wikipedia download is in progress locally. No RunPod resource has been created and project RunPod spend remains $0.
