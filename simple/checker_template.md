# OpenVerifiableLLM: check one training segment

Recomputes a random segment of the published OpenVerifiableLLM training run bit for bit and
prints MATCH or MISMATCH. Use an **RTX 5090** (the training GPU type). It takes about an hour
(a 22 GB download, which depends on the host's network, then about 20 minutes of training) and
costs about US$1–1.50 at RTX 5090 prices.

1. Deploy on an RTX 5090 with this template (60 GB container disk, no volume needed).
2. Wait about an hour, then open the pod's **Logs** (or run `tail /workspace/check.log` in its
   web terminal). At the end it prints one line starting with `OVL check`.
3. Paste that line in the project Discord, then **terminate the pod**.

Optional environment variables: `K=3` checks three random segments; `SEGMENTS="7 8"` checks
chosen ones.

What runs: `simple/checker_pod.sh` from the OpenVerifiableLLM repository at a fixed commit. It
downloads the release's own code from Hugging Face
(`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence`) at a fixed revision, then runs
its `verify_segment.sh`, which hash-checks the data and the checkpoint against the record. Nothing is uploaded, and no account or token is needed.

First real run (21 Oct 2026, fresh RTX 5090, driver 595.91.07): segment 8 MATCH in 1139 s.

Template start command:

```
bash -c "(curl -fsSL https://raw.githubusercontent.com/AOSSIE-Org/OpenVerifiableLLM/<commit>/simple/checker_pod.sh | bash 2>&1 | tee /workspace/check.log) & exec /start.sh"
```
