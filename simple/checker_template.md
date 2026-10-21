# OpenVerifiableLLM: check one training segment

Recomputes a random segment of the published OpenVerifiableLLM training run bit for bit and
prints MATCH or MISMATCH. Use an **RTX 5090** (the training GPU type). It takes about 30 minutes
(22 GB download, then about 20 minutes of training) and costs well under a dollar.

1. Deploy on an RTX 5090 with this template (60 GB container disk, no volume needed).
2. Open the pod's **Logs**. At the end it prints one line starting with `OVL check`.
3. Paste that line in the project Discord, then **terminate the pod**.

Optional environment variables: `K=3` checks three random segments; `SEGMENTS="7 8"` checks
chosen ones.

What runs: `simple/checker_pod.sh` from the OpenVerifiableLLM repository at a fixed commit. It
downloads the release's own code and data from Hugging Face
(`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence`) at a fixed revision and runs
`verify_segment.sh`. Nothing is uploaded, and no account or token is needed.
