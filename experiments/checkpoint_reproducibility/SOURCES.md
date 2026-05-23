# Bibliography

Sources that were actually useful for this experiment.

---

**PyTorch — Reproducibility docs**  
https://pytorch.org/docs/stable/notes/randomness.html  
The main reference. Covers `use_deterministic_algorithms`, DataLoader worker
seeding, and the fork vs spawn difference that caused the Windows issue.

**PyTorch source — `torch/serialization.py`**  
https://github.com/pytorch/pytorch/blob/main/torch/serialization.py  
Where the hardcoded ZIP date (`2011-01-01`) lives. Explains why `torch.save`
gave identical hashes in the experiment.

**Hugging Face — Safetensors**  
https://github.com/huggingface/safetensors  
The format `save_deterministic()` approximates. No pickle, no ZIP, raw tensor
bytes in a fixed layout. Worth reading if this experiment gets extended.

**Reproducible Builds project**  
https://reproducible-builds.org/  
Background on why byte-level reproducibility is harder than it looks. The
timestamp problem in ZIP archives is a well-known issue in compiled software —
same class of problem, different domain.

**Bouthillier et al. (2019) — Unreproducible Research is Reproducible. ICML.**  
https://proceedings.mlr.press/v97/bouthillier19a.html  
Argues that undocumented variation is the real problem in ML reproducibility, not
variation itself. Relevant to the results where training is repeatable but
different from baseline — those aren't wrong, they're just undocumented states.

**Pineau et al. (2021) — Improving Reproducibility in ML Research. JMLR.**  
https://jmlr.org/papers/v22/20-1364.html  
The NeurIPS reproducibility checklist. Useful reference for what the
OpenVerifiableLLM training manifest should eventually include.