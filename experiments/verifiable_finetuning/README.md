# Verifiable Fine-Tuning PoC

### What this proves
If you treat fine-tuning as just another step in the training pipeline " with controlled inputs ", then it becomes fully deterministic and reproducible. That means you can cryptographically link every stage of a model's life: base → fine-tune → final weights. No black boxes.

### How to run
```bash
python train_base.py
python finetune.py
python manifest.py

