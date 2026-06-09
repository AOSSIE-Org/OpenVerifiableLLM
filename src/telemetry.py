import json
from xml.parsers.expat import model
import torch
import os
import hashlib

class TelemetryLogger:
    def __init__(self, filepath="audit_log.jsonl"):
        self.filepath = filepath

        open(self.filepath, 'w').close()

    def log_step(self, step, loss, model):
        grad_norm = 0.0
        param_norm = 0.0

        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
            param_norm += p.data.norm(2).item() ** 2

        grad_norm = grad_norm ** 0.5
        param_norm = param_norm ** 0.5

        record ={
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "param_norm": param_norm
        }

        with open(self.filepath, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        return record
    
    def hash_model(self, model):
        h = hashlib.sha256()
        for p in model.parameters():
            h.update(p.data.cpu().numpy().tobytes())
        return h.hexdigest()