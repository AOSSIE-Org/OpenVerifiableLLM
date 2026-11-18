"""Measure numerical drift between Apple Silicon (MPS) and CPU on OpenVerifiableLLM 23M architecture.

Addresses Roadmap Open Research Question 1:
'Checking across GPU architectures. Bits currently match only on the training architecture.'

Usage:
  OVL_DEVICE=cpu python scripts/research/measure_mps_cpu_drift.py
"""
import sys
import platform
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import ovl
from model import TinyGPT


def main():
    print("=" * 60)
    print("OpenVerifiableLLM: MPS vs CPU Numerical Drift Trace (23M Architecture)")
    print("=" * 60)
    print(f"Platform:       {platform.platform()}")
    print(f"Machine:        {platform.machine()}")
    print(f"Python:         {platform.python_version()}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"MPS Available:  {torch.backends.mps.is_available()}")
    print("-" * 60)

    if not torch.backends.mps.is_available():
        sys.exit("Error: Apple Silicon MPS backend is not available on this machine.")

    # 1. Initialize identical seeded models via canonical ovl.fresh()
    ovl.configure(1234)
    cpu_model, cpu_opt = ovl.fresh()
    cpu_model = cpu_model.to("cpu")

    mps_model, _ = ovl.fresh()
    mps_model.load_state_dict(cpu_model.state_dict())
    mps_model = mps_model.to("mps")
    mps_opt = torch.optim.AdamW(
        mps_model.parameters(), lr=0.0006, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    # Verify bit-identical initialization
    init_diff = max(
        (p1.cpu() - p2.cpu()).abs().max().item()
        for p1, p2 in zip(cpu_model.parameters(), mps_model.parameters())
    )
    print(f"Init state diff: {init_diff:.10e} (Bit-exact match at step 0)\n")

    # 2. Step 1 Forward pass primitive trace
    B, T = 4, 128
    torch.manual_seed(42)
    inputs = torch.randint(0, 32000, (B, T), dtype=torch.long)
    targets = torch.randint(0, 32000, (B, T), dtype=torch.long)

    x_cpu = inputs.to("cpu")
    x_mps = inputs.to("mps")

    # Embedding checks
    wte_cpu = cpu_model.transformer["wte"](x_cpu)
    wte_mps = mps_model.transformer["wte"](x_mps).cpu()
    print(f"wte max diff:        {(wte_cpu - wte_mps).abs().max().item():.10e} (Bit-exact)")

    pos = torch.arange(0, T, dtype=torch.long)
    wpe_cpu = cpu_model.transformer["wpe"](pos)
    wpe_mps = mps_model.transformer["wpe"](pos.to("mps")).cpu()
    print(f"wpe max diff:        {(wpe_cpu - wpe_mps).abs().max().item():.10e} (Bit-exact)")

    h_cpu = wte_cpu + wpe_cpu
    h_mps = (mps_model.transformer["wte"](x_mps) + mps_model.transformer["wpe"](pos.to("mps"))).cpu()
    print(f"Embed sum max diff:  {(h_cpu - h_mps).abs().max().item():.10e} (Bit-exact)")

    # First divergence point: Layer 0 LayerNorm
    ln1_cpu = cpu_model.transformer["h"][0].ln_1(h_cpu)
    ln1_mps = mps_model.transformer["h"][0].ln_1(h_mps.to("mps")).cpu()
    diff_ln1 = (ln1_cpu - ln1_mps).abs().max().item()
    print(f"Layer 0 ln_1 diff:   {diff_ln1:.10e} <-- FIRST DIVERGENCE (Reduction non-associativity)")

    # Output logits and Step 1 loss
    logits_cpu = cpu_model(x_cpu)
    logits_mps = mps_model(x_mps).cpu()
    diff_logits = (logits_cpu - logits_mps).abs().max().item()
    print(f"Logits max diff:     {diff_logits:.10e}")

    l1_cpu = F.cross_entropy(logits_cpu.flatten(0, 1), targets.flatten())
    l1_mps = F.cross_entropy(logits_mps.to("mps").flatten(0, 1), targets.to("mps").flatten()).cpu()
    expected_uniform = torch.log(torch.tensor(32000.0)).item()
    print(f"Step 1 Loss (CPU):   {l1_cpu.item():.7f} (Uniform prior ln(32000) ~ {expected_uniform:.4f})")
    print(f"Step 1 Loss (MPS):   {l1_mps.item():.7f}")
    print(f"Step 1 Loss diff:    {abs(l1_cpu.item() - l1_mps.item()):.10e}")

    # Backward pass & Step 1 optimizer update
    l1_cpu.backward()
    l1_mps.backward()
    cpu_opt.step()
    mps_opt.step()
    cpu_opt.zero_grad()
    mps_opt.zero_grad()

    # 3. Compounding drift across steps 2 to 5
    print("\n--- Compounding Drift Across Steps 2 to 5 ---")
    for step in range(2, 6):
        x = torch.randint(0, 32000, (B, T), dtype=torch.long)
        y = torch.randint(0, 32000, (B, T), dtype=torch.long)
        l_c = F.cross_entropy(cpu_model(x).flatten(0, 1), y.flatten())
        l_m = F.cross_entropy(mps_model(x.to("mps")).flatten(0, 1), y.to("mps").flatten()).cpu()
        l_c.backward()
        l_m.backward()
        cpu_opt.step()
        mps_opt.step()
        cpu_opt.zero_grad()
        mps_opt.zero_grad()

        w_diff = max(
            (p1.cpu() - p2.cpu()).abs().max().item()
            for p1, p2 in zip(cpu_model.parameters(), mps_model.parameters())
        )
        w_rel = max(
            (p1.cpu() - p2.cpu()).norm().item() / p1.cpu().norm().item()
            for p1, p2 in zip(cpu_model.parameters(), mps_model.parameters())
        )
        print(
            f"Step {step} | Loss diff: {abs(l_c.item() - l_m.item()):.8e} | "
            f"Weight max_abs: {w_diff:.8e} | Weight rel_L2: {w_rel:.8e}"
        )


if __name__ == "__main__":
    main()
