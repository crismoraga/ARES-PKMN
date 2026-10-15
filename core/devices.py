"""Validate complete model updates before selecting optional DirectML hardware."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.autograd import backward
from torch.optim.adam import Adam


@dataclass(frozen=True)
class DeviceSelection:
    """Measured backend selection including explicit compatibility failures."""
    device: torch.device
    backend: str
    reason: str
    cpu_seconds: float
    directml_seconds: float | None


ModelProbe = Callable[[nn.Module, torch.device], Tensor]


def _benchmark(models: Sequence[tuple[nn.Module, ModelProbe]], device: torch.device) -> tuple[float, list[float]]:
    """Require finite losses, gradients and changed parameters for every model."""
    start = time.perf_counter()
    losses: list[float] = []
    for original, probe in models:
        model = copy.deepcopy(original).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        before = [parameter.detach().cpu().clone() for parameter in model.parameters() if parameter.requires_grad]
        optimizer.zero_grad()
        loss = probe(model, device)
        if not bool(torch.isfinite(loss).all().cpu()):
            raise RuntimeError("Nonfinite model probe loss")
        backward(loss)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or any(not bool(torch.isfinite(gradient).all().cpu()) for gradient in gradients):
            raise RuntimeError("Missing or nonfinite model gradients")
        optimizer.step()
        after = [parameter.detach().cpu() for parameter in model.parameters() if parameter.requires_grad]
        if all(torch.equal(a, b) for a, b in zip(before, after, strict=True)):
            raise RuntimeError("Optimizer probe did not update parameters")
        losses.append(float(loss.detach().cpu()))
    return time.perf_counter() - start, losses


def choose_training_device(models: Sequence[tuple[nn.Module, ModelProbe]], requested: str = "auto") -> DeviceSelection:
    """Select GPU only after whole-model parity/update and measured-speed checks."""
    if requested not in {"auto", "cpu", "directml"}:
        raise ValueError("device must be auto, cpu or directml")
    cpu = torch.device("cpu")
    cpu_seconds, cpu_losses = _benchmark(models, cpu)
    if requested == "cpu":
        return DeviceSelection(cpu, "cpu", "CPU explicitly selected", cpu_seconds, None)
    try:
        import torch_directml
        device = torch_directml.device()
        seconds, losses = _benchmark(models, device)
        if any(abs(a - b) > 0.01 + 0.005 * abs(a) for a, b in zip(cpu_losses, losses, strict=True)):
            raise RuntimeError("Complete-model CPU/DirectML loss parity failed")
        if requested == "directml" or seconds < cpu_seconds:
            return DeviceSelection(device, "directml", "All model updates passed", cpu_seconds, seconds)
        return DeviceSelection(cpu, "cpu", "Validated GPU slower for these models", cpu_seconds, seconds)
    except (ImportError, RuntimeError, NotImplementedError) as error:
        if requested == "directml":
            raise RuntimeError(f"DirectML failed complete-model verification: {error}") from error
        return DeviceSelection(cpu, "cpu", f"DirectML compatibility test failed: {error}", cpu_seconds, None)
