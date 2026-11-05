"""Shared vector perturbation and uncoded 16-QAM link functions."""

from dataclasses import dataclass
from math import pi, sqrt

import torch

from cdl import RX, frequency_response, ofdm_channels


TAU = 8 / sqrt(10)  # Per-axis modulo period for unit-energy 16-QAM.
EPS = 1e-12


@dataclass(frozen=True)
class Channel:
    h: torch.Tensor
    w: torch.Tensor
    basis: torch.Tensor
    projector: torch.Tensor
    spectrum: torch.Tensor


def prepare_channel(h):
    # If H = U S V^H, its ZF pseudoinverse is V S^-1 U^H.
    u, singular, vh = torch.linalg.svd(h, full_matrices=False)
    inverse = singular.reciprocal()
    w = (vh.mH * inverse.unsqueeze(-2)) @ u.mH
    # SVD of W in descending singular-value order.
    directions = u.flip(-1)
    weights = inverse.flip(-1)
    pivot_index = directions.abs().argmax(dim=-2, keepdim=True)
    pivot = directions.gather(-2, pivot_index)
    directions = directions * (pivot.conj() / pivot.abs())
    basis = directions.flip(-1)[..., :2]
    projector = weights.unsqueeze(-1) * directions.mH
    return Channel(h, w, basis, projector, weights / weights[..., :1])


def action_table(device="cpu"):
    """Return all 192 coefficient pairs and the indices of the original 32."""
    magnitudes = torch.arange(1, 9, dtype=torch.float64) / 4
    phases = torch.arange(16, dtype=torch.float64) * (2 * pi / 16)
    first = torch.polar(magnitudes[:, None], phases).flatten()
    one_direction = torch.stack((first, torch.zeros_like(first)), dim=-1)

    phases = phases[::4]
    left = torch.polar(torch.tensor([0.5, 1.0], dtype=torch.float64)[:, None], phases)
    right = torch.polar(torch.tensor([0.25, 0.5], dtype=torch.float64)[:, None], phases)
    two_directions = torch.cartesian_prod(torch.arange(8), torch.arange(8))
    additional = torch.stack(
        (left.flatten()[two_directions[:, 0]], right.flatten()[two_directions[:, 1]]),
        dim=-1,
    )
    original = (torch.arange(1, 8, 2)[:, None] * 16 + torch.arange(0, 16, 2)).flatten()
    table = torch.cat((one_direction, additional)).to(device=device, dtype=torch.complex64)
    return table, original.to(device)


def cdl_samples(channel_count, symbols_per_channel, generator, device):
    """Generate independent CDL-C realizations and 16-QAM symbols on their OFDM grid."""
    h = ofdm_channels(channel_count, symbols_per_channel, generator, device)
    state = prepare_channel(h.flatten(0, 1))
    levels = torch.randint(4, (len(state.h), RX, 2), generator=generator).to(device)
    coords = (2 * levels - 3).float() / sqrt(10)
    symbols = torch.complex(coords[..., 0], coords[..., 1])
    return state, symbols, levels


def features(state, symbols):
    first, second = state.basis.unbind(dim=-1)
    scaled = symbols / TAU
    return torch.cat((
        scaled.real, scaled.imag, first.real, first.imag,
        second.real, second.imag, state.spectrum,
    ), dim=-1)


def candidate_signals(state, symbols, coefficients):
    """Return s + tau*l for each action, after componentwise integer rounding."""
    points = coefficients @ state.basis.transpose(-2, -1)
    points = points * symbols.norm(dim=-1)[:, None, None]
    offsets = (points - symbols[:, None, :]) / TAU
    integer = torch.complex(offsets.real.round(), offsets.imag.round())
    return symbols[:, None, :] + TAU * integer


def costs(state, signals):
    projected = signals @ state.projector.transpose(-2, -1)
    return projected.abs().square().sum(dim=-1)


def select_signal(state, symbols, coefficients):
    candidates = candidate_signals(state, symbols, coefficients)
    candidates = torch.cat((symbols[:, None, :], candidates), dim=1)
    energy = costs(state, candidates)
    best = energy.argmin(dim=-1)
    row = torch.arange(len(symbols), device=symbols.device)
    return candidates[row, best], energy[row, best]


def modulo(values):
    return torch.complex(
        (values.real + TAU / 2).remainder(TAU) - TAU / 2,
        (values.imag + TAU / 2).remainder(TAU) - TAU / 2,
    )


def transmit(state, selected, energy, noise):
    gain = energy.sqrt()
    x = (state.w @ selected.unsqueeze(-1)).squeeze(-1) / gain[:, None]
    received = (state.h @ x.unsqueeze(-1)).squeeze(-1) + noise
    return modulo(received * gain[:, None])


def bit_errors(recovered, levels):
    coords = torch.stack((recovered.real, recovered.imag), dim=-1)
    estimated = ((coords * sqrt(10) + 3) / 2).round().long().clamp(0, 3)
    diff = (estimated ^ (estimated >> 1)) ^ (levels ^ (levels >> 1))
    return int(((diff & 1) + (diff >> 1)).sum().item())


def check_vp():
    response = frequency_response(
        1, torch.tensor([0.0, 0.0, 0.01, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 360_000.0]),
        torch.Generator().manual_seed(42), "cpu",
    )[0]
    torch.testing.assert_close(response[0], response[1])
    assert not torch.allclose(response[0], response[2])
    assert not torch.allclose(response[0], response[3])
    state, symbols, levels = cdl_samples(4, 8, torch.Generator().manual_seed(42), "cpu")
    table, original = action_table()
    assert table.shape == (192, 2) and original.numel() == 32
    expected = torch.polar(
        torch.arange(1, 5)[:, None] / 2,
        torch.arange(8) * (2 * pi / 8),
    ).flatten()
    torch.testing.assert_close(table[original, 0], expected)
    torch.testing.assert_close(table[original, 1], torch.zeros_like(expected))
    identity = torch.eye(RX, dtype=torch.complex64).expand(len(symbols), RX, RX)
    torch.testing.assert_close(state.h @ state.w, identity, atol=3e-6, rtol=3e-6)
    selected, full_cost = select_signal(state, symbols, table)
    _, original_cost = select_signal(state, symbols, table[original])
    _, zero_cost = select_signal(state, symbols, table[:0])
    direct = (state.w @ selected.unsqueeze(-1)).abs().square().sum(dim=(-2, -1))
    torch.testing.assert_close(full_cost, direct, atol=3e-6, rtol=3e-6)
    assert torch.all(full_cost <= original_cost + 1e-5)
    assert torch.all(original_cost <= zero_cost + 1e-5)
    restored = transmit(state, selected, full_cost, torch.zeros_like(symbols))
    torch.testing.assert_close(restored, symbols, atol=3e-6, rtol=3e-6)
    assert bit_errors(restored, levels) == 0
    print("PASS: CDL-C time/frequency response, ZF, SVD cost, action nesting, and noiseless recovery")


if __name__ == "__main__":
    check_vp()
