"""Single-polarized CDL-C coefficients from 3GPP TR 38.901 v15.0.0."""

from math import pi, sqrt

import torch


TX_ARRAY = (8, 8)
RX_ARRAY = (2, 4)
TX = TX_ARRAY[0] * TX_ARRAY[1]
RX = RX_ARRAY[0] * RX_ARRAY[1]
CDL_CONFIG = {
    "profile": "CDL-C",
    "spec": "3GPP TR 38.901 v15.0.0",
    "carrier_hz": 3.5e9,
    "subcarrier_spacing_hz": 30e3,
    "delay_spread_s": 100e-9,
    "speed_kmh": 3.0,
    "tx_array": TX_ARRAY,
    "rx_array": RX_ARRAY,
    "array_spacing_yz_wavelengths": (0.3, 0.9),
    "ofdm_symbol_stride": 900,
    "subcarrier_stride": 16,
    "antenna_pattern": "isotropic, co-polar theta",
    "channel_scale": "1/sqrt(TX)",
}

# Table 7.7.1-3: normalized delay, dB power, AoD, AoA, ZoD, ZoA.
_CLUSTERS = (
    (0.0000, -4.4, -46.6, -101.0, 97.2, 87.6),
    (0.2099, -1.2, -22.8, 120.0, 98.6, 72.1),
    (0.2219, -3.5, -22.8, 120.0, 98.6, 72.1),
    (0.2329, -5.2, -22.8, 120.0, 98.6, 72.1),
    (0.2176, -2.5, -40.7, -127.5, 100.6, 70.1),
    (0.6366, 0.0, 0.3, 170.4, 99.2, 75.3),
    (0.6448, -2.2, 0.3, 170.4, 99.2, 75.3),
    (0.6560, -3.9, 0.3, 170.4, 99.2, 75.3),
    (0.6584, -7.4, 73.1, 55.4, 105.2, 67.4),
    (0.7935, -7.1, -64.5, 66.5, 95.3, 63.8),
    (0.8213, -10.7, 80.2, -48.1, 106.1, 71.4),
    (0.9336, -11.1, -97.1, 46.9, 93.5, 60.5),
    (1.2285, -5.1, -55.3, 68.1, 103.7, 90.6),
    (1.3083, -6.8, -64.3, -68.7, 104.2, 60.1),
    (2.1704, -8.7, -78.5, 81.5, 93.0, 61.0),
    (2.7105, -13.2, 102.7, 30.7, 104.2, 100.7),
    (4.2589, -13.9, 99.2, -16.4, 94.9, 62.3),
    (4.6003, -13.9, 88.8, 3.8, 93.1, 66.7),
    (5.4902, -15.8, -101.9, -13.7, 92.2, 52.9),
    (5.6077, -17.1, 92.2, 9.7, 106.7, 61.8),
    (6.3065, -16.0, 93.3, 5.6, 93.0, 51.9),
    (6.6374, -15.7, 106.6, 0.7, 92.9, 61.7),
    (7.0427, -21.6, 119.5, -21.9, 105.2, 58.0),
    (8.6523, -22.8, -123.8, 33.6, 107.8, 57.0),
)
# Table 7.5-3: the two rays in each pair use opposite signs.
_OFFSETS = (0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797,
            0.8844, 1.1481, 1.5195, 2.1551)
_SPREADS = (2.0, 15.0, 3.0, 7.0)


def _planar_positions(shape, device):
    """Return centered y-z plane coordinates in carrier wavelengths."""
    rows, cols = shape
    y = torch.arange(cols, device=device, dtype=torch.float32) - (cols - 1) / 2
    z = torch.arange(rows, device=device, dtype=torch.float32) - (rows - 1) / 2
    spacing_yz = y.new_tensor(CDL_CONFIG["array_spacing_yz_wavelengths"])
    yz = torch.cartesian_prod(y, z) * spacing_yz
    return torch.cat((torch.zeros_like(yz[:, :1]), yz), dim=-1)


def _unit_direction(azimuth, zenith):
    sin_zenith = zenith.sin()
    return torch.stack((sin_zenith * azimuth.cos(), sin_zenith * azimuth.sin(), zenith.cos()), dim=-1)


def _steering(directions, positions):
    phase = (2 * pi) * (directions @ positions.T)
    return torch.polar(torch.ones_like(phase), phase)


def frequency_response(realizations, time_s, frequency_hz, generator, device):
    """Evaluate each realization at paired (time, baseband frequency) points.

    This is the theta-to-theta element of the CDL polarization matrix for
    isotropic, aligned single-polarized elements. Cross-polar terms vanish.
    """
    cluster = torch.tensor(_CLUSTERS, device=device, dtype=torch.float32)
    clusters = len(_CLUSTERS)
    offsets = torch.tensor([sign * value for value in _OFFSETS for sign in (1, -1)])
    rays = offsets.numel()
    # Keep AoD order, independently couple the other three angles within each cluster.
    shuffled = torch.rand((realizations, clusters, 3, rays), generator=generator).argsort(dim=-1)
    ordered = offsets.expand(realizations, clusters, 1, rays)
    ray_offsets = torch.cat((ordered, offsets[shuffled]), dim=2).to(device)
    spreads = torch.tensor(_SPREADS, device=device)[None, None, :, None]
    angles = (cluster[None, :, 2:, None] + spreads * ray_offsets).deg2rad()
    aod, aoa, zod, zoa = (part.flatten(1) for part in angles.unbind(dim=2))

    departure = _unit_direction(aod, zod)
    arrival = _unit_direction(aoa, zoa)
    tx = _steering(departure, _planar_positions(TX_ARRAY, device))
    rx = _steering(arrival, _planar_positions(RX_ARRAY, device))

    delay = (cluster[:, 0] * CDL_CONFIG["delay_spread_s"]).repeat_interleave(rays)
    power = torch.pow(10.0, cluster[:, 1] / 10)
    amplitude = (power / (power.sum() * rays)).sqrt().repeat_interleave(rays)
    initial = (2 * pi) * torch.rand((realizations, clusters * rays), generator=generator).to(device)
    heading = (2 * pi) * torch.rand((realizations, 1), generator=generator).to(device)
    velocity_direction = torch.cat((heading.cos(), heading.sin(), torch.zeros_like(heading)), dim=-1)
    max_doppler = CDL_CONFIG["carrier_hz"] * (CDL_CONFIG["speed_kmh"] / 3.6) / 299_792_458
    doppler = max_doppler * (arrival * velocity_direction[:, None, :]).sum(dim=-1)

    phase = initial[:, None, :] + (2 * pi) * (
        time_s.to(device)[None, :, None] * doppler[:, None, :]
        - frequency_hz.to(device)[None, :, None] * delay
    )
    gain = torch.polar(amplitude.expand_as(phase), phase)
    weighted_rx = gain[:, :, None, :] * rx.transpose(-1, -2)[:, None, :, :]
    response = weighted_rx.flatten(1, 2) @ tx
    return response.reshape(realizations, -1, RX, TX) / sqrt(TX)


def ofdm_channels(realizations, samples_per_realization, generator, device):
    index = torch.arange(samples_per_realization, dtype=torch.float32)
    spacing = CDL_CONFIG["subcarrier_spacing_hz"]
    # Sample selected resource elements while keeping the physical OFDM numerology.
    time_s = index * CDL_CONFIG["ofdm_symbol_stride"] / spacing
    frequency_hz = (index - samples_per_realization // 2) * (spacing * CDL_CONFIG["subcarrier_stride"])
    return frequency_response(realizations, time_s, frequency_hz, generator, device)
