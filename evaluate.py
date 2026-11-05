"""Compare VP-PA policies on paired, held-out CDL-C channels and 16-QAM."""

import argparse
import csv
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cdl import CDL_CONFIG
from train import CHECKPOINT, build_model
from vp import EPS, RX, action_table, bit_errors, cdl_samples, features, select_signal, transmit


def save_comparison(rows, output_path):
    by_method = {}
    for row in rows:
        by_method.setdefault(row[0], []).append(row)

    figure = Figure(figsize=(11.5, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    ber_ax, cost_ax = figure.subplots(1, 2, width_ratios=(1.35, 1))
    figure.suptitle("VP-PA comparison | 3GPP CDL-C, 16-QAM", fontsize=14)

    markers = ("o", "s", "D", "^", "P")
    has_zero_errors = False
    for position, (method, results) in enumerate(by_method.items()):
        results = sorted(results, key=lambda row: row[1])
        snr = [row[1] for row in results]
        ber = [row[4] if row[2] else float("nan") for row in results]
        line, = ber_ax.plot(snr, ber, marker=markers[position % len(markers)], label=method)
        color = line.get_color()
        zero_error_points = [(row[1], 1 / row[3]) for row in results if row[2] == 0]
        if zero_error_points:
            has_zero_errors = True
            zero_snr, resolution = zip(*zero_error_points)
            ber_ax.scatter(zero_snr, resolution, color=color, marker="v", zorder=3)

        normalized_cost = results[0][5]
        cost_ax.scatter(normalized_cost, position, color=color, s=42, zorder=3)
        cost_ax.annotate(
            f"{normalized_cost:.4f}", (normalized_cost, position),
            xytext=(7, 0), textcoords="offset points", va="center",
        )

    ber_ax.set(title="Bit error rate", xlabel="SNR (dB)", ylabel="BER", yscale="log")
    if has_zero_errors:
        ber_ax.set_xlabel("SNR (dB)\nDownward triangles: zero errors, shown at 1 / tested bits")
    ber_ax.grid(True, which="both", alpha=0.25)
    ber_ax.legend(fontsize=8)

    cost_ax.set(title="Mean precoding cost", xlabel=r"Mean $J/J_0$ (lower is better)")
    cost_ax.set_yticks(range(len(by_method)), labels=list(by_method))
    cost_ax.invert_yaxis()
    cost_ax.margins(x=0.23, y=0.15)
    cost_ax.grid(True, axis="x", alpha=0.25)
    figure.savefig(output_path, dpi=180)


@torch.inference_mode()
def evaluate(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("channel_config") != CDL_CONFIG:
        raise ValueError("Checkpoint does not match the current CDL-C configuration. Run train.py first.")
    table_cpu, original = action_table()
    if not torch.equal(saved["actions"], table_cpu):
        raise ValueError("Checkpoint action table differs from vp.py. Retrain the model.")
    table = table_cpu.to(args.device)
    original = original.to(args.device)
    fixed = saved["fixed_indices"].to(args.device)
    top_k = saved["top_k"]
    model = build_model(len(table)).to(args.device)
    model.load_state_dict(saved["model"])
    model.eval()

    data_rng = torch.Generator().manual_seed(saved["seed"] + 2)
    noise_rng = torch.Generator().manual_seed(saved["seed"] + 3)
    names = ("ZF", "VP-PA-32", "Full-192", f"Fixed-{top_k}", f"NN-Top-{top_k}")
    counts = {name: [0] * len(args.snr_db) for name in names}
    energies = dict.fromkeys(names, 0.0)
    gaps = dict.fromkeys(names, 0.0)
    total_symbols = 0
    for start in range(0, args.test_channels, 32):
        state, symbols, levels = cdl_samples(
            min(32, args.test_channels - start), args.symbols_per_channel, data_rng, args.device
        )
        shortlist = model(features(state, symbols)).topk(top_k, dim=-1).indices
        subsets = (table[:0], table[original], table, table[fixed], table[shortlist])
        choices = {
            name: select_signal(state, symbols, subset) for name, subset in zip(names, subsets)
        }
        zero = choices["ZF"][1]
        full = choices["Full-192"][1]
        for name, (_, energy) in choices.items():
            energies[name] += (energy / (zero + EPS)).sum().item()
            gaps[name] += ((energy - full) / (zero + EPS)).sum().item()

        # Unit total transmit power. SNR here is 1/E[|n_m|^2].
        noise_unit = torch.randn(symbols.shape, dtype=symbols.dtype, generator=noise_rng).to(args.device)
        for index, snr in enumerate(args.snr_db):
            noise = noise_unit * 10 ** (-snr / 20)
            for name, (selected, energy) in choices.items():
                recovered = transmit(state, selected, energy, noise)
                counts[name][index] += bit_errors(recovered, levels)
        total_symbols += len(symbols)

    bits = total_symbols * RX * 4
    rows = []
    print(f"Tested {total_symbols:,} symbol vectors, {bits:,} bits per SNR per policy.")
    print(f"{'Method':<14} {'Mean J/J0':>12} {'Gap to full':>13}")
    for name in names:
        mean_energy, mean_gap = energies[name] / total_symbols, gaps[name] / total_symbols
        print(f"{name:<14} {mean_energy:12.6f} {mean_gap:13.6f}")
        for snr, errors in zip(args.snr_db, counts[name]):
            rows.append((name, snr, errors, bits, errors / bits, mean_energy, mean_gap))

    print("\nSNR (dB)  " + "  ".join(f"{name:>12}" for name in names))
    for index, snr in enumerate(args.snr_db):
        print(f"{snr:8g}  " + "  ".join(f"{counts[name][index] / bits:12.4g}" for name in names))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "snr_db", "errors", "bits", "ber", "mean_j_over_j0", "gap_to_full"))
        writer.writerows(rows)
    print(f"Saved: {args.csv}")
    comparison_path = args.csv.with_name(f"{args.csv.stem}_comparison.png")
    save_comparison(rows, comparison_path)
    print(f"Saved: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--test-channels", type=int, default=256)
    parser.add_argument("--symbols-per-channel", type=int, default=16)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[5, 10, 15, 20, 25])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", type=Path, default=CHECKPOINT.with_name("ber.csv"))
    args = parser.parse_args()
    if min(args.test_channels, args.symbols_per_channel) < 1:
        parser.error("test channels and symbols per channel must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Run train.py first.")
    evaluate(args)


if __name__ == "__main__":
    main()
