"""Generate CDL-C training data and train the VP-PA action scorer."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cdl import CDL_CONFIG
from vp import EPS, RX, action_table, candidate_signals, cdl_samples, costs, features


CHECKPOINT = Path(__file__).resolve().parent / "results" / "cdl-c" / "vp_model.pt"


def build_model(action_count):
    return nn.Sequential(
        nn.Linear(7 * RX, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, action_count),
    )


@torch.no_grad()
def make_dataset(channels, symbols_per_channel, seed, table, device):
    generator = torch.Generator().manual_seed(seed)
    inputs, action_costs, zero_costs = [], [], []
    for start in range(0, channels, 32):
        state, s, _ = cdl_samples(
            min(32, channels - start), symbols_per_channel, generator, device
        )
        inputs.append(features(state, s))
        action_costs.append(costs(state, candidate_signals(state, s, table)))
        zero_costs.append(costs(state, s[:, None, :]).squeeze(-1))
    return TensorDataset(torch.cat(inputs), torch.cat(action_costs), torch.cat(zero_costs))


@torch.no_grad()
def screening_gap(model, loader, top_k):
    model.eval()
    total, count = 0.0, 0
    for inputs, action_costs, zero in loader:
        shortlist = model(inputs).topk(top_k, dim=-1).indices
        selected = torch.minimum(action_costs.gather(1, shortlist).amin(dim=-1), zero)
        full = torch.minimum(action_costs.amin(dim=-1), zero)
        total += ((selected - full) / (zero + EPS)).sum().item()
        count += len(inputs)
    return total / count


@torch.no_grad()
def fixed_subset(validation, top_k):
    """Choose a strong fixed subset on validation data, without test data."""
    _, action_costs, zero = validation.tensors
    normalized = action_costs / (zero[:, None] + EPS)
    best = zero / (zero + EPS)
    chosen = []
    for _ in range(top_k):
        means = torch.minimum(best[:, None], normalized).mean(dim=0)
        means[chosen] = torch.inf
        index = int(means.argmin().item())
        chosen.append(index)
        best = torch.minimum(best, normalized[:, index])
    return torch.tensor(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-channels", type=int, default=1024)
    parser.add_argument("--val-channels", type=int, default=256)
    parser.add_argument("--symbols-per-channel", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if min(args.train_channels, args.val_channels, args.symbols_per_channel,
           args.epochs, args.batch_size) < 1:
        parser.error("channel counts, symbols, epochs and batch size must be positive")
    if args.temperature <= 0 or args.learning_rate <= 0:
        parser.error("temperature and learning rate must be positive")
    table, _ = action_table(args.device)
    if not 1 <= args.top_k <= len(table):
        parser.error("top-k must be between 1 and the action table size")

    torch.manual_seed(args.seed)
    print(f"Generating CDL-C train and validation data on {args.device}...", flush=True)
    train = make_dataset(
        args.train_channels, args.symbols_per_channel, args.seed, table, args.device
    )
    validation = make_dataset(
        args.val_channels, args.symbols_per_channel, args.seed + 1, table, args.device
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    model = build_model(len(table)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    fixed = fixed_subset(validation, args.top_k)
    best_gap = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for inputs, action_costs, zero in train_loader:
            gaps = (action_costs - action_costs.amin(dim=1, keepdim=True)) / (zero[:, None] + EPS)
            targets = (-gaps / args.temperature).softmax(dim=1)
            log_prob = model(inputs).log_softmax(dim=1)
            loss = -(targets * log_prob).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(inputs)

        gap = screening_gap(model, val_loader, args.top_k)
        if gap < best_gap:
            best_gap = gap
            torch.save({
                "model": model.state_dict(),
                "actions": table.cpu(),
                "fixed_indices": fixed,
                "top_k": args.top_k,
                "seed": args.seed,
                "val_gap": gap,
                "channel_config": CDL_CONFIG,
            }, args.checkpoint)
        print(f"Epoch {epoch:02d}  loss={loss_total / len(train):.5f}  "
              f"validation Top-{args.top_k} gap={gap:.5f}", flush=True)
    print(f"Best validation gap: {best_gap:.5f}. Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
