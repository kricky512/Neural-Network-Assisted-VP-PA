# Neural Network Assisted VP-PA

This project uses PyTorch to evaluate neural network assisted vector perturbation precoding. The simulation uses 64 transmit antennas, eight receive streams, 16-QAM, and a moderately correlated CDL-C channel. The network ranks 192 actions. By default, the precoder evaluates the exact cost of the eight highest-ranked actions and the zero perturbation.

## Environment setup

The project uses Python and PyTorch. It was tested on Windows with Python 3.12 and PyTorch 2.11.0. Training and evaluation use CUDA when PyTorch detects it and fall back to CPU otherwise. Pass `--device cpu` to select CPU explicitly.


## Run the project

From the repository root, run the following commands in order.

```sh
python vp.py
python train.py
python evaluate.py
```

| Script | Purpose |
| --- | --- |
| [`vp.py`](vp.py) | Checks ZF, agreement between the direct and SVD-based cost formulas, inclusion of the original 32 actions in the 192-action table, and noiseless symbol recovery. Prints `PASS` if the checks succeed. |
| [`train.py`](train.py) | Generates action costs from simulated channels, trains the scoring network, and saves the checkpoint with the lowest validation screening gap to `results/cdl-c/vp_model.pt`. |
| [`evaluate.py`](evaluate.py) | Loads the checkpoint and compares ZF, 32-action full search, 192-action full search, a fixed 8-action subset, and neural Top-8 screening. Writes `results/cdl-c/ber.csv`. |

If Conda activation fails in Windows PowerShell, call your environment's Python executable directly. Replace the example path with your local interpreter path.

```powershell
$py = "D:\miniconda\envs\nf-noma\python.exe"
& $py .\vp.py
& $py .\train.py
& $py .\evaluate.py
```

## Quick CPU check

These commands use a small sample and one training epoch to verify the full workflow. The checkpoint and CSV go to `results/smoke/`, separate from the default experiment outputs.

```sh
python vp.py
python train.py --device cpu --train-channels 32 --val-channels 8 --symbols-per-channel 4 --epochs 1 --batch-size 32 --checkpoint results/smoke/vp_model.pt
python evaluate.py --device cpu --test-channels 8 --symbols-per-channel 4 --snr-db 10 20 --checkpoint results/smoke/vp_model.pt --csv results/smoke/ber.csv
```

## Interpret the results

Each row of `ber.csv` reports one method at one SNR.

- `ber` is the number of bit errors (`errors`) divided by the number of tested bits (`bits`).
- `mean_j_over_j0` is the mean of the per-sample ratio `J/J0`, where `J0` is the zero-perturbation cost. Lower values indicate lower precoding energy before power normalization.
- `gap_to_full` is the mean of the per-sample cost difference from the 192-action full search, normalized by `J0`. A value closer to zero indicates that screening retains more of the full search's benefit.

Compare neural Top-8 with the fixed 8-action subset to assess screening at the same action budget. Compare the two full searches to isolate the effect of expanding the action table.

## Repository notes

`results/` and `__pycache__/` are generated locally. The example `.vscode/settings.json` points to a machine-specific Windows interpreter. On another machine, select your own interpreter in VS Code.

## Reference

X. Wang, S. Fang, and S. Liao, “A Low Complexity Vector Perturbation Precoding with Projection Analysis Assisted Subspace Searching,” IEEE ICCCAS, 2024, pp. 497–502.

