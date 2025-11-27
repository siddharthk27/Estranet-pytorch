# evaluate_results.py
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import special
import argparse
import os
import evaluation_utils as eu  # the file you showed (evaluation_utils.py)

# ---- User options ----
TARGET_BYTE = 2         # the byte index within plaintext/key used for S-box (matches your snippet)
NUM_REPEATS = 20        # repeat ranking N times to average (because compute_key_rank shuffles internally)
PLOT = True
# ----------------------

def load_predictions(path):
    """
    Robust loader for results.txt that expects 256 values per prediction.
    Behaviors:
      - If file parses as a 2D array with second dim == 256, return it.
      - Otherwise, parse all numeric tokens and reshape into full (n,256) rows,
        dropping any trailing incomplete values (with a warning).
    """
    import numpy as np
    import io

    # Try the standard fast approach first (works if each line has 256 numbers)
    try:
        arr = np.loadtxt(path, dtype=float)
        # If loadtxt returned 2D and second dim == 256, we're done
        if arr.ndim == 2 and arr.shape[1] == 256:
            print(f"[load_predictions] Loaded {arr.shape[0]} rows x 256 cols with np.loadtxt.")
            return arr
        # If loadtxt returned 1D, or 2D with unexpected columns, fallthrough to robust parsing
    except Exception as e:
        # np.loadtxt sometimes fails for ragged rows or strange spacing; fall back
        print(f"[load_predictions] np.loadtxt failed or returned unexpected shape: {e}. Falling back to tolerant parser.")

    # Tolerant parser: read all numeric tokens and reshape into full 256-vectors
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    if text.strip() == "":
        raise ValueError(f"{path} is empty or contains no numeric data.")

    # Parse all floats from the file (whitespace separated)
    try:
        all_vals = np.fromstring(text, sep=' ')
        # np.fromstring accepts any whitespace as separator if sep=' ' is given
    except Exception as e:
        raise ValueError(f"Could not parse numeric values from {path}: {e}")

    total = all_vals.size
    full_rows = total // 256
    remainder = total - full_rows * 256

    if full_rows == 0:
        raise ValueError(f"No full 256-length prediction rows found in {path}. Total numeric tokens = {total}")

    if remainder != 0:
        print(f"[load_predictions] WARNING: Found {total} numeric tokens. Keeping {full_rows} full rows "
              f"({full_rows*256} values) and dropping {remainder} trailing values that don't make a full 256-vector.")
    else:
        print(f"[load_predictions] Parsed {full_rows} full rows (exact division).")

    preds = all_vals[: full_rows * 256].reshape((full_rows, 256)).astype(float)
    return preds


def load_metadata_bytes(h5file, group='Profiling_traces', byte_index=2):
    """
    Load plaintext and key bytes for each trace from metadata.
    Returns plaintexts (N,) and keys (N,) arrays of ints (0..255).
    Tries Profiling_traces first, then Attack_traces if not found.
    """
    f = h5py.File(h5file, 'r')
    if group not in f:
        # fallback to Attack_traces
        if 'Attack_traces' in f:
            group = 'Attack_traces'
        else:
            raise KeyError(f"Neither 'Profiling_traces' nor 'Attack_traces' found in {h5file}")
    meta = f[f'{group}/metadata']
    # metadata is typically a compound array. We'll iterate and extract fields.
    N = len(meta)
    plaintexts = np.zeros(N, dtype=np.int32)
    keys = np.zeros(N, dtype=np.int32)
    for i in range(N):
        rec = meta[i]
        # try common field names; adapt if your file is different
        plaintext = rec['plaintext']  # array-like (16 bytes)
        key = rec['key']              # array-like (16 bytes)
        plaintexts[i] = int(plaintext[byte_index])
        keys[i] = int(key[byte_index])
    f.close()
    return plaintexts, keys

def evaluate(predictions, plaintexts, keys, repeats=20):
    """
    Run compute_key_rank multiple times to average out randomness.
    Returns:
      - mean_rank_per_trace: (N,) average rank after each trace
      - success_rate_per_trace: (N,) average fraction with rank==0 after each trace
      - raw_ranks: list of arrays (repeats x N)
    """
    n_samples = predictions.shape[0]
    all_ranks = np.zeros((repeats, n_samples), dtype=int)
    for r in range(repeats):
        # compute_key_rank internally shuffles in-place, so pass copies
        preds_copy = predictions.copy()
        p_copy = plaintexts.copy()
        k_copy = keys.copy()
        ranks = eu.compute_key_rank(preds_copy, p_copy, k_copy)
        all_ranks[r] = ranks
        print(f"Repeat {r+1}/{repeats} done. final rank = {ranks[-1]}")
    mean_rank = all_ranks.mean(axis=0)
    success_rate = (all_ranks == 0).mean(axis=0)
    return mean_rank, success_rate, all_ranks

def plot_results(mean_rank, success_rate, out_prefix='eval'):
    x = np.arange(1, len(mean_rank)+1)
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(x, mean_rank)
    ax[0].set_xlabel("Number of traces (cumulative)")
    ax[0].set_ylabel("Guessing Entropy (mean key rank)")
    ax[0].set_title("Mean Key Rank vs #traces")
    ax[0].grid(True)

    ax[1].plot(x, success_rate)
    ax[1].set_xlabel("Number of traces (cumulative)")
    ax[1].set_ylabel("Success Rate (rank==0)")
    ax[1].set_title("Success Rate vs #traces")
    ax[1].grid(True)

    plt.tight_layout()
    png = f"{out_prefix}_performance.png"
    plt.savefig(png, dpi=150)
    print(f"Saved plot to {png}")

def main(args):
    preds = load_predictions(args.results)
    print("Predictions loaded, shape:", preds.shape)
    plaintexts, keys = load_metadata_bytes(args.h5, byte_index=args.byte)
    # Ensure lengths match
    Npred = preds.shape[0]
    Nmeta = plaintexts.shape[0]
    if Npred > Nmeta:
        print(f"Warning: predictions has {Npred} rows but metadata has {Nmeta} rows. Truncating predictions.")
        preds = preds[:Nmeta]
        Npred = Nmeta
    elif Nmeta > Npred:
        print(f"Warning: metadata has {Nmeta} rows but predictions has {Npred} rows. Truncating metadata.")
        plaintexts = plaintexts[:Npred]
        keys = keys[:Npred]

    # Convert to floats (softmax/log will be handled in compute_key_rank)
    preds = preds.astype(float)

    mean_rank, success_rate, all_ranks = evaluate(preds, plaintexts, keys, repeats=args.repeats)

    # Print summary numbers
    print("\nSummary:")
    print(f"Final mean guessing entropy (after {len(mean_rank)} traces): {mean_rank[-1]:.4f}")
    print(f"Final success rate (rank==0): {success_rate[-1]*100:.2f}%")

    # Optionally save per-trace results
    np.savetxt(args.out_prefix + "_mean_rank.txt", mean_rank, fmt="%.6f")
    np.savetxt(args.out_prefix + "_success_rate.txt", success_rate, fmt="%.6f")
    print(f"Saved per-trace results to {args.out_prefix}_mean_rank.txt and {args.out_prefix}_success_rate.txt")

    if args.plot:
        plot_results(mean_rank, success_rate, out_prefix=args.out_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "-r", default="results.txt", help="Path to results.txt (flattened logits)")
    parser.add_argument("--h5", "-d", default="ascadf.h5", help="Path to ASCADf h5 file")
    parser.add_argument("--byte", "-b", type=int, default=TARGET_BYTE, help="Target plaintext/key byte index")
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS, help="Number of repeats to average over")
    parser.add_argument("--out_prefix", default="eval", help="Prefix for output files")
    parser.add_argument("--plot", action="store_true", help="Save plots")
    args = parser.parse_args()
    main(args)
