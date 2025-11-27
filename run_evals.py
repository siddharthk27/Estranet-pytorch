import numpy as np
import h5py
import matplotlib.pyplot as plt
from evaluation_utils import compute_key_rank

# --- CONFIGURATION ---
RESULTS_FILE = 'results.txt'   # Your model predictions
DATASET_FILE = 'ASCAD.h5'      # The original H5 dataset file (UPDATE THIS PATH)
TARGET_BYTE = 2                # The byte you attacked (Matches your utils code)
# ---------------------

def load_predictions(file_path):
    print(f"Loading predictions from {file_path}...")
    # Assuming the text file is whitespace or comma separated values
    # resulting in a matrix of shape (N_traces, 256)
    try:
        preds = np.loadtxt(file_path)
    except ValueError:
        # Fallback if it's comma separated
        preds = np.loadtxt(file_path, delimiter=',')

    print(f"Predictions shape: {preds.shape}")
    return preds

def run_evaluation():
    # 1. Load Model Predictions
    predictions = load_predictions(RESULTS_FILE)
    num_traces = predictions.shape[0]

    # 2. Load Ground Truth (Plaintext and Key) from H5
    # Note: Usually evaluation is done on 'Attack_traces', not 'Profiling_traces'
    print(f"Loading ground truth from {DATASET_FILE}...")
    with h5py.File(DATASET_FILE, 'r') as f:
        # We slice [:num_traces] to match the number of predictions you have
        plaintexts = f['Attack_traces']['metadata']['plaintext'][:num_traces, TARGET_BYTE]
        keys = f['Attack_traces']['metadata']['key'][:num_traces, TARGET_BYTE]

    # 3. Compute Key Rank using your provided utility
    print("Computing key rank (this may take a moment)...")
    ranks = compute_key_rank(predictions, plaintexts, keys)

    # 4. Output Results
    final_rank = ranks[-1]
    print("-" * 30)
    print(f"Final Key Rank after {num_traces} traces: {final_rank}")

    # A rank of 0 means the model successfully predicted the correct key.
    if final_rank == 0:
        print("SUCCESS: The key was fully recovered!")
    else:
        print(f"FAILURE: The correct key is at rank {final_rank} (0 is best).")

    # 5. (Optional) Plotting Guessing Entropy
    # Guessing Entropy is the average rank. For a single run, it's just the rank over time.
    plt.plot(ranks)
    plt.title('Key Rank Evolution (Guessing Entropy)')
    plt.xlabel('Number of Traces')
    plt.ylabel('Rank of Correct Key')
    plt.yscale('log') # Log scale is standard for SCA
    plt.grid(True, which="both", ls="-")
    plt.savefig('guessing_entropy.png')
    print("Plot saved to guessing_entropy.png")

if __name__ == "__main__":
    run_evaluation()
