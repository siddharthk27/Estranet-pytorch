import numpy as np
import torch
import torch.nn.functional as F
import sys


def sbox_layer(x):
    """CHES20 S-box layer."""
    y1 = (x[0] & x[1]) ^ x[2]
    y0 = (x[3] & x[0]) ^ x[1]
    y3 = (y1 & x[3]) ^ x[0]
    y2 = (y0 & y1) ^ x[3]
    return np.stack([y0, y1, y2, y3], axis=1)


def shuffle_all(predictions, nonces):
    """Shuffle predictions and nonces together."""
    perm = np.random.permutation(predictions.shape[0])
    predictions = predictions[perm]
    nonces = nonces[perm]

    return predictions, nonces


def gen_key_bits():
    """Generate all possible 4-bit key combinations."""
    values = np.arange(16, dtype=np.uint8).reshape(-1, 1)
    key_bits = np.unpackbits(values, axis=1)[:, -4:]

    # Swap bits according to the specific bit order
    for k in range(16):
        t = key_bits[k, 0]
        key_bits[k, 0] = key_bits[k, 3]
        key_bits[k, 3] = t
        t = key_bits[k, 1]
        key_bits[k, 1] = key_bits[k, 2]
        key_bits[k, 2] = t

    return key_bits


def compute_key_rank(predictions, nonces, keys):
    """Compute key rank (guessing entropy) for CHES20 dataset.

    Args:
        predictions: Model predictions [N, 4] (logits)
        nonces: Nonce values [N, 4]
        keys: True key values [4]

    Returns:
        key_ranks: Rank of correct key at each trace [N]
    """
    n_samples, n_classes = predictions.shape
    nonces = (nonces[:n_samples] & 0x1)
    keys = np.squeeze(keys)

    predictions, nonces = shuffle_all(predictions, nonces)

    # Get correct key value
    def get_corr_key(keys):
        corr_key = ((keys[0] & 0x1) << 0)
        corr_key |= ((keys[1] & 0x1) << 1)
        corr_key |= ((keys[2] & 0x1) << 2)
        corr_key |= ((keys[3] & 0x1) << 3)
        return corr_key

    corr_key = get_corr_key(keys)

    # Generate all possible key bits
    key_bits = gen_key_bits()
    n_keys = key_bits.shape[0]

    # Compute negative log probability for each possible key
    neg_log_prob = np.zeros((n_samples, n_keys))

    predictions_tensor = torch.from_numpy(predictions)

    for k in range(n_keys):
        key_rep = np.reshape(key_bits[k, :], [1, -1])
        sbox_in = (nonces ^ key_rep).T
        sbox_out = (sbox_layer(sbox_in) & 0x1)
        sbox_out = sbox_out.astype(np.float32)

        sbox_out_tensor = torch.from_numpy(sbox_out)

        # Compute cross entropy loss
        scores = F.binary_cross_entropy_with_logits(
            predictions_tensor,
            sbox_out_tensor,
            reduction='none'
        ).mean(dim=1).numpy()

        neg_log_prob[:, k] = scores

    # Compute cumulative negative log probabilities
    cum_neg_log_prob = np.zeros((n_samples, n_keys))
    last_neg_log_prob = np.zeros((1, n_keys))
    for i in range(n_samples):
        last_neg_log_prob += neg_log_prob[i]
        cum_neg_log_prob[i, :] = last_neg_log_prob

    # Find rank of correct key (lower loss = better)
    sorted_keys = np.argsort(cum_neg_log_prob, axis=1)
    key_ranks = np.zeros((n_samples), dtype=int) - 1
    for i in range(n_samples):
        for j in range(n_keys):
            if sorted_keys[i, j] == corr_key:
                key_ranks[i] = j
                break

    # Verify all ranks were found
    for i in range(n_samples):
        assert key_ranks[i] >= 0, f"Assertion failed at index {i}"

    return key_ranks


if __name__ == '__main__':
    # Test key bit generation
    key_bits = gen_key_bits()
    print("Key bits shape:", key_bits.shape)
    print("First 5 key combinations:")
    print(key_bits[:5])
