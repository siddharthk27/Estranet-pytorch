# EstraNet: An Efficient Shift-Invariant Transformer Network for Side Channel Analysis (PyTorch)

This repository contains the **PyTorch implementation** of **EstraNet**, an efficient shift-invariant transformer network for Side-Channel Analysis.  
For more details, refer to the [paper](https://tches.iacr.org/index.php/TCHES/article/view/11255).

> **Note**: This is a PyTorch conversion of the original TensorFlow implementation.

---

## Repository Structure
- **`fast_attention.py`** – Implements the proposed GaussiP attention layer
- **`normalization.py`** – Implements the layer-centering normalization
- **`transformer.py`** – Defines the EstraNet model architecture
- **`train_trans.py`** – Training and evaluation script for EstraNet
- **`data_utils.py`** – Utilities for loading ASCADf and ASCADr datasets
- **`data_utils_ches20.py`** – Utilities for loading the CHES20 dataset
- **`evaluation_utils.py`** – Computes guessing entropy for ASCAD datasets
- **`evaluation_utils_ches20.py`** – Computes guessing entropy for CHES20 dataset
- **`run_trans_\<dataset\>.sh`** – Bash scripts with predefined hyperparameters for specific datasets:
  - **ASCADf** ([fixed key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key))
  - **ASCADr** ([random key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key))
  - **CHES20** ([CHES CTF 2020](https://ctf.spook.dev/))

---

## Key Differences from TensorFlow Version

### Framework Changes
- Converted from TensorFlow 2.x to PyTorch 2.x
- Uses PyTorch's `nn.Module` instead of `tf.keras.layers.Layer`
- Uses PyTorch's DataLoader instead of `tf.data.Dataset`
- Native PyTorch optimizers and learning rate schedulers

### Implementation Notes
- Random feature projection matrices are generated using PyTorch's QR decomposition
- Attention mechanisms use PyTorch's `einsum` operations
- Gradient clipping uses `torch.nn.utils.clip_grad_norm_`
- Model checkpointing uses PyTorch's native format (`.pt` files)

---

## Data Pre-processing
- For the **CHES CTF 2020** dataset, the traces are multiplied by a constant `0.004` to keep the feature value range within **[-120, 120]**.

---

## Requirements

### Tested On
- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- SciPy 1.10+
- h5py 3.8+

### Installation

```bash
pip install -r requirements.txt
```

For GPU support, ensure you have CUDA installed and install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/EstraNet-PyTorch.git
cd EstraNet-PyTorch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset
Download one of the supported datasets:
- [ASCAD Fixed Key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key)
- [ASCAD Random Key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key)
- [CHES20 CTF](https://ctf.spook.dev/)

### 4. Set dataset path
Edit the bash script for your dataset:
```bash
# For ASCAD fixed key
nano run_trans_ascadf.sh

# Set DATA_PATH to your dataset location
DATA_PATH=/path/to/ASCAD.h5
```

### 5. Train EstraNet
```bash
# For ASCAD fixed key
bash run_trans_ascadf.sh train

# For ASCAD random key
bash run_trans_ascadr.sh train

# For CHES20
bash run_trans_ches20.sh train
```

### 6. Evaluate Model
```bash
# For ASCAD fixed key
bash run_trans_ascadf.sh test

# For ASCAD random key
bash run_trans_ascadr.sh test

# For CHES20
bash run_trans_ches20.sh test
```

---

## Training from Python

You can also train directly using Python:

```bash
python train_trans.py \
    --data_path=/path/to/dataset.h5 \
    --dataset=ASCAD \
    --input_length=10000 \
    --data_desync=200 \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --train_steps=400000 \
    --n_layer=2 \
    --d_model=128 \
    --n_head=8 \
    --do_train
```

---

## Model Architecture

EstraNet consists of:
1. **Convolutional Layers**: Extract local features from traces
2. **Positional Features**: Generate shift-invariant positional encodings
3. **Transformer Layers**: Self-attention layers with GaussiP attention
4. **Output Layer**: Classification head with optional softmax attention pooling

### Key Features
- **GaussiP Attention**: Efficient linear attention using random Fourier features
- **Shift Invariance**: Handles trace desynchronization naturally
- **Layer Centering**: Novel normalization technique for better training

---

## Results

Results are saved in the format specified by `--result_path`:
- `results.txt`: Key ranks for each trace
- `checkpoint.pt`: Model checkpoint
- `loss.pkl`: Training loss history

---

## GPU Usage

The implementation automatically uses GPU if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

For multi-GPU training, you can use `torch.nn.DataParallel`:
```python
model = nn.DataParallel(model)
```

---

## Citation

```bibtex
@article{DBLP:journals/tches/HajraCM24,
  author       = {Suvadeep Hajra and
                  Siddhartha Chowdhury and
                  Debdeep Mukhopadhyay},
  title        = {EstraNet: An Efficient Shift-Invariant Transformer Network for Side-Channel
                  Analysis},
  journal      = {{IACR} Trans. Cryptogr. Hardw. Embed. Syst.},
  volume       = {2024},
  number       = {1},
  pages        = {336--374},
  year         = {2024},
  url          = {https://doi.org/10.46586/tches.v2024.i1.336-374},
  doi          = {10.46586/TCHES.V2024.I1.336-374}
}
```

---

## License

This project maintains the same license as the original TensorFlow implementation.

---

## Acknowledgments

This is a PyTorch conversion of the original TensorFlow implementation. All credit for the model architecture and methodology goes to the original authors.
