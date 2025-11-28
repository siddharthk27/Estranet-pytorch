import os
import math
import time
import pickle
import argparse
import numpy as np
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data_utils
import data_utils_ches20
from transformer import Transformer
import evaluation_utils
import evaluation_utils_ches20


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min', verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        # Check if improved
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  → Validation improved! New best: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  → No improvement for {self.counter}/{self.patience} checks (best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n⚠ Early stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop


class ProgressTracker:
    """Track training progress with time estimates."""
    
    def __init__(self, total_steps, log_interval):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_time = time.time()  # Initialize last_time
        self.step_times = []
        self.max_history = 100  # Keep last 100 step times for ETA calculation
        
    def update(self, current_step):
        """Update progress and return formatted string."""
        current_time = time.time()
        
        # Calculate elapsed time
        elapsed = current_time - self.start_time
        
        # Store step time for better ETA estimation
        step_time = current_time - self.last_time
        self.step_times.append(step_time)
        if len(self.step_times) > self.max_history:
            self.step_times.pop(0)
        
        self.last_time = current_time
        
        # Calculate progress
        progress = (current_step / self.total_steps) * 100
        
        # Estimate time remaining
        if len(self.step_times) >= 2:  # Need at least 2 measurements
            avg_step_time = sum(self.step_times) / len(self.step_times)
            steps_remaining = self.total_steps - current_step
            eta_seconds = avg_step_time * steps_remaining
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."
        
        # Format elapsed time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # Create progress bar
        bar_length = 30
        filled = int(bar_length * current_step / self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        return f"Progress: {bar} {progress:.1f}% | Elapsed: {elapsed_str} | ETA: {eta}"
    
    def reset(self):
        """Reset timer for new training session."""
        self.start_time = time.time()
        self.last_time = self.start_time
        self.step_times = []


class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""
    
    def __init__(self, optimizer, max_lr, total_steps, warmup_steps=0, min_lr_ratio=0.0):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_lr(self):
        """Compute current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return (self.current_step / self.warmup_steps) * self.max_lr
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed = (1.0 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio
            return self.max_lr * decayed


def create_model(args, n_classes, device):
    """Create EstraNet model."""
    model = Transformer(
        n_layer=args.n_layer,
        d_model=args.d_model,
        d_head=args.d_head,
        n_head=args.n_head,
        d_inner=args.d_inner,
        n_head_softmax=args.n_head_softmax,
        d_head_softmax=args.d_head_softmax,
        dropout=args.dropout,
        n_classes=n_classes,
        conv_kernel_size=args.conv_kernel_size,
        n_conv_layer=args.n_conv_layer,
        pool_size=args.pool_size,
        d_kernel_map=args.d_kernel_map,
        beta_hat_2=args.beta_hat_2,
        model_normalization=args.model_normalization,
        head_initialization=args.head_initialization,
        softmax_attn=args.softmax_attn,
        output_attn=args.output_attn
    )
    model = model.to(device)
    return model


def train(args, train_loader, eval_loader, device):
    """Train the model."""
    
    # Create model
    n_classes = 4 if args.dataset == 'CHES20' else 256
    model = create_model(args, n_classes, device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineWarmupScheduler(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.train_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    
    # Load checkpoint if warm start
    start_step = 0
    loss_history = {}
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    
    if args.warm_start and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        scheduler.current_step = start_step
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']
        print(f"Resumed from step {start_step}")
    
    # Quick sanity check before training loop
    print("\n" + "=" * 80)
    print("DATA VALIDATION CHECK")
    print("=" * 80)
    
    train_iter_check = iter(train_loader)
    traces_check, labels_check = next(train_iter_check)
    traces_check = traces_check.to(device)
    labels_check = labels_check.to(device)
    
    print(f"Traces shape: {traces_check.shape}")
    print(f"Labels shape: {labels_check.shape}")
    print(f"Traces dtype: {traces_check.dtype}")
    print(f"Labels dtype: {labels_check.dtype}")
    print(f"Traces range: [{traces_check.min().item():.4f}, {traces_check.max().item():.4f}]")
    print(f"Traces mean: {traces_check.mean().item():.4f}, std: {traces_check.std().item():.4f}")
    print(f"Labels range: [{labels_check.min().item()}, {labels_check.max().item()}]")
    print(f"Unique labels: {torch.unique(labels_check).numel()} (should be close to 256)")
    print(f"First 10 labels: {labels_check[:10].tolist()}")
    
    if torch.allclose(traces_check[0], traces_check[1]):
        print("⚠️  WARNING: First two traces are identical!")
    else:
        print("✓ Traces appear to be different")
    
    model.eval()
    with torch.no_grad():
        outputs = model(traces_check[:2], training=False)
        logits = outputs[0]
        probs = F.softmax(logits, dim=1)
        print(f"\nModel output check:")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
        print(f"First sample probs (first 10): {probs[0, :10].tolist()}")
        print(f"Max prob: {probs.max().item():.4f}, Min prob: {probs.min().item():.6f}")
        
        loss_manual = F.cross_entropy(logits[:2], labels_check[:2])
        print(f"Manual loss for 2 samples: {loss_manual.item():.4f}")
    
    model.train()
    print("=" * 80 + "\n")
    
    # Training loop
    model.train()
    train_iter = iter(train_loader)
    
    running_loss = 0.0
    grad_norm_sum = 0.0
    iteration_count = 0
    
    for step in range(start_step, args.train_steps):
        # Get batch
        try:
            traces, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            traces, labels = next(train_iter)
        
        traces = traces.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        softmax_attn_smoothing = 1.0  # min(float(step) / args.train_steps, 1.0)
        outputs = model(traces, softmax_attn_smoothing=softmax_attn_smoothing, training=True)
        logits = outputs[0]
        
        # Compute loss
        if args.dataset == 'CHES20':
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        else:
            loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step()
        lr = scheduler.step()
        
        # Accumulate metrics
        running_loss += loss.item()
        grad_norm_sum += grad_norm.item()
        iteration_count += 1
        
        # Log and evaluate periodically
        if (step + 1) % args.iterations == 0:
            avg_loss = running_loss / iteration_count
            avg_grad_norm = grad_norm_sum / iteration_count
            
            print(f"[{step + 1:6d}] | gnorm {avg_grad_norm:5.2f} lr {lr:9.6f} | loss {avg_loss:5.2f}")
            
            # Evaluate on training set
            train_eval_loss = evaluate_loss(model, train_loader, device, args, 
                                           max_batches=args.max_eval_batch)
            print(f"Train batches[{args.max_eval_batch:5d}]                | loss {train_eval_loss:5.2f}")
            
            # Evaluate on validation set
            eval_loss = evaluate_loss(model, eval_loader, device, args,
                                     max_batches=args.max_eval_batch)
            print(f"Eval  batches[{args.max_eval_batch:5d}]                | loss {eval_loss:5.2f}")
            
            # Store loss history
            loss_history[step + 1] = {
                'gnorm': avg_grad_norm,
                'running_train_loss': avg_loss,
                'train_loss': train_eval_loss,
                'test_loss': eval_loss
            }
            
            # Reset accumulators
            running_loss = 0.0
            grad_norm_sum = 0.0
            iteration_count = 0
            
            model.train()
        
        # Save checkpoint periodically
        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            save_checkpoint(model, optimizer, step + 1, loss_history, args)
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, args.train_steps, loss_history, args)


def evaluate_loss(model, data_loader, device, args, max_batches=-1):
    """Evaluate loss on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, (traces, labels) in enumerate(data_loader):
            if max_batches > 0 and i >= max_batches:
                break
            
            traces = traces.to(device)
            labels = labels.to(device)
            
            outputs = model(traces, training=False)
            logits = outputs[0]
            
            if args.dataset == 'CHES20':
                loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
            else:
                loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, step, loss_history, args, best_eval_loss=None, elapsed_time=0):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint.pt')
    checkpoint_data = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'elapsed_time': elapsed_time
    }
    
    if best_eval_loss is not None:
        checkpoint_data['best_eval_loss'] = best_eval_loss
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  ✓ Saved checkpoint at step {step}")
    
    # Save loss history
    loss_path = os.path.join(args.checkpoint_dir, 'loss.pkl')
    with open(loss_path, 'wb') as f:
        pickle.dump(loss_history, f)


def evaluate(args, test_loader, test_dataset, device):
    """Evaluate the model and compute key ranks."""
    
    # Create model
    n_classes = 4 if args.dataset == 'CHES20' else 256
    model = create_model(args, n_classes, device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from step {checkpoint['step']}")
    
    # Run inference
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for i, (traces, labels) in enumerate(test_loader):
            if args.max_eval_batch > 0 and i >= args.max_eval_batch:
                break
            
            traces = traces.to(device)
            outputs = model(traces, training=False)
            logits = outputs[0]
            
            all_predictions.append(logits.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Get metadata
    if args.max_eval_batch > 0:
        nsamples = args.max_eval_batch * args.eval_batch_size
    else:
        nsamples = test_dataset.num_samples
    
    if args.dataset == 'ASCAD':
        plaintexts = test_dataset.plaintexts[:nsamples]
        keys = test_dataset.keys[:nsamples]
    elif args.dataset == 'CHES20':
        nonces = test_dataset.nonces[:nsamples]
        keys = test_dataset.umsk_keys
    
    # Compute key ranks
    key_rank_list = []
    for i in range(100):
        if args.dataset == 'ASCAD':
            key_ranks = evaluation_utils.compute_key_rank(predictions, plaintexts, keys)
        elif args.dataset == 'CHES20':
            key_ranks = evaluation_utils_ches20.compute_key_rank(predictions, nonces, keys)
        
        key_rank_list.append(key_ranks)
    
    key_ranks = np.stack(key_rank_list, axis=0)
    
    # Save results
    result_file = args.result_path + '.txt'
    with open(result_file, 'w') as fout:
        for i in range(key_ranks.shape[0]):
            for r in key_ranks[i]:
                fout.write(str(r) + '\t')
            fout.write('\n')
        
        mean_ranks = np.mean(key_ranks, axis=0)
        for r in mean_ranks:
            fout.write(str(r) + '\t')
        fout.write('\n')
    
    print(f"Results written to {result_file}")


def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate EstraNet')
    
    # Experiment config
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='ASCAD', choices=['ASCAD', 'CHES20'])
    parser.add_argument('--checkpoint_dir', type=str, default='./', help='Checkpoint directory')
    parser.add_argument('--warm_start', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--result_path', type=str, default='results', help='Path for results')
    parser.add_argument('--do_train', action='store_true', help='Train model')
    
    # Optimization config
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--clip', type=float, default=0.25, help='Gradient clipping')
    parser.add_argument('--min_lr_ratio', type=float, default=0.004)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--input_length', type=int, default=700)
    parser.add_argument('--data_desync', type=int, default=0)
    
    # Training config
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--train_steps', type=int, default=100000)
    parser.add_argument('--iterations', type=int, default=500, help='Log interval')
    parser.add_argument('--save_steps', type=int, default=10000)
    
    # Model config
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_head', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_inner', type=int, default=256)
    parser.add_argument('--n_head_softmax', type=int, default=4)
    parser.add_argument('--d_head_softmax', type=int, default=32)
    parser.add_argument('--d_kernel_map', type=int, default=128)
    parser.add_argument('--beta_hat_2', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--conv_kernel_size', type=int, default=3)
    parser.add_argument('--n_conv_layer', type=int, default=1)
    parser.add_argument('--pool_size', type=int, default=2)
    parser.add_argument('--model_normalization', type=str, default='preLC')
    parser.add_argument('--head_initialization', type=str, default='forward')
    parser.add_argument('--softmax_attn', action='store_true', default=True)
    
    # Evaluation config
    parser.add_argument('--max_eval_batch', type=int, default=-1)
    parser.add_argument('--output_attn', action='store_true')
    
    # Early stopping config
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Number of evaluations without improvement before stopping')
    parser.add_argument('--early_stopping_delta', type=float, default=0.0001,
                       help='Minimum change in validation loss to qualify as improvement')
    parser.add_argument('--disable_early_stopping', action='store_true',
                       help='Disable early stopping')
    
    args = parser.parse_args()
    
    # Disable early stopping if requested
    if args.disable_early_stopping:
        args.early_stopping_patience = float('inf')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load datasets
    if args.dataset == 'ASCAD':
        if args.do_train:
            train_dataset = data_utils.ASCADDataset(
                args.data_path, 'train', args.input_length, args.data_desync
            )
            test_dataset = data_utils.ASCADDataset(
                args.data_path, 'test', args.input_length, 0
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=args.train_batch_size,
                shuffle=True, num_workers=4, pin_memory=True, drop_last=True
            )
            eval_loader = DataLoader(
                test_dataset, batch_size=args.eval_batch_size,
                shuffle=False, num_workers=4, pin_memory=True, drop_last=True
            )
        else:
            test_dataset = data_utils.ASCADDataset(
                args.data_path, 'test', args.input_length, 0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.eval_batch_size,
                shuffle=False, num_workers=4, pin_memory=True, drop_last=True
            )
    
    elif args.dataset == 'CHES20':
        if args.do_train:
            train_dataset = data_utils_ches20.CHES20Dataset(
                args.data_path + '.npz', 'train', args.input_length, args.data_desync
            )
            test_dataset = data_utils_ches20.CHES20Dataset(
                args.data_path + '_valid.npz', 'test', args.input_length, 0
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=args.train_batch_size,
                shuffle=True, num_workers=4, pin_memory=True, drop_last=True
            )
            eval_loader = DataLoader(
                test_dataset, batch_size=args.eval_batch_size,
                shuffle=False, num_workers=4, pin_memory=True, drop_last=True
            )
        else:
            test_dataset = data_utils_ches20.CHES20Dataset(
                args.data_path + '.npz', 'test', args.input_length, 0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.eval_batch_size,
                shuffle=False, num_workers=4, pin_memory=True, drop_last=True
            )
    
    # Train or evaluate
    if args.do_train:
        print("Starting training...")
        train(args, train_loader, eval_loader, device)
    else:
        print("Starting evaluation...")
        evaluate(args, test_loader, test_dataset, device)


if __name__ == '__main__':
    main()