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

    # ==========================================================================
    # START DIAGNOSTIC INSERTION
    # ==========================================================================
    print("\n" + "="*80)
    print("DATA VALIDATION CHECK")
    print("="*80)

    # Get one batch
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

    # Check if all traces are the same
    if torch.allclose(traces_check[0], traces_check[1]):
        print("⚠️  WARNING: First two traces are identical!")
    else:
        print("✓ Traces appear to be different")

    # Check model output
    model.eval()
    with torch.no_grad():
        outputs = model(traces_check[:2], training=False)
        logits = outputs[0]
        # Note: If using CHES20, this softmax might need to be sigmoid depending on n_classes
        if args.dataset != 'CHES20':
            probs = F.softmax(logits, dim=1)
            print(f"\nModel output check:")
            print(f"Logits shape: {logits.shape}")
            print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
            print(f"First sample probs (first 10): {probs[0, :10].tolist()}")
            print(f"Max prob: {probs.max().item():.4f}, Min prob: {probs.min().item():.6f}")
            
            # Compute loss manually
            loss_manual = F.cross_entropy(logits[:2], labels_check[:2])
            print(f"Manual loss for 2 samples: {loss_manual.item():.4f}")
        else:
            print("Skipping detailed Softmax check (CHES20 dataset detected).")

    model.train()
    print("="*80 + "\n")
    # ==========================================================================
    # END DIAGNOSTIC INSERTION
    # ==========================================================================

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