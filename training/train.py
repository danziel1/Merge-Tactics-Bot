from ultralytics import YOLO

# Load YOLOv11 pretrained weights (e.g., nano version)
model = YOLO("yolo11n.pt")  # You can also try yolo11s.pt, yolo11m.pt, etc.

# Train the model
model.train(
    data="./data.yaml",         # Path to your dataset config
    epochs=1000,                 # Sufficient for convergence, but monitor val loss
    imgsz=640,                  # Standard resolution; increase if objects are small
    lr0=0.0005,                 # Low starting LR for stability
    lrf=0.01,                   # Final LR fraction (cosine decay target)
    cos_lr=True,                # Cosine learning rate schedule
    batch=9,                   # Adjust based on memory; odd batch sizes are fine
    augment=True,              # Enables default augmentations
    cache=True,                # Speeds up training by caching images
    device="cpu",              # Switch to "cuda" if you have a GPU
    # resume=True,             # Uncomment to resume from last checkpoint
    # patience=25,             # Use with early stopping if needed

    # 🔧 Advanced Augmentations
    mosaic=True,               # Combines 4 images — boosts generalization
    mixup=0.2,                 # Blends images and labels — good for small datasets
    hsv_h=0.015,               # Hue jitter
    hsv_s=0.7,                 # Saturation jitter
    hsv_v=0.4,                 # Value jitter
    flipud=0.5,                # Vertical flip
    fliplr=0.5,                # Horizontal flip
    degrees=10,                # Rotation
    scale=0.5,                 # Random scaling
    shear=2.0,                 # Shearing — adds geometric distortion

    # 🧠 Optimization & Regularization
    dropout=0.1,               # Helps prevent overfitting
    label_smoothing=0.1,       # Softens hard labels — improves generalization
    warmup_epochs=3,           # Stabilizes early training
    weight_decay=0.0005,       # Regularization to reduce overfitting

    # 📊 Logging & Evaluation
    val=True,                  # Run validation after training
    plots=True,                # Save training plots (loss, mAP, etc.)
    # save_period=10             # Save checkpoints every N epochs
)

print("Finished Training")
# epoch 292 in 4.833 hours