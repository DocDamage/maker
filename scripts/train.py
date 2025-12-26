"""
MusicGen LoRA Training Script

Fine-tunes MusicGen on your music collection using LoRA for memory efficiency.
Optimized for RTX 3060 (12GB VRAM).

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 20 --lr 5e-5
    python scripts/train.py --resume models/lora/checkpoint-500
"""

import argparse
import json
import sys
import random
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROCESSED_AUDIO_DIR, METADATA_DIR, LORA_DIR,
    MODEL_NAME, USE_LORA, LORA_CONFIG, TRAINING_CONFIG, SAMPLE_RATE
)

import torch
import numpy as np
from tqdm import tqdm

# Check CUDA availability
if not torch.cuda.is_available():
    print("⚠️  Warning: CUDA not available. Training will be very slow on CPU.")
    print("   Make sure you have NVIDIA drivers and CUDA toolkit installed.")


def check_dependencies():
    """Verify all required packages are installed."""
    missing = []
    
    try:
        import audiocraft
    except ImportError:
        missing.append("audiocraft")
    
    try:
        import peft
    except ImportError:
        missing.append("peft")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import accelerate
    except ImportError:
        missing.append("accelerate")
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def load_dataset(processed_dir: Path, metadata_dir: Path) -> list[dict]:
    """Load audio paths and descriptions."""
    jsonl_path = metadata_dir / "data.jsonl"
    
    if not jsonl_path.exists():
        print(f"❌ Metadata not found: {jsonl_path}")
        print(f"   Run auto-labeling first: python scripts/autolabel.py")
        sys.exit(1)
    
    dataset = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            audio_path = Path(entry["path"])
            if audio_path.exists():
                dataset.append(entry)
            else:
                print(f"   Warning: Audio file not found: {audio_path}")
    
    return dataset


def setup_model_and_lora():
    """Load MusicGen and apply LoRA adapters."""
    from audiocraft.models import MusicGen
    
    print(f"📦 Loading model: {MODEL_NAME}")
    model = MusicGen.get_pretrained(MODEL_NAME)
    
    if USE_LORA:
        print("🔧 Applying LoRA adapters...")
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Get the language model component
        lm = model.lm
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            target_modules=LORA_CONFIG["target_modules"],
            bias=LORA_CONFIG["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model.lm = get_peft_model(lm, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.lm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.lm.parameters())
        print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def load_audio(file_path: str, duration: float = 30.0) -> torch.Tensor:
    """Load audio file as tensor."""
    import torchaudio
    
    waveform, sr = torchaudio.load(file_path)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Trim or pad to exact duration
    target_length = int(duration * SAMPLE_RATE)
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform


class MusicDataset(torch.utils.data.Dataset):
    """Dataset for MusicGen training."""
    
    def __init__(self, data: list[dict], duration: float = 30.0):
        self.data = data
        self.duration = duration
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        audio = load_audio(entry["path"], self.duration)
        return {
            "audio": audio,
            "description": entry["description"],
        }


def train(
    model,
    dataset: list[dict],
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    save_steps: int = 500,
    logging_steps: int = 50,
    resume_from: str = None,
):
    """Main training loop."""
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Training on: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create dataset and dataloader
    train_dataset = MusicDataset(dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep at 0 for Windows compatibility
        pin_memory=True,
    )
    
    # Move model to device
    model.lm.to(device)
    
    # Enable gradient checkpointing
    if TRAINING_CONFIG.get("gradient_checkpointing", True):
        if hasattr(model.lm, "gradient_checkpointing_enable"):
            model.lm.gradient_checkpointing_enable()
            print("   Gradient checkpointing: enabled")
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.lm.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # Scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if TRAINING_CONFIG.get("fp16", True) else None
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if resume_from:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            print(f"📂 Resuming from: {checkpoint_path}")
            # Load LoRA weights
            model.lm.load_adapter(str(checkpoint_path), adapter_name="default")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Total steps: {total_steps}")
    print(f"   Dataset size: {len(dataset)} samples")
    
    model.lm.train()
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for step, batch in enumerate(progress):
            # Get audio and descriptions
            audio = batch["audio"].to(device)
            descriptions = batch["description"]
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                # Encode audio with compression model
                with torch.no_grad():
                    codes, scale = model.compression_model.encode(audio)
                
                # Get text conditioning
                attributes, _ = model._prepare_tokens_and_attributes(descriptions, None)
                conditions = model._prepare_conditions(attributes)
                
                # Language model forward
                # Note: This is a simplified version - actual implementation
                # depends on MusicGen internals which may vary by version
                try:
                    output = model.lm.compute_predictions(
                        codes=codes,
                        conditions=conditions,
                    )
                    loss = output.loss if hasattr(output, 'loss') else output[0]
                except Exception as e:
                    # Fallback for different MusicGen versions
                    logits = model.lm(codes, conditions=conditions)
                    # Simple cross-entropy loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        codes.view(-1),
                        ignore_index=-100,
                    )
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Optimizer step
            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.lm.parameters(),
                        TRAINING_CONFIG.get("max_grad_norm", 1.0)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.lm.parameters(),
                        TRAINING_CONFIG.get("max_grad_norm", 1.0)
                    )
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.lm.save_pretrained(str(checkpoint_dir))
                    print(f"\n   💾 Checkpoint saved: {checkpoint_dir}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"   Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(str(final_dir))
    print(f"\n✅ Training complete!")
    print(f"   Final model saved to: {final_dir}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train MusicGen with LoRA")
    parser.add_argument("--epochs", "-e", type=int, default=TRAINING_CONFIG["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--batch-size", "-b", type=int, default=TRAINING_CONFIG["batch_size"],
                        help="Batch size")
    parser.add_argument("--resume", "-r", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--output", "-o", type=Path, default=LORA_DIR,
                        help="Output directory for model")
    args = parser.parse_args()
    
    print(f"\n🎵 MusicGen LoRA Trainer")
    print(f"{'=' * 50}")
    
    # Check dependencies
    check_dependencies()
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    dataset = load_dataset(PROCESSED_AUDIO_DIR, METADATA_DIR)
    
    if len(dataset) == 0:
        print("❌ No valid training samples found!")
        print("   Make sure you've run both preprocess.py and autolabel.py")
        sys.exit(1)
    
    print(f"   Found {len(dataset)} training samples")
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Setup model
    print()
    model = setup_model_and_lora()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "use_lora": USE_LORA,
            "lora_config": LORA_CONFIG,
            "training_config": TRAINING_CONFIG,
            "dataset_size": len(dataset),
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
        }, f, indent=2)
    
    # Train
    train(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=args.lr,
        save_steps=TRAINING_CONFIG["save_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        resume_from=args.resume,
    )
    
    print(f"\n   Next step: python scripts/generate.py --model {output_dir / 'final'}")


if __name__ == "__main__":
    main()
