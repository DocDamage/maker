"""
Music Generation Script

Generate new music using your fine-tuned MusicGen model.

Usage:
    python scripts/generate.py --prompt "upbeat electronic dance music"
    python scripts/generate.py --model models/lora/final --prompt "calm piano melody"
    python scripts/generate.py --prompt "energetic rock" --duration 60 --count 5
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_NAME, LORA_DIR, OUTPUT_DIR,
    USE_LORA, GENERATION_CONFIG
)

import torch


def load_model(lora_path: Path = None):
    """Load MusicGen with optional LoRA adapter."""
    from audiocraft.models import MusicGen
    
    print(f"📦 Loading base model: {MODEL_NAME}")
    model = MusicGen.get_pretrained(MODEL_NAME)
    
    # Load LoRA adapter if specified
    if lora_path and lora_path.exists():
        print(f"🔧 Loading LoRA adapter: {lora_path}")
        try:
            from peft import PeftModel
            model.lm = PeftModel.from_pretrained(
                model.lm,
                str(lora_path),
                is_trainable=False,
            )
            print("   ✓ LoRA adapter loaded successfully")
        except Exception as e:
            print(f"   ⚠️ Could not load LoRA adapter: {e}")
            print("   Falling back to base model")
    elif lora_path:
        print(f"   ⚠️ LoRA path not found: {lora_path}")
        print("   Using base model instead")
    
    return model


def generate_music(
    model,
    prompts: list[str],
    duration: float = 30.0,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.95,
    cfg_coef: float = 3.0,
) -> list:
    """Generate music from text prompts."""
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef,
    )
    
    print(f"\n🎵 Generating {len(prompts)} track(s)...")
    print(f"   Duration: {duration} seconds")
    print(f"   Temperature: {temperature}")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Generate
    with torch.no_grad():
        wav = model.generate(prompts)
    
    return wav


def save_audio(waveforms, prompts: list[str], output_dir: Path, sample_rate: int = 32000):
    """Save generated audio to files."""
    import torchaudio
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = []
    
    for i, (wav, prompt) in enumerate(zip(waveforms, prompts)):
        # Create safe filename from prompt
        safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)
        safe_prompt = safe_prompt[:50].strip().replace(" ", "_")
        
        filename = f"{timestamp}_{i:02d}_{safe_prompt}.wav"
        filepath = output_dir / filename
        
        # Ensure correct shape (channels, samples)
        if wav.dim() == 3:
            wav = wav.squeeze(0)  # Remove batch dimension
        
        # Save
        torchaudio.save(str(filepath), wav.cpu(), sample_rate)
        saved_files.append(filepath)
        
        print(f"   💾 Saved: {filename}")
    
    return saved_files


def extend_audio(model, audio_path: Path, prompt: str, num_extensions: int = 2):
    """Extend an existing audio file using the model."""
    import torchaudio
    
    print(f"\n🔄 Extending: {audio_path}")
    
    # Load existing audio
    waveform, sr = torchaudio.load(str(audio_path))
    
    # Resample if needed
    if sr != 32000:
        resampler = torchaudio.transforms.Resample(sr, 32000)
        waveform = resampler(waveform)
    
    # Generate continuation
    model.set_generation_params(
        duration=30 * num_extensions,
        extend_stride=GENERATION_CONFIG.get("extend_stride", 18),
    )
    
    with torch.no_grad():
        extended = model.generate_continuation(
            waveform.unsqueeze(0),
            32000,
            [prompt],
        )
    
    return extended


def main():
    parser = argparse.ArgumentParser(description="Generate music with MusicGen")
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="Text description of the music to generate")
    parser.add_argument("--model", "-m", type=Path, default=None,
                        help="Path to LoRA model (default: latest in models/lora/)")
    parser.add_argument("--output", "-o", type=Path, default=OUTPUT_DIR,
                        help="Output directory for generated audio")
    parser.add_argument("--duration", "-d", type=float, default=GENERATION_CONFIG["duration"],
                        help="Duration in seconds")
    parser.add_argument("--count", "-n", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--temperature", "-t", type=float, default=GENERATION_CONFIG["temperature"],
                        help="Creativity (0.5-1.5)")
    parser.add_argument("--cfg", type=float, default=GENERATION_CONFIG["cfg_coef"],
                        help="Classifier-free guidance strength")
    parser.add_argument("--extend", type=Path, default=None,
                        help="Extend an existing audio file")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    print(f"\n🎵 MusicGen Generator")
    print(f"{'=' * 50}")
    
    # Set seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"   Seed: {args.seed}")
    
    # Find LoRA model
    lora_path = args.model
    if lora_path is None and USE_LORA:
        # Look for latest training run
        if LORA_DIR.exists():
            runs = sorted([d for d in LORA_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")])
            if runs:
                latest_run = runs[-1]
                final_path = latest_run / "final"
                if final_path.exists():
                    lora_path = final_path
                    print(f"   Using latest model: {lora_path}")
    
    # Load model
    model = load_model(lora_path)
    
    # Generate or extend
    if args.extend:
        # Extension mode
        wav = extend_audio(model, args.extend, args.prompt)
        prompts = [f"extended_{args.prompt}"]
    else:
        # Regular generation
        prompts = [args.prompt] * args.count
        wav = generate_music(
            model=model,
            prompts=prompts,
            duration=args.duration,
            temperature=args.temperature,
            top_k=GENERATION_CONFIG["top_k"],
            top_p=GENERATION_CONFIG["top_p"],
            cfg_coef=args.cfg,
        )
    
    # Save outputs
    print(f"\n💾 Saving to: {args.output}")
    saved = save_audio(wav, prompts, args.output)
    
    print(f"\n{'=' * 50}")
    print(f"✅ Generation complete!")
    print(f"   Generated {len(saved)} file(s)")
    for f in saved:
        print(f"   • {f}")


if __name__ == "__main__":
    main()
