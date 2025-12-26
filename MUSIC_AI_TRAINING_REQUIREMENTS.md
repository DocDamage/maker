# Music AI Training Requirements Research

A comprehensive guide to fine-tuning music generation AI models on a custom music collection.

---

## Your Hardware Assessment

| Component | Your Specs | Assessment |
|-----------|------------|------------|
| **GPU** | RTX 3060 (12GB VRAM) | ✅ **Viable for LoRA training** |
| **CPU** | i5-14600K | ✅ Excellent for preprocessing |
| **RAM** | 48GB DDR4 | ✅ More than sufficient |
| **Storage** | ~5.5TB total SSD | ✅ Plenty for datasets |

> [!TIP]
> Your RTX 3060 with 12GB VRAM is sufficient for **LoRA fine-tuning** of MusicGen-small and even medium models with optimizations. Full fine-tuning of larger models would require cloud resources.

---

## Model Options for Your Hardware

### 🎯 Recommended: MusicGen with LoRA

**Why MusicGen?**
- Open source (Meta/Facebook)
- Active community support
- LoRA training works on consumer GPUs
- Good quality results with small datasets

| Model Variant | Parameters | Your GPU Feasibility |
|---------------|------------|---------------------|
| `musicgen-small` | 300M | ✅ Full fine-tune possible |
| `musicgen-medium` | 1.5B | ⚡ LoRA only (recommended) |
| `musicgen-large` | 3.3B | ⚠️ LoRA with aggressive optimization |
| `musicgen-melody` | 1.5B | ⚡ LoRA only |

### Alternative Options

| Model | Feasibility | Notes |
|-------|-------------|-------|
| **Riffusion** | ✅ Excellent | Spectrogram-based, very consumer-friendly |
| **AudioLDM 2** | ⚡ Moderate | Needs optimization for 12GB |
| **Jukebox** | ❌ Not practical | Requires 15GB+ VRAM, very slow |
| **Stable Audio** | ⚠️ Limited | Inference only, training not public |

---

## Dataset Requirements

### Audio Specifications

```
Format:       .mp3, .wav, or .flac
Sample Rate:  32,000 Hz (mono recommended)
Duration:     5-30 seconds per clip (longer files auto-chunked)
Quality:      High bitrate, minimal noise/artifacts
```

### Dataset Size Guidelines

| Goal | Minimum Dataset | Recommended | Notes |
|------|-----------------|-------------|-------|
| **Style transfer** | 10-20 songs | 50+ songs | Learn specific artist/genre |
| **Genre training** | 100+ hours | 500+ hours | Broader musical understanding |
| **Production model** | 1,000+ hours | 20,000+ hours | MusicGen was trained on 20k hours |

> [!NOTE]
> For fine-tuning to capture *your* collection's style, **50-200 songs** is often sufficient with LoRA. You're teaching stylistic nuances, not building from scratch.

### Metadata Requirements

Each audio file needs a corresponding JSON description:

```json
{
  "description": "upbeat electronic dance track with synthesizer leads and driving bass",
  "genre": "EDM",
  "mood": "energetic",
  "bpm": 128,
  "key": "A minor",
  "instruments": ["synthesizer", "drums", "bass"]
}
```

**Auto-labeling tools available:**
- Essentia (open-source audio analysis)
- Demucs (for stem separation)
- Built-in auto-labeling in some training scripts

---

## Training Time Estimates (RTX 3060)

| Scenario | Dataset | Training Time |
|----------|---------|---------------|
| Quick test | 10 songs (~30 min audio) | 2-4 hours |
| Style capture | 50 songs (~2.5 hours) | 8-16 hours |
| Thorough training | 200 songs (~10 hours) | 2-4 days |

> [!WARNING]
> Training times can vary significantly based on:
> - Number of training epochs
> - Batch size (lower = slower but less VRAM)
> - LoRA rank (higher = more parameters to train)

---

## VRAM Optimization Techniques

To maximize what you can train on 12GB:

### Essential Optimizations

1. **LoRA (Low-Rank Adaptation)** - Reduces VRAM from 40GB+ to ~10GB
2. **Half Precision (FP16)** - Cuts memory usage roughly in half
3. **Gradient Checkpointing** - Trades compute for memory
4. **Gradient Accumulation** - Simulates larger batches

### Example Configuration

```python
# Optimized settings for RTX 3060 12GB
training_config = {
    "model": "facebook/musicgen-medium",
    "use_lora": True,
    "lora_rank": 8,          # Lower = less VRAM
    "batch_size": 1,         # Accumulate gradients instead
    "gradient_accumulation": 8,
    "fp16": True,
    "gradient_checkpointing": True,
    "max_audio_length": 30,  # seconds
}
```

---

## Step-by-Step Workflow

### Phase 1: Environment Setup

```bash
# Create virtual environment
python -m venv musicgen-env
musicgen-env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install audiocraft transformers accelerate peft
pip install essentia  # For auto-labeling
```

### Phase 2: Dataset Preparation

1. **Organize your music** into a single directory
2. **Filter for quality** - Remove low-quality recordings
3. **Remove vocals** (optional but recommended)
4. **Auto-label** using Essentia or manual descriptions
5. **Chunk** long files into 30-second segments

### Phase 3: Training

```python
# Simplified training script outline
from audiocraft.models import MusicGen
from peft import LoraConfig, get_peft_model

# Load base model
model = MusicGen.get_pretrained('facebook/musicgen-medium')

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# Train on your dataset
# ... (full implementation varies by repository)
```

### Phase 4: Generation

```python
# Generate with your fine-tuned model
model.set_generation_params(duration=30)
descriptions = ["your style description here"]
wav = model.generate(descriptions)
```

---

## Recommended Repositories

| Repository | Description | Best For |
|------------|-------------|----------|
| [audiocraft](https://github.com/facebookresearch/audiocraft) | Official MusicGen | Base model, inference |
| [musicgen-trainer](https://github.com/chavinlo/musicgen-trainer) | Community trainer | LoRA fine-tuning |
| [audiocraft-lora](https://huggingface.co/docs/peft/task_guides/musicgen) | HuggingFace guide | PEFT integration |
| [riffusion](https://github.com/riffusion/riffusion) | Spectrogram approach | Alternative method |

---

## Cost Comparison: Local vs Cloud

| Option | Cost | Speed | Notes |
|--------|------|-------|-------|
| **Your RTX 3060** | Electricity only | Slow-Medium | Best for experimentation |
| **RunPod A40** | ~$0.79/hr | Fast | Good for larger training |
| **Lambda A100** | ~$1.50/hr | Very Fast | Best for production |
| **Replicate** | Pay per run | Variable | Easy to use, pre-built |

> [!TIP]
> **Recommended approach:** Experiment locally on your RTX 3060, then use cloud GPUs (RunPod, Lambda) for final training runs when you've validated your dataset and approach.

---

## Legal Considerations

> [!CAUTION]
> Before training on your music collection:
> 
> - **Your own music**: ✅ No issues
> - **Royalty-free/CC licensed**: ✅ Check specific license
> - **Purchased music**: ⚠️ Personal use may be gray area
> - **Copyrighted music**: ❌ Training may violate copyright
> - **Distributing the model**: ⚠️ Additional legal concerns
>
> If your collection includes copyrighted material, consult legal advice before training or distributing any resulting model.

---

## Next Steps

1. **Start small**: Test with 10-20 songs to validate your pipeline
2. **Iterate on descriptions**: Quality metadata improves results significantly
3. **Monitor training**: Watch for overfitting (model just memorizes songs)
4. **Experiment with LoRA rank**: Higher rank = more capacity but more VRAM
5. **Try Riffusion too**: Different approach, might suit your needs

---

## Summary

| Requirement | Your Status |
|-------------|-------------|
| GPU VRAM (12GB minimum for LoRA) | ✅ RTX 3060 (12GB) |
| System RAM (16GB+) | ✅ 48GB |
| Storage for dataset | ✅ 5.5TB+ |
| Dataset (50+ songs recommended) | ❓ Depends on your collection |
| Technical comfort with Python | ❓ You'll need basic skills |

**Bottom line**: Your hardware can absolutely fine-tune music AI models using LoRA. Start with MusicGen-small or Riffusion for the smoothest experience, then scale up to MusicGen-medium once you've validated your workflow.
