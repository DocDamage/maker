# MusicGen LoRA Training Pipeline

A complete pipeline for fine-tuning MusicGen on your music collection using LoRA (Low-Rank Adaptation), optimized for RTX 3060 (12GB VRAM).

## Quick Start

### Option 1: Web UI (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the studio
python app.py

# 3. Open http://localhost:5000 in your browser
```

### Option 2: Command Line

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your music (put your audio files in data/raw/)
python scripts/preprocess.py

# 3. Auto-label your dataset
python scripts/autolabel.py

# 4. Train the model
python scripts/train.py

# 5. Generate new music
python scripts/generate.py --prompt "your music description"
```

## Project Structure

```
musicmaker/
├── data/
│   ├── raw/              # Put your original music files here
│   ├── processed/        # Preprocessed audio chunks
│   └── metadata/         # Generated labels and descriptions
├── scripts/
│   ├── preprocess.py     # Audio preprocessing
│   ├── autolabel.py      # Automatic music analysis & labeling
│   ├── train.py          # LoRA fine-tuning
│   └── generate.py       # Generate new music
├── models/
│   └── lora/             # Saved LoRA adapters
├── output/               # Generated music files
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Hardware Requirements

- **GPU**: NVIDIA RTX 3060 12GB (or equivalent)
- **RAM**: 16GB+ (you have 48GB ✓)
- **Storage**: 10GB+ free for models and data

## Configuration

Edit `config.py` to customize:

- Model variant (small/medium/melody)
- LoRA settings (rank, alpha)
- Training hyperparameters
- Output directories

## Tips for Best Results

1. **Quality over quantity**: 50 high-quality songs beats 500 noisy ones
2. **Consistent style**: Train on similar genres for focused results
3. **Good descriptions**: Detailed metadata improves generation
4. **Start small**: Test with 10-20 songs first, then scale up

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size in `config.py`
- Lower LoRA rank (e.g., from 8 to 4)
- Enable gradient checkpointing

### Poor Results

- Add more training data
- Improve metadata descriptions
- Train for more epochs
- Check for corrupted audio files

## License

This pipeline is for personal/educational use. Ensure your training data is properly licensed.
