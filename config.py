"""
MusicGen LoRA Training Pipeline - Configuration

Optimized settings for RTX 3060 (12GB VRAM)
Adjust these values based on your hardware and needs.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw"
PROCESSED_AUDIO_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"
LORA_DIR = MODELS_DIR / "lora"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
for dir_path in [RAW_AUDIO_DIR, PROCESSED_AUDIO_DIR, METADATA_DIR, LORA_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL SETTINGS
# =============================================================================
# Options: "facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-melody"
# For RTX 3060 with LoRA: medium is recommended
# For full fine-tune: use small only
MODEL_NAME = "facebook/musicgen-small"  # Start with small for testing

# =============================================================================
# AUDIO PREPROCESSING
# =============================================================================
SAMPLE_RATE = 32000  # MusicGen native sample rate
AUDIO_CHANNELS = 1   # Mono (recommended)
CHUNK_DURATION = 30  # Seconds per training sample
MIN_CHUNK_DURATION = 5  # Minimum valid duration
AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"]

# Vocal removal (requires demucs)
REMOVE_VOCALS = False  # Set True if your music has vocals

# =============================================================================
# LORA SETTINGS (Memory-efficient fine-tuning)
# =============================================================================
USE_LORA = True  # Set False for full fine-tune (requires musicgen-small)

LORA_CONFIG = {
    "r": 8,               # Rank - lower = less VRAM, less capacity (try 4-16)
    "lora_alpha": 16,     # Scaling factor (usually 2x rank)
    "lora_dropout": 0.05, # Regularization
    "target_modules": [   # Which layers to adapt
        "q_proj",
        "v_proj", 
        "k_proj",
        "out_proj",
    ],
    "bias": "none",
}

# =============================================================================
# TRAINING SETTINGS (Optimized for 12GB VRAM)
# =============================================================================
TRAINING_CONFIG = {
    # Batch settings
    "batch_size": 1,              # Keep low for 12GB VRAM
    "gradient_accumulation_steps": 8,  # Effective batch = 8
    
    # Training duration
    "num_epochs": 10,             # Adjust based on dataset size
    "max_steps": None,            # Set to override epochs (e.g., 1000)
    
    # Learning rate
    "learning_rate": 1e-4,        # LoRA can use higher LR
    "lr_scheduler": "cosine",     # Options: cosine, linear, constant
    "warmup_steps": 100,
    
    # Memory optimization
    "fp16": True,                 # Half precision (essential for 12GB)
    "gradient_checkpointing": True,  # Trade compute for memory
    "max_grad_norm": 1.0,         # Gradient clipping
    
    # Saving
    "save_steps": 500,            # Save checkpoint every N steps
    "save_total_limit": 3,        # Keep only last N checkpoints
    "logging_steps": 50,          # Log metrics every N steps
    
    # Reproducibility
    "seed": 42,
}

# =============================================================================
# GENERATION SETTINGS
# =============================================================================
GENERATION_CONFIG = {
    "duration": 30,           # Output duration in seconds
    "temperature": 1.0,       # Creativity (0.5-1.5, higher = more random)
    "top_k": 250,             # Limit token choices
    "top_p": 0.95,            # Nucleus sampling
    "cfg_coef": 3.0,          # Classifier-free guidance strength
    "extend_stride": 18,      # For extending beyond 30s
}

# =============================================================================
# AUTO-LABELING
# =============================================================================
AUTOLABEL_CONFIG = {
    "analyze_tempo": True,
    "analyze_key": True,
    "analyze_mood": True,
    "analyze_genre": True,
    "analyze_instruments": True,
    "use_filename_hints": True,  # Extract info from filenames
}

# Default description template (used if auto-labeling fails)
DEFAULT_DESCRIPTION = "instrumental music track"
