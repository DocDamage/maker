"""
Audio Preprocessing Script

Prepares your music collection for MusicGen training:
1. Converts all audio to 32kHz mono WAV
2. Chunks long files into 30-second segments
3. Normalizes audio levels
4. Optionally removes vocals

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --input path/to/music --remove-vocals
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_AUDIO_DIR, PROCESSED_AUDIO_DIR,
    SAMPLE_RATE, AUDIO_CHANNELS, CHUNK_DURATION, MIN_CHUNK_DURATION,
    AUDIO_EXTENSIONS, REMOVE_VOCALS
)

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Try to import optional dependencies
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. Install with: pip install librosa")

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("Warning: pydub not installed. Some formats may not work.")


def find_audio_files(directory: Path) -> list[Path]:
    """Find all audio files in directory recursively."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(directory.rglob(f"*{ext}"))
        audio_files.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(set(audio_files))


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, handling various formats."""
    try:
        # Try librosa first (handles most formats)
        if HAS_LIBROSA:
            audio, sr = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=(AUDIO_CHANNELS == 1))
            return audio, sr
    except Exception as e:
        print(f"  Librosa failed: {e}")
    
    try:
        # Fallback to soundfile
        audio, sr = sf.read(str(file_path))
        if len(audio.shape) > 1 and AUDIO_CHANNELS == 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLE_RATE:
            if HAS_LIBROSA:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            else:
                raise ValueError(f"Cannot resample without librosa (file is {sr}Hz, need {SAMPLE_RATE}Hz)")
        return audio, SAMPLE_RATE
    except Exception as e:
        print(f"  Soundfile failed: {e}")
    
    try:
        # Last resort: pydub
        if HAS_PYDUB:
            audio_seg = AudioSegment.from_file(str(file_path))
            audio_seg = audio_seg.set_frame_rate(SAMPLE_RATE)
            if AUDIO_CHANNELS == 1:
                audio_seg = audio_seg.set_channels(1)
            samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
            samples = samples / (2 ** 15)  # Normalize to [-1, 1]
            return samples, SAMPLE_RATE
    except Exception as e:
        print(f"  Pydub failed: {e}")
    
    raise ValueError(f"Could not load audio file: {file_path}")


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio to target dB level."""
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    
    # Calculate target RMS
    target_rms = 10 ** (target_db / 20)
    
    # Scale audio
    audio = audio * (target_rms / rms)
    
    # Clip to prevent distortion
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def chunk_audio(audio: np.ndarray, sr: int, chunk_duration: float, min_duration: float) -> list[np.ndarray]:
    """Split audio into fixed-size chunks."""
    chunk_samples = int(chunk_duration * sr)
    min_samples = int(min_duration * sr)
    
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        
        # Pad last chunk if needed (but only if it's long enough)
        if len(chunk) >= min_samples:
            if len(chunk) < chunk_samples:
                # Pad with zeros
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)
    
    return chunks


def remove_vocals_demucs(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove vocals using Demucs (if installed)."""
    try:
        import torch
        import torchaudio
        from demucs import pretrained
        from demucs.apply import apply_model
        
        # Load model
        model = pretrained.get_model('htdemucs')
        model.eval()
        
        # Prepare audio
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio])  # Mono to stereo
        
        waveform = torch.tensor(audio).unsqueeze(0).float()
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(model, waveform, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get everything except vocals (bass, drums, other)
        # Sources order: drums, bass, other, vocals
        no_vocals = sources[0, 0] + sources[0, 1] + sources[0, 2]
        
        # Convert back to mono if needed
        if AUDIO_CHANNELS == 1:
            no_vocals = no_vocals.mean(dim=0)
        
        return no_vocals.numpy()
    
    except ImportError:
        print("  Demucs not installed. Skipping vocal removal.")
        print("  Install with: pip install demucs")
        return audio
    except Exception as e:
        print(f"  Vocal removal failed: {e}")
        return audio


def process_file(file_path: Path, output_dir: Path, remove_vocals: bool = False) -> list[Path]:
    """Process a single audio file."""
    output_files = []
    
    try:
        # Load audio
        audio, sr = load_audio(file_path)
        
        # Skip very short files
        if len(audio) < MIN_CHUNK_DURATION * sr:
            print(f"  Skipping (too short): {file_path.name}")
            return []
        
        # Remove vocals if requested
        if remove_vocals:
            print(f"  Removing vocals...")
            audio = remove_vocals_demucs(audio, sr)
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Chunk
        chunks = chunk_audio(audio, sr, CHUNK_DURATION, MIN_CHUNK_DURATION)
        
        # Save chunks
        base_name = file_path.stem
        for i, chunk in enumerate(chunks):
            output_name = f"{base_name}_chunk{i:03d}.wav"
            output_path = output_dir / output_name
            sf.write(str(output_path), chunk, sr)
            output_files.append(output_path)
        
        print(f"  ✓ Created {len(chunks)} chunks from {file_path.name}")
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path.name}: {e}")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for MusicGen training")
    parser.add_argument("--input", "-i", type=Path, default=RAW_AUDIO_DIR,
                        help="Input directory with audio files")
    parser.add_argument("--output", "-o", type=Path, default=PROCESSED_AUDIO_DIR,
                        help="Output directory for processed chunks")
    parser.add_argument("--remove-vocals", "-v", action="store_true", default=REMOVE_VOCALS,
                        help="Remove vocals using Demucs")
    args = parser.parse_args()
    
    # Find audio files
    print(f"\n🎵 MusicGen Audio Preprocessor")
    print(f"{'=' * 50}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Remove vocals: {args.remove_vocals}")
    print(f"{'=' * 50}\n")
    
    audio_files = find_audio_files(args.input)
    
    if not audio_files:
        print(f"❌ No audio files found in {args.input}")
        print(f"   Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        print(f"\n   Please add your music files to: {args.input}")
        return
    
    print(f"Found {len(audio_files)} audio files\n")
    
    # Process files
    args.output.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    
    for file_path in tqdm(audio_files, desc="Processing"):
        chunks = process_file(file_path, args.output, args.remove_vocals)
        total_chunks += len(chunks)
    
    print(f"\n{'=' * 50}")
    print(f"✅ Preprocessing complete!")
    print(f"   Total audio files: {len(audio_files)}")
    print(f"   Total chunks created: {total_chunks}")
    print(f"   Output directory: {args.output}")
    print(f"\n   Next step: python scripts/autolabel.py")


if __name__ == "__main__":
    main()
