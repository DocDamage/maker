"""
Auto-Labeling Script

Analyzes your processed audio files and generates metadata/descriptions
for MusicGen training. Uses audio feature extraction to determine:
- Tempo (BPM)
- Key and mode
- Mood/energy
- Estimated genre
- Instrument detection

Usage:
    python scripts/autolabel.py
    python scripts/autolabel.py --input path/to/processed --analyze-all
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROCESSED_AUDIO_DIR, METADATA_DIR,
    AUTOLABEL_CONFIG, DEFAULT_DESCRIPTION, SAMPLE_RATE
)

import numpy as np
from tqdm import tqdm

# Try to import analysis libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. Some analysis features unavailable.")


# Genre keywords for filename-based detection
GENRE_KEYWORDS = {
    "electronic": ["electronic", "edm", "techno", "house", "trance", "dubstep", "dnb", "drum and bass"],
    "rock": ["rock", "metal", "punk", "grunge", "alternative"],
    "jazz": ["jazz", "swing", "bebop", "fusion"],
    "classical": ["classical", "orchestra", "symphony", "piano", "violin", "baroque"],
    "hip-hop": ["hip-hop", "hiphop", "rap", "trap", "beats"],
    "pop": ["pop", "dance", "disco"],
    "ambient": ["ambient", "chill", "relax", "meditation", "drone"],
    "folk": ["folk", "acoustic", "country", "bluegrass"],
    "r&b": ["rnb", "r&b", "soul", "funk", "groove"],
    "world": ["world", "ethnic", "tribal", "latin", "african"],
}

# Mood keywords
MOOD_KEYWORDS = {
    "energetic": ["energetic", "upbeat", "fast", "hype", "intense", "powerful"],
    "calm": ["calm", "peaceful", "relaxing", "chill", "mellow", "soft"],
    "dark": ["dark", "moody", "sad", "melancholy", "minor"],
    "happy": ["happy", "uplifting", "joyful", "bright", "cheerful"],
    "aggressive": ["aggressive", "heavy", "hard", "angry", "intense"],
    "dreamy": ["dreamy", "ethereal", "atmospheric", "floating"],
}

# Key mappings
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODE_NAMES = ['minor', 'major']


def analyze_tempo(audio: np.ndarray, sr: int) -> Optional[float]:
    """Estimate tempo in BPM."""
    if not HAS_LIBROSA:
        return None
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    except:
        return None


def analyze_key(audio: np.ndarray, sr: int) -> Optional[tuple[str, str]]:
    """Estimate musical key and mode."""
    if not HAS_LIBROSA:
        return None
    try:
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        
        # Sum across time
        chroma_sum = np.sum(chroma, axis=1)
        
        # Find dominant pitch class
        key_idx = int(np.argmax(chroma_sum))
        key_name = KEY_NAMES[key_idx]
        
        # Estimate mode using major/minor templates
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        # Rotate profiles to match key
        major_rotated = np.roll(major_profile, -key_idx)
        minor_rotated = np.roll(minor_profile, -key_idx)
        
        # Correlate with chromagram
        major_corr = np.corrcoef(chroma_sum, major_rotated)[0, 1]
        minor_corr = np.corrcoef(chroma_sum, minor_rotated)[0, 1]
        
        mode = "major" if major_corr > minor_corr else "minor"
        
        return key_name, mode
    except:
        return None


def analyze_energy(audio: np.ndarray, sr: int) -> Optional[str]:
    """Estimate energy/mood from audio features."""
    if not HAS_LIBROSA:
        return None
    try:
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        avg_rms = float(np.mean(rms))
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        avg_centroid = float(np.mean(centroid))
        
        # Classify based on features
        if avg_rms > 0.15 and avg_centroid > 3000:
            return "energetic"
        elif avg_rms < 0.05:
            return "calm"
        elif avg_centroid < 2000:
            return "dark"
        elif avg_centroid > 4000:
            return "bright"
        else:
            return "moderate"
    except:
        return None


def analyze_spectral_features(audio: np.ndarray, sr: int) -> dict:
    """Extract additional spectral features."""
    if not HAS_LIBROSA:
        return {}
    try:
        features = {}
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
        features["spectral_rolloff"] = float(np.mean(rolloff))
        
        # Zero crossing rate (percussiveness indicator)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        features["zero_crossing_rate"] = float(np.mean(zcr))
        
        # Spectral flatness (noise-like vs tonal)
        flatness = librosa.feature.spectral_flatness(y=audio)
        features["spectral_flatness"] = float(np.mean(flatness))
        
        return features
    except:
        return {}


def guess_genre_from_filename(filename: str) -> Optional[str]:
    """Try to extract genre from filename."""
    filename_lower = filename.lower()
    
    for genre, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return genre
    
    return None


def guess_mood_from_filename(filename: str) -> Optional[str]:
    """Try to extract mood from filename."""
    filename_lower = filename.lower()
    
    for mood, keywords in MOOD_KEYWORDS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return mood
    
    return None


def generate_description(analysis: dict) -> str:
    """Generate a natural language description from analysis."""
    parts = []
    
    # Start with mood/energy
    if analysis.get("mood"):
        parts.append(analysis["mood"])
    
    # Add genre
    if analysis.get("genre"):
        parts.append(analysis["genre"])
    else:
        parts.append("instrumental")
    
    parts.append("track")
    
    # Add tempo info
    if analysis.get("bpm"):
        bpm = analysis["bpm"]
        if bpm < 80:
            parts.append("with slow tempo")
        elif bpm < 120:
            parts.append("with moderate tempo")
        elif bpm < 150:
            parts.append("with upbeat tempo")
        else:
            parts.append("with fast tempo")
    
    # Add key info
    if analysis.get("key") and analysis.get("mode"):
        parts.append(f"in {analysis['key']} {analysis['mode']}")
    
    # Add instrument hints based on spectral features
    if analysis.get("spectral_flatness", 0) > 0.1:
        parts.append("with electronic/synthesized elements")
    elif analysis.get("zero_crossing_rate", 0) > 0.1:
        parts.append("with percussive elements")
    
    return " ".join(parts)


def analyze_file(file_path: Path) -> dict:
    """Analyze a single audio file."""
    analysis = {
        "filename": file_path.name,
        "path": str(file_path),
    }
    
    # Load audio
    try:
        if HAS_LIBROSA:
            audio, sr = librosa.load(str(file_path), sr=SAMPLE_RATE)
        else:
            import soundfile as sf
            audio, sr = sf.read(str(file_path))
    except Exception as e:
        analysis["error"] = str(e)
        analysis["description"] = DEFAULT_DESCRIPTION
        return analysis
    
    # Filename-based guesses
    if AUTOLABEL_CONFIG.get("use_filename_hints", True):
        analysis["genre"] = guess_genre_from_filename(file_path.stem)
        filename_mood = guess_mood_from_filename(file_path.stem)
        if filename_mood:
            analysis["mood"] = filename_mood
    
    # Audio analysis
    if AUTOLABEL_CONFIG.get("analyze_tempo", True):
        tempo = analyze_tempo(audio, sr)
        if tempo:
            analysis["bpm"] = round(tempo, 1)
    
    if AUTOLABEL_CONFIG.get("analyze_key", True):
        key_result = analyze_key(audio, sr)
        if key_result:
            analysis["key"], analysis["mode"] = key_result
    
    if AUTOLABEL_CONFIG.get("analyze_mood", True) and "mood" not in analysis:
        mood = analyze_energy(audio, sr)
        if mood:
            analysis["mood"] = mood
    
    # Additional spectral features
    spectral = analyze_spectral_features(audio, sr)
    analysis.update(spectral)
    
    # Generate description
    analysis["description"] = generate_description(analysis)
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Auto-label audio files for MusicGen training")
    parser.add_argument("--input", "-i", type=Path, default=PROCESSED_AUDIO_DIR,
                        help="Input directory with processed audio")
    parser.add_argument("--output", "-o", type=Path, default=METADATA_DIR,
                        help="Output directory for metadata")
    parser.add_argument("--analyze-all", "-a", action="store_true",
                        help="Re-analyze all files (ignore existing)")
    args = parser.parse_args()
    
    print(f"\n🏷️  MusicGen Auto-Labeler")
    print(f"{'=' * 50}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"{'=' * 50}\n")
    
    # Find audio files
    audio_files = list(args.input.glob("*.wav"))
    
    if not audio_files:
        print(f"❌ No WAV files found in {args.input}")
        print(f"   Run preprocessing first: python scripts/preprocess.py")
        return
    
    print(f"Found {len(audio_files)} audio files to analyze\n")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Analyze files
    all_metadata = []
    
    for file_path in tqdm(audio_files, desc="Analyzing"):
        analysis = analyze_file(file_path)
        all_metadata.append(analysis)
    
    # Save individual JSON files
    for analysis in all_metadata:
        json_path = args.output / f"{Path(analysis['filename']).stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Save combined JSONL file (for training)
    jsonl_path = args.output / "data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for analysis in all_metadata:
            entry = {
                "path": analysis["path"],
                "description": analysis["description"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Print summary
    print(f"\n{'=' * 50}")
    print(f"✅ Auto-labeling complete!")
    print(f"   Files analyzed: {len(all_metadata)}")
    print(f"   Metadata saved to: {args.output}")
    print(f"   Training data file: {jsonl_path}")
    
    # Show sample descriptions
    print(f"\n📝 Sample descriptions:")
    for analysis in all_metadata[:5]:
        print(f"   • {analysis['filename']}")
        print(f"     \"{analysis['description']}\"")
    
    print(f"\n   Next step: python scripts/train.py")


if __name__ == "__main__":
    main()
