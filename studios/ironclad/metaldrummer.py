import random
from pathlib import Path

class MetalDrummerWrapper:
    """
    Generates aggressive metal drum patterns.
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        # No DL model for this one yet, purely algorithmic for now
        print(f"🥁 MetalDrummer Initialized")

    def generate_pattern(self, bpm: int, intensity: str) -> str:
        
        patterns = [
            "Blast Beat (Snare/Kick alternating 16th notes)",
            "Double Bass Run (Constant 16th kick notes)",
            "D-Beat (Discharge style punk/crust rhythm)",
            "Half-time Groove with China Cymbal accents"
        ]
        
        if bpm > 180 and intensity == "Extreme":
            return "Gravity Blasts with rapid double bass"
            
        return random.choice(patterns)
