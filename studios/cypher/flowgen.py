import random
from pathlib import Path

class FlowGenWrapper:
    """
    Wrapper for Rap-Lyrics-Generator.
    Generates lyrical flow patterns and rhyme schemes to guide the melody.
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.model_path = models_dir / "flowgen.pt"
        self.loaded = False
        
        if self.model_path.exists():
             self.loaded = True
        else:
             print(f"⚠️ FlowGen model not found at {self.model_path}. Using algorithmic fallback.")

    def generate_flow(self, mood: str) -> str:
        """
        Generates a description of the vocal flow/cadence.
        """
        if self.loaded:
            return self._generate_neural()
        else:
            return self._generate_algorithmic(mood)

    def _generate_algorithmic(self, mood: str) -> str:
        flows = [
            "Double-time triplets (Migos flow)",
            "Laid back, behind the beat (Snoop style)",
            "Aggressive staccato 16th notes",
            "Melodic sing-song cadence (Drake style)"
        ]
        
        rhyme_schemes = ["AABB", "ABAB", "AAAA", "Multi-syllabic stacking"]
        
        return f"{random.choice(flows)} with {random.choice(rhyme_schemes)} rhyme scheme"
        
    def _generate_neural(self):
        pass
