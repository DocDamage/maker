from pathlib import Path

class BlueNoteEnsemble:
    """
    The Blue Note Studio Ensemble (Jazz/Blues).
    Coordinates DeepJazz (Harmony) and JazzLoRA.
    """
    
    def __init__(self):
        self.name = "Blue Note Studio"
        self.models_dir = Path("models/bluenote")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        print(f"🎷 {self.name} Initialized")

    def generate(self, prompt: str, duration: int = 30, **kwargs):
        enhanced_prompt = f"{prompt}, jazz, smooth jazz, saxophone, piano trio, improvisation, swing rhythm, complex harmony"
        
        return {
            "prompt": enhanced_prompt,
            "duration": duration,
            "metadata": {
                "key": "Bb Major",
                "style": "Cool Jazz",
                "studio": "Blue Note"
            }
        }
