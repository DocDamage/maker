from pathlib import Path
import json

class CypherEnsemble:
    """
    The Cypher Studio Ensemble (Hip Hop/Trap).
    Coordinates BeatMaker (Rhythm) and FlowGen (Lyrics/Cadence).
    """
    
    def __init__(self):
        self.name = "Cypher Studio"
        self.models_dir = Path("models/cypher")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from studios.cypher.beatmaker import BeatMakerWrapper
            from studios.cypher.flowgen import FlowGenWrapper
            
            self.beatmaker = BeatMakerWrapper(self.models_dir)
            self.flowgen = FlowGenWrapper(self.models_dir)
        except ImportError as e:
            print(f"❌ Failed to import Cypher modules: {e}")
            
        print(f"🎤 {self.name} Initialized")

    def generate(self, prompt: str, duration: int = 30, **kwargs):
        print(f"🎤 Cypher Studio generating: {prompt}")
        
        # 1. Determine Style from Prompt
        style = "Boom Bap"
        if "trap" in prompt.lower(): style = "Trap"
        elif "lo-fi" in prompt.lower(): style = "Lo-Fi"
        elif "west" in prompt.lower(): style = "West Coast"
        
        bpm = 90
        if style == "Trap": bpm = 140
        if style == "Lo-Fi": bpm = 80
        
        # 2. Generate Beat Structure
        beat_info = self.beatmaker.generate_beat(bpm, style)
        
        # 3. Generate Flow Guide
        flow_info = self.flowgen.generate_flow("Energetic")
        
        # 4. Construct Enhanced Prompt
        description = f"Hip Hop track, {style} style, {bpm} BPM. "
        description += f"Drum Pattern: {beat_info['pattern_name']}. "
        description += f"Bass: {beat_info['bass_sound']}. "
        description += f"Vocal Flow capability: {flow_info}. "
        description += f"Vibe: {prompt}"
        
        return {
            "prompt": description,
            "duration": duration,
            "metadata": {
                "bpm": bpm,
                "style": style,
                "beat": beat_info,
                "flow": flow_info,
                "studio": "Cypher",
                "technical_prompt": description
            }
        }

