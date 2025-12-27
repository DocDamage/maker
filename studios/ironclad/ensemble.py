from pathlib import Path

class IroncladEnsemble:
    """
    The Ironclad Studio Ensemble (Metal/Rock).
    Coordinates ShredGP (Guitar/Tab) and MetalDrummer.
    """
    
    def __init__(self):
        self.name = "Ironclad Studio"
        self.models_dir = Path("models/ironclad")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from studios.ironclad.shredgp import ShredGPWrapper
            from studios.ironclad.metaldrummer import MetalDrummerWrapper
            
            self.shredgp = ShredGPWrapper(self.models_dir)
            self.drummer = MetalDrummerWrapper(self.models_dir)
        except ImportError as e:
             print(f"❌ Failed to import Ironclad modules: {e}")
             
        print(f"🎸 {self.name} Initialized")

    def generate(self, prompt: str, duration: int = 30, **kwargs):
        print(f"🎸 Ironclad Studio generating: {prompt}")
        
        # 1. Determine Sub-Genre
        style = "Heavy Metal"
        tuning = "Standard E"
        bpm = 120
        intensity = "High"
        
        if "djent" in prompt.lower(): 
            style = "Djent"
            tuning = "Drop A" # Low tuning for djent
            bpm = 130
        elif "thrash" in prompt.lower():
            style = "Thrash"
            bpm = 190
            intensity = "Extreme"
        elif "doom" in prompt.lower():
            style = "Doom"
            bpm = 60
            tuning = "Drop C"
            
        # 2. Generate Riffs
        riff_info = self.shredgp.generate_riff(style, tuning)
        
        # 3. Generate Drums
        drum_info = self.drummer.generate_pattern(bpm, intensity)
        
        enhanced_prompt = f"{style} track, {bpm} BPM. "
        enhanced_prompt += f"Guitar: {riff_info}. "
        enhanced_prompt += f"Drums: {drum_info}. "
        enhanced_prompt += f"Vibe: {prompt}"
        
        return {
            "prompt": enhanced_prompt,
            "duration": duration,
            "metadata": {
                "tuning": tuning,
                "distortion": "High",
                "studio": "Ironclad",
                "riff": riff_info,
                "technical_prompt": enhanced_prompt
            }
        }

