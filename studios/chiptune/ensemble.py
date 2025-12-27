from pathlib import Path
import json
import random

class ChiptuneEnsemble:
    """
    The Chiptune Studio Ensemble.
    Coordinates LakhNES (Composition) and 8bit-RNN (Melody) 
    to drive the MusicGen generation.
    """
    
    def __init__(self):
        self.name = "Chiptune Studio"
        self.models_dir = Path("models/chiptune")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load specialized agents
        try:
            from studios.chiptune.lakhnes import LakhNESWrapper
            from studios.chiptune.rnn import ChiptuneRNN
            
            self.lakhnes = LakhNESWrapper(self.models_dir)
            self.rnn = ChiptuneRNN(self.models_dir)
        except ImportError as e:
            print(f"❌ Failed to import Chiptune modules: {e}")
        
        print(f"👾 {self.name} Initialized")

    def generate(self, prompt: str, duration: int = 30, **kwargs):
        """
        Full generation pipeline for Chiptune.
        """
        print(f"👾 Chiptune Studio generating: {prompt}")
        
        # 1. Structural Generation (LakhNES)
        structure = self.lakhnes.generate()
        
        # 2. Melodic Improv (RNN)
        melody = self.rnn.generate_melody(structure['progression'], structure['key'])
        
        # 3. Construct Enhanced Prompt
        # We inject the structured musical data into the text prompt
        
        description = f"chiptune track in {structure['key']} {structure['scale']} at {structure['tempo']} BPM. "
        description += f"Structure: {', '.join([s['name'] for s in structure['sections']])}. "
        description += f"Chord Progression: {', '.join(structure['progression'])}. "
        description += f"Melody Style: {melody}. "
        description += "Sound: NES console 2A03 chip, square waves, triangle bass, noise drums. "
        description += f"Vibe: {prompt}"
        
        return {
            "prompt": description,
            "duration": duration,
            "metadata": {
                "structure": structure,
                "melody": melody,
                "studio": "Chiptune",
                "technical_prompt": description
            }
        }



