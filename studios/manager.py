from enum import Enum
from typing import Dict, Optional, Any
import importlib

class StudioType(Enum):
    STANDARD = "standard"
    CHIPTUNE = "chiptune"
    CYPHER = "cypher"
    IRONCLAD = "ironclad"
    BLUENOTE = "bluenote"
    VIRTUOSO = "virtuoso"

class StudioManager:
    """Manages the lifecycle of specialized music generation studios."""
    
    def __init__(self):
        self.active_studio: StudioType = StudioType.STANDARD
        self.loaded_modules: Dict[str, Any] = {}
        
    def switch_studio(self, studio_type: str) -> Dict[str, Any]:
        """Switch to a different studio, loading necessary resources."""
        try:
            new_studio = StudioType(studio_type.lower())
        except ValueError:
            return {"error": f"Invalid studio type: {studio_type}"}
            
        if self.active_studio == new_studio:
            return {"status": "already_active", "studio": new_studio.value}
            
        print(f"🔄 Switching studio from {self.active_studio.value} to {new_studio.value}...")
        
        # Unload previous studio resources (if aggressive memory management needed)
        self._unload_resources(self.active_studio)
        
        # Load new studio resources
        try:
            self._load_resources(new_studio)
            self.active_studio = new_studio
            return {"status": "success", "studio": new_studio.value}
        except Exception as e:
            print(f"❌ Failed to load studio {new_studio.value}: {e}")
            return {"error": str(e)}

    def get_active_studio(self) -> str:
        return self.active_studio.value

    def _load_resources(self, studio: StudioType):
        """Lazy load resources for specific studios."""
        current_ensemble = None
        
        if studio == StudioType.CHIPTUNE:
            from studios.chiptune.ensemble import ChiptuneEnsemble
            current_ensemble = ChiptuneEnsemble()
            
        elif studio == StudioType.CYPHER:
            from studios.cypher.ensemble import CypherEnsemble
            current_ensemble = CypherEnsemble()
            
        elif studio == StudioType.IRONCLAD:
            from studios.ironclad.ensemble import IroncladEnsemble
            current_ensemble = IroncladEnsemble()
            
        elif studio == StudioType.BLUENOTE:
            from studios.bluenote.ensemble import BlueNoteEnsemble
            current_ensemble = BlueNoteEnsemble()
            
        elif studio == StudioType.VIRTUOSO:
            from studios.virtuoso.ensemble import VirtuosoEnsemble
            current_ensemble = VirtuosoEnsemble()
            
        if current_ensemble:
            self.loaded_modules[studio.value] = current_ensemble

    def process_generation_request(self, prompt: str, duration: int, **kwargs):
        """Route generation request to the active studio."""
        active_resource = self.loaded_modules.get(self.active_studio.value)
        
        # If active studio is standard or not loaded, just return the raw prompt
        if not active_resource or self.active_studio == StudioType.STANDARD:
            return {"prompt": prompt, "duration": duration, "studio": "standard"}
            
        # Delegate to the specialized ensemble
        return active_resource.generate(prompt, duration, **kwargs)


    def _unload_resources(self, studio: StudioType):
        """Cleanup resources to save RAM."""
        # In a real scenario, we would explicitly delete models from GPU/RAM here
        if studio.value in self.loaded_modules:
            del self.loaded_modules[studio.value]
            import gc
            gc.collect()
        print(f"🧹 Unloaded {studio.value} resources")
