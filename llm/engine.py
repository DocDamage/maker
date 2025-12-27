import os
from pathlib import Path
from typing import Dict, Any, Optional, List
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class LLMEngine:
    """Manages local LLM inference with llama.cpp"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Any] = {}
        
        if Llama is None:
            print("WARNING: llama-cpp-python not installed. LLM features will be disabled.")
        
    def load_model(self, model_name: str, n_ctx: int = 4096, n_threads: int = 8):
        """Load a GGUF model into memory. Unloads others if needed to save RAM."""
        if Llama is None:
            return

        model_path = self.models_dir / f"{model_name}"
        if not model_path.suffix:
            model_path = model_path.with_suffix(".gguf")
            
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        if model_name in self.loaded_models:
            return
            
        # Basic memory management: Unload all models before loading a new one?
        # For now, let's keep it simple. If we run out of RAM, we might need to unload.
        # Given 48GB RAM, we can probably hold 1-2 quantized models.
        
        print(f"Loading LLM: {model_name}...")
        self.loaded_models[model_name] = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU inference since we are optimizing for 48GB System RAM
            verbose=False
        )
        print(f"✅ Loaded {model_name}")
        
    def generate(self, model_name: str, prompt: str, 
                 max_tokens: int = 512, temperature: float = 0.7, 
                 stop: Optional[List[str]] = None) -> str:
        """Generate text from a loaded model"""
        if Llama is None:
            return "Error: llama-cpp-python not installed."

        if model_name not in self.loaded_models:
            self.load_model(model_name)
            
        if stop is None:
            stop = ["User:", "Assistant:", "\n\n"]
            
        response = self.loaded_models[model_name](
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        return response['choices'][0]['text'].strip()

    def unload_all(self):
        """Unload all models to free RAM"""
        self.loaded_models.clear()
        # triggering garbage collection might be needed in some python envs
        import gc
        gc.collect()
