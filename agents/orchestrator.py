from pathlib import Path
from typing import Dict, Any, Optional
from llm.engine import LLMEngine
from agents.base_agent import BaseAgent
import json

class PromptAgent(BaseAgent):
    """Enhances user prompts with musical terminology"""
    
    def get_model_name(self) -> str:
        # Utilizing the available local model
        return "llama-3-8b"
    
    def get_system_prompt(self) -> str:
        return """You are a music production expert and prompt engineer. Your task is to expand brief music descriptions into detailed, technical prompts for AI music generation architectures (like MusicGen).

Include specific details about:
- Instruments and their playing techniques (e.g., "palm-muted distortion guitar", "brushed jazz drums")
- Tempo (BPM) and time signature
- Key and chord progressions
- Mood and energy level
- Genre and Sub-genre specifics

Output ONLY valid JSON with this structure:
{
  "enhanced_prompt": "detailed, descriptive prompt string...",
  "technical_specs": {
    "tempo_bpm": 120,
    "key": "C major",
    "instruments": ["instrument1", "instrument2"]
  },
  "suggested_studio": "Chiptune" | "Cypher" | "Ironclad" | "Blue Note" | "Virtuoso" | "Standard"
}"""

    def parse_response(self, response: str) -> Dict[str, Any]:
        result = super().parse_response(response)
        if "enhanced_prompt" not in result:
             # Fallback if valid JSON wasn't found
            return {
                "enhanced_prompt": response, 
                "technical_specs": {},
                "suggested_studio": "Standard"
            }
        return result

class AgentOrchestrator:
    """Coordinates multiple agents for music generation"""
    
    def __init__(self, models_dir: Path):
        self.llm_engine = LLMEngine(models_dir)
        
        # Initialize agents
        self.prompt_agent = PromptAgent(self.llm_engine)
        
        # Preload the main model
        # self.llm_engine.load_model("llama-3-8b")
    
    def enhance_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """Enhance a user prompt with musical detail"""
        return self.prompt_agent.process({"prompt": user_prompt})
