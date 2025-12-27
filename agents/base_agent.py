from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from llm.engine import LLMEngine
import json

class BaseAgent(ABC):
    """Base class for specialized agents"""
    
    def __init__(self, llm_engine: LLMEngine):
        self.llm = llm_engine
        self.model_name = self.get_model_name()
        
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model filename this agent uses (e.g., 'llama-3-8b')"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return result"""
        prompt = self.build_prompt(input_data)
        
        # Fallback to local LLM
        response = self.llm.generate(self.model_name, prompt)
        return self.parse_response(response)
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the full prompt from input data (Llama-3 Chat Format)"""
        system = self.get_system_prompt()
        user_input = input_data.get('prompt', '')
        
        # Llama-3 specific formatting
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured output"""
        try:
            # Basic JSON extraction
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
            return {"raw_output": response}
        except Exception as e:
            return {"error": str(e), "raw_output": response}
