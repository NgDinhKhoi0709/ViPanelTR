"""
Base agent interface for PanelTR-ViTabQA.

Defines abstract agent class and output structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...utils.json_parser import parse_json_response


@dataclass
class AgentOutput:
    """
    Standard output structure for all agents.
    
    Contains both parsed structured output and raw response.
    """
    
    # Parsed structured output
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Raw LLM response
    raw_response: str = ""
    
    # Metadata
    agent_name: str = ""
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from data dict."""
        return self.data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get value from data dict."""
        return self.data[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data,
            "raw_response": self.raw_response,
            "agent_name": self.agent_name,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Agents process prompts through LLM and return structured outputs.
    """
    
    def __init__(
        self,
        name: str,
        llm_client: Any,  # LLMClient
        output_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize agent.
        
        Args:
            name: Agent name for logging/tracing
            llm_client: LLM client for generating responses
            output_schema: Expected output JSON schema for validation
        """
        self.name = name
        self.llm_client = llm_client
        self.output_schema = output_schema
    
    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        """
        Build prompt for LLM.
        
        Args:
            **kwargs: Agent-specific arguments
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def parse_output(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured output.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed dictionary
        """
        pass
    
    def invoke(self, **kwargs) -> AgentOutput:
        """
        Invoke agent with given inputs.
        
        Args:
            **kwargs: Agent-specific arguments
            
        Returns:
            AgentOutput with parsed data
        """
        import time
        
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self.build_prompt(**kwargs)
            
            # Call LLM
            response = self.llm_client.generate(prompt)
            
            # Parse output
            parsed = self.parse_output(response)
            
            latency = (time.time() - start_time) * 1000
            
            return AgentOutput(
                data=parsed,
                raw_response=response,
                agent_name=self.name,
                success=True,
                latency_ms=latency,
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            
            return AgentOutput(
                data={},
                raw_response="",
                agent_name=self.name,
                success=False,
                error=str(e),
                latency_ms=latency,
            )
    
    def _parse_json(self, response: str) -> Dict[str, Any]:
        """
        Helper to parse JSON from response.
        
        Args:
            response: LLM response potentially containing JSON
            
        Returns:
            Parsed dictionary
        """
        return parse_json_response(response)
    
    def _validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate output against schema.
        
        Args:
            data: Parsed output data
            
        Returns:
            True if valid
        """
        if self.output_schema is None:
            return True
        
        # Simple validation: check required keys exist
        required = self.output_schema.get("required", [])
        for key in required:
            if key not in data:
                return False
        
        return True
