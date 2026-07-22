"""
Configuration management for PanelTR-ViTabQA.

Supports YAML config files and environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import dataset_dir as default_dataset_dir
from .paths import repo_root

# Load .env file if exists
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = repo_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try current directory
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars


@dataclass
class ModelConfig:
    """LLM model configuration."""
    provider: str = "openai"  # openai | gemini | claude | openrouter
    name: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 10000
    timeout: int = 60  # Request timeout in seconds
    api_key: Optional[str] = None  # Will use env var if not set
    api_base: Optional[str] = None  # For custom endpoints
    
    def __post_init__(self):
        """
        Normalize fields only.

        NOTE: Do NOT auto-fill `api_key` from environment here.
        The CLI can override `provider` at runtime (e.g. `--model openrouter/...`),
        and auto-filling the key during dataclass init would bind the wrong key.

        Each provider client is responsible for reading the correct env var if
        `model.api_key` is not explicitly set in YAML.
        """
        if isinstance(self.provider, str):
            self.provider = self.provider.strip()
        if isinstance(self.name, str):
            self.name = self.name.strip()


@dataclass
class PanelTRConfig:
    """PanelTR algorithm configuration."""
    tmax_self: int = 1  # Max self-review iterations
    tmax_peer: int = 1  # Max peer-review deliberation rounds (paper default=1)
    consensus_threshold: float = 1.0  # 5/5 identical for strict consensus
    confidence_threshold: float = 0.9  # Min confidence for early stop
    unanswerable_threshold: int = 3  # Phase 1 early stop if >= this many unanswerable
    enable_self_review: bool = True
    enable_peer_review: bool = True
    enable_semantic_consensus: bool = True  # Semantic consensus check when strict fails
    

@dataclass
class PersonasConfig:
    """Personas configuration for peer-review."""
    enabled: List[str] = field(default_factory=lambda: [
        "logician",
        "calculator", 
        "verifier",
        "structuralist",
        "synthesizer"
    ])


@dataclass 
class TableConfig:
    """Table representation configuration."""
    representation: str = "flattened"  # flattened | structured
    max_rows: int = 1500  # Limit rows
    max_cols: int = 50  # Limit columns
    normalize_vietnamese: bool = True  # Normalize Vietnamese text
    include_span_info: bool = True  # Include colspan/rowspan info
    include_html: bool = False  # Include raw HTML in representation


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "DEBUG"  # DEBUG | INFO | WARNING | ERROR
    output_dir: str = "./outputs"
    save_traces: bool = True
    save_raw_responses: bool = True  # Save raw LLM responses


@dataclass
class ThreadingConfig:
    """Threading configuration for parallel processing."""
    enabled: bool = True
    max_workers: int = 8  # Number of parallel workers for batch inference
    parallel_personas: bool = True  # Run personas in parallel during peer review


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: ["f1", "em", "rouge1", "meteor"])
    use_vietnamese_tokenizer: bool = True
    fail_on_metric_error: bool = False


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    paneltr: PanelTRConfig = field(default_factory=PanelTRConfig)
    personas: PersonasConfig = field(default_factory=PersonasConfig)
    table: TableConfig = field(default_factory=TableConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    threading: ThreadingConfig = field(default_factory=ThreadingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Dataset paths
    dataset_dir: str = field(default_factory=lambda: str(default_dataset_dir()))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        model_data = dict(data.get("model", {}))
        # `openrouter_provider` is now a CLI-only runtime flag, not YAML config.
        model_data.pop("openrouter_provider", None)
        return cls(
            model=ModelConfig(**model_data),
            paneltr=PanelTRConfig(**data.get("paneltr", {})),
            personas=PersonasConfig(**data.get("personas", {})),
            table=TableConfig(**data.get("table", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            threading=ThreadingConfig(**data.get("threading", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            dataset_dir=data.get("dataset_dir", str(default_dataset_dir())),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "model": {
                "provider": self.model.provider,
                "name": self.model.name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "timeout": self.model.timeout,
                "api_base": self.model.api_base,
            },
            "paneltr": {
                "tmax_self": self.paneltr.tmax_self,
                "tmax_peer": self.paneltr.tmax_peer,
                "consensus_threshold": self.paneltr.consensus_threshold,
                "unanswerable_threshold": self.paneltr.unanswerable_threshold,
                "enable_semantic_consensus": self.paneltr.enable_semantic_consensus,
            },
            "personas": {
                "enabled": self.personas.enabled,
            },
            "table": {
                "representation": self.table.representation,
                "max_rows": self.table.max_rows,
                "include_html": self.table.include_html,
            },
            "logging": {
                "level": self.logging.level,
                "save_raw_responses": self.logging.save_raw_responses,
                "output_dir": self.logging.output_dir,
                "save_traces": self.logging.save_traces,
            },
            "evaluation": {
                "use_vietnamese_tokenizer": self.evaluation.use_vietnamese_tokenizer,
                "fail_on_metric_error": self.evaluation.fail_on_metric_error,
            },
            "dataset_dir": self.dataset_dir,
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default config.
        
    Returns:
        Config object
    """
    if config_path is None:
        return Config()
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return Config.from_dict(data or {})


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        config_path: Path to save YAML file
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
