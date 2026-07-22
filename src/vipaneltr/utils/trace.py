"""
Trace builder for PanelTR-ViTabQA.

Builds comprehensive execution traces for debugging and analysis.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config


class TraceBuilder:
    """
    Builder for execution traces.
    
    Collects outputs from all phases and builds a comprehensive trace.
    """
    
    def __init__(self, config: Config):
        """
        Initialize trace builder.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._reset()
    
    def _reset(self):
        """Reset trace state."""
        self._qa_id = ""
        self._start_time = None
        self._investigation = None
        self._self_review = None
        self._peer_review = None
        self._errors: List[str] = []
        self._raw_responses: Dict[str, Any] = {}
    
    def start_trace(self, qa_id: str):
        """
        Start a new trace.
        
        Args:
            qa_id: QA identifier
        """
        self._reset()
        self._qa_id = qa_id
        self._start_time = datetime.now()
    
    def add_investigation(self, output):
        """
        Add Investigation phase output.
        
        Args:
            output: MultiAgentInvestigationOutput instance
        """
        # MultiAgentInvestigationOutput has to_dict()
        self._investigation = output.to_dict()
        
        # Store raw responses if enabled
        if self.config.logging.save_raw_responses:
            for persona_name, persona_result in output.persona_results.items():
                if persona_result.analyze_output and persona_result.analyze_output.raw_response:
                    self._raw_responses[f"investigation_{persona_name}_A"] = {
                        "response": persona_result.analyze_output.raw_response,
                        "latency_ms": persona_result.analyze_output.latency_ms,
                    }
                if persona_result.solve_output and persona_result.solve_output.raw_response:
                    self._raw_responses[f"investigation_{persona_name}_S"] = {
                        "response": persona_result.solve_output.raw_response,
                        "latency_ms": persona_result.solve_output.latency_ms,
                    }
    
    def add_self_review(self, output):
        """
        Add Self-Review phase output.
        
        Args:
            output: SelfReviewOutput or Dict[str, SelfReviewOutput]
        """
        if isinstance(output, dict):
            self._self_review = {
                persona_name: persona_output.to_dict()
                for persona_name, persona_output in output.items()
            }
        else:
            self._self_review = output.to_dict()
        
        # Store raw responses if enabled
        if self.config.logging.save_raw_responses:
            if isinstance(output, dict):
                for persona_name, persona_output in output.items():
                    for i, round_data in enumerate(persona_output.rounds):
                        if round_data.raw_output:
                            self._raw_responses[f"self_review_{persona_name}_{i+1}"] = {
                                "response": round_data.raw_output.raw_response,
                                "latency_ms": round_data.raw_output.latency_ms,
                            }
            else:
                for i, round_data in enumerate(output.rounds):
                    if round_data.raw_output:
                        self._raw_responses[f"self_review_{i+1}"] = {
                            "response": round_data.raw_output.raw_response,
                            "latency_ms": round_data.raw_output.latency_ms,
                        }
    
    def add_peer_review(self, output):
        """
        Add Peer-Review phase output.
        
        Args:
            output: PeerReviewOutput instance
        """
        self._peer_review = output.to_dict()
        
        # Store raw responses if enabled
        if self.config.logging.save_raw_responses:
            # Presentation round
            if output.presentation_round:
                for name, resp in output.presentation_round.responses.items():
                    if resp:
                        self._raw_responses[f"peer_review_r0_{name}"] = {
                            "response": resp.raw_response,
                            "latency_ms": resp.latency_ms,
                        }
            
            # Deliberation rounds
            for i, round_data in enumerate(output.deliberation_rounds):
                for name, resp in round_data.responses.items():
                    if resp:
                        self._raw_responses[f"peer_review_r{i+1}_{name}"] = {
                            "response": resp.raw_response,
                            "latency_ms": resp.latency_ms,
                        }
            
            # Final synthesis
            if output.final_synthesis:
                self._raw_responses["peer_review_synthesis"] = {
                    "response": output.final_synthesis.raw_response,
                    "latency_ms": output.final_synthesis.latency_ms,
                }
    
    def add_error(self, error: str):
        """
        Add an error to the trace.
        
        Args:
            error: Error message
        """
        self._errors.append(error)
    
    def build(self) -> Dict[str, Any]:
        """
        Build the final trace.
        
        Returns:
            Complete trace dictionary
        """
        end_time = datetime.now()
        
        trace = {
            "meta": {
                "qa_id": self._qa_id,
                "timestamp": self._start_time.isoformat() if self._start_time else None,
                "duration_ms": (
                    (end_time - self._start_time).total_seconds() * 1000
                    if self._start_time else 0
                ),
                "model": f"{self.config.model.provider}/{self.config.model.name}",
                "config_hash": self._compute_config_hash(),
                "paneltr_version": "1.0.0",
            },
            "investigation": self._investigation,
            "self_review": self._self_review,
            "peer_review": self._peer_review,
            "errors": self._errors if self._errors else None,
        }
        
        # Add raw responses if enabled
        if self.config.logging.save_raw_responses and self._raw_responses:
            trace["raw_llm_responses"] = self._raw_responses
        
        return trace
    
    def _compute_config_hash(self) -> str:
        """Compute hash of relevant config for reproducibility."""
        config_str = json.dumps({
            "model": self.config.model.name,
            "provider": self.config.model.provider,
            "temperature": self.config.model.temperature,
            "tmax_self": self.config.paneltr.tmax_self,
            "tmax_peer": self.config.paneltr.tmax_peer,
            "consensus_threshold": self.config.paneltr.consensus_threshold,
            "unanswerable_threshold": self.config.paneltr.unanswerable_threshold,
            "personas": self.config.personas.enabled,
            "table_representation": self.config.table.representation,
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_trace(trace: Dict[str, Any], output_path: str):
    """
    Save trace to JSON file.
    
    Args:
        trace: Trace dictionary
        output_path: Output file path
    """
    from pathlib import Path
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)


def load_trace(input_path: str) -> Dict[str, Any]:
    """
    Load trace from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Trace dictionary
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


class StructuredOutputSaver:
    """
    Saves PanelTR results in a structured format with separate files per phase.
    
    Output structure:
        {output_dir}/{run_id}/
        ├── results.json          # Final results (compact)
        ├── investigation.json    # Investigation phase details
        ├── self_review.json      # Self-Review phase details
        ├── peer_review.json      # Peer-Review phase details
        ├── meta.json             # Run metadata (config, stats)
        └── raw_responses.json    # Raw LLM responses (optional)
    """
    
    def __init__(self, output_dir: str, run_id: Optional[str] = None):
        """
        Initialize structured output saver.
        
        Args:
            output_dir: Base output directory
            run_id: Run identifier (auto-generated if not provided)
        """
        from pathlib import Path
        
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        
        # Accumulators for each phase
        self._results: List[Dict[str, Any]] = []
        self._investigations: List[Dict[str, Any]] = []
        self._self_reviews: List[Dict[str, Any]] = []
        self._peer_reviews: List[Dict[str, Any]] = []
        self._raw_responses: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []
        
        # Metadata
        self._start_time = datetime.now()
        self._config: Optional[Dict[str, Any]] = None
    
    def set_config(self, config):
        """Set config for metadata."""
        from dataclasses import asdict
        self._config = asdict(config)
    
    def add_result(self, result: Dict[str, Any]):
        """
        Add a single result to the structured output.
        
        Extracts and separates trace data into appropriate phase files.
        
        Args:
            result: Full result dictionary with trace
        """
        qa_id = result.get("qa_id", "unknown")
        
        # Extract summary for results.json (compact)
        summary = {
            "qa_id": result.get("qa_id"),
            "table_id": result.get("table_id"),
            "question": result.get("question"),
            "pred_answer": result.get("pred_answer"),
            "formatted_answer": result.get("formatted_answer"),
            "groundtruth": result.get("groundtruth"),
            "answerable": result.get("answerable"),
            "confidence": result.get("confidence"),
            "final_rationale": result.get("final_rationale"),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "cost_usd": round(float(result.get("cost_usd", 0.0) or 0.0), 6),
        }
        self._results.append(summary)
        
        # Extract trace data
        trace = result.get("trace", {})
        if not trace:
            return
        
        meta = trace.get("meta", {})
        
        # Investigation phase
        if trace.get("investigation"):
            self._investigations.append({
                "qa_id": qa_id,
                "timestamp": meta.get("timestamp"),
                "data": trace["investigation"]
            })
        
        # Self-Review phase
        if trace.get("self_review"):
            self._self_reviews.append({
                "qa_id": qa_id,
                "timestamp": meta.get("timestamp"),
                "data": trace["self_review"]
            })
        
        # Peer-Review phase
        if trace.get("peer_review"):
            self._peer_reviews.append({
                "qa_id": qa_id,
                "timestamp": meta.get("timestamp"),
                "data": trace["peer_review"]
            })
        
        # Raw LLM responses
        if trace.get("raw_llm_responses"):
            self._raw_responses.append({
                "qa_id": qa_id,
                "responses": trace["raw_llm_responses"]
            })
        
        # Errors
        if trace.get("errors"):
            self._errors.append({
                "qa_id": qa_id,
                "errors": trace["errors"]
            })
    
    def save_all(self, save_raw: bool = False) -> Dict[str, str]:
        """
        Save all accumulated results to structured files.
        
        Args:
            save_raw: Whether to save raw LLM responses
            
        Returns:
            Dictionary mapping file type to file path
        """
        saved_files = {}
        
        # 1. Save results.json (compact final results)
        results_path = self.output_dir / "results.json"
        self._save_json(results_path, {
            "run_id": self.run_id,
            "count": len(self._results),
            "predictions": self._results
        })
        saved_files["results"] = str(results_path)
        
        # 2. Save investigation.json
        if self._investigations:
            investigation_path = self.output_dir / "investigation.json"
            self._save_json(investigation_path, {
                "phase": "investigation",
                "count": len(self._investigations),
                "items": self._investigations
            })
            saved_files["investigation"] = str(investigation_path)
        
        # 3. Save self_review.json
        if self._self_reviews:
            self_review_path = self.output_dir / "self_review.json"
            self._save_json(self_review_path, {
                "phase": "self_review",
                "count": len(self._self_reviews),
                "items": self._self_reviews
            })
            saved_files["self_review"] = str(self_review_path)
        
        # 4. Save peer_review.json
        if self._peer_reviews:
            peer_review_path = self.output_dir / "peer_review.json"
            self._save_json(peer_review_path, {
                "phase": "peer_review",
                "count": len(self._peer_reviews),
                "items": self._peer_reviews
            })
            saved_files["peer_review"] = str(peer_review_path)
        
        # 5. Save meta.json
        end_time = datetime.now()
        meta_path = self.output_dir / "meta.json"
        
        # Compute statistics
        stats = self._compute_stats()
        
        self._save_json(meta_path, {
            "run_id": self.run_id,
            "started_at": self._start_time.isoformat(),
            "finished_at": end_time.isoformat(),
            "duration_seconds": (end_time - self._start_time).total_seconds(),
            "total_samples": len(self._results),
            "statistics": stats,
            "config": self._config,
            "errors": self._errors if self._errors else None
        })
        saved_files["meta"] = str(meta_path)
        
        # 6. Save raw_responses.json (optional)
        if save_raw and self._raw_responses:
            raw_path = self.output_dir / "raw_responses.json"
            self._save_json(raw_path, {
                "count": len(self._raw_responses),
                "items": self._raw_responses
            })
            saved_files["raw_responses"] = str(raw_path)
        
        return saved_files
    
    def _save_json(self, path, data: Dict[str, Any]):
        """Save data to JSON file with pretty formatting."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _compute_stats(self) -> Dict[str, Any]:
        """Compute statistics from results."""
        if not self._results:
            return {}
        
        answerable_count = sum(1 for r in self._results if r.get("answerable"))
        unanswerable_count = len(self._results) - answerable_count
        
        confidences = [r.get("confidence", 0) for r in self._results if r.get("confidence") is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        total_prompt = sum(int(r.get("prompt_tokens", 0) or 0) for r in self._results)
        total_completion = sum(int(r.get("completion_tokens", 0) or 0) for r in self._results)
        total_tokens = sum(int(r.get("total_tokens", 0) or 0) for r in self._results)
        total_cost = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in self._results)
        count = len(self._results)
        
        return {
            "answerable_count": answerable_count,
            "unanswerable_count": unanswerable_count,
            "answerable_ratio": answerable_count / len(self._results) if self._results else 0,
            "avg_confidence": round(avg_confidence, 4),
            "min_confidence": min(confidences) if confidences else None,
            "max_confidence": max(confidences) if confidences else None,
            "total_prompt_tokens": total_prompt,
            "average_prompt_tokens": round(total_prompt / count, 1),
            "total_completion_tokens": total_completion,
            "average_completion_tokens": round(total_completion / count, 1),
            "total_tokens": total_tokens,
            "average_total_tokens": round(total_tokens / count, 1),
            "total_cost_usd": round(total_cost, 6),
            "average_cost_usd": round(total_cost / count, 6),
        }
