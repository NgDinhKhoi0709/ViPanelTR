"""
Dataset loader for Open-ViTabQA.

Loads table.json and qas_*.json files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vipaneltr.evaluation.normalization import is_unanswerable_reference


class DatasetLoader:
    """
    Loader for Open-ViTabQA dataset.
    
    Dataset structure:
        dataset/
            table.json      - All tables
            qas_train.json  - Training QA pairs
            qas_dev.json    - Development QA pairs
            qas_test.json   - Test QA pairs
    """
    
    def __init__(self, dataset_dir: str = "./dataset"):
        """
        Initialize loader.
        
        Args:
            dataset_dir: Path to dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self._tables_cache = None
        self._qas_cache = {}
    
    def load_tables(self) -> dict:
        if self._tables_cache is not None:
            return self._tables_cache
        table_path = self.dataset_dir / "table.json"
        return self.load_tables_path(str(table_path))

    def load_tables_path(self, tables_path: str) -> dict:
        import json
        path = Path(tables_path)
        if not path.is_absolute():
            path = (self.dataset_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Table file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tables = data.get("table", data) if isinstance(data, dict) else data
        if isinstance(tables, list):
            self._tables_cache = {str(t.get("table_id", i)): t for i, t in enumerate(tables)}
        elif isinstance(tables, dict):
            self._tables_cache = {str(k): v for k, v in tables.items() if isinstance(v, dict)}
        return self._tables_cache
    
    def load_qas(self, split: str) -> list:
        """
        Load QA pairs for a specific split.
        
        Args:
            split: One of 'train', 'dev', 'test'
            
        Returns:
            List of QA pair dictionaries
        """
        if split in self._qas_cache:
            return self._qas_cache[split]
        
        qas_path = self.dataset_dir / f"qas_{split}.json"
        
        if not qas_path.exists():
            raise FileNotFoundError(f"QAs file not found: {qas_path}")
        
        with open(qas_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both formats: {"qas": [...]} or [...]
        qas = data.get("qas", data) if isinstance(data, dict) else data
        
        self._qas_cache[split] = qas
        return qas

    def load_qas_path(self, qas_path: str) -> List[Dict[str, Any]]:
        """Load QA pairs from an explicit JSON file path.

        Supports both formats:
        - {"qas": [...]} 
        - [...]

        Args:
            qas_path: Path to a qas_*.json file (absolute or relative)

        Returns:
            List of QA pair dictionaries
        """
        path = Path(qas_path)
        if not path.is_absolute():
            path = (self.dataset_dir / path).resolve()
        else:
            path = path.resolve()

        cache_key = str(path)
        if cache_key in self._qas_cache:
            return self._qas_cache[cache_key]

        if not path.exists():
            raise FileNotFoundError(f"QAs file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qas = data.get("qas", data) if isinstance(data, dict) else data
        self._qas_cache[cache_key] = qas
        return qas
    
    def get_table(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific table by ID.
        
        Args:
            table_id: Table identifier
            
        Returns:
            Table data or None if not found
        """
        tables = self.load_tables()
        return tables.get(table_id)
    
    def get_qa_with_table(
        self, 
        split: str, 
        qa_id: Optional[str] = None,
        index: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a QA pair with its associated table.
        
        Args:
            split: Dataset split
            qa_id: QA identifier (optional)
            index: Index in the split (optional)
            
        Returns:
            Dictionary with 'qa' and 'table' keys, or None
        """
        qas = self.load_qas(split)
        tables = self.load_tables()
        
        qa = None
        if qa_id:
            for q in qas:
                if q["qa_id"] == qa_id:
                    qa = q
                    break
        elif index is not None and 0 <= index < len(qas):
            qa = qas[index]
        
        if qa is None:
            return None
        
        table = tables.get(qa["table_id"])
        
        return {
            "qa": qa,
            "table": table
        }
    
    def iterate_split(
        self, 
        split: str, 
        limit: Optional[int] = None
    ):
        """
        Iterate over QA pairs with their tables.
        
        Args:
            split: Dataset split
            limit: Maximum number of items to yield
            
        Yields:
            Dictionaries with 'qa' and 'table' keys
        """
        qas = self.load_qas(split)
        tables = self.load_tables()
        
        count = 0
        for qa in qas:
            if limit and count >= limit:
                break
            
            table = tables.get(qa["table_id"])
            if table:
                yield {"qa": qa, "table": table}
                count += 1
    
    def get_split_stats(self, split: str) -> Dict[str, Any]:
        """
        Get statistics for a dataset split.
        
        Args:
            split: Dataset split
            
        Returns:
            Dictionary with statistics
        """
        qas = self.load_qas(split)
        tables = self.load_tables()
        
        # Count by hints
        hint_counts: Dict[str, int] = {}
        answerable = 0
        unanswerable = 0
        
        for qa in qas:
            # Count hints
            for hint in qa.get("hints", []):
                hint_counts[hint] = hint_counts.get(hint, 0) + 1
            
            # Count answerable/unanswerable
            if is_unanswerable_reference(qa.get("answer")):
                unanswerable += 1
            else:
                answerable += 1
        
        # Count table types
        table_ids_in_split = set(qa["table_id"] for qa in qas)
        table_types: Dict[str, int] = {}
        
        for table_id in table_ids_in_split:
            table = tables.get(table_id, {})
            for t_type in table.get("table_type", []):
                table_types[t_type] = table_types.get(t_type, 0) + 1
        
        return {
            "total_qas": len(qas),
            "unique_tables": len(table_ids_in_split),
            "answerable": answerable,
            "unanswerable": unanswerable,
            "hint_counts": hint_counts,
            "table_types": table_types,
        }
