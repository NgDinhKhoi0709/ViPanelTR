"""
Question type detector for PanelTR-ViTabQA.

Detects question types (how/why/other) from hints and keywords.
Used to activate Explainer agent when needed.
"""

from __future__ import annotations

from typing import List

# Vietnamese keywords for HOW questions
HOW_KEYWORDS = [
    "làm thế nào",
    "như thế nào",
    "bằng cách nào",
    "cách nào",
    "làm sao",
    "thế nào",
    "how",
]

# Vietnamese keywords for WHY questions
WHY_KEYWORDS = [
    "tại sao",
    "vì sao",
    "lý do",
    "nguyên nhân",
    "do đâu",
    "sao lại",
    "why",
]

# Hints that indicate HOW questions
HOW_HINTS = [
    "Làm thế nào, như thế nào,...",
    "How",
]

# Hints that indicate WHY questions
WHY_HINTS = [
    "Vì sao, tại sao,...",
    "Why",
]


def detect_question_type(question: str, hints: List[str] = None) -> str:
    """
    Detect question type from question text and hints.
    
    Args:
        question: Question text
        hints: List of hint strings (from dataset)
        
    Returns:
        One of: "how", "why", "other"
    """
    hints = hints or []
    question_lower = question.lower()
    
    # Check hints first (higher priority)
    for hint in hints:
        hint_lower = hint.lower()
        
        # Check HOW hints
        for how_hint in HOW_HINTS:
            if how_hint.lower() in hint_lower or hint_lower in how_hint.lower():
                return "how"
        
        # Check WHY hints
        for why_hint in WHY_HINTS:
            if why_hint.lower() in hint_lower or hint_lower in why_hint.lower():
                return "why"
    
    # Check keywords in question
    for keyword in HOW_KEYWORDS:
        if keyword.lower() in question_lower:
            return "how"
    
    for keyword in WHY_KEYWORDS:
        if keyword.lower() in question_lower:
            return "why"
    
    return "other"


def should_activate_explainer(
    question: str,
    hints: List[str] = None,
    is_unanswerable: bool = False,
) -> bool:
    """
    Determine if Explainer agent should be activated.
    
    Explainer is activated when:
    1. Question type is how/why (needs explanation)
    2. Question is unanswerable (needs to explain why)
    
    Args:
        question: Question text
        hints: List of hint strings
        is_unanswerable: Flag indicating question may be unanswerable
        
    Returns:
        True if Explainer should be activated
    """
    # Always activate for unanswerable questions
    if is_unanswerable:
        return True
    
    # Activate for how/why questions
    question_type = detect_question_type(question, hints)
    if question_type in ["how", "why"]:
        return True
    
    return False


def get_question_category(question: str, hints: List[str] = None) -> str:
    """
    Get broader question category from hints.
    
    Categories based on Open-ViTabQA hints:
    - who: Ai, người nào
    - what: Cái gì, cái nào
    - when: Khi nào, thời gian
    - where: Ở đâu, nơi nào
    - how: Làm thế nào
    - why: Tại sao, vì sao
    - yes_no: Câu hỏi Yes/No
    - list: Liệt kê
    - calculate: Tính toán
    - multi_condition: Hỏi kết hợp
    
    Args:
        question: Question text
        hints: List of hint strings
        
    Returns:
        Question category string
    """
    hints = hints or []
    question_lower = question.lower()
    
    # Mapping from hint keywords to categories
    hint_mappings = {
        "ai, người nào": "who",
        "who": "who",
        "cái gì, cái nào": "what",
        "what": "what",
        "khi nào, thời gian": "when",
        "when": "when",
        "ở đâu": "where",
        "where": "where",
        "làm thế nào": "how",
        "how": "how",
        "vì sao, tại sao": "why",
        "why": "why",
        "yes/no": "yes_no",
        "liệt kê": "list",
        "tính toán": "calculate",
        "sử dụng tính toán": "calculate",
        "kết hợp": "multi_condition",
    }
    
    # Check hints
    for hint in hints:
        hint_lower = hint.lower()
        for pattern, category in hint_mappings.items():
            if pattern in hint_lower:
                return category
    
    # Fallback: detect from question text
    who_keywords = ["ai", "người nào", "who"]
    what_keywords = ["cái gì", "gì", "what"]
    when_keywords = ["khi nào", "lúc nào", "năm nào", "when"]
    where_keywords = ["ở đâu", "nơi nào", "where"]
    
    for kw in who_keywords:
        if kw in question_lower:
            return "who"
    
    for kw in when_keywords:
        if kw in question_lower:
            return "when"
    
    for kw in where_keywords:
        if kw in question_lower:
            return "where"
    
    for kw in what_keywords:
        if kw in question_lower:
            return "what"
    
    # Check how/why
    q_type = detect_question_type(question, hints)
    if q_type != "other":
        return q_type
    
    return "other"
