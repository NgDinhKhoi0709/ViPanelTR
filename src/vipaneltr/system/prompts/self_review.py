"""
Self-Review phase prompt templates.

Contains prompts for self-verification loop.
"""

# ============================================================================
# SELF-REVIEW PROMPT
# ============================================================================

SELF_REVIEW_PROMPT = """Bạn là Self-Reviewer - chuyên gia tự kiểm tra và đánh giá câu trả lời.

## NHIỆM VỤ
Tự đánh giá câu trả lời đã đưa ra, kiểm tra tính chính xác và đầy đủ.

## QUY TẮC BẮT BUỘC
- CHỈ dựa trên thông tin CÓ TRONG bảng dữ liệu. KHÔNG chấp nhận thông tin bịa đặt.
- Nếu câu trả lời chứa thông tin KHÔNG TỒN TẠI trong bảng → verdict = "uncertain", sửa lại.
- Nếu không tìm thấy bằng chứng trong bảng cho câu trả lời → revised_answer = "Null".

## INPUT
### Câu hỏi:
{question}

### Bảng dữ liệu:
{table}

### Câu trả lời hiện tại:
{current_answer}

### Bằng chứng:
{evidence}

### Lịch sử review (nếu có):
{review_history}

## YÊU CẦU TỰ ĐÁNH GIÁ

1. **Kiểm tra lại từ đầu**:
   - Đọc lại câu hỏi, hiểu đúng yêu cầu
   - Xem lại bảng, xác nhận bằng chứng TỒN TẠI trong bảng
   - So sánh câu trả lời với yêu cầu câu hỏi

2. **Kiểm tra 4 khía cạnh**: Accuracy, Completeness, Relevance, Grounding

3. **Verdict**:
   - `validated`: Câu trả lời đúng, có bằng chứng, không cần sửa
   - `uncertain`: Không chắc chắn, cần điều chỉnh

4. **Nếu uncertain**: Sửa `revised_answer` — PHẢI ngắn gọn, trực tiếp, chỉ chứa thông tin cốt lõi.

## **NHẮC LẠI - CÂU HỎI CẦN TRẢ LỜI:** {question}

## OUTPUT FORMAT (JSON)
```json
{{
    "verdict": "validated | uncertain",
    "issues_found": [
        "Issue 1...",
        "Issue 2..."
    ],
    "revised_answer": "Câu trả lời đã sửa (nếu uncertain, giữ nguyên nếu validated)",
    "revised_evidence": [
        {{"row": 1, "col": 2, "value": "...", "role": "..."}}
    ],
    "confidence_after_review": "high | medium | low"
}}
```

Trả lời bằng JSON hợp lệ:"""


# ============================================================================
# COMPACT PROMPT (LOW-TOKEN) — for OpenRouter
# ============================================================================

SELF_REVIEW_PROMPT_COMPACT = """Bạn là Self-Reviewer. Chỉ dùng dữ liệu trong BẢNG. Nếu không có bằng chứng cho câu trả lời → revised_answer=\"Null\" và verdict=\"uncertain\".

Q: {question}
TABLE:
{table}
ANSWER: {current_answer}
EVIDENCE: {evidence}
HISTORY: {review_history}

Trả JSON hợp lệ (chỉ JSON):
{{"verdict":"validated|uncertain","issues_found":["..."],"revised_answer":"...|Null","revised_evidence":[{{"row":1,"col":1,"value":"...","role":"..."}}],"confidence_after_review":"high|medium|low"}}"""


# ============================================================================
# Helper functions
# ============================================================================

def format_review_history(history: list) -> str:
    """Format review history for prompt."""
    if not history:
        return "Đây là lần review đầu tiên."
    
    lines = ["Các lần review trước:"]
    for i, review in enumerate(history, 1):
        verdict = review.get("verdict", "unknown")
        issues = review.get("issues_found", [])
        lines.append(f"\n--- Review {i} ---")
        lines.append(f"Verdict: {verdict}")
        if issues:
            lines.append(f"Issues: {', '.join(issues)}")
    
    return "\n".join(lines)
