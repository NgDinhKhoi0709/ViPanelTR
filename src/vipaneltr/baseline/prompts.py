from __future__ import annotations

PROMPT_VERSION = "v1"

def build_tableqa_prompt(
    *,
    question: str,
    table_str: str,
    answer_language: str = "vi",
) -> str:
    if answer_language == "vi":
        instr = (
            "Bạn là hệ thống hỏi-đáp dựa trên bảng.\n"
            "Bảng được cung cấp dưới dạng Flatten V1 string trong TABLE_STR.\n"
            "CHỈ được dùng thông tin trong TABLE_STR để trả lời.\n"
            "\n"
            "GHI CHÚ ĐỊNH DẠNG TABLE_STR (FLATTEN V1):\n"
            "- Mỗi dòng có dạng: row_header|column_header|value.\n"
            "- Header hàng và header cột luôn có hậu tố '<header>'.\n"
            "- Giá trị ô dữ liệu là thành phần thứ ba của mỗi dòng.\n"
            "\n"
            "QUY TẮC TRẢ LỜI (BẮT BUỘC):\n"
            "- Chỉ xuất ra DUY NHẤT 1 dòng: đáp án cuối cùng.\n"
            "- Không giải thích. Không suy luận từng bước. Không ghi nhãn như 'Đáp án:', 'ANSWER:', 'Giải thích:', không dùng markdown.\n"
            "- Không lặp lại câu hỏi hay trích lại dữ liệu bảng.\n"
            "- Nếu không thể tìm thấy thông tin phù hợp để trích xuất hoặc suy luận từ bảng, câu trả lời được trả về là \"Null\".\n"
        )
    else:
        instr = (
            "You are a table question-answering system.\n"
            "The table is provided as Flatten V1 string in TABLE_STR.\n"
            "Use ONLY TABLE_STR to answer.\n"
            "\n"
            "TABLE_STR FORMAT NOTES (FLATTEN V1):\n"
            "- Each line has the form: row_header|column_header|value.\n"
            "- Row and column headers always end with '<header>'.\n"
            "- The data value is the third field of each line.\n"
            "\n"
            "ANSWERING RULES (MANDATORY):\n"
            "- Output EXACTLY one line: the final answer only.\n"
            "- No explanations. No step-by-step reasoning. No labels like 'Answer:'/'Explanation:'. No markdown.\n"
            "- Do not repeat the question or quote the table.\n"
            "- If the table is insufficient to answer, output: Null.\n"
        )

    table_str = str(table_str or "").strip()

    return (
        f"{instr}\n"
        f"TABLE_STR:\n{table_str}\n\n"
        f"QUESTION: {question}\n"
        f"FINAL ANSWER:"
    )
