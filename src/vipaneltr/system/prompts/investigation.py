"""
Investigation Prompt Templates for PanelTR.

Contains atomic prompts that are persona-agnostic:
- ANALYZE: Phân tích câu hỏi từ góc nhìn chuyên môn
- SOLVE: Lập kế hoạch giải quyết và đưa ra draft answer  
- VERIFY: Xác minh câu trả lời từ góc độ chuyên môn
- PRESENT: Trình bày kết quả cho hội đồng
- DELIBERATE: Thảo luận và phản hồi ý kiến đồng nghiệp

Each prompt uses placeholders for persona-specific content:
- {persona_name}: Tên persona (e.g., "Logician")
- {persona_role}: Mô tả vai trò (e.g., "chuyên gia logic và suy luận")
- {analysis_focus}: Focus khi phân tích
- {verification_criteria}: Tiêu chí xác minh
- {deliberation_style}: Phong cách thảo luận
"""

# ============================================================================
# ANALYZE (A) - Phân tích câu hỏi từ góc nhìn persona
# ============================================================================

ANALYZE_PROMPT = """Bạn là {persona_name} - {persona_role}.

## NHIỆM VỤ
Phân tích câu hỏi về bảng dữ liệu từ GÓC NHÌN CHUYÊN MÔN của bạn.
**Trả lời NGẮN GỌN, đi thẳng vào trọng tâm. Mỗi observation/risk chỉ 1 câu.**

## QUY TẮC BẮT BUỘC
- CHỈ sử dụng thông tin CÓ TRONG bảng dữ liệu. KHÔNG suy diễn hoặc bịa đặt thông tin không tồn tại.
- Nếu thông tin không có trong bảng → đánh giá `unanswerable`.
- Chú ý merged cells, header hierarchy, rowspan/colspan khi đọc bảng.

## GÓC NHÌN CỦA BẠN
{analysis_focus}

## INPUT
### Câu hỏi:
{question}

### Bảng dữ liệu:
{table}

### Gợi ý (hints):
{hints}

## YÊU CẦU PHÂN TÍCH

Với tư cách là {persona_name}, hãy phân tích:

1. **Đánh giá độ phức tạp** (complexity): `basic` | `intermediate` | `complex`

2. **Loại câu hỏi** (question_type):
   - `who` | `what` | `when` | `where` | `how` | `why` | `yes_no` | `list` | `calculate` | `multi_conditions`

3. **Các điểm quan trọng** (key_observations): Tối đa 3 điểm, mỗi điểm 1 câu ngắn.

4. **Rủi ro tiềm ẩn** (potential_risks): Tối đa 2 rủi ro, mỗi cái 1 câu.

5. **Khả năng trả lời** (answerable_assessment): `answerable` | `uncertain` | `unanswerable`

## **NHẮC LẠI - CÂU HỎI CẦN TRẢ LỜI:** {question}

## OUTPUT FORMAT (JSON)
```json
{{
    "complexity": "basic | intermediate | complex",
    "question_type": "who | what | when | ...",
    "key_observations": ["...", "..."],
    "potential_risks": ["..."],
    "answerable_assessment": "answerable | uncertain | unanswerable"
}}
```

Trả lời bằng JSON hợp lệ:"""



# ============================================================================
# SOLVE (S) - Lập kế hoạch và đưa ra draft answer
# ============================================================================

SOLVE_PROMPT = """Bạn là {persona_name} - {persona_role}.

## NHIỆM VỤ
Dựa trên phân tích, đưa ra câu trả lời từ GÓC NHÌN CHUYÊN MÔN.
**Trả lời NGẮN GỌN. Mỗi bước giải quyết chỉ 1 câu.**

## QUY TẮC BẮT BUỘC
- CHỈ sử dụng thông tin CÓ TRONG bảng dữ liệu. KHÔNG suy diễn hoặc bịa đặt.
- Mọi bằng chứng (evidence_cells) PHẢI trích xuất chính xác từ bảng — ghi đúng row, col, value.
- Nếu không tìm thấy thông tin trong bảng → draft_answer = "Null".

## GÓC NHÌN CỦA BẠN
{analysis_focus}

## INPUT
### Câu hỏi:
{question}

### Bảng dữ liệu:
{table}

### Phân tích của bạn:
{analysis}

## YÊU CẦU

1. **Kế hoạch giải quyết** (solution_plan): Tối đa 3 bước, mỗi bước 1 câu.

2. **Câu trả lời nháp** (draft_answer):
   - **PHẢI ngắn gọn, trực tiếp, chỉ chứa thông tin cốt lõi trả lời câu hỏi.**
   - Hỏi "Ai?" → chỉ tên. Hỏi "Bao nhiêu?" → chỉ số. Hỏi "Có/Không?" → Có/Không/Phải/Đúng.
   - Hỏi "Liệt kê" → liệt kê ngắn, cách nhau bởi dấu phẩy.
   - Nếu không tìm được → "Null"

3. **Bằng chứng** (evidence_cells): Chỉ các ô then chốt nhất (tối đa 3).

4. **Độ tin cậy** (confidence): `high` | `medium` | `low`

## **NHẮC LẠI - CÂU HỎI CẦN TRẢ LỜI:** {question}

## OUTPUT FORMAT (JSON)
```json
{{
    "solution_plan": ["Bước 1: ...", "Bước 2: ..."],
    "draft_answer": "Câu trả lời ngắn gọn, sát câu hỏi",
    "evidence_cells": [
        {{"row": 1, "col": 2, "value": "...", "role": "..."}}
    ],
    "confidence": "high | medium | low"
}}
```

Trả lời bằng JSON hợp lệ:"""


# ============================================================================
# VERIFY (V) - Xác minh từ góc độ chuyên môn
# ============================================================================

VERIFY_PROMPT = """Bạn là {persona_name} - {persona_role}.

## NHIỆM VỤ
Xác minh câu trả lời từ GÓC ĐỘ CHUYÊN MÔN của bạn.
**Trả lời NGẮN GỌN. Mỗi issue chỉ 1 câu.**

## NGUYÊN TẮC CỐT LÕI
Bảng dữ liệu là nguồn sự thật duy nhất.
`draft_answer`, `evidence` và `solution_plan` chỉ là các đề xuất cần kiểm tra; không được mặc định là đúng.
Không chỉ kiểm tra đáp án nháp có xuất hiện nguyên văn trong bảng hay không: đáp án được phép là kết quả lọc, đếm, so sánh hoặc tổng hợp nếu có thể tự suy ra đầy đủ từ bảng.

## QUY TẮC BẮT BUỘC
- Kiểm tra theo chuỗi: **bảng dữ liệu → bằng chứng → kế hoạch → suy luận → đáp án nháp**.
- Đối chiếu từng evidence với bảng: đúng `row`, `col`, `value`, ngữ cảnh header/merged cells và đúng vai trò mà evidence tuyên bố hỗ trợ.
- Kiểm tra kế hoạch có xác định đúng dữ liệu cần dùng, điều kiện lọc, phép tính và kết luận hay không; không chấp nhận bước nhảy cóc hoặc kiến thức ngoài bảng.
- Tự thực hiện lại phép lọc/đếm/so sánh/tổng hợp từ bảng khi cần, rồi so sánh kết quả với `draft_answer`.
- Một ví dụ đơn lẻ không chứng minh được tổng số; với claim tổng hợp, phải kiểm tra toàn bộ tập dữ liệu hoặc phép tính tương ứng.
- Không coi một đáp án là hallucination chỉ vì nó không xuất hiện nguyên văn trong bảng; chỉ coi là hallucination khi không thể suy ra trực tiếp và đầy đủ từ bảng.
- Nếu bằng chứng hoặc kế hoạch sai nhưng bảng vẫn đủ dữ liệu để tự suy ra đáp án, trả về đáp án đã suy luận lại và dùng `needs_refinement` hoặc `invalid` tùy mức độ sai lệch.
- Nếu bảng không đủ dữ liệu để xác định chắc chắn, chọn `unanswerable` và `refined_answer` là `Null`.
- Chỉ dùng thông tin có trong bảng; không suy diễn vượt quá dữ liệu.
- Chú ý merged cells, header hierarchy, rowspan/colspan khi đọc bảng.

## TIÊU CHÍ XÁC MINH CỦA {persona_name}
{verification_criteria}

## INPUT
### Câu hỏi:
{question}

### Bảng dữ liệu:
{table}

### Câu trả lời nháp:
{draft_answer}

### Bằng chứng:
{evidence}

### Kế hoạch giải quyết:
{solution_plan}

## TRÌNH TỰ XÁC MINH
1. Xác định chính xác câu hỏi yêu cầu dữ liệu nào và điều kiện nào.
2. Kiểm tra `evidence` và `solution_plan` có khớp bảng, đủ để dẫn đến kết luận hay không.
3. Tự suy luận lại từ bảng nếu evidence/plan/draft chưa đủ hoặc mâu thuẫn.
4. Đối chiếu `draft_answer` với kết quả suy luận lại.
5. Nêu tối đa 3 issues, mỗi issue đúng 1 câu.

## QUY TẮC STATUS
- `valid`: evidence đúng, kế hoạch hợp lý và draft_answer khớp kết quả suy luận từ bảng.
- `needs_refinement`: kết quả cốt lõi đúng nhưng đáp án diễn đạt chưa đúng định dạng/yêu cầu, hoặc evidence/kế hoạch thiếu nhẹ nhưng không làm đổi kết quả.
- `invalid`: draft_answer, evidence hoặc kế hoạch có claim sai, mâu thuẫn bảng hoặc không thể dẫn đến kết quả được nêu.
- `unanswerable`: bảng không đủ thông tin để suy ra đáp án chắc chắn.

## QUY TẮC refined_answer
- Chỉ chứa thông tin cốt lõi trả lời câu hỏi, không giải thích.
- Hỏi "Ai?" → chỉ tên.
- Hỏi "Bao nhiêu?" → chỉ số hoặc số lượng cần thiết.
- Hỏi "Có/Không?" → chỉ `Có` hoặc `Không`.
- Nếu `unanswerable` → `Null`.

## **NHẮC LẠI - CÂU HỎI CẦN TRẢ LỜI:** {question}

## OUTPUT FORMAT (JSON)
```json
{{
    "verification_status": "valid | needs_refinement | invalid | unanswerable",
    "issues_found": ["..."],
    "refined_answer": "Câu trả lời ngắn gọn đã sửa (hoặc giữ nguyên)",
    "refined_evidence": [
        {{"row": 1, "col": 2, "value": "...", "role": "..."}}
    ],
    "is_answerable": true,
    "final_confidence": "high | medium | low"
}}
```

Trả lời bằng JSON hợp lệ:"""

# ============================================================================
# PRESENT - Trình bày kết quả cho hội đồng
# ============================================================================

PRESENT_PROMPT = """Bạn là {persona_name} - {persona_role}.

## NHIỆM VỤ
Trình bày KẾT QUẢ ĐIỀU TRA cho hội đồng.
**Trả lời NGẮN GỌN. Tối đa 2-3 items mỗi mục.**

## QUY TẮC BẮT BUỘC
- CHỈ trình bày thông tin CÓ TRONG bảng dữ liệu. KHÔNG thêm thông tin bịa đặt.
- Mọi evidence PHẢI trích xuất chính xác từ bảng.

## PHONG CÁCH TRÌNH BÀY CỦA {persona_name}
{deliberation_style}

## INPUT
### Câu hỏi gốc:
{question}

### Bảng dữ liệu:
{table}

### Kết quả điều tra của bạn:
{investigation_result}

## YÊU CẦU TRÌNH BÀY

1. **proposed_answer**: **PHẢI ngắn gọn, trực tiếp, chỉ chứa thông tin cốt lõi trả lời câu hỏi.**
   - Hỏi "Ai?" → chỉ tên. Hỏi "Bao nhiêu?" → chỉ số. Hỏi danh sách → liệt kê ngắn.
2. **key_evidence**: Chỉ các ô then chốt nhất (tối đa 3).
3. **reasoning**: Lập luận ngắn gọn 1-2 câu.

## **NHẮC LẠI - CÂU HỎI CẦN TRẢ LỜI:** {question}

## OUTPUT FORMAT (JSON)
```json
{{
    "proposed_answer": "Câu trả lời ngắn gọn, sát câu hỏi",
    "answerable": true,
    "confidence": 0.85,
    "key_evidence": [
        {{"row": 1, "col": 2, "value": "...", "importance": "critical"}}
    ],
    "reasoning": "Lập luận ngắn gọn 1-2 câu..."
}}
```

Trả lời bằng JSON hợp lệ:"""


# ============================================================================
# DELIBERATE - Thảo luận và phản hồi đồng nghiệp
# ============================================================================

DELIBERATE_PROMPT = """Bạn là {persona_name} - {persona_role}.

## NHIỆM VỤ  
Phản hồi ý kiến ĐỒNG NGHIỆP trong hội đồng.
**Trả lời NGẮN GỌN. Không cần đạt đồng thuận giả tạo.**

## QUY TẮC BẮT BUỘC
- CHỈ dựa trên thông tin CÓ TRONG bảng dữ liệu để đánh giá ý kiến đồng nghiệp.
- Nếu đồng nghiệp đưa ra thông tin KHÔNG TỒN TẠI trong bảng → dissent.
- KHÔNG thay đổi câu trả lời chỉ vì áp lực đa số. Chỉ revise khi có bằng chứng thuyết phục từ bảng.

## PHONG CÁCH THẢO LUẬN CỦA {persona_name}
{deliberation_style}

## INPUT
### Câu hỏi gốc:
{question}

### Bảng dữ liệu:
{table}

### Vị trí hiện tại của bạn:
{current_position}

### Ý kiến của các đồng nghiệp:
{colleagues_opinions}

### Lịch sử thảo luận (vòng trước):
{discussion_history}

## YÊU CẦU

1. **Stance**: `maintain` | `revise` | `support` | `dissent`
2. **updated_answer**: **PHẢI ngắn gọn, trực tiếp, chỉ chứa thông tin cốt lõi trả lời câu hỏi.**
   - Hỏi "Ai?" → chỉ tên. Hỏi "Bao nhiêu?" → chỉ số. Hỏi "Có/Không?" → 1 từ.
   - Đây là đáp án cuối cùng BẠN đề xuất cho câu hỏi, phải khớp với bằng chứng trong bảng.
   - Nếu `stance="dissent"` vì phản bác đáp án sai của đồng nghiệp, KHÔNG chép lại đáp án sai đó trong `updated_answer`; hãy giữ/đưa đáp án đúng được bảng hỗ trợ, hoặc `"Null"` nếu bảng không đủ dữ liệu.
3. **stance_justification**: 1-2 câu giải thích lý do chọn stance.
   - Lý do phải nhất quán với `updated_answer`; không được nói một đáp án sai rồi trả về chính đáp án sai đó.

## **NHẮC LẠI - CÂU HỎI CẦN TRẢ LỜI:** {question}

## OUTPUT FORMAT (JSON)
```json
{{
    "stance": "maintain | revise | support | dissent",
    "updated_answer": "Câu trả lời ngắn gọn, sát câu hỏi",
    "answerable": true,
    "confidence": 0.80,
    "stance_justification": "1-2 câu giải thích..."
}}
```

Trả lời bằng JSON hợp lệ:"""


# ============================================================================
# SYNTHESIZE - Tổng hợp quyết định cuối (chỉ cho Synthesizer)
# ============================================================================

SYNTHESIZE_PROMPT = """Bạn là Synthesizer - nhà tổng hợp và quyết định cho hội đồng.

## NHIỆM VỤ
Tổng hợp ý kiến và đưa ra QUYẾT ĐỊNH CUỐI CÙNG.
**Trả lời NGẮN GỌN. Không ép buộc đồng thuận 100%.**

## INPUT
### Câu hỏi gốc:
{question}

### Bảng dữ liệu:
{table}

### Tất cả ý kiến từ hội đồng:
{all_opinions}

### Lịch sử thảo luận:
{discussion_history}

## YÊU CẦU

1. **Bảng điểm**: Chấm mỗi persona theo grounding, logic, math, structural (0-1).
2. **Phân tích đồng thuận**: Đếm vote, ghi nhận bất đồng.
3. **final_answer**: **PHẢI ngắn gọn, trực tiếp, chỉ chứa thông tin cốt lõi trả lời câu hỏi.**
   - Hỏi "Ai?" → chỉ tên. Hỏi "Bao nhiêu?" → chỉ số. Hỏi "Có/Không?" → 1 từ.
   - Hỏi "Liệt kê" → liệt kê ngắn, cách nhau bởi dấu phẩy.
   - Nếu unanswerable → "Null"
4. **why_this_answer** và **why_not_others**: Mỗi cái 1-2 câu.

## OUTPUT FORMAT (JSON)
```json
{{
    "scoreboard": [
        {{
            "persona": "Logician",
            "answer": "...",
            "answerable": true,
            "scores": {{"grounding": 0.9, "logic": 0.95, "math": 0.8, "structural": 0.85}},
            "total": 0.875,
            "key_contribution": "1 câu..."
        }}
    ],
    "vote_summary": {{
        "answer_counts": {{"Answer A": 4}},
        "answerable_votes": {{"true": 5, "false": 1}},
        "consensus_level": 0.67
    }},
    "dissenting_opinions": [
        {{
            "persona": "Calculator",
            "dissent_answer": "...",
            "reason": "1 câu...",
            "weight": "minor | significant"
        }}
    ],
    "consensus_reached": true,
    "final_answer": "Câu trả lời ngắn gọn, sát câu hỏi",
    "answerable": true,
    "confidence": 0.88,
    "why_this_answer": "1-2 câu...",
    "why_not_others": "1-2 câu...",
    "acknowledged_uncertainties": ["..."],
    "final_rationale": "Tóm tắt ngắn gọn..."
}}
```

Trả lời bằng JSON hợp lệ:"""


# ============================================================================
# FORMAT_ANSWER (F) - Format đáp án cuối cùng cho khớp groundtruth style
# ============================================================================

FORMAT_ANSWER_PROMPT = """Bạn là Answer Normalizer — chuẩn hóa câu trả lời thô thành dạng ngắn gọn, khớp Exact Match với đáp án mẫu.
Trả về DANH SÁCH các phương án có thể (phần tử đầu = dạng chuẩn nhất, các phần tử sau = biến thể hợp lệ).

## QUY TẮC BẮT BUỘC
- KHÔNG thay đổi NỘI DUNG / Ý NGHĨA của Pred Answer. CHỈ chuẩn hóa ĐỊNH DẠNG.
- KHÔNG bịa đặt thông tin mới. KHÔNG suy diễn vượt quá Pred Answer.
- LUÔN nhìn vào NỘI DUNG của Pred Answer TRƯỚC. Question Type chỉ là gợi ý, có thể sai.

## TASK INPUT
**Question:** {question}
**Pred Answer:** {pred_answer}
**Question Type (gợi ý):** {question_type}

## CÁC QUY TẮC (theo thứ tự ưu tiên)

### R0. NGUYÊN TẮC TỔNG QUÁT
- Nếu Pred Answer chứa tên thực thể / con số / danh sách cụ thể → TRÍCH XUẤT thực thể đó.
- CHỈ áp dụng quy tắc Yes/No khi Pred Answer THỰC SỰ chỉ là một từ Có/Không/Đúng/Sai/Phải.
- Nếu Pred Answer rỗng hoặc nói "không có thông tin" / "không thể trả lời" → trả về ["Null"].
- Phần tử ĐẦU TIÊN trong list = dạng chuẩn nhất; các phần tử SAU = biến thể hợp lệ khác.

### R1. SO SÁNH ("A hay B ...?")
- Câu hỏi dạng "A hay B có/là/...?" → Trích xuất CHỈ tên thực thể được chọn trong Pred Answer. Viết Hoa tên riêng.
- VD: Q: "Bắc Ninh hay Từ Sơn có ít hơn?" | Pred: "từ sơn có ít hơn" → ["Từ Sơn"]

### R2. YES/NO
- CHỈ áp dụng khi Pred Answer THỰC SỰ trả lời Có/Không/Đúng/Sai (1 từ hoặc kèm giải thích ngắn).
- Nếu Pred Answer phủ định → trả ["Không","Sai","Không Phải"].
- Nếu Pred Answer khẳng định → trả ["Có", "Đúng", "Phải"].

### R3. CẮT TIỀN TỐ / TỪ LOẠI
- Loại bỏ từ chỉ loại lặp lại giữa Question và Pred Answer.
- "Câu lạc bộ nào?" + "Câu lạc bộ X" → "X". "Thế kỷ thứ mấy?" + "Thế kỷ thứ 6" → "6".
- Giữ nguyên nếu tên riêng gắn liền (VD: "Bitexco Financial Tower").

### R4. VIẾT HOA TÊN RIÊNG
- Tên người, địa danh, tổ chức → Viết Hoa Chữ Cái Đầu mỗi từ (Title Case).

### R5. SỐ LIỆU, ĐƠN VỊ VÀ TIỀN TỐ THỜI GIAN
- Khi Pred Answer có SỐ kèm ĐƠN VỊ → tạo các biến thể: chỉ số, số+đơn vị (viết liền), số + đơn vị (cách dấu cách).
- Khi Pred Answer có tiền tố thời gian ("Năm", "Ngày", "Tháng") → tạo biến thể CÓ và KHÔNG CÓ tiền tố.
- Giữ nguyên định dạng số gốc. "Thứ mấy/đứng thứ mấy" → Chỉ lấy số thứ tự.

### R6. DANH SÁCH VÀ NULL
- Nhiều đáp án → nối bằng ", " (bỏ "và", "hoặc"). Mỗi phần tử Title Case nếu là tên riêng. Toàn bộ chuỗi đã nối là MỘT phương án trong list.
- Không có thông tin → ["Null"] (chữ N hoa). Số 0 → ["0"] (không phải Null).

## VÍ DỤ

| Question | Pred Answer | formatted_answer |
|----------|------------|------------------|
| Câu lạc bộ nào...? | câu lạc bộ công an hà nội | ["Công An Hà Nội"] |
| Bắc Ninh hay Từ Sơn ít hơn? | từ sơn có ít hơn | ["Từ Sơn"] |
| ...đặt tại Quận 1 phải không? | không, đặt tại quận 10 | ["Không", "Sai", "Không phải"] |
| ...đúng không? | đúng. | ["Đúng", "Có", "Phải"] |
| Kishau Dam cao bao nhiêu? | Nó cao 236 m | ["236", "236m", "236 m"] |
| Xây dựng năm nào? | năm 2005 | ["2005", "Năm 2005"] |
| Sinh ngày nào? | ngày 15 tháng 3 | ["15 tháng 3", "Ngày 15 tháng 3"] |
| Nặng bao nhiêu? | 50 kg | ["50", "50kg", "50 kg"] |
| Tòa nhà nào hạng 1? | buenos aires forum | ["Buenos Aires Forum"] |
| Ai là Đại tá? | phùng văn khầu, chu văn mùi | ["Phùng Văn Khầu, Chu Văn Mùi"] |
| Tại sao X không nặng nhất? | Vì Y nặng nhất đội | ["Vì Y nặng nhất đội"] |
| Liệt kê các tuyến...? | không có tuyến nào | ["Null"] |

## **NHẮC LẠI - CẦN XỬ LÝ:**
**Question:** {question}
**Pred Answer:** {pred_answer}

Trả về JSON:
```json
{{
  "formatted_answer": ["<phương án chuẩn nhất>", "<biến thể 2>", "..."]
}}
```"""


# ============================================================================
# Helper functions
# ============================================================================

def format_hints(hints: list) -> str:
    """Format hints list for prompt."""
    if not hints:
        return "Không có gợi ý"
    return "\n".join([f"- {h}" for h in hints])


def format_evidence(evidence: list) -> str:
    """Format evidence list for prompt."""
    if not evidence:
        return "Không có bằng chứng cụ thể"
    
    lines = []
    for e in evidence:
        if isinstance(e, dict):
            lines.append(f"- Row {e.get('row')}, Col {e.get('col')}: '{e.get('value')}' ({e.get('role', '')})")
        else:
            lines.append(f"- {e}")
    
    return "\n".join(lines)


def format_solution_plan(plan: list) -> str:
    """Format solution plan for prompt."""
    if not plan:
        return "Không có kế hoạch"
    return "\n".join(plan)
