from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


NO_INFO_KO = "문서에 명시되지 않음"
NO_DOCS_KO = "관련 문서를 찾지 못했습니다. 조건을 조금 완화해서 다시 질문해 주세요."
LLM_DISABLED_KO = "LLM 비활성화 상태입니다. 후보 문서 목록:\n"
NO_INFO_EN = "Not specified in the documents."
NO_DOCS_EN = "No relevant documents were found. Please relax the conditions and try again."


def _llm_complete(llm: Any, prompt: str, system_prompt: Optional[str] = None) -> str:
    if llm is None:
        return ""
    if hasattr(llm, "complete"):
        return str(llm.complete(prompt, system_prompt=system_prompt, max_tokens=1600)).strip()
    if hasattr(llm, "generate"):
        return str(llm.generate(prompt, system_prompt or "", max_tokens=1600)).strip()
    return ""


def _detect_answer_language(question: str) -> str:
    return "ko"


def _history_block(history: List[Dict[str, Any]], limit: int = 4) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for item in history[-limit:]:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role and content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


def _nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_url(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    if text.startswith("//"):
        return f"https:{text}"
    if text.startswith("www."):
        return f"https://{text}"
    if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", text):
        return f"https://{text}"
    return ""


def _best_link(md: Dict[str, Any]) -> str:
    for candidate in (
        md.get("apply_url"),
        md.get("biz_aply_url"),
        md.get("guide_url"),
        md.get("biz_gdnc_url"),
        md.get("detail_url"),
        md.get("detl_pg_url"),
        md.get("lctr_pg_url"),
        md.get("hmpg"),
    ):
        normalized = _normalize_url(candidate)
        if normalized:
            return normalized
    return ""


def _apply_period(md: Dict[str, Any]) -> str:
    start = _nonempty(md.get("apply_start"), md.get("pbanc_rcpt_bgng_dt"))
    end = _nonempty(md.get("apply_end"), md.get("pbanc_rcpt_end_dt"), md.get("deadline"))
    if start and end:
        return f"{start} ~ {end}"
    return _nonempty(md.get("apply_period"), start, end)


def _relevance_score(doc: Dict[str, Any]) -> str:
    for key in ("cross_encoder_score", "rrf_score", "combined_score", "score"):
        value = doc.get(key)
        if value is not None:
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return str(value)
    return ""


def _fact_pairs_for_doc(doc: Dict[str, Any]) -> List[tuple[str, str]]:
    md = doc.get("metadata", {}) or {}
    doc_type = str(md.get("type") or "").strip().lower()
    title = str(doc.get("title") or "").strip()
    summary = str(doc.get("text") or "").strip()
    detail_url = _best_link(md)

    pairs: List[tuple[str, str]] = [("문서 유형", doc_type or NO_INFO_KO)]
    if title:
        pairs.append(("제목", title))
    score = _relevance_score(doc)
    if score:
        pairs.append(("연관도 점수", score))

    if doc_type in {"announcement", "business"}:
        pairs.extend(
            [
                ("사업명", _nonempty(title, md.get("biz_pbanc_nm"), md.get("supt_biz_titl_nm"))),
                (
                    "지원 대상",
                    _nonempty(
                        md.get("apply_target"),
                        md.get("aply_trgt"),
                        md.get("apply_target_desc"),
                        md.get("aply_trgt_ctnt"),
                        md.get("biz_supt_trgt_info"),
                    ),
                ),
                ("지원 내용", _nonempty(md.get("pbanc_ctnt"), md.get("biz_supt_ctnt"), summary[:500])),
                ("분야", _nonempty(md.get("field"), md.get("supt_biz_clsfc"), md.get("biz_category_cd"))),
                ("지역", _nonempty(md.get("region"), md.get("supt_regin"))),
                ("연령", _nonempty(md.get("age_limit"), md.get("biz_trgt_age"))),
                ("창업기간", _nonempty(md.get("startup_period"), md.get("biz_enyy"))),
                ("접수기간", _apply_period(md)),
                ("마감일", _nonempty(md.get("deadline"), md.get("pbanc_rcpt_end_dt"))),
                ("상태", _nonempty(md.get("status"), md.get("rcrt_prgs_yn"))),
                ("주관기관", _nonempty(md.get("host_org"), md.get("pbanc_ntrp_nm"))),
                ("운영기관", _nonempty(md.get("supervisor_org"), md.get("sprv_inst"), md.get("biz_prch_dprt_nm"))),
                ("안내 링크", _nonempty(md.get("guide_url"), md.get("biz_gdnc_url"))),
                ("신청 링크", detail_url),
            ]
        )
    elif doc_type == "lecture":
        pairs.extend(
            [
                ("강좌명", _nonempty(title, md.get("lctr_nm"))),
                ("강좌 설명", _nonempty(md.get("lctr_istc"), summary[:500])),
                ("키워드", _nonempty(md.get("kywrd"))),
                ("재생시간", _nonempty(md.get("play_time"))),
                ("등록일", _nonempty(md.get("reg_dt"))),
                ("수정일", _nonempty(md.get("mdfcn_dt"))),
                ("추천 여부", _nonempty(md.get("rcmd_yn"))),
                ("패키지 여부", _nonempty(md.get("pckg_use_yn"))),
                ("상세", detail_url),
            ]
        )
    elif doc_type in {"space", "center"}:
        pairs.extend(
            [
                ("이름", _nonempty(title, md.get("spce_nm"), md.get("cntr_nm"))),
                ("위치", _nonempty(md.get("address"), md.get("addr"), md.get("region"), md.get("regin_clss"))),
                ("우편번호", _nonempty(md.get("pstno"))),
                ("센터 유형", _nonempty(md.get("cntr_type_nm"), md.get("cntr_intrd_type_nm"))),
                ("공간 유형", _nonempty(md.get("spce_type_nm"), md.get("seat_type_nm"))),
                (
                    "이용 정보",
                    _nonempty(
                        md.get("rsvt_psbl_clss"),
                        md.get("seat_clss"),
                        md.get("seat_type_nm"),
                        md.get("rent"),
                        md.get("guam"),
                    ),
                ),
                ("좌석 수", _nonempty(md.get("seat_co"), md.get("spce_cnt"))),
                ("임대료", _nonempty(md.get("rent"))),
                ("보증금", _nonempty(md.get("guam"))),
                ("예약 가능 여부", _nonempty(md.get("rsvt_psbl_clss"))),
                ("홈페이지", detail_url),
            ]
        )
    elif doc_type == "institution":
        pairs.extend(
            [
                ("기관명", _nonempty(title, md.get("inst_nm"))),
                ("영문명", _nonempty(md.get("inst_eng_nm"))),
                (
                    "기관 분류",
                    _nonempty(
                        md.get("inst_clsfc_clss_cd"),
                        md.get("inst_chr_clss_cd"),
                        md.get("inst_chrr_clss_cd"),
                        md.get("inds_clsf_clss_cd"),
                        md.get("inds_clsfc_clss_cd"),
                    ),
                ),
                ("지역", _nonempty(md.get("region"), md.get("regin_clss_cd"))),
                ("설립일", _nonempty(md.get("fndn_dt"))),
                ("설립 구분", _nonempty(md.get("fndn_clss_cd"))),
                ("국가 구분", _nonempty(md.get("natn_clss_cd"))),
                ("상세", detail_url),
            ]
        )
    elif doc_type in {"product", "corporate"}:
        pairs.extend(
            [
                ("이름", _nonempty(title, md.get("manu_nm"), md.get("ntrp_nm"))),
                ("기업 유형", _nonempty(md.get("ntrp_type_nm"))),
                ("발급일", _nonempty(md.get("confmdoc_isu_dt"))),
                ("유효기간", _nonempty(md.get("confmdoc_expr_dt"))),
                ("제품 설명", _nonempty(md.get("manu_intrd"))),
                ("제품 규격", _nonempty(md.get("manu_stnds"))),
                ("가격", _nonempty(md.get("manu_prc"))),
                ("상세", detail_url),
            ]
        )
    else:
        pairs.extend(
            [
                ("문서 제목", _nonempty(title, md.get("titl_nm"), md.get("file_nm"))),
                ("핵심 설명", _nonempty(summary[:500], md.get("cntn"), md.get("content"))),
                ("등록일", _nonempty(md.get("fstm_reg_dt"), md.get("reg_dt"))),
                ("수정일", _nonempty(md.get("last_mdfcn_dt"), md.get("mdfcn_dt"))),
                ("상세", detail_url),
            ]
        )

    cleaned: List[tuple[str, str]] = []
    for key, value in pairs:
        text = str(value or "").strip()
        if text:
            cleaned.append((key, text))
    return cleaned


def _facts_block(docs: List[Dict[str, Any]], limit: int = 8) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(docs[:limit], start=1):
        pairs = _fact_pairs_for_doc(doc)
        body = "\n".join(f"- {key}: {value}" for key, value in pairs)
        blocks.append(f"[{i}]\n{body}")
    return "\n\n".join(blocks)


def _citation_guide_block(docs: List[Dict[str, Any]], limit: int = 8) -> str:
    lines: List[str] = []
    for i, doc in enumerate(docs[:limit], start=1):
        title = str(doc.get("title") or "").strip()
        if title:
            lines.append(f"[{i}] {title}")
    return "\n".join(lines)


def _postprocess_answer(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    text = text.replace("<|im_end|>", "").strip()
    text = re.sub(r"\[(https?://[^\]\s]+)\]\(\1\)", r"\1", text)
    text = re.sub(r"(https?://[^\s\]]+)\]\(\1\)", r"\1", text)
    return text.strip()


def _contains_hangul(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7a3]", text or ""))


def _translate_answer_to_english(
    llm: Any,
    answer: str,
    question: str,
    docs: List[Dict[str, Any]],
) -> str:
    if not answer.strip():
        return ""
    prompt = f"""
User question:
{question}

Current answer:
{answer}

Structured evidence:
{_facts_block(docs)}

Instruction:
1. Rewrite the current answer entirely in English.
2. Preserve the original meaning and supported facts only.
3. Do not add new facts.
4. Keep inline citations like [1], [2].
5. Output plain English only.
""".strip()
    translated = _llm_complete(
        llm,
        prompt,
        "Faithful translation only. Output English only. Preserve citations.",
    )
    return _postprocess_answer(translated)


def _normalized_for_match(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _answer_references_docs(answer: str, docs: List[Dict[str, Any]]) -> bool:
    normalized_answer = _normalized_for_match(_postprocess_answer(answer))
    if not normalized_answer:
        return False

    for doc in docs[:5]:
        title = str(doc.get("title") or "").strip()
        if not title:
            continue
        compact_title = _normalized_for_match(title)
        if len(compact_title) >= 4 and compact_title in normalized_answer:
            return True
    return False


def _is_low_information_answer(answer: str) -> bool:
    text = _postprocess_answer(answer)
    if not text:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) <= 2 and len(text) < 180


def _is_direct_no_answer(answer: str, answer_lang: str) -> bool:
    text = _postprocess_answer(answer)
    if not text:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    first = lines[0]
    if answer_lang == "ko":
        return first in {
            NO_INFO_KO,
            f"{NO_INFO_KO}.",
            "질문에 대한 직접적인 답은 문서에 명시되지 않았습니다.",
        }
    lowered = first.lower()
    return lowered in {
        "not specified in the documents",
        "not specified in the documents.",
    }


def _value_from_pairs(pairs: List[tuple[str, str]], key: str) -> str:
    for k, v in pairs:
        if k == key and str(v or "").strip():
            return str(v).strip()
    return ""


def _fallback_candidate_block(docs: List[Dict[str, Any]], answer_lang: str) -> str:
    if not docs:
        return ""

    lines: List[str] = []
    header = (
        "질문에 대한 직접적인 답은 명시되지 않았습니다. 다만 관련 정보는 다음과 같습니다:"
        if answer_lang == "ko"
        else "The data does not state the exact answer directly, but the following related information is available:"
    )
    lines.append(header)

    for idx, doc in enumerate(docs[:5], start=1):
        pairs = _fact_pairs_for_doc(doc)
        title = _value_from_pairs(pairs, "사업명") or _value_from_pairs(pairs, "강좌명") or _value_from_pairs(pairs, "이름") or _value_from_pairs(pairs, "기관명") or _value_from_pairs(pairs, "제목") or _value_from_pairs(pairs, "문서 제목")
        location = _value_from_pairs(pairs, "위치") or _value_from_pairs(pairs, "지역")
        detail = _value_from_pairs(pairs, "신청 링크") or _value_from_pairs(pairs, "홈페이지") or _value_from_pairs(pairs, "상세") or _value_from_pairs(pairs, "안내 링크")

        item = f"{idx}. {title or ('제목 없음' if answer_lang == 'ko' else 'Untitled')}"
        extras: List[str] = []
        if location:
            extras.append(location)
        if detail:
            extras.append(detail)
        if extras:
            item += " | " + " | ".join(extras)
        lines.append(item)

    return "\n".join(lines)


def _deadline_advisory_block(docs: List[Dict[str, Any]], answer_lang: str) -> str:
    lines: List[str] = []
    policy_docs = []
    for doc in docs:
        md = doc.get("metadata", {}) or {}
        if str(md.get("type") or "").strip().lower() in {"announcement", "business"}:
            policy_docs.append(doc)
    if not policy_docs:
        return ""

    for doc in policy_docs[:3]:
        pairs = _fact_pairs_for_doc(doc)
        title = (
            _value_from_pairs(pairs, "사업명")
            or _value_from_pairs(pairs, "제목")
            or _value_from_pairs(pairs, "이름")
        )
        status = _value_from_pairs(pairs, "상태")
        period = _value_from_pairs(pairs, "접수기간")
        deadline = _value_from_pairs(pairs, "마감일")
        parts: List[str] = []
        if status and status != NO_INFO_KO:
            parts.append(f"상태: {status}" if answer_lang == "ko" else f"Status: {status}")
        if period and period != NO_INFO_KO:
            parts.append(f"접수기간: {period}" if answer_lang == "ko" else f"Application period: {period}")
        elif deadline and deadline != NO_INFO_KO:
            parts.append(f"마감일: {deadline}" if answer_lang == "ko" else f"Deadline: {deadline}")
        if parts:
            if answer_lang == "ko":
                lines.append(f"- {title or '관련 공고'} | " + " | ".join(parts))
            else:
                lines.append(f"- {title or 'Related program'} | " + " | ".join(parts))

    if not lines:
        return ""

    header = (
        "참고: 아래 접수/마감 정보는 문서에 표시된 상태를 바탕으로 정리했습니다."
        if answer_lang == "ko"
        else "Note: The following application/deadline information is summarized from the document metadata."
    )
    return header + "\n" + "\n".join(lines)


def _needs_related_info_append(answer: str, docs: List[Dict[str, Any]], answer_lang: str) -> bool:
    if not docs:
        return False
    if _is_direct_no_answer(answer, answer_lang):
        return True
    if _is_low_information_answer(answer) and not _answer_references_docs(answer, docs):
        return True
    return False


def generate_node(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    question = str(state.get("question", ""))
    user_question = str(state.get("user_question", "")).strip()
    chat_history = list(state.get("chat_history") or [])
    docs = list(
        state.get("final_docs")
        or state.get("filtered_docs")
        or state.get("reranked_docs")
        or state.get("retrieved_docs")
        or []
    )
    trace = list(state.get("reasoning_trace", []))
    answer_lang = _detect_answer_language(user_question or question)
    print(
        f"[generate] answer_lang={answer_lang} "
        f"user_question={user_question[:120]!r} "
        f"question={question[:120]!r}"
    )

    no_info_text = NO_INFO_KO if answer_lang == "ko" else NO_INFO_EN

    if not docs:
        out = dict(state)
        out["answer"] = NO_DOCS_KO if answer_lang == "ko" else NO_DOCS_EN
        trace.append("generate: no docs")
        out["reasoning_trace"] = trace
        return out

    language_instruction = (
        "Answer entirely in Korean because the user's question is in Korean."
        if answer_lang == "ko"
        else (
            "Answer entirely in English because the user's question is in English. "
            "Do not answer in Korean. "
            "Even if the evidence is written in Korean, translate the answer into natural English."
        )
    )

    prompt = f"""
Question:
{question}

Conversation history:
{_history_block(chat_history)}

Reasoning trace:
{chr(10).join(trace[-12:])}

Structured evidence:
{_facts_block(docs)}

Citation map:
{_citation_guide_block(docs)}

Instruction:
1. Answer the user's current question using the already-prepared conversation context above.
2. Use only the structured evidence above. Do not invent missing facts.
3. If information is missing, explicitly say "{no_info_text}".
4. Treat deadline, open or closed status, and eligibility as preprocessed pipeline outputs. Do not reinterpret or guess them beyond the evidence.
5. Match the answer style to the question:
   - recommendation questions: recommend the most relevant items first
   - eligibility or status questions: focus on the exact condition being asked
   - comparison questions: organize by similarities and differences
   - explanation questions: explain concisely without forcing a recommendation list
6. When links exist, keep them concise and natural.
7. Output links as plain URLs, not markdown links.
8. Show all relevant items supported by the structured evidence, ordered by relevance score from highest to lowest.
9. If there are many relevant items, keep each item concise rather than dropping items silently.
10. Do not output internal reasoning or mention the reasoning trace.
11. After every factual claim, add one or more inline citations in [X] format using the citation map above.
12. Do not cite documents that do not support the claim.
13. The answer language must exactly match the user's question language.
{language_instruction}
""".strip()

    answer = _llm_complete(
        llm,
        prompt,
        (
            "You are a grounded startup-support assistant. "
            "Use only the provided structured evidence and conversation context. "
            "Do not hallucinate facts, deadlines, links, eligibility, institutions, or statuses. "
            "Always answer in the same language as the user's question. "
            + (
                "Use Korean only."
                if answer_lang == "ko"
                else "Use English only. Do not switch to Korean. Translate Korean evidence into English when answering."
            )
        ),
    )
    answer = _postprocess_answer(answer)
    if answer_lang == "en" and _contains_hangul(answer):
        translated = _translate_answer_to_english(llm, answer, user_question or question, docs)
        if translated:
            answer = translated
            trace.append("generate: english translation fallback applied")
    if answer and _needs_related_info_append(answer, docs, answer_lang):
        candidate_block = _fallback_candidate_block(docs, answer_lang)
        if candidate_block:
            if _is_direct_no_answer(answer, answer_lang):
                answer = candidate_block
            else:
                answer = f"{answer}\n\n{candidate_block}".strip()
    if not answer:
        llm_err = ""
        if llm is not None:
            last_error = getattr(llm, "last_error", None)
            init_error = getattr(llm, "init_error", None)
            if last_error:
                llm_err = f" last_error={last_error}"
            elif init_error:
                llm_err = f" init_error={init_error}"

        head = (
            LLM_DISABLED_KO
            if answer_lang == "ko"
            else "LLM is unavailable. Candidate documents:\n"
        )
        untitled = "제목없음" if answer_lang == "ko" else "Untitled"
        answer = head + "\n".join([f"- {d.get('title', untitled)}" for d in docs[:5]])
        trace.append(f"generate: llm empty output.{llm_err}")

    advisory = _deadline_advisory_block(docs, answer_lang)
    if advisory and advisory not in answer:
        answer = f"{answer}\n\n{advisory}".strip()

    trace.append(f"generate: answer generated from {min(len(docs), 8)} docs")
    out = dict(state)
    out["answer"] = answer
    out["reasoning_trace"] = trace
    out["structured_facts"] = [_fact_pairs_for_doc(doc) for doc in docs[:8]]
    out["citation_map"] = [
        {"index": i, "title": str(doc.get("title") or "").strip()}
        for i, doc in enumerate(docs[:8], start=1)
    ]
    return out
