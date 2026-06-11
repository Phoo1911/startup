"""
RAG 인덱스 구축 에이전트
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from legacy_agents.base import AgenticAgent
from models.data import Document
from utils.text import clean_text, safe_get


class RAGBuilderAgent(AgenticAgent):
    """RAG 벡터 인덱스 구축"""

    def __init__(self, rag_system: object, llm_client: Optional[object] = None):
        super().__init__("RAGBuilder", llm_client)
        self.rag = rag_system

    # ---------------- 날짜 처리 유틸 ----------------
    @staticmethod
    def _clean_deadline(raw: str) -> str:
        """
        날짜 문자열 정리:
        - '20250101'   -> '2025-01-01'
        - '2025-01-01' -> '2025-01-01'
        - '2025. 11. 26. 18:00까지' 등은 숫자만 뽑아서 앞 8자리 사용
        """
        raw = clean_text(raw or "")
        if not raw:
            return ""

        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 8:
            y = digits[0:4]
            m = digits[4:6]
            d = digits[6:8]
            return f"{y}-{m}-{d}"

        return raw.strip()

    @staticmethod
    def _extract_deadline_from_content(pbanc_ctnt: str) -> str:
        """
        공고 본문에서 날짜 패턴으로 마감일 추정:
        - '2025. 11. 5. ~ 2025. 11. 26. 17:00까지'
        - '2025-11-05 ~ 2025-11-26'
        같은 문장에서 '마지막 날짜'를 뽑아서 YYYY-MM-DD로 리턴
        """
        text = clean_text(pbanc_ctnt or "")
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)

        pattern = r"(20\d{2})[.\-/년\s]*([01]?\d)[.\-/월\s]*([0-3]?\d)"
        matches = re.findall(pattern, text)

        if not matches:
            return ""

        year, month, day = matches[-1]
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    @staticmethod
    def _build_source_doc_id(category: str, raw_id: object, item_idx: int) -> str:
        raw_text = str(raw_id).strip() if raw_id not in (None, "") else str(item_idx)
        prefix = f"{category}_"
        if raw_text.startswith(prefix):
            return raw_text
        return f"{category}_{raw_text}"

    @staticmethod
    def _parse_iso_date(raw: str):
        text = clean_text(raw or "")
        if not text:
            return None
        digits = re.sub(r"\D", "", text)
        if len(digits) < 8:
            return None
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date()
        except ValueError:
            return None

    @staticmethod
    def _parse_year(raw: str):
        text = clean_text(raw or "")
        if not text:
            return None
        digits = re.sub(r"\D", "", text)
        if len(digits) < 4:
            return None
        year = int(digits[:4])
        if 2000 <= year <= 2100:
            return year
        return None

    # ---------------- 인덱스 구축 ----------------
    def build_index(self, raw_data: Dict[str, List[Dict]]) -> int:
        """RAG 인덱스 구축"""
        self.think("RAG 벡터 인덱스 구축", action="문서 임베딩", confidence=0.95)

        documents: List[Document] = []
        doc_id = 0
        today = datetime.now().date()
        cutoff = today - timedelta(days=90)

        # 1) 통합공고 지원사업 정보 (business)
        for item_idx, item in enumerate(raw_data.get("business", [])):
            source_doc_id = self._build_source_doc_id("business", item.get("id"), item_idx)
            title = clean_text(safe_get(item, "supt_biz_titl_nm", default=""))
            target = clean_text(safe_get(item, "biz_supt_trgt_info", default=""))
            budget = clean_text(safe_get(item, "biz_supt_bdgt_info", default=""))
            content = clean_text(safe_get(item, "biz_supt_ctnt", default=""))
            chrct = clean_text(safe_get(item, "supt_biz_chrct", default=""))
            intro = clean_text(safe_get(item, "supt_biz_intrd_info", default=""))
            category = clean_text(safe_get(item, "biz_category_cd", default=""))
            year = clean_text(safe_get(item, "biz_yr", default=""))
            raw_detail_url = clean_text(safe_get(item, "detl_pg_url", default=""))

            biz_year = self._parse_year(year)
            if biz_year is not None and biz_year < today.year:
                continue

            text_parts = [
                title,
                f"지원 대상: {target}" if target else "",
                f"지원 예산 및 규모: {budget}" if budget else "",
                f"지원 내용: {content}" if content else "",
                f"사업 특성: {chrct}" if chrct else "",
                f"사업 소개: {intro}" if intro else "",
                f"사업 구분 코드: {category}" if category else "",
                f"사업 연도: {year}" if year else "",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"BIZ_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "business",
                        "doc_id": source_doc_id,
                        "policy_id": source_doc_id,
                        "title": title,
                        "supt_biz_titl_nm": title,
                        "category_cd": category,
                        "biz_category_cd": category,
                        "year": year,
                        "biz_yr": year,
                        "target": target,
                        "biz_supt_trgt_info": target,
                        "apply_target": target,
                        "apply_target_desc": target,
                        "budget": budget,
                        "biz_supt_bdgt_info": budget,
                        "content": content,
                        "biz_supt_ctnt": content,
                        "character": chrct,
                        "supt_biz_chrct": chrct,
                        "intro": intro,
                        "supt_biz_intrd_info": intro,
                        "deadline": "",
                        "detail_url": raw_detail_url,
                        "detl_pg_url": raw_detail_url,
                        "guide_url": raw_detail_url,
                        "apply_url": "",
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 2) 지원사업 공고 정보 (announcements)
        for item_idx, item in enumerate(raw_data.get("announcements", [])):
            source_doc_id = self._build_source_doc_id("announcements", item.get("id"), item_idx)
            rcrt_prgs_yn = clean_text(safe_get(item, "rcrt_prgs_yn", default=""))
            aply_trgt = clean_text(safe_get(item, "aply_trgt", default=""))
            biz_enyy = clean_text(safe_get(item, "biz_enyy", default=""))
            biz_trgt_age = clean_text(safe_get(item, "biz_trgt_age", default=""))
            prfn_matr = clean_text(safe_get(item, "prfn_matr", default=""))
            intg_pbanc_yn = clean_text(safe_get(item, "intg_pbanc_yn", default=""))
            biz_pbanc_nm = clean_text(safe_get(item, "biz_pbanc_nm", default=""))
            intg_pbanc_biz_nm = clean_text(
                safe_get(item, "intg_pbanc_biz_nm", default="")
            )
            pbanc_ctnt = clean_text(safe_get(item, "pbanc_ctnt", default=""))
            supt_biz_clsfc = clean_text(safe_get(item, "supt_biz_clsfc", default=""))
            aply_trgt_ctnt = clean_text(safe_get(item, "aply_trgt_ctnt", default=""))
            supt_regin = clean_text(safe_get(item, "supt_regin", default=""))
            pbanc_rcpt_bgng_dt = clean_text(
                safe_get(item, "pbanc_rcpt_bgng_dt", default="")
            )
            pbanc_rcpt_end_dt = clean_text(
                safe_get(item, "pbanc_rcpt_end_dt", default="")
            )
            pbanc_ntrp_nm = clean_text(safe_get(item, "pbanc_ntrp_nm", default=""))
            sprv_inst = clean_text(safe_get(item, "sprv_inst", default=""))
            biz_prch_dprt_nm = clean_text(
                safe_get(item, "biz_prch_dprt_nm", default="")
            )
            biz_gdnc_url = clean_text(safe_get(item, "biz_gdnc_url", default=""))
            biz_aply_url = clean_text(safe_get(item, "biz_aply_url", default=""))
            prch_cnpl_no = clean_text(safe_get(item, "prch_cnpl_no", default=""))
            detl_pg_url = clean_text(safe_get(item, "detl_pg_url", default=""))
            aply_mthd_vst_rcpt_istc = clean_text(
                safe_get(item, "aply_mthd_vst_rcpt_istc", default="")
            )
            aply_mthd_pssr_rcpt_istc = clean_text(
                safe_get(item, "aply_mthd_pssr_rcpt_istc", default="")
            )
            aply_mthd_fax_rcpt_istc = clean_text(
                safe_get(item, "aply_mthd_fax_rcpt_istc", default="")
            )
            aply_mthd_onli_rcpt_istc = clean_text(
                safe_get(item, "aply_mthd_onli_rcpt_istc", default="")
            )
            aply_mthd_etc_istc = clean_text(
                safe_get(item, "aply_mthd_etc_istc", default="")
            )
            aply_excl_trgt_ctnt = clean_text(
                safe_get(item, "aply_excl_trgt_ctnt", default="")
            )
            pbanc_sn = clean_text(safe_get(item, "pbanc_sn", default=""))

            # 날짜 정리
            start_date_clean = self._clean_deadline(pbanc_rcpt_bgng_dt)
            deadline_from_api = self._clean_deadline(pbanc_rcpt_end_dt)
            deadline_from_text = self._extract_deadline_from_content(pbanc_ctnt)
            final_deadline = deadline_from_api or deadline_from_text
            start_date_obj = self._parse_iso_date(start_date_clean)
            deadline_obj = self._parse_iso_date(final_deadline)

            if rcrt_prgs_yn.strip().upper() in {"N", "마감", "종료", "CLOSED"}:
                continue
            if deadline_obj is not None and deadline_obj < today:
                continue
            anchor_date = deadline_obj or start_date_obj
            if anchor_date is None or anchor_date < cutoff:
                continue

            # 접수기간 문자열
            if start_date_clean and final_deadline:
                apply_period = f"{start_date_clean} ~ {final_deadline}"
            elif final_deadline:
                apply_period = f"~ {final_deadline}"
            elif start_date_clean:
                apply_period = f"{start_date_clean} ~"
            else:
                # 둘 다 없으면 상태에 따라 기본값
                apply_period = "상시 모집" if rcrt_prgs_yn == "Y" else "모집 기간 미정"

            # 관련 링크: detl_pg_url이 없으면 신청/안내 URL로 대체
            primary_url = detl_pg_url or biz_aply_url or biz_gdnc_url

            text_parts = [
                biz_pbanc_nm,
                f"통합 공고 여부: {intg_pbanc_yn}",
                f"통합 공고 사업 명: {intg_pbanc_biz_nm}",
                f"지원 분야: {supt_biz_clsfc}",
                f"지원 지역: {supt_regin}",
                f"신청 대상(요약): {aply_trgt}",
                f"신청 대상(상세): {aply_trgt_ctnt}",
                f"신청 제외 대상: {aply_excl_trgt_ctnt}",
                f"창업 기간 조건: {biz_enyy}",
                f"대상 연령: {biz_trgt_age}",
                f"우대 사항: {prfn_matr}",
                f"공고 내용: {pbanc_ctnt}",
                f"접수 기간: {apply_period}",
                f"모집 진행 여부: {rcrt_prgs_yn}",
                f"모집 주체: {pbanc_ntrp_nm}",
                f"주관 기관: {sprv_inst}",
                f"담당 부서: {biz_prch_dprt_nm}",
                f"신청 방법(방문): {aply_mthd_vst_rcpt_istc}",
                f"신청 방법(우편): {aply_mthd_pssr_rcpt_istc}",
                f"신청 방법(FAX): {aply_mthd_fax_rcpt_istc}",
                f"신청 방법(온라인): {aply_mthd_onli_rcpt_istc}",
                f"신청 방법(기타): {aply_mthd_etc_istc}",
                f"안내 URL: {biz_gdnc_url}",
                f"신청 URL: {biz_aply_url}",
                f"담당 연락처/번호: {prch_cnpl_no}",
                f"공고 일련번호: {pbanc_sn}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"ANN_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "announcement",
                        "doc_id": source_doc_id,
                        "policy_id": pbanc_sn or source_doc_id,
                        "title": biz_pbanc_nm,
                        "biz_pbanc_nm": biz_pbanc_nm,
                        "intg_flag": intg_pbanc_yn,
                        "intg_pbanc_yn": intg_pbanc_yn,
                        "intg_biz_name": intg_pbanc_biz_nm,
                        "intg_pbanc_biz_nm": intg_pbanc_biz_nm,
                        "field": supt_biz_clsfc,
                        "supt_biz_clsfc": supt_biz_clsfc,
                        "region": supt_regin,
                        "supt_regin": supt_regin,
                        "apply_target": aply_trgt,
                        "aply_trgt": aply_trgt,
                        "apply_target_desc": aply_trgt_ctnt,
                        "aply_trgt_ctnt": aply_trgt_ctnt,
                        "exclude_target": aply_excl_trgt_ctnt,
                        "aply_excl_trgt_ctnt": aply_excl_trgt_ctnt,
                        "startup_period": biz_enyy,
                        "biz_enyy": biz_enyy,
                        "age_limit": biz_trgt_age,
                        "biz_trgt_age": biz_trgt_age,
                        "preferential": prfn_matr,
                        "prfn_matr": prfn_matr,
                        "content": pbanc_ctnt,
                        "pbanc_ctnt": pbanc_ctnt,
                        "start_date": start_date_clean,
                        "pbanc_rcpt_bgng_dt": pbanc_rcpt_bgng_dt,
                        "deadline": final_deadline,
                        "pbanc_rcpt_end_dt": pbanc_rcpt_end_dt,
                        "apply_period": apply_period,
                        "status": rcrt_prgs_yn,
                        "rcrt_prgs_yn": rcrt_prgs_yn,
                        "host_org": pbanc_ntrp_nm,
                        "pbanc_ntrp_nm": pbanc_ntrp_nm,
                        "supervisor_org": sprv_inst,
                        "sprv_inst": sprv_inst,
                        "dept_name": biz_prch_dprt_nm,
                        "biz_prch_dprt_nm": biz_prch_dprt_nm,
                        "guide_url": biz_gdnc_url,
                        "biz_gdnc_url": biz_gdnc_url,
                        "apply_url": biz_aply_url,
                        "biz_aply_url": biz_aply_url,
                        "detail_url": primary_url,
                        "detl_pg_url": detl_pg_url,
                        "contact_no": prch_cnpl_no,
                        "prch_cnpl_no": prch_cnpl_no,
                        "apply_method_visit": aply_mthd_vst_rcpt_istc,
                        "aply_mthd_vst_rcpt_istc": aply_mthd_vst_rcpt_istc,
                        "apply_method_post": aply_mthd_pssr_rcpt_istc,
                        "aply_mthd_pssr_rcpt_istc": aply_mthd_pssr_rcpt_istc,
                        "apply_method_fax": aply_mthd_fax_rcpt_istc,
                        "aply_mthd_fax_rcpt_istc": aply_mthd_fax_rcpt_istc,
                        "apply_method_online": aply_mthd_onli_rcpt_istc,
                        "aply_mthd_onli_rcpt_istc": aply_mthd_onli_rcpt_istc,
                        "apply_method_etc": aply_mthd_etc_istc,
                        "aply_mthd_etc_istc": aply_mthd_etc_istc,
                        "pbanc_sn": pbanc_sn,
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 3) 콘텐츠 정보
        for item_idx, item in enumerate(raw_data.get("content", [])):
            source_doc_id = self._build_source_doc_id("content", item.get("id"), item_idx)
            title = clean_text(safe_get(item, "titl_nm", default=""))
            class_code = clean_text(safe_get(item, "clss_cd", default=""))
            file_name = clean_text(safe_get(item, "file_nm", default=""))
            detail_url = clean_text(safe_get(item, "detl_pg_url", default=""))
            reg_date = clean_text(safe_get(item, "fstm_reg_dt", default=""))
            view_cnt = clean_text(str(safe_get(item, "view_cnt", default="")))

            text_parts = [
                title,
                f"콘텐츠 구분 코드: {class_code}",
                f"파일명: {file_name}",
                f"등록일: {reg_date}",
                f"조회수: {view_cnt}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"CNT_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "content",
                        "doc_id": source_doc_id,
                        "policy_id": source_doc_id,
                        "title": title,
                        "titl_nm": title,
                        "class_code": class_code,
                        "clss_cd": class_code,
                        "file_name": file_name,
                        "file_nm": file_name,
                        "reg_date": reg_date,
                        "fstm_reg_dt": reg_date,
                        "view_cnt": view_cnt,
                        "detail_url": detail_url,
                        "detl_pg_url": detail_url,
                        "deadline": "",
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 4) 통계보고서 정보
        for item_idx, item in enumerate(raw_data.get("statistical", [])):
            source_doc_id = self._build_source_doc_id("statistical", item.get("id"), item_idx)
            title = clean_text(safe_get(item, "titl_nm", default=""))
            content = clean_text(safe_get(item, "cntn", default=""))
            file_name = clean_text(safe_get(item, "file_nm", default=""))
            first_reg_dt = clean_text(safe_get(item, "fstm_reg_dt", default=""))
            last_mod_dt = clean_text(safe_get(item, "last_mdfcn_dt", default=""))
            detail_url = clean_text(safe_get(item, "detl_pg_url", default=""))

            text_parts = [
                title,
                f"통계 자료 내용: {content}",
                f"파일명: {file_name}",
                f"최초 등록일: {first_reg_dt}",
                f"최종 수정일: {last_mod_dt}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"STAT_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "statistical",
                        "doc_id": source_doc_id,
                        "policy_id": source_doc_id,
                        "title": title,
                        "titl_nm": title,
                        "content": content,
                        "cntn": content,
                        "file_name": file_name,
                        "file_nm": file_name,
                        "first_reg_dt": first_reg_dt,
                        "fstm_reg_dt": first_reg_dt,
                        "last_mdfcn_dt": last_mod_dt,
                        "last_mdfcn_dt_raw": last_mod_dt,
                        "detail_url": detail_url,
                        "detl_pg_url": detail_url,
                        "deadline": "",
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 5) 창업에듀 강좌 정보
        for item_idx, item in enumerate(raw_data.get("edu_lectures", [])):
            source_doc_id = self._build_source_doc_id("edu_lectures", item.get("id"), item_idx)
            lctr_nm = clean_text(safe_get(item, "lctr_nm", default=""))
            lctr_istc = clean_text(safe_get(item, "lctr_istc", default=""))
            kywrd = clean_text(safe_get(item, "kywrd", default=""))
            l_lclss = clean_text(safe_get(item, "lctr_lclss_cd", default=""))
            l_mclss = clean_text(safe_get(item, "lctr_mclss_cd", default=""))
            l_sclss = clean_text(safe_get(item, "lctr_sclss_cd", default=""))
            sbtitl_yn = clean_text(safe_get(item, "sbtitl_expsr_use_yn", default=""))
            rcmd_yn = clean_text(safe_get(item, "rcmd_yn", default=""))
            pckg_yn = clean_text(safe_get(item, "pckg_use_yn", default=""))
            aliac_yn = clean_text(safe_get(item, "aliac_yn", default=""))
            play_time = safe_get(item, "play_time", default=0)
            view_cnt = safe_get(item, "view_cnt", default=0)
            lctr_url = clean_text(safe_get(item, "lctr_pg_url", default=""))
            reg_dt = clean_text(safe_get(item, "reg_dt", default=""))
            mdfcn_dt = clean_text(safe_get(item, "mdfcn_dt", default=""))

            text_parts = [
                f"강좌명: {lctr_nm}",
                f"강좌 설명: {lctr_istc}",
                f"키워드: {kywrd}",
                f"강좌 대분류 코드: {l_lclss}",
                f"강좌 중분류 코드: {l_mclss}",
                f"강좌 소분류 코드: {l_sclss}",
                f"자막 노출 여부: {sbtitl_yn}",
                f"추천 강좌 여부: {rcmd_yn}",
                f"패키지 강좌 여부: {pckg_yn}",
                f"제휴 강좌 여부: {aliac_yn}",
                f"재생 시간(초): {play_time}",
                f"조회수: {view_cnt}",
                f"등록일: {reg_dt}",
                f"수정일: {mdfcn_dt}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"EDU_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "lecture",
                        "doc_id": source_doc_id,
                        "policy_id": source_doc_id,
                        "title": lctr_nm,
                        "lctr_nm": lctr_nm,
                        "desc": lctr_istc,
                        "lctr_istc": lctr_istc,
                        "keywords": kywrd,
                        "kywrd": kywrd,
                        "lctr_lclss_cd": l_lclss,
                        "lctr_mclss_cd": l_mclss,
                        "lctr_sclss_cd": l_sclss,
                        "sbtitl_expsr_use_yn": sbtitl_yn,
                        "rcmd_yn": rcmd_yn,
                        "pckg_use_yn": pckg_yn,
                        "aliac_yn": aliac_yn,
                        "play_time": play_time,
                        "view_cnt": view_cnt,
                        "reg_dt": reg_dt,
                        "mdfcn_dt": mdfcn_dt,
                        "detail_url": lctr_url,
                        "lctr_pg_url": lctr_url,
                        "deadline": "",
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 6) 창업공간 정보 (spaces)
        for item_idx, item in enumerate(raw_data.get("spaces", [])):
            source_doc_id = self._build_source_doc_id("spaces", item.get("id"), item_idx)
            lgtde = safe_get(item, "lgtde", default=0.0)
            latde = safe_get(item, "latde", default=0.0)
            spce_cnt = safe_get(item, "spce_cnt", default=0)
            spce_id = safe_get(item, "spce_id", default=0)
            spce_nm = clean_text(safe_get(item, "spce_nm", default=""))
            spce_type = clean_text(safe_get(item, "spce_type_nm", default=""))
            seat_type = clean_text(safe_get(item, "seat_type_nm", default=""))
            seat_clss = clean_text(safe_get(item, "seat_clss", default=""))
            cntr_id = safe_get(item, "cntr_id", default=0)
            buld_id = safe_get(item, "buld_id", default=0)
            excuse_ar = clean_text(safe_get(item, "excuse_ar", default=""))
            cmnus_ar = clean_text(safe_get(item, "cmnus_ar", default=""))
            rent = safe_get(item, "rent", default=0)
            guam = safe_get(item, "guam", default=0)
            rsvt_cls = clean_text(safe_get(item, "rsvt_psbl_clss", default=""))
            seat_co = clean_text(safe_get(item, "seat_co", default=""))
            cntr_nm = clean_text(safe_get(item, "cntr_nm", default=""))
            cntr_type = clean_text(safe_get(item, "cntr_type_nm", default=""))
            buld_nm = clean_text(safe_get(item, "buld_nm", default=""))
            pstno = clean_text(safe_get(item, "pstno", default=""))
            addr = clean_text(safe_get(item, "addr", default=""))
            hmpg = clean_text(safe_get(item, "hmpg", default=""))
            cntr_intrd = clean_text(
                safe_get(item, "cntr_intrd_type_nm", default="")
            )

            text_parts = [
                f"공간명: {spce_nm}",
                f"센터명: {cntr_nm}",
                f"공간 유형: {spce_type}",
                f"좌석 유형: {seat_type}",
                f"좌석 구분: {seat_clss}",
                f"좌석 수: {seat_co}",
                f"임대료: {rent}",
                f"보증금: {guam}",
                f"전용 면적: {excuse_ar}",
                f"공용 면적: {cmnus_ar}",
                f"예약 가능 여부: {rsvt_cls}",
                f"주소: {addr} ({pstno})",
                f"건물명: {buld_nm}",
                f"센터 유형: {cntr_type}",
                f"센터 소개 유형: {cntr_intrd}",
                f"홈페이지: {hmpg}",
                f"위도: {latde}, 경도: {lgtde}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"SPC_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "space",
                        "doc_id": source_doc_id,
                        "policy_id": source_doc_id,
                        "title": spce_nm,
                        "space_id": spce_id,
                        "spce_id": spce_id,
                        "spce_nm": spce_nm,
                        "space_count": spce_cnt,
                        "spce_cnt": spce_cnt,
                        "space_type": spce_type,
                        "spce_type_nm": spce_type,
                        "seat_type": seat_type,
                        "seat_type_nm": seat_type,
                        "seat_class": seat_clss,
                        "seat_clss": seat_clss,
                        "seat_count": seat_co,
                        "seat_co": seat_co,
                        "center_id": cntr_id,
                        "cntr_id": cntr_id,
                        "center_name": cntr_nm,
                        "cntr_nm": cntr_nm,
                        "center_type": cntr_type,
                        "cntr_type_nm": cntr_type,
                        "building_id": buld_id,
                        "buld_id": buld_id,
                        "building_name": buld_nm,
                        "buld_nm": buld_nm,
                        "exclusive_area": excuse_ar,
                        "excuse_ar": excuse_ar,
                        "common_area": cmnus_ar,
                        "cmnus_ar": cmnus_ar,
                        "rent": rent,
                        "deposit": guam,
                        "guam": guam,
                        "reservation_class": rsvt_cls,
                        "rsvt_psbl_clss": rsvt_cls,
                        "postcode": pstno,
                        "pstno": pstno,
                        "address": addr,
                        "addr": addr,
                        "homepage": hmpg,
                        "hmpg": hmpg,
                        "center_intro_type": cntr_intrd,
                        "cntr_intrd_type_nm": cntr_intrd,
                        "latitude": latde,
                        "latde": latde,
                        "longitude": lgtde,
                        "lgtde": lgtde,
                        "deadline": "",
                        "detail_url": hmpg,
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 7) 센터 정보 (centers)
        for item_idx, item in enumerate(raw_data.get("centers", [])):
            source_doc_id = self._build_source_doc_id("centers", item.get("id"), item_idx)
            lgtde = safe_get(item, "lgtde", default=0)
            latde = safe_get(item, "latde", default=0)
            spce_cnt = safe_get(item, "spce_cnt", default=0)

            cntr_id = clean_text(safe_get(item, "cntr_id", default=""))
            cntr_nm = clean_text(safe_get(item, "cntr_nm", default=""))
            cntr_type = clean_text(safe_get(item, "cntr_type_nm", default=""))

            buld_id = clean_text(safe_get(item, "buld_id", default=""))
            buld_nm = clean_text(safe_get(item, "buld_nm", default=""))

            pstno = clean_text(safe_get(item, "pstno", default=""))
            regin = clean_text(safe_get(item, "regin_clss", default=""))
            addr = clean_text(safe_get(item, "addr", default=""))
            hmpg = clean_text(safe_get(item, "hmpg", default=""))
            intrd_typ = clean_text(
                safe_get(item, "cntr_intrd_type_nm", default="")
            )

            text_parts = [
                f"센터명: {cntr_nm}",
                f"센터 유형: {cntr_type}",
                f"보유 공간 수: {spce_cnt}",
                f"건물명: {buld_nm}",
                f"주소: {addr}",
                f"우편번호: {pstno}",
                f"지역 구분: {regin}",
                f"홈페이지: {hmpg}",
                f"센터 소개 유형: {intrd_typ}",
                f"위도(lat): {latde}",
                f"경도(lng): {lgtde}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"CNTR_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "center",
                        "doc_id": source_doc_id,
                        "policy_id": source_doc_id,
                        "title": cntr_nm,
                        "center_id": cntr_id,
                        "cntr_id": cntr_id,
                        "cntr_nm": cntr_nm,
                        "center_type": cntr_type,
                        "cntr_type_nm": cntr_type,
                        "space_count": spce_cnt,
                        "spce_cnt": spce_cnt,
                        "building_id": buld_id,
                        "buld_id": buld_id,
                        "building_name": buld_nm,
                        "buld_nm": buld_nm,
                        "postcode": pstno,
                        "pstno": pstno,
                        "region": regin,
                        "regin_clss": regin,
                        "address": addr,
                        "addr": addr,
                        "homepage": hmpg,
                        "hmpg": hmpg,
                        "intro_type": intrd_typ,
                        "cntr_intrd_type_nm": intrd_typ,
                        "latitude": latde,
                        "latde": latde,
                        "longitude": lgtde,
                        "lgtde": lgtde,
                        "detail_url": hmpg,
                        "deadline": "",
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 8) 창업기업 확인서 제품 정보 (products)
        for item_idx, item in enumerate(raw_data.get("products", [])):
            source_doc_id = self._build_source_doc_id("products", item.get("id"), item_idx)
            manu_nm = clean_text(safe_get(item, "manu_nm", default=""))
            manu_intrd = clean_text(safe_get(item, "manu_intrd", default=""))
            manu_stnds = clean_text(safe_get(item, "manu_stnds", default=""))
            manu_prc = clean_text(safe_get(item, "manu_prc", default=""))
            hmpg = clean_text(safe_get(item, "hmpg", default=""))
            ntrp_nm = clean_text(safe_get(item, "ntrp_nm", default=""))
            confmdoc_isu_no = clean_text(
                safe_get(item, "confmdoc_isu_no", default="")
            )
            confmdoc_isu_dt = clean_text(
                safe_get(item, "confmdoc_isu_dt", default="")
            )
            confmdoc_expr_dt = clean_text(
                safe_get(item, "confmdoc_expr_dt", default="")
            )
            brno = clean_text(safe_get(item, "brno", default=""))
            lwdg_nm = clean_text(safe_get(item, "lwdg_nm", default=""))

            manu_category = clean_text(
                safe_get(item, "manu_category", default="")
            )
            manu_lclss = clean_text(safe_get(item, "manu_lclss", default=""))
            manu_mclss = clean_text(safe_get(item, "manu_mclss", default=""))
            manu_sclss = clean_text(safe_get(item, "manu_sclss", default=""))

            text_parts = [
                f"제품명: {manu_nm}",
                f"제품 소개: {manu_intrd}",
                f"제품 규격: {manu_stnds}",
                f"제품 가격: {manu_prc}",
                f"제품 분류: {manu_category}",
                f"제품 대분류: {manu_lclss}",
                f"제품 중분류: {manu_mclss}",
                f"제품 소분류: {manu_sclss}",
                f"기업명: {ntrp_nm}",
                f"지자체명: {lwdg_nm}",
                f"사업자등록번호: {brno}",
                f"확인서 발급 번호: {confmdoc_isu_no}",
                f"확인서 발급 일자: {confmdoc_isu_dt}",
                f"확인서 만료 일자: {confmdoc_expr_dt}",
                f"홈페이지: {hmpg}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"PRD_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "product",
                        "doc_id": source_doc_id,
                        "policy_id": confmdoc_isu_no or source_doc_id,
                        "title": manu_nm,
                        "manu_nm": manu_nm,
                        "company_name": ntrp_nm,
                        "ntrp_nm": ntrp_nm,
                        "price": manu_prc,
                        "manu_prc": manu_prc,
                        "intro": manu_intrd,
                        "manu_intrd": manu_intrd,
                        "spec": manu_stnds,
                        "manu_stnds": manu_stnds,
                        "homepage": hmpg,
                        "hmpg": hmpg,
                        "local_gov": lwdg_nm,
                        "lwdg_nm": lwdg_nm,
                        "biz_reg_no": brno,
                        "brno": brno,
                        "confmdoc_isu_no": confmdoc_isu_no,
                        "confmdoc_isu_dt": confmdoc_isu_dt,
                        "confmdoc_expr_dt": confmdoc_expr_dt,
                        "manu_category": manu_category,
                        "manu_lclss": manu_lclss,
                        "manu_mclss": manu_mclss,
                        "manu_sclss": manu_sclss,
                        "deadline": confmdoc_expr_dt,
                        "detail_url": hmpg,
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 9) 창업기업 확인서 기업 정보 (corporates)
        for item_idx, item in enumerate(raw_data.get("corporates", [])):
            source_doc_id = self._build_source_doc_id("corporates", item.get("id"), item_idx)
            ntrp_nm = clean_text(safe_get(item, "ntrp_nm", default=""))
            ntrp_type_nm = clean_text(safe_get(item, "ntrp_type_nm", default=""))
            repr_nm = clean_text(safe_get(item, "repr_nm", default=""))
            unin_repr_nm = clean_text(safe_get(item, "unin_repr_nm", default=""))
            brno = clean_text(safe_get(item, "brno", default=""))
            crno = clean_text(safe_get(item, "crno", default=""))
            confmdoc_isu_no = clean_text(
                safe_get(item, "confmdoc_isu_no", default="")
            )
            confmdoc_isu_dt = clean_text(
                safe_get(item, "confmdoc_isu_dt", default="")
            )
            confmdoc_expr_dt = clean_text(
                safe_get(item, "confmdoc_expr_dt", default="")
            )

            text_parts = [
                f"기업명: {ntrp_nm}",
                f"기업 구분: {ntrp_type_nm}",
                f"대표자명: {repr_nm}",
                f"공동대표/단체대표: {unin_repr_nm}",
                f"사업자등록번호: {brno}",
                f"법인등록번호: {crno}",
                f"창업기업확인서 발급 번호: {confmdoc_isu_no}",
                f"창업기업확인서 발급 일시: {confmdoc_isu_dt}",
                f"창업기업확인서 만료 일시: {confmdoc_expr_dt}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"CORP_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "corporate",
                        "doc_id": source_doc_id,
                        "policy_id": confmdoc_isu_no or source_doc_id,
                        "title": ntrp_nm,
                        "ntrp_nm": ntrp_nm,
                        "corp_type": ntrp_type_nm,
                        "ntrp_type_nm": ntrp_type_nm,
                        "ceo_name": repr_nm,
                        "repr_nm": repr_nm,
                        "co_rep_name": unin_repr_nm,
                        "unin_repr_nm": unin_repr_nm,
                        "brno": brno,
                        "crno": crno,
                        "confmdoc_isu_no": confmdoc_isu_no,
                        "confmdoc_isu_dt": confmdoc_isu_dt,
                        "confmdoc_expr_dt": confmdoc_expr_dt,
                        "detail_url": "",
                        "deadline": confmdoc_expr_dt,
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # 10) 주관기관 정보 (institutions)
        for item_idx, item in enumerate(raw_data.get("institutions", [])):
            source_doc_id = self._build_source_doc_id("institutions", item.get("id"), item_idx)
            regin_clss_cd = clean_text(
                safe_get(item, "regin_clss_cd", default="")
            )
            inds_clsf_clss_cd = clean_text(
                safe_get(item, "inds_clsf_clss_cd", default="")
                or safe_get(item, "inds_clsfc_clss_cd", default="")
            )
            brno = clean_text(safe_get(item, "brno", default=""))
            inst_nm = clean_text(safe_get(item, "inst_nm", default=""))
            repr_nm = clean_text(safe_get(item, "repr_nm", default=""))
            inst_eng_nm = clean_text(
                safe_get(item, "inst_eng_nm", default="")
            )
            crno = clean_text(safe_get(item, "crno", default=""))
            fndn_clss_cd = clean_text(
                safe_get(item, "fndn_clss_cd", default="")
            )
            fndn_dt = clean_text(safe_get(item, "fndn_dt", default=""))
            inst_chr_clss_cd = clean_text(
                safe_get(item, "inst_chr_clss_cd", default="")
                or safe_get(item, "inst_chrr_clss_cd", default="")
            )
            inst_clsf_clss_cd = clean_text(
                safe_get(item, "inst_clsf_clss_cd", default="")
            )
            natn_clss_cd = clean_text(
                safe_get(item, "natn_clss_cd", default="")
            )

            text_parts = [
                inst_nm,
                f"기관 영문 명: {inst_eng_nm}",
                f"대표자 명: {repr_nm}",
                f"사업자등록번호: {brno}",
                f"법인등록번호: {crno}",
                f"설립 구분 코드: {fndn_clss_cd}",
                f"설립 일시: {fndn_dt}",
                f"기관 특성 코드: {inst_chr_clss_cd}",
                f"기관 분류 코드: {inst_clsf_clss_cd}",
                f"산업 분류 코드: {inds_clsf_clss_cd}",
                f"국가 구분 코드: {natn_clss_cd}",
                f"지역 구분 코드: {regin_clss_cd}",
            ]
            text = "\n".join([t for t in text_parts if t])

            documents.append(
                Document(
                    id=f"INST_{doc_id:06d}",
                    text=text,
                    metadata={
                        "type": "institution",
                        "doc_id": source_doc_id,
                        "policy_id": source_doc_id,
                        "title": inst_nm,
                        "inst_nm": inst_nm,
                        "inst_eng_nm": inst_eng_nm,
                        "repr_nm": repr_nm,
                        "brno": brno,
                        "crno": crno,
                        "fndn_clss_cd": fndn_clss_cd,
                        "fndn_dt": fndn_dt,
                        "inst_chr_clss_cd": inst_chr_clss_cd,
                        "inst_chrr_clss_cd": inst_chr_clss_cd,
                        "inst_clsf_clss_cd": inst_clsf_clss_cd,
                        "inds_clsf_clss_cd": inds_clsf_clss_cd,
                        "inds_clsfc_clss_cd": inds_clsf_clss_cd,
                        "natn_clss_cd": natn_clss_cd,
                        "regin_clss_cd": regin_clss_cd,
                        "detail_url": "",
                        "deadline": "",
                        "raw": item,
                    },
                )
            )
            doc_id += 1

        # RAG 인덱스에 추가
        if documents:
            self.rag.add_documents(documents)

        self.think(
            "RAG 인덱스 구축 완료", result=f"{len(documents)}개", confidence=1.0
        )
        return len(documents)
