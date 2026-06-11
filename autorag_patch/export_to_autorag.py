"""
기존 RAG 데이터를 AutoRAG parquet 형식으로 변환.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config
from agents.data_collector import DataCollectionAgent
from utils.text import clean_text, safe_get

CANONICAL_TYPES = {
    "announcements": "announcements",
    "announcement": "announcements",
    "business": "business",
    "content": "content",
    "statistical": "statistical",
    "edu_lectures": "edu_lectures",
    "lecture": "edu_lectures",
    "lectures": "edu_lectures",
    "spaces": "spaces",
    "space": "spaces",
    "centers": "centers",
    "center": "centers",
    "products": "products",
    "product": "products",
    "corporates": "corporates",
    "corporate": "corporates",
    "institutions": "institutions",
    "institution": "institutions",
}


def _build_unique_doc_id(category: str, raw_id: object, idx: int) -> tuple[str, str]:
    raw_text = str(raw_id).strip() if raw_id not in (None, "") else str(idx)
    prefix = f"{category}_"
    if raw_text.startswith(prefix):
        return raw_text, raw_text[len(prefix):]
    return f"{category}_{raw_text}", raw_text


def _extract_title(item: dict, category: str) -> str:
    candidates = [
        item.get("title"),
        safe_get(item, "supt_biz_titl_nm", default=""),
        safe_get(item, "biz_pbanc_nm", default=""),
        safe_get(item, "titl_nm", default=""),
        safe_get(item, "lctr_nm", default=""),
        safe_get(item, "spce_nm", default=""),
        safe_get(item, "cntr_nm", default=""),
        safe_get(item, "manu_nm", default=""),
        safe_get(item, "ntrp_nm", default=""),
        safe_get(item, "inst_nm", default=""),
    ]
    for value in candidates:
        text = clean_text(str(value or ""))
        if text:
            return text
    return f"{category}_{item.get('id', '')}".strip("_")


def export_to_parquet() -> Path:
    """수집 데이터를 autorag_workspace/parsed/parsed.parquet로 저장."""
    output_path = PROJECT_ROOT / "autorag_workspace" / "parsed" / "parsed.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("📦 데이터 수집 중...")
    collector = DataCollectionAgent(service_key=Config.SERVICE_KEY, llm_client=None)
    raw_data = collector.collect_all(max_pages=3)

    all_docs = []
    for category, items in raw_data.items():
        normalized_cat = CANONICAL_TYPES.get(category.lower(), category.lower())
        for idx, item in enumerate(items):
            raw_id = item.get("id") or idx
            doc_id, source_raw_id = _build_unique_doc_id(normalized_cat, raw_id, idx)
            title = _extract_title(item, normalized_cat)
            metadata = {
                "category": category,
                "data_type": normalized_cat,
                "type": item.get("type", normalized_cat),
                "title": title,
                "region": item.get("region", ""),
                "field": item.get("field", ""),
                "deadline": item.get("deadline", ""),
                "url": item.get("detail_url", ""),
                "doc_id": doc_id,
                "source_doc_id": doc_id,
                "raw_id": source_raw_id,
                "source_raw_id": source_raw_id,
            }
            doc = {
                "doc_id": doc_id,
                "contents": _extract_content(item, normalized_cat),
                "metadata": metadata,
                "data_type": normalized_cat,
            }
            all_docs.append(doc)

    df = pd.DataFrame(all_docs)
    if "texts" not in df.columns:
        df["texts"] = df["contents"]
    df.to_parquet(output_path)
    print(f"✅ {len(all_docs)}개 문서를 {output_path}에 저장 완료")
    return output_path


def _extract_content(item: dict, category: str) -> str:
    """카테고리별 핵심 텍스트를 RAGBuilder와 유사하게 구성."""
    cat = category.lower()
    parts: list[str] = []

    if cat == "business":
        title = clean_text(safe_get(item, "supt_biz_titl_nm", default=""))
        target = clean_text(safe_get(item, "biz_supt_trgt_info", default=""))
        budget = clean_text(safe_get(item, "biz_supt_bdgt_info", default=""))
        content = clean_text(safe_get(item, "biz_supt_ctnt", default=""))
        chrct = clean_text(safe_get(item, "supt_biz_chrct", default=""))
        intro = clean_text(safe_get(item, "supt_biz_intrd_info", default=""))
        category_cd = clean_text(safe_get(item, "biz_category_cd", default=""))
        year = clean_text(safe_get(item, "biz_yr", default=""))
        parts = [
            title,
            f"지원 대상: {target}" if target else "",
            f"지원 예산 및 규모: {budget}" if budget else "",
            f"지원 내용: {content}" if content else "",
            f"사업 특성: {chrct}" if chrct else "",
            f"사업 소개: {intro}" if intro else "",
            f"사업 구분 코드: {category_cd}" if category_cd else "",
            f"사업 연도: {year}" if year else "",
        ]

    elif cat == "announcements":
        title = clean_text(safe_get(item, "biz_pbanc_nm", default=""))
        supt_biz_clsfc = clean_text(safe_get(item, "supt_biz_clsfc", default=""))
        supt_regin = clean_text(safe_get(item, "supt_regin", default=""))
        aply_trgt = clean_text(safe_get(item, "aply_trgt", default=""))
        aply_trgt_ctnt = clean_text(safe_get(item, "aply_trgt_ctnt", default=""))
        aply_excl_trgt_ctnt = clean_text(safe_get(item, "aply_excl_trgt_ctnt", default=""))
        biz_enyy = clean_text(safe_get(item, "biz_enyy", default=""))
        biz_trgt_age = clean_text(safe_get(item, "biz_trgt_age", default=""))
        prfn_matr = clean_text(safe_get(item, "prfn_matr", default=""))
        pbanc_ctnt = clean_text(safe_get(item, "pbanc_ctnt", default=""))
        rcrt_prgs_yn = clean_text(safe_get(item, "rcrt_prgs_yn", default=""))
        pbanc_ntrp_nm = clean_text(safe_get(item, "pbanc_ntrp_nm", default=""))
        sprv_inst = clean_text(safe_get(item, "sprv_inst", default=""))
        biz_prch_dprt_nm = clean_text(safe_get(item, "biz_prch_dprt_nm", default=""))
        apply_period = clean_text(safe_get(item, "apply_period", default=""))
        parts = [
            title,
            f"지원 분야: {supt_biz_clsfc}" if supt_biz_clsfc else "",
            f"지원 지역: {supt_regin}" if supt_regin else "",
            f"신청 대상(요약): {aply_trgt}" if aply_trgt else "",
            f"신청 대상(상세): {aply_trgt_ctnt}" if aply_trgt_ctnt else "",
            f"신청 제외 대상: {aply_excl_trgt_ctnt}" if aply_excl_trgt_ctnt else "",
            f"창업 기간 조건: {biz_enyy}" if biz_enyy else "",
            f"대상 연령: {biz_trgt_age}" if biz_trgt_age else "",
            f"우대 사항: {prfn_matr}" if prfn_matr else "",
            f"공고 내용: {pbanc_ctnt}" if pbanc_ctnt else "",
            f"접수 기간: {apply_period}" if apply_period else "",
            f"모집 진행 여부: {rcrt_prgs_yn}" if rcrt_prgs_yn else "",
            f"모집 주체: {pbanc_ntrp_nm}" if pbanc_ntrp_nm else "",
            f"주관 기관: {sprv_inst}" if sprv_inst else "",
            f"담당 부서: {biz_prch_dprt_nm}" if biz_prch_dprt_nm else "",
        ]

    elif cat == "content":
        title = clean_text(safe_get(item, "titl_nm", default=""))
        class_code = clean_text(safe_get(item, "clss_cd", default=""))
        file_name = clean_text(safe_get(item, "file_nm", default=""))
        reg_date = clean_text(safe_get(item, "fstm_reg_dt", default=""))
        view_cnt = clean_text(str(safe_get(item, "view_cnt", default="")))
        parts = [
            title,
            f"콘텐츠 구분 코드: {class_code}" if class_code else "",
            f"파일명: {file_name}" if file_name else "",
            f"등록일: {reg_date}" if reg_date else "",
            f"조회수: {view_cnt}" if view_cnt else "",
        ]

    elif cat == "statistical":
        title = clean_text(safe_get(item, "titl_nm", default=""))
        content = clean_text(safe_get(item, "cntn", default=""))
        file_name = clean_text(safe_get(item, "file_nm", default=""))
        first_reg_dt = clean_text(safe_get(item, "fstm_reg_dt", default=""))
        last_mdfcn_dt = clean_text(safe_get(item, "last_mdfcn_dt", default=""))
        parts = [
            title,
            f"통계 자료 내용: {content}" if content else "",
            f"파일명: {file_name}" if file_name else "",
            f"최초 등록일: {first_reg_dt}" if first_reg_dt else "",
            f"최종 수정일: {last_mdfcn_dt}" if last_mdfcn_dt else "",
        ]

    elif cat == "edu_lectures":
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
        reg_dt = clean_text(safe_get(item, "reg_dt", default=""))
        mdfcn_dt = clean_text(safe_get(item, "mdfcn_dt", default=""))
        parts = [
            f"강좌명: {lctr_nm}" if lctr_nm else "",
            f"강좌 설명: {lctr_istc}" if lctr_istc else "",
            f"키워드: {kywrd}" if kywrd else "",
            f"강좌 대분류 코드: {l_lclss}" if l_lclss else "",
            f"강좌 중분류 코드: {l_mclss}" if l_mclss else "",
            f"강좌 소분류 코드: {l_sclss}" if l_sclss else "",
            f"자막 노출 여부: {sbtitl_yn}" if sbtitl_yn else "",
            f"추천 강좌 여부: {rcmd_yn}" if rcmd_yn else "",
            f"패키지 강좌 여부: {pckg_yn}" if pckg_yn else "",
            f"제휴 강좌 여부: {aliac_yn}" if aliac_yn else "",
            f"재생 시간(초): {play_time}" if play_time else "",
            f"조회수: {view_cnt}" if view_cnt else "",
            f"등록일: {reg_dt}" if reg_dt else "",
            f"수정일: {mdfcn_dt}" if mdfcn_dt else "",
        ]

    elif cat == "spaces":
        spce_nm = clean_text(safe_get(item, "spce_nm", default=""))
        cntr_nm = clean_text(safe_get(item, "cntr_nm", default=""))
        spce_type = clean_text(safe_get(item, "spce_type_nm", default=""))
        seat_type = clean_text(safe_get(item, "seat_type_nm", default=""))
        seat_clss = clean_text(safe_get(item, "seat_clss", default=""))
        seat_co = clean_text(safe_get(item, "seat_co", default=""))
        rent = safe_get(item, "rent", default=0)
        guam = safe_get(item, "guam", default=0)
        excuse_ar = clean_text(safe_get(item, "excuse_ar", default=""))
        cmnus_ar = clean_text(safe_get(item, "cmnus_ar", default=""))
        rsvt_cls = clean_text(safe_get(item, "rsvt_psbl_clss", default=""))
        addr = clean_text(safe_get(item, "addr", default=""))
        pstno = clean_text(safe_get(item, "pstno", default=""))
        cntr_type = clean_text(safe_get(item, "cntr_type_nm", default=""))
        cntr_intrd = clean_text(safe_get(item, "cntr_intrd_type_nm", default=""))
        hmpg = clean_text(safe_get(item, "hmpg", default=""))
        parts = [
            f"공간명: {spce_nm}" if spce_nm else "",
            f"센터명: {cntr_nm}" if cntr_nm else "",
            f"공간 유형: {spce_type}" if spce_type else "",
            f"좌석 유형: {seat_type}" if seat_type else "",
            f"좌석 구분: {seat_clss}" if seat_clss else "",
            f"좌석 수: {seat_co}" if seat_co else "",
            f"임대료: {rent}" if rent else "",
            f"보증금: {guam}" if guam else "",
            f"전용 면적: {excuse_ar}" if excuse_ar else "",
            f"공용 면적: {cmnus_ar}" if cmnus_ar else "",
            f"예약 가능 여부: {rsvt_cls}" if rsvt_cls else "",
            f"주소: {addr} ({pstno})" if addr else "",
            f"센터 유형: {cntr_type}" if cntr_type else "",
            f"센터 소개 유형: {cntr_intrd}" if cntr_intrd else "",
            f"홈페이지: {hmpg}" if hmpg else "",
        ]

    elif cat == "centers":
        cntr_nm = clean_text(safe_get(item, "cntr_nm", default=""))
        cntr_type = clean_text(safe_get(item, "cntr_type_nm", default=""))
        spce_cnt = safe_get(item, "spce_cnt", default=0)
        buld_nm = clean_text(safe_get(item, "buld_nm", default=""))
        addr = clean_text(safe_get(item, "addr", default=""))
        pstno = clean_text(safe_get(item, "pstno", default=""))
        regin = clean_text(safe_get(item, "regin_clss", default=""))
        hmpg = clean_text(safe_get(item, "hmpg", default=""))
        intrd_typ = clean_text(safe_get(item, "cntr_intrd_type_nm", default=""))
        parts = [
            f"센터명: {cntr_nm}" if cntr_nm else "",
            f"센터 유형: {cntr_type}" if cntr_type else "",
            f"보유 공간 수: {spce_cnt}" if spce_cnt else "",
            f"건물명: {buld_nm}" if buld_nm else "",
            f"주소: {addr}" if addr else "",
            f"우편번호: {pstno}" if pstno else "",
            f"지역 구분: {regin}" if regin else "",
            f"홈페이지: {hmpg}" if hmpg else "",
            f"센터 소개 유형: {intrd_typ}" if intrd_typ else "",
        ]

    elif cat == "products":
        manu_nm = clean_text(safe_get(item, "manu_nm", default=""))
        manu_intrd = clean_text(safe_get(item, "manu_intrd", default=""))
        manu_stnds = clean_text(safe_get(item, "manu_stnds", default=""))
        manu_prc = clean_text(safe_get(item, "manu_prc", default=""))
        manu_category = clean_text(safe_get(item, "manu_category", default=""))
        manu_lclss = clean_text(safe_get(item, "manu_lclss", default=""))
        manu_mclss = clean_text(safe_get(item, "manu_mclss", default=""))
        manu_sclss = clean_text(safe_get(item, "manu_sclss", default=""))
        ntrp_nm = clean_text(safe_get(item, "ntrp_nm", default=""))
        lwdg_nm = clean_text(safe_get(item, "lwdg_nm", default=""))
        brno = clean_text(safe_get(item, "brno", default=""))
        confmdoc_isu_no = clean_text(safe_get(item, "confmdoc_isu_no", default=""))
        confmdoc_isu_dt = clean_text(safe_get(item, "confmdoc_isu_dt", default=""))
        confmdoc_expr_dt = clean_text(safe_get(item, "confmdoc_expr_dt", default=""))
        parts = [
            f"제품명: {manu_nm}" if manu_nm else "",
            f"제품 소개: {manu_intrd}" if manu_intrd else "",
            f"제품 규격: {manu_stnds}" if manu_stnds else "",
            f"제품 가격: {manu_prc}" if manu_prc else "",
            f"제품 분류: {manu_category}" if manu_category else "",
            f"제품 대분류: {manu_lclss}" if manu_lclss else "",
            f"제품 중분류: {manu_mclss}" if manu_mclss else "",
            f"제품 소분류: {manu_sclss}" if manu_sclss else "",
            f"기업명: {ntrp_nm}" if ntrp_nm else "",
            f"지자체명: {lwdg_nm}" if lwdg_nm else "",
            f"사업자등록번호: {brno}" if brno else "",
            f"확인서 번호: {confmdoc_isu_no}" if confmdoc_isu_no else "",
            f"확인서 발급 일자: {confmdoc_isu_dt}" if confmdoc_isu_dt else "",
            f"확인서 만료 일자: {confmdoc_expr_dt}" if confmdoc_expr_dt else "",
        ]

    elif cat == "corporates":
        ntrp_nm = clean_text(safe_get(item, "ntrp_nm", default=""))
        ntrp_type_nm = clean_text(safe_get(item, "ntrp_type_nm", default=""))
        repr_nm = clean_text(safe_get(item, "repr_nm", default=""))
        unin_repr_nm = clean_text(safe_get(item, "unin_repr_nm", default=""))
        brno = clean_text(safe_get(item, "brno", default=""))
        crno = clean_text(safe_get(item, "crno", default=""))
        confmdoc_isu_no = clean_text(safe_get(item, "confmdoc_isu_no", default=""))
        confmdoc_isu_dt = clean_text(safe_get(item, "confmdoc_isu_dt", default=""))
        confmdoc_expr_dt = clean_text(safe_get(item, "confmdoc_expr_dt", default=""))
        parts = [
            f"기업명: {ntrp_nm}" if ntrp_nm else "",
            f"기업 구분: {ntrp_type_nm}" if ntrp_type_nm else "",
            f"대표자명: {repr_nm}" if repr_nm else "",
            f"공동대표/단체대표: {unin_repr_nm}" if unin_repr_nm else "",
            f"사업자등록번호: {brno}" if brno else "",
            f"법인등록번호: {crno}" if crno else "",
            f"확인서 번호: {confmdoc_isu_no}" if confmdoc_isu_no else "",
            f"확인서 발급 일시: {confmdoc_isu_dt}" if confmdoc_isu_dt else "",
            f"확인서 만료 일시: {confmdoc_expr_dt}" if confmdoc_expr_dt else "",
        ]

    elif cat == "institutions":
        inst_nm = clean_text(safe_get(item, "inst_nm", default=""))
        inst_eng_nm = clean_text(safe_get(item, "inst_eng_nm", default=""))
        repr_nm = clean_text(safe_get(item, "repr_nm", default=""))
        brno = clean_text(safe_get(item, "brno", default=""))
        crno = clean_text(safe_get(item, "crno", default=""))
        fndn_clss_cd = clean_text(safe_get(item, "fndn_clss_cd", default=""))
        fndn_dt = clean_text(safe_get(item, "fndn_dt", default=""))
        inst_chr_clss_cd = clean_text(safe_get(item, "inst_chr_clss_cd", default=""))
        inst_clsf_clss_cd = clean_text(safe_get(item, "inst_clsf_clss_cd", default=""))
        inds_clsf_clss_cd = clean_text(safe_get(item, "inds_clsf_clss_cd", default=""))
        natn_clss_cd = clean_text(safe_get(item, "natn_clss_cd", default=""))
        regin_clss_cd = clean_text(safe_get(item, "regin_clss_cd", default=""))
        parts = [
            inst_nm,
            f"기관 영문 명: {inst_eng_nm}" if inst_eng_nm else "",
            f"대표자 명: {repr_nm}" if repr_nm else "",
            f"사업자등록번호: {brno}" if brno else "",
            f"법인등록번호: {crno}" if crno else "",
            f"설립 구분 코드: {fndn_clss_cd}" if fndn_clss_cd else "",
            f"설립 일시: {fndn_dt}" if fndn_dt else "",
            f"기관 특성 코드: {inst_chr_clss_cd}" if inst_chr_clss_cd else "",
            f"기관 분류 코드: {inst_clsf_clss_cd}" if inst_clsf_clss_cd else "",
            f"산업 분류 코드: {inds_clsf_clss_cd}" if inds_clsf_clss_cd else "",
            f"국가 구분 코드: {natn_clss_cd}" if natn_clss_cd else "",
            f"지역 구분 코드: {regin_clss_cd}" if regin_clss_cd else "",
        ]

    else:
        for field in ("title", "summary", "desc", "content", "cntn"):
            value = item.get(field)
            if value:
                parts.append(str(value))

    return "\n".join([p for p in parts if p]) if parts else str(item)


if __name__ == "__main__":
    export_to_parquet()
