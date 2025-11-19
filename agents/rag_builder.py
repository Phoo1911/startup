"""
RAG 인덱스 구축 에이전트
"""
from typing import List, Dict, Optional
from agents.base import AgenticAgent
from models.data import Document
from utils.text import clean_text, safe_get

class RAGBuilderAgent(AgenticAgent):
    """RAG 벡터 인덱스 구축 에이전트"""
    
    def __init__(self, rag_system: object, llm_client: Optional[object] = None):
        super().__init__("RAGBuilder", llm_client)
        self.rag = rag_system
    
    def build_index(self, raw_data: Dict[str, List[Dict]]) -> int:
        """RAG 인덱스 구축"""
        self.think("RAG 벡터 인덱스 구축", action="문서 임베딩", confidence=0.95)
        
        documents: List[Document] = []
        doc_id = 0
        
        # 1) 통합공고 지원사업 정보
        for item in raw_data.get('business', []):
            text_parts = [
                safe_get(item, 'supt_biz_titl_nm', default=''),
                safe_get(item, 'biz_supt_trgt_info', default=''),
                safe_get(item, 'biz_supt_bdgt_info', default=''),
                safe_get(item, 'biz_supt_ctnt', default=''),
                safe_get(item, 'supt_biz_chrct', default=''),
                safe_get(item, 'supt_biz_intrd_info', default='')
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"BIZ_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'business',
                    'title': clean_text(safe_get(item, 'supt_biz_titl_nm', default='')),
                    'deadline': '',
                    'detail_url': clean_text(safe_get(item, 'detl_pg_url', default='')),
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 2) 지원사업 공고 정보
        for item in raw_data.get('announcements', []):
            text_parts = [
                safe_get(item, 'biz_pbanc_nm', default=''),
                safe_get(item, 'pbanc_ctnt', default=''),
                safe_get(item, 'supt_biz_clsfc', default=''),
                safe_get(item, 'aply_trgt_ctnt', default=''),
                safe_get(item, 'supt_regin', default=''),
                safe_get(item, 'aply_trgt', default=''),
                safe_get(item, 'biz_enyy', default=''),
                safe_get(item, 'biz_trgt_age', default=''),
                safe_get(item, 'prfn_matr', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"ANN_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'announcement',
                    'title': clean_text(safe_get(item, 'biz_pbanc_nm', default='')),
                    'deadline': clean_text(safe_get(item, 'pbanc_rcpt_end_dt', default='')),
                    'detail_url': clean_text(safe_get(item, 'detl_pg_url', default='')),
                    'region': clean_text(safe_get(item, 'supt_regin', default='')),
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 3) 콘텐츠 정보
        for item in raw_data.get('content', []):
            text_parts = [
                safe_get(item, 'titl_nm', default=''),
                safe_get(item, 'clss_cd', default=''),
                safe_get(item, 'file_nm', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"CNT_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'content',
                    'title': clean_text(safe_get(item, 'titl_nm', default='')),
                    'deadline': '',
                    'detail_url': clean_text(safe_get(item, 'detl_pg_url', default='')),
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 4) 통계보고서 정보
        for item in raw_data.get('statistical', []):
            text_parts = [
                safe_get(item, 'titl_nm', default=''),
                safe_get(item, 'ctnt', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"STAT_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'statistical',
                    'title': clean_text(safe_get(item, 'titl_nm', default='')),
                    'deadline': '',
                    'detail_url': clean_text(safe_get(item, 'detl_pg_url', default='')),
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 5) 창업에듀 강좌 정보
        for item in raw_data.get('edu_lectures', []):
            text_parts = [
                safe_get(item, 'lctr_nm', default=''),
                safe_get(item, 'lctr_istc', default=''),
                safe_get(item, 'kywrd', default=''),
                safe_get(item, 'cntr_nm', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"EDU_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'lecture',
                    'title': clean_text(safe_get(item, 'lctr_nm', default='')),
                    'deadline': '',
                    'detail_url': clean_text(safe_get(item, 'detl_pg_url', default='')),
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 6) 창업공간 정보
        for item in raw_data.get('spaces', []):
            text_parts = [
                safe_get(item, 'spce_nm', default=''),
                safe_get(item, 'regin_clss', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"SPC_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'space',
                    'title': clean_text(safe_get(item, 'spce_nm', default='')),
                    'deadline': '',
                    'detail_url': '',
                    'region': clean_text(safe_get(item, 'regin_clss', default='')),
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 7) 센터 정보
        for item in raw_data.get('centers', []):
            text_parts = [
                safe_get(item, 'cntr_nm', default=''),
                safe_get(item, 'regin_clss', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"CNTR_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'center',
                    'title': clean_text(safe_get(item, 'cntr_nm', default='')),
                    'deadline': '',
                    'detail_url': '',
                    'region': clean_text(safe_get(item, 'regin_clss', default='')),
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 8) 창업기업 확인서 제품 정보
        for item in raw_data.get('products', []):
            text_parts = [
                safe_get(item, 'manu_nm', default=''),
                safe_get(item, 'manu_category', default=''),
                safe_get(item, 'manu_lclss', default=''),
                safe_get(item, 'manu_mclss', default=''),
                safe_get(item, 'manu_sclss', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"PRD_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'product',
                    'title': clean_text(safe_get(item, 'manu_nm', default='')),
                    'deadline': '',
                    'detail_url': '',
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 9) 창업기업 확인서 기업 정보
        for item in raw_data.get('corporates', []):
            text_parts = [
                safe_get(item, 'ntrp_nm', default=''),
                safe_get(item, 'ntrp_type_nm', default=''),
                safe_get(item, 'addr', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"CORP_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'corporate',
                    'title': clean_text(safe_get(item, 'ntrp_nm', default='')),
                    'deadline': '',
                    'detail_url': '',
                    'raw': item
                }
            ))
            doc_id += 1
        
        # 10) 주관기관 정보
        for item in raw_data.get('institutions', []):
            text_parts = [
                safe_get(item, 'inst_nm', default=''),
                safe_get(item, 'inst_eng_nm', default=''),
                safe_get(item, 'addr', default=''),
                safe_get(item, 'inst_chr_clss_cd', default=''),
                safe_get(item, 'inst_clsf_clss_cd', default=''),
                safe_get(item, 'regin_clss_cd', default=''),
            ]
            text = "\n".join([clean_text(t) for t in text_parts if t])
            
            documents.append(Document(
                id=f"INST_{doc_id:06d}",
                text=text,
                metadata={
                    'type': 'institution',
                    'title': clean_text(safe_get(item, 'inst_nm', default='')),
                    'deadline': '',
                    'detail_url': '',
                    'region': clean_text(safe_get(item, 'regin_clss_cd', default='')),
                    'raw': item
                }
            ))
            doc_id += 1
        
        # RAG 인덱스에 추가
        if documents:
            self.rag.add_documents(documents)
        
        self.think("RAG 인덱스 구축 완료", result=f"{len(documents)}개", confidence=1.0)
        return len(documents)
