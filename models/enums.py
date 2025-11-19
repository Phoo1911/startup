"""
Enum 정의
"""
from enum import Enum

class APIEndpoint(Enum):
    """공공API 엔드포인트"""
    # kisedKstartupService01
    ANNOUNCEMENT = "/kisedKstartupService01/getAnnouncementInformation01"
    BUSINESS = "/kisedKstartupService01/getBusinessInformation01"
    CONTENT = "/kisedKstartupService01/getContentInformation01"
    STATISTICAL = "/kisedKstartupService01/getStatisticalInformation01"
    
    # kisedEduService (창업에듀 강좌)
    EDU_LECTURE = "/kisedEduService/getEducationInformation"
    
    # kisedSlpService (창업공간/센터)
    SLP_SPACE = "/kisedSlpService/getCenterSpaceList"
    SLP_CENTER = "/kisedSlpService/getCenterList"
    
    # kisedCertService (창업기업 확인서)
    CERT_PRODUCT = "/kisedCertService/getProductInformation"
    CERT_CORPORATE = "/kisedCertService/getCorporateInformation"
    
    # kisedInsttInfoService (창업지원기관)
    INSTITUTION = "/kisedPmsService/getInstitutionInformation"
