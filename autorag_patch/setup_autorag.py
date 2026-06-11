"""
AutoRAG 평가 데이터 생성을 위한 디렉터리 구조 생성.
"""

from pathlib import Path


WORKSPACE_DIRS = [
    "autorag_workspace/raw",
    "autorag_workspace/parsed",
    "autorag_workspace/corpus",
    "autorag_workspace/qa",
    "autorag_workspace/configs",
    "autorag_workspace/results",
]


def setup_autorag_dirs() -> None:
    """필요한 워크스페이스 폴더들을 생성한다."""
    for rel in WORKSPACE_DIRS:
        Path(rel).mkdir(parents=True, exist_ok=True)
        print(f"✅ {rel} 생성 완료")


if __name__ == "__main__":
    setup_autorag_dirs()

