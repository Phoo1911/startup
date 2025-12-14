# core/experiment_logger.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

from config.settings import Config


class ExperimentLogger:
    """추천 실행 결과 + 평가 지표를 JSON으로 저장"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Config.EVAL_LOG_DIR
        self.base_dir.mkdir(exist_ok=True)

    def log_run(
        self,
        mode: str,
        profile: Dict[str, Any],
        report: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{mode}_run_{ts}.json"
        path = self.base_dir / filename

        payload = {
            "mode": mode,
            "timestamp": ts,
            "profile": profile,
            "summary": {
                "status": report.get("status"),
                "total_matches": report.get("total_matches", 0),
            },
            "report": report,
            "metrics": metrics or {},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"💾 실험 로그 저장: {path}")
        return path
