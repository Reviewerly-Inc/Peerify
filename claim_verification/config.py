
import os
from pathlib import Path
from typing import Dict, Optional


def get_env(var_name: str, default: str = "") -> str:
    return os.getenv(var_name, default)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    base = get_env("BENCHMARK_DATA_ROOT", str(repo_root() / "data"))
    return Path(base)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def openreview_credentials() -> Dict[str, str]:
    return {
        "baseurl": get_env("OPENREVIEW_BASEURL", "https://api2.openreview.net"),
        "username": get_env("OPENREVIEW_USERNAME"),
        "password": get_env("OPENREVIEW_PASSWORD"),
    }


def as_str(path: Path) -> str:
    return str(path)


VENUES = {
    "neurips2024": {
        "venue_id": "NeurIPS.cc/2024/Conference",
        "short_name": "neurips2024",
    },
    "iclr2024": {
        "venue_id": "ICLR.cc/2024/Conference",
        "short_name": "iclr2024",
    },
}

DEFAULT_MODEL = "gpt-5-mini"

BENCHMARK_TARGETS = {
    "B1_golden_supported": 500,
    "B2_reviewer_author": 800,
    "B3_reviewer_paper": 800,
    "B4_agreement": 800,
    "B5_verifiable": None,
}

MAX_PAPERS_TO_CRAWL = 500
NUM_PAPERS_TO_SELECT = 200
ONE_PAIR_PER_PAPER = True

DECISION_CATEGORIES = {
    "Reject": ["Reject", "reject", "Rejected", "rejected"],
    "Poster": ["Poster", "poster", "Accept (Poster)", "Accept as poster"],
    "Oral": ["Oral", "oral", "Accept (Oral)", "Accept as oral"],
    "Spotlight": ["Spotlight", "spotlight", "Accept (Spotlight)", "Accept as spotlight"],
}


def normalize_decision(decision: Optional[str]) -> str:
    if not decision:
        return "Unknown"

    decision_str = str(decision).strip()
    for category, variants in DECISION_CATEGORIES.items():
        if any(variant.lower() in decision_str.lower() for variant in variants):
            return category

    decision_lower = decision_str.lower()
    if "reject" in decision_lower or "not accept" in decision_lower:
        return "Reject"
    elif "oral" in decision_lower:
        return "Oral"
    elif "spotlight" in decision_lower:
        return "Spotlight"
    elif "poster" in decision_lower or "accept" in decision_lower:
        return "Poster"

    return "Unknown"


class DataStructure:
    def __init__(self):
        self.base_dir = data_root()
        self._create_structure()

    def _create_structure(self):
        for sub in ["raw", "paired", "benchmark", "logs", "pdfs", "markdown", "chunks"]:
            ensure_dir(self.base_dir / sub)

    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "raw"

    @property
    def paired_dir(self) -> Path:
        return self.base_dir / "paired"

    @property
    def benchmark_dir(self) -> Path:
        return self.base_dir / "benchmark"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def pdfs_dir(self) -> Path:
        return self.base_dir / "pdfs"

    @property
    def markdown_dir(self) -> Path:
        return self.base_dir / "markdown"

    @property
    def chunks_dir(self) -> Path:
        return self.base_dir / "chunks"

    def get_raw_submissions_file(self, venue: str) -> Path:
        return self.raw_dir / f"{venue}_submissions.pkl"

    def get_raw_metadata_file(self, venue: str) -> Path:
        return self.raw_dir / f"{venue}_metadata.json"

    def get_raw_threads_file(self, venue: str) -> Path:
        return self.raw_dir / f"{venue}_threads.json"

    def get_paired_file(self) -> Path:
        return self.paired_dir / "review_response_pairs.json"

    def get_selected_papers_file(self) -> Path:
        return self.paired_dir / "selected_papers.json"

    def get_benchmark_file(self, benchmark_name: str, decision: Optional[str] = None) -> Path:
        if decision:
            decision_normalized = normalize_decision(decision)
            return self.benchmark_dir / f"{benchmark_name}_{decision_normalized.lower()}.jsonl"
        return self.benchmark_dir / f"{benchmark_name}.jsonl"

    def get_claims_file(self) -> Path:
        return self.benchmark_dir / "extracted_claims.json"

    def get_verifiable_claims_file(self) -> Path:
        return self.benchmark_dir / "verifiable_claims.json"
