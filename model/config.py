from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FINBERT_CACHE = DATA_DIR / "finbert_scores.parquet"
LLM_ANNOTATIONS_PATH = DATA_DIR / "headline_annotations.parquet"
OUTPUT_DIR = ROOT / "web" / "public" / "data"
