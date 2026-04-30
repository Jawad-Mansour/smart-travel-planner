from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

PIPELINE_STEPS: list[tuple[str, Path]] = [
    (
        "Phase 10A - Setup database",
        PROJECT_ROOT / "backend" / "rag" / "scripts" / "setup_database.py",
    ),
    (
        "Phase 8 - Collect content",
        PROJECT_ROOT / "backend" / "rag" / "scripts" / "collect_content.py",
    ),
    (
        "Phase 9 - Chunk documents",
        PROJECT_ROOT / "backend" / "rag" / "scripts" / "chunk_documents.py",
    ),
    (
        "Phase 10B - Embed and store",
        PROJECT_ROOT / "backend" / "rag" / "scripts" / "embed_and_store.py",
    ),
    (
        "Phase 11 - Test retrieval",
        PROJECT_ROOT / "backend" / "rag" / "scripts" / "test_retrieval.py",
    ),
    (
        "Phase 11b - Relevance suite",
        PROJECT_ROOT / "backend" / "rag" / "scripts" / "relevance_test.py",
    ),
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_step(step_name: str, script_path: Path) -> None:
    logger = logging.getLogger(__name__)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    logger.info("=" * 70)
    logger.info("Running %s", step_name)
    logger.info("Script: %s", script_path)
    logger.info("=" * 70)
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    logger.info("Completed %s", step_name)


def run_pipeline() -> None:
    logger = logging.getLogger(__name__)
    failures: list[str] = []
    for step_name, script_path in PIPELINE_STEPS:
        try:
            run_step(step_name, script_path)
        except subprocess.CalledProcessError as exc:
            message = f"{step_name} failed with exit code {exc.returncode}"
            logger.exception(message)
            failures.append(message)
            break
        except Exception:
            message = f"{step_name} failed unexpectedly"
            logger.exception(message)
            failures.append(message)
            break

    if failures:
        raise RuntimeError("RAG pipeline aborted: " + "; ".join(failures))

    logger.info("=" * 70)
    logger.info("RAG pipeline completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    configure_logging()
    try:
        run_pipeline()
    except Exception:
        logging.getLogger(__name__).exception("Pipeline execution failed")
        raise
