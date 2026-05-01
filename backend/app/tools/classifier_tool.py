"""
Phase 15: ML-backed travel-style classification + destination matching.

Loads sklearn/joblib artifacts when present under ``Settings.ml_models_dir``;
otherwise uses CSV keyword routing only.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from backend.app.core.config import Settings
from backend.app.schemas.tools import ToolEnvelope, ToolError

logger = structlog.get_logger(__name__)

FEATURE_COLUMNS = [
    "avg_annual_temp_c",
    "seasonal_range_c",
    "cost_per_day_avg_usd",
    "meal_budget_usd",
    "hotel_night_avg_usd",
    "flight_cost_usd",
    "museum_count",
    "monument_count",
    "festival_score",
    "beach_score",
    "scenic_score",
    "wellness_score",
    "culture_score",
    "hiking_score",
    "nightlife_score",
    "family_score",
    "luxury_score",
    "safety_score",
    "tourist_density_score",
    "adventure_sports_score",
    "near_mountains",
    "near_beach",
    "english_friendly_score",
    "public_transport_score",
    "latitude",
    "longitude",
]


def _keyword_boosts(text: str) -> dict[str, float]:
    """Infer numeric boosts from free text activities."""
    t = text.lower()
    b: dict[str, float] = {}
    if any(k in t for k in ("hike", "trek", "ski", "climb", "adventure", "outdoor")):
        b["hiking_score"] = 9.0
        b["adventure_sports_score"] = 9.0
        b["near_mountains"] = 1.0
    if any(k in t for k in ("beach", "snorkel", "dive", "surf")):
        b["beach_score"] = 10.0
        b["near_beach"] = 1.0
    if any(k in t for k in ("museum", "history", "culture", "temple", "art")):
        b["culture_score"] = 10.0
        b["museum_count"] = 80.0
    if any(k in t for k in ("food", "dining", "street food", "wine")):
        b["culture_score"] = max(b.get("culture_score", 5.0), 8.0)
    if any(k in t for k in ("nightlife", "club", "bar")):
        b["nightlife_score"] = 9.0
    if any(k in t for k in ("family", "kids")):
        b["family_score"] = 9.0
    if any(k in t for k in ("luxury", "resort", "spa")):
        b["luxury_score"] = 9.0
        b["wellness_score"] = max(b.get("wellness_score", 5.0), 8.0)
    if "quiet" in t or "not tourist" in t:
        b["tourist_density_score"] = 3.0
    if "warm" in t or "beach" in t:
        b["avg_annual_temp_c"] = 26.0
    if "cool" in t or "snow" in t:
        b["avg_annual_temp_c"] = 5.0
    return b


def _infer_style_keyword(text: str) -> str:
    t = text.lower()
    scores = {
        "Adventure": 0,
        "Culture": 0,
        "Luxury": 0,
        "Relax": 0,
        "Family": 0,
        "Food": 0,
    }
    for k in ("hike", "trek", "ski", "adventure", "outdoor", "diving", "bungee"):
        if k in t:
            scores["Adventure"] += 2
    for k in ("museum", "history", "cathedral", "art", "architecture"):
        if k in t:
            scores["Culture"] += 2
    for k in ("luxury", "resort", "spa", "honeymoon"):
        if k in t:
            scores["Luxury"] += 2
    for k in ("relax", "beach", "wellness", "yoga"):
        if k in t:
            scores["Relax"] += 2
    for k in ("family", "kids"):
        if k in t:
            scores["Family"] += 2
    for k in ("food", "restaurant", "street food", "wine"):
        if k in t:
            scores["Food"] += 2
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Culture"


class DestinationClassifier:
    """Loads CSV + optional sklearn pipeline for travel_style alignment."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._df: pd.DataFrame | None = None
        self._model: Any = None
        self._preprocessor: Any = None
        self._label_encoder: Any = None

    async def _ensure_data(self) -> None:
        if self._df is not None:
            return
        path = Path(self._settings.ml_destinations_csv)
        loop = asyncio.get_event_loop()
        self._df = await loop.run_in_executor(None, pd.read_csv, path)

    async def _maybe_load_sklearn(self) -> None:
        if self._model is not None:
            return
        d = Path(self._settings.ml_models_dir)
        model_f = d / "travel_classifier_final.joblib"
        pre_f = d / "preprocessor.joblib"
        le_f = d / "label_encoder.joblib"
        if not all(p.exists() for p in (model_f, pre_f, le_f)):
            logger.warning(
                "classifier.sklearn_missing",
                expected=str(model_f),
            )
            return

        import joblib

        loop = asyncio.get_event_loop()

        def _load() -> None:
            self._model = joblib.load(model_f)
            self._preprocessor = joblib.load(pre_f)
            self._label_encoder = joblib.load(le_f)

        await loop.run_in_executor(None, _load)

    def _build_feature_row(
        self,
        *,
        activities_text: str,
        duration_days: int | None,
        budget_usd: float | None,
    ) -> pd.DataFrame:
        assert self._df is not None
        means = self._df[FEATURE_COLUMNS].median(numeric_only=True).to_dict()
        boosts = _keyword_boosts(activities_text)
        row: dict[str, Any] = {**means, **boosts}
        if duration_days:
            daily = budget_usd / float(duration_days) if budget_usd else None
            if daily is not None:
                row["cost_per_day_avg_usd"] = float(np.clip(daily, 20.0, 800.0))
        elif budget_usd:
            row["cost_per_day_avg_usd"] = float(np.clip(budget_usd / 7.0, 20.0, 800.0))

        # Align region/categorical columns expected by preprocessor if present in training
        extra_cols = [c for c in self._df.columns if c not in FEATURE_COLUMNS + ["travel_style"]]
        for c in extra_cols:
            if c not in row:
                mode = self._df[c].mode(dropna=True)
                row[c] = mode.iloc[0] if len(mode) else ""

        frame = pd.DataFrame([row])
        # Ensure column union matches training export — preprocessor stores column names
        return frame


async def classify_destinations(
    settings: Settings,
    *,
    activities: list[str],
    duration_days: int | None,
    budget_usd: float | None,
    destination_hint: str | None,
    top_k: int = 8,
) -> ToolEnvelope:
    """
    Returns ranked destinations matching inferred travel style.
    """
    t0 = time.perf_counter()
    tool_name = "classify_destinations"
    try:
        svc = DestinationClassifier(settings)
        await svc._ensure_data()
        await svc._maybe_load_sklearn()

        assert svc._df is not None
        df = svc._df
        text = " ".join(activities) + " " + (destination_hint or "")

        predicted_style: str | None = None
        confidence = 0.0

        if (
            svc._model is not None
            and svc._preprocessor is not None
            and svc._label_encoder is not None
        ):
            try:
                frame = svc._build_feature_row(
                    activities_text=text,
                    duration_days=duration_days,
                    budget_usd=budget_usd,
                )
                Xp = svc._preprocessor.transform(frame)
                pred_code = svc._model.predict(Xp)[0]
                predicted_style = str(svc._label_encoder.inverse_transform([pred_code])[0])
                try:
                    proba = svc._model.predict_proba(Xp)[0]
                    confidence = float(np.max(proba))
                except Exception:
                    confidence = 0.5
            except Exception as exc:
                logger.warning("classifier.sklearn_fallback", detail=str(exc))
                predicted_style = _infer_style_keyword(text)
                confidence = 0.55
        else:
            predicted_style = _infer_style_keyword(text)
            confidence = 0.55

        known_styles = set(df["travel_style"].astype(str).unique())
        if predicted_style not in known_styles:
            predicted_style = _infer_style_keyword(text)

        pool = df[df["travel_style"] == predicted_style].copy()
        if pool.empty:
            pool = df.copy()

        def score_row(r: pd.Series) -> float:
            cost = float(r["cost_per_day_avg_usd"])
            target_daily = None
            if duration_days and budget_usd:
                target_daily = budget_usd / float(duration_days)
            elif budget_usd:
                target_daily = budget_usd / 7.0
            pen = 0.0
            if target_daily is not None:
                pen += abs(cost - target_daily) / max(target_daily, 1.0)
            return -pen

        pool["_rank"] = pool.apply(score_row, axis=1)
        pool = pool.sort_values("_rank", ascending=False)

        picks = pool.head(top_k)
        out_rows = picks[
            [
                "destination_city",
                "country",
                "travel_style",
                "cost_per_day_avg_usd",
                "flight_cost_usd",
            ]
        ].to_dict(orient="records")

        ms = int((time.perf_counter() - t0) * 1000)
        boosts = _keyword_boosts(text)
        boost_keys = sorted(boosts.keys(), key=lambda k: float(boosts[k]), reverse=True)[:3]
        payload = {
            "travel_style": predicted_style,
            "confidence": round(confidence, 4),
            "destinations": out_rows,
            "duration_ms": ms,
            "signal_features": boost_keys,
        }
        logger.info("tool.classify_destinations.ok", style=predicted_style, n=len(out_rows))
        return ToolEnvelope(ok=True, tool=tool_name, payload=payload, error=None)
    except Exception as exc:
        logger.exception("tool.classify_destinations.error")
        return ToolEnvelope(
            ok=False,
            tool=tool_name,
            payload=None,
            error=ToolError(tool=tool_name, error="classification_failed", detail=str(exc)),
        )
