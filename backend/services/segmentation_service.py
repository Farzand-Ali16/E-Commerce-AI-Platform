"""
Customer segment prediction service.

- Loads the SAVED preprocessing pipeline and the SAVED K-Means model once.
- Never trains or modifies anything. Every request only calls .transform() / .predict().
- Does NOT re-implement preprocessing: the saved ClusteringPreprocessor does impute -> transform -> clip -> scale.
"""
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# The saved preprocessor was pickled from src/preprocessing/clustering_preprocessor.py,
# so the project root must be importable to load it.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("segmentation_service")

DEFAULT_MODEL_PATH = ROOT / "models" / "customer_segmentation_kmeans_core5.joblib"
DEFAULT_PREPROCESSOR_PATH = ROOT / "models" / "cluster_preprocessor_core5.joblib"
DEFAULT_METADATA_PATH = ROOT / "models" / "segment_metadata_core5.json"

FEATURES: List[str] = ["Total_Orders", "Average_Order_Value", "Days_Since_Last_Order",
                       "Average_Days_Between_Orders", "Preferred_Category_Share"]

# Margin bands. Rough bands taken from the 5,000 training customers:
# below 0.10 is about the lowest 20% of margins, 0.25 is about the median.
BORDERLINE_BELOW = 0.10
CLEAR_FROM = 0.25

# Range of each feature among the training customers (model card). Used only to warn about extrapolation.
TRAINING_RANGE = {
    "Total_Orders": (1, 10),
    "Average_Order_Value": (14.28, 25314.08),
    "Days_Since_Last_Order": (1, 450),
    "Average_Days_Between_Orders": (0, 444),
    "Preferred_Category_Share": (0.14, 1.0),
}

# Business labels approved after reading the Stage 5.1 cluster profiles (cluster ids are numbered by mean orders).
# "typical" numbers come from data/processed/cluster_profiles_core5.csv (checked by the tests).
SEGMENT_CATALOG: Dict[int, Dict[str, Any]] = {
    0: {
        "label": "Loyal & Active",
        "description": ("Frequent buyers who ordered recently. A typical customer has about 6 orders, ordered "
                        "about 50 days ago and orders every ~70 days. About one third of customers, and the "
                        "source of roughly 59% of revenue."),
        "suggested_actions": ["Reward loyalty (early access, loyalty tier).",
                              "Cross-sell other categories.",
                              "Avoid unnecessary discounting: these customers already buy often."],
        "typical": {"customers_pct": 32.7, "avg_orders": 5.82, "avg_order_value": 1391.0,
                    "avg_days_since_last_order": 50.0, "avg_days_between_orders": 70.6, "revenue_share_pct": 59.2},
    },
    1: {
        "label": "Lapsed Repeat Buyers",
        "description": ("Repeat customers who used to order roughly every two months but have been quiet for about "
                        "3 times their usual gap (typically ~200 days since the last order). The main win-back "
                        "group, about 19% of revenue."),
        "suggested_actions": ["Run a win-back campaign with a personal reminder.",
                              "Offer an incentive on the category they bought before.",
                              "Ask why they stopped ordering."],
        "typical": {"customers_pct": 23.3, "avg_orders": 2.90, "avg_order_value": 1233.0,
                    "avg_days_since_last_order": 203.7, "avg_days_between_orders": 60.6, "revenue_share_pct": 18.7},
    },
    2: {
        "label": "Slow-Cycle Repeat Buyers",
        "description": ("Repeat customers with long natural gaps between orders (about 190 days) who are still "
                        "inside their normal cycle (last order ~80 days ago). Not at risk yet."),
        "suggested_actions": ["Time reminders to their own ordering rhythm.",
                              "Do not treat them as lapsed too early.",
                              "Encourage a slightly shorter cycle with a replenishment offer."],
        "typical": {"customers_pct": 23.9, "avg_orders": 2.54, "avg_order_value": 1196.0,
                    "avg_days_since_last_order": 79.5, "avg_days_between_orders": 190.1, "revenue_share_pct": 16.1},
    },
    3: {
        "label": "One-Time / Mostly One-Time Customers",
        "description": ("Customers with a single order (about 89% of this segment). Some are new customers who have "
                        "not had time to reorder, others bought once and never returned. A few repeat customers "
                        "who bought everything in one category also land here by coincidence."),
        "suggested_actions": ["Check the order date: a recent first order means a new customer, not a lost one.",
                              "Send a second-order incentive or onboarding message.",
                              "Look at the order count before treating them as one-time buyers."],
        "typical": {"customers_pct": 20.2, "avg_orders": 1.13, "avg_order_value": 1168.0,
                    "avg_days_since_last_order": 213.1, "avg_days_between_orders": 104.5, "revenue_share_pct": 6.0},
    },
}


# ----------------------------------------------------------------------------- errors
class SegmentationError(Exception):
    """Base class for service errors."""


class ModelNotLoadedError(SegmentationError):
    """Raised when a prediction is requested but the artifacts are not loaded/valid."""


class PredictionError(SegmentationError):
    """Raised when the saved pipeline fails on a request."""


# ----------------------------------------------------------------------------- service
class SegmentationService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
                 metadata_path: Path = DEFAULT_METADATA_PATH):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.metadata_path = Path(metadata_path)
        self._reset()

    def _reset(self):
        self._model = None
        self._preprocessor = None
        self._metadata: Optional[dict] = None
        self._consistent = False
        self._errors: List[str] = []
        self._notes: List[str] = []

    # ---- loading
    def load(self) -> "SegmentationService":
        """Load the three saved files and check they belong together. Never raises: problems are recorded
        and reported by health(), so the API can still start and answer /health."""
        self._reset()
        self._model = self._load_artifact(self.model_path, "model", joblib.load)
        self._preprocessor = self._load_artifact(self.preprocessor_path, "preprocessor", joblib.load)
        self._metadata = self._load_artifact(self.metadata_path, "metadata", self._read_json)
        if self._model is not None and self._preprocessor is not None and self._metadata is not None:
            self._check_consistency()
        if self.is_ready:
            logger.info("Segmentation service ready (K=%s)", self._model.n_clusters)
        else:
            logger.error("Segmentation service NOT ready: %s", "; ".join(self._errors))
        return self

    @staticmethod
    def _read_json(path: Path) -> dict:
        import json
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _load_artifact(self, path: Path, name: str, loader):
        if not path.exists():
            self._errors.append(f"{name} file not found: {path.name}")
            return None
        try:
            return loader(path)
        except Exception as exc:  # corrupted file, version mismatch, missing module ...
            self._errors.append(f"{name} could not be loaded ({type(exc).__name__}: {exc})")
            logger.exception("Failed to load %s", name)
            return None

    def _check_consistency(self):
        """Make sure the model, preprocessor, metadata and label catalog all agree."""
        problems = []
        pre_features = list(getattr(self._preprocessor, "features", []))
        if pre_features != FEATURES:
            problems.append(f"preprocessor features {pre_features} differ from expected {FEATURES}")
        if self._metadata.get("features") != FEATURES:
            problems.append("metadata features differ from expected features")
        k = getattr(self._model, "n_clusters", None)
        centers = getattr(self._model, "cluster_centers_", None)
        if centers is None or centers.shape != (k, len(FEATURES)):
            problems.append("model cluster centres do not match the 5 features")
        if k != self._metadata.get("selected_k") or k != len(SEGMENT_CATALOG):
            problems.append(f"model has K={k}, metadata K={self._metadata.get('selected_k')}, "
                            f"label catalog has {len(SEGMENT_CATALOG)} segments")
        meta_ids = sorted(int(i) for i in self._metadata.get("cluster_labels", {}))
        if meta_ids != sorted(SEGMENT_CATALOG):
            problems.append("metadata cluster ids differ from the label catalog")
        if problems:
            self._errors.extend(problems)
            return
        # Metadata labels were written in Stage 5.1; the catalog holds the approved final names.
        for i, meta_label in self._metadata["cluster_labels"].items():
            used = SEGMENT_CATALOG[int(i)]["label"]
            if meta_label != used:
                self._notes.append(f"Cluster {i}: metadata label '{meta_label}' is overridden by approved label '{used}'.")
        self._consistent = True

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._preprocessor is not None and self._metadata is not None and self._consistent

    def health(self) -> Dict[str, Any]:
        ready = self.is_ready
        return {
            "status": "ok" if ready else "unavailable",
            "model_loaded": self._model is not None,
            "preprocessor_loaded": self._preprocessor is not None,
            "metadata_loaded": self._metadata is not None,
            "consistency_checked": self._consistent,
            "n_clusters": int(self._model.n_clusters) if self._model is not None else None,
            "features": list(FEATURES) if ready else [],
            "segments": {str(i): v["label"] for i, v in SEGMENT_CATALOG.items()} if ready else {},
            "errors": list(self._errors),
            "notes": list(self._notes),
        }

    # ---- prediction
    def _require_ready(self):
        if not self.is_ready:
            raise ModelNotLoadedError("Segmentation model is not available: " + "; ".join(self._errors or ["not loaded"]))

    @staticmethod
    def _to_frame(records: List[Mapping[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(list(records), columns=FEATURES)
        return df.astype(float)          # None -> NaN (the saved preprocessor fills the gap of single-order customers)

    def _score(self, df: pd.DataFrame):
        """The only place where data meets the models: saved preprocessor -> saved K-Means (no fitting)."""
        scaled = self._preprocessor.transform(df)
        arr = scaled.to_numpy()
        if not np.isfinite(arr).all():
            raise PredictionError("Preprocessing produced non-finite values.")
        clusters = self._model.predict(arr)
        distances = self._model.transform(arr)
        return arr, clusters, distances

    def predict_clusters(self, df: pd.DataFrame) -> np.ndarray:
        """Batch helper (used by the tests): cluster ids for a DataFrame with the 5 feature columns."""
        self._require_ready()
        try:
            return self._score(df[FEATURES].astype(float))[1]
        except SegmentationError:
            raise
        except Exception as exc:
            raise PredictionError(f"Prediction failed: {exc}") from exc

    def predict(self, features: Mapping[str, Any]) -> Dict[str, Any]:
        """Predict the segment of ONE customer. `features` must contain the 5 features
        (validated before by the Pydantic request model)."""
        self._require_ready()
        missing = [f for f in FEATURES if f not in features]
        if missing:
            raise PredictionError(f"Missing features: {missing}")
        try:
            df = self._to_frame([{f: features[f] for f in FEATURES}])
            arr, clusters, distances = self._score(df)
            cluster = int(clusters[0])
            dist = distances[0]
            order = np.argsort(dist)
            d1, d2 = float(dist[order[0]]), float(dist[order[1]])
            if int(order[0]) != cluster:      # cannot happen; safety net
                raise PredictionError("Nearest centre and predicted cluster disagree.")
            margin = (d2 - d1) / (d1 + d2) if (d1 + d2) > 0 else 0.0
            strength = "borderline" if margin < BORDERLINE_BELOW else ("clear" if margin >= CLEAR_FROM else "moderate")
            runner_up = int(order[1])
        except SegmentationError:
            raise
        except Exception as exc:
            logger.exception("Prediction failed")
            raise PredictionError(f"Prediction failed: {exc}") from exc

        info = SEGMENT_CATALOG[cluster]
        n_orders = int(features["Total_Orders"])
        return {
            "cluster": cluster,
            "segment": info["label"],
            "description": info["description"],
            "suggested_actions": list(info["suggested_actions"]),
            "is_single_order_customer": n_orders == 1,
            "input_features": {f: (None if pd.isna(features[f]) else float(features[f])) for f in FEATURES},
            "assignment": {
                "margin": round(margin, 4),
                "strength": strength,
                "closest_alternative": {"cluster": runner_up, "segment": SEGMENT_CATALOG[runner_up]["label"]},
                "distances": {SEGMENT_CATALOG[i]["label"]: round(float(dist[i]), 4) for i in range(len(dist))},
            },
            "segment_profile": dict(info["typical"]),
            "warnings": self._warnings(features, cluster, strength, runner_up),
        }

    @staticmethod
    def _warnings(features, cluster, strength, runner_up) -> List[str]:
        out = []
        for f, (lo, hi) in TRAINING_RANGE.items():
            v = features[f]
            if v is not None and not pd.isna(v) and (v < lo or v > hi):
                out.append(f"{f}={v:g} is outside the range seen in training ({lo:g} to {hi:g}); "
                           f"the prediction is an extrapolation.")
        if cluster == 3 and int(features["Total_Orders"]) > 1:
            out.append("This customer has more than one order but was placed in the one-time segment because all "
                       "orders are in the same category (Preferred_Category_Share is high). Treat them as a repeat "
                       "customer.")
        if strength == "borderline":
            out.append(f"Borderline assignment: this customer is almost as close to "
                       f"'{SEGMENT_CATALOG[runner_up]['label']}'. Use the segment as indicative only.")
        return out


@lru_cache(maxsize=1)
def get_segmentation_service() -> SegmentationService:
    """One shared service for the whole app (created here, loaded once at API start-up)."""
    return SegmentationService()
