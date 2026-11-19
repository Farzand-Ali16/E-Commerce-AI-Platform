"""
Pydantic models (request validation + response shape) for the segmentation API.

The request contains EXACTLY the 5 production features. Extra fields are rejected.
"""
import math
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The training data has 8 product categories. With n orders, the most common category
# must therefore appear at least ceil(n / 8) times. Used to reject impossible inputs.
N_PRODUCT_CATEGORIES = 8


class SegmentationRequest(BaseModel):
    """One customer, described by the 5 features the production model was trained on."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"Total_Orders": 6, "Average_Order_Value": 1400.0, "Days_Since_Last_Order": 45,
                 "Average_Days_Between_Orders": 70, "Preferred_Category_Share": 0.33},
                {"Total_Orders": 1, "Average_Order_Value": 1150.0, "Days_Since_Last_Order": 215,
                 "Average_Days_Between_Orders": None, "Preferred_Category_Share": 1.0},
            ]
        },
    )

    Total_Orders: int = Field(..., ge=1, le=1000, description="Number of orders the customer has placed (1 or more).")
    Average_Order_Value: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False,
                                       description="Average amount per order (total spending / orders).")
    Days_Since_Last_Order: float = Field(..., ge=0, le=10_000, allow_inf_nan=False,
                                         description="Days between the customer's last order and today.")
    Average_Days_Between_Orders: Optional[float] = Field(
        None, ge=0, le=10_000, allow_inf_nan=False,
        description="Average days between consecutive orders. Must be null/omitted when Total_Orders is 1.")
    Preferred_Category_Share: float = Field(
        ..., gt=0, le=1, allow_inf_nan=False,
        description="Share of the customer's orders that are in their most common product category (0 to 1).")

    @model_validator(mode="after")
    def check_consistency(self):
        n = self.Total_Orders
        gap = self.Average_Days_Between_Orders
        share = self.Preferred_Category_Share

        # A gap between orders needs at least two orders.
        if n == 1 and gap is not None:
            raise ValueError("Average_Days_Between_Orders must be null (or omitted) when Total_Orders is 1, "
                             "because a gap needs at least 2 orders.")
        if n > 1 and gap is None:
            raise ValueError("Average_Days_Between_Orders is required when Total_Orders is 2 or more.")

        # share = (orders in top category) / n, and the top category has at least ceil(n/8) orders.
        min_count = math.ceil(n / N_PRODUCT_CATEGORIES)
        counts = range(min_count, n + 1)
        nearest = min((c / n for c in counts), key=lambda p: abs(p - share))
        if abs(nearest - share) > 0.005 + 1e-9:        # 0.005 = allows rounding to 2 decimals
            if n <= 12:
                options = ", ".join(f"{c / n:.2f}" for c in counts)
                hint = f"Possible values with {n} order(s): {options}."
            else:
                hint = (f"It must be (orders in the top category) / {n}, with at least {min_count} orders "
                        f"in that category.")
            raise ValueError(f"Preferred_Category_Share={share} is not possible with Total_Orders={n}. {hint}")
        return self


# ----------------------------------------------------------------------------- responses
class SegmentProfile(BaseModel):
    """Typical customer of the segment (averages from the training customers)."""
    customers_pct: float
    avg_orders: float
    avg_order_value: float
    avg_days_since_last_order: float
    avg_days_between_orders: float
    revenue_share_pct: float


class AlternativeSegment(BaseModel):
    cluster: int
    segment: str


class Assignment(BaseModel):
    """K-Means gives no probabilities. This shows how clear the assignment is instead."""
    margin: float = Field(..., description="0 = exactly between two segments, close to 1 = very clear. "
                                           "(d2 - d1) / (d2 + d1) using distances to the two nearest centres.")
    strength: Literal["clear", "moderate", "borderline"]
    closest_alternative: AlternativeSegment
    distances: Dict[str, float] = Field(..., description="Distance to each segment centre in standardized space.")


class SegmentationResponse(BaseModel):
    cluster: int
    segment: str
    description: str
    suggested_actions: List[str]
    is_single_order_customer: bool
    input_features: Dict[str, Optional[float]]
    assignment: Assignment
    segment_profile: SegmentProfile
    warnings: List[str] = []


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    preprocessor_loaded: bool
    metadata_loaded: bool
    consistency_checked: bool
    n_clusters: Optional[int] = None
    features: List[str] = []
    segments: Dict[str, str] = {}
    errors: List[str] = []
    notes: List[str] = []


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[List[ErrorDetail]] = None
