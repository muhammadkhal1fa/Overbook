"""
Preprocessing pipeline for Task A (regression, target = ADR).

PREDICTION POINT: the moment the booking is confirmed in the PMS.
Only columns that are known at that moment are used. The builder keeps a
WHITELIST of source columns; every other column (IsCanceled, ReservationStatus,
ReservationStatusDate, AssignedRoomType, BookingChanges, TotalOfSpecialRequests,
RequiredCarParkingSpaces, DepositType, ArrivalDateDayOfMonth) is ignored even if
it is present in the input. Ignoring by whitelist means a leaking column can never
sneak in by accident.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Source columns (exact labels from the spec) that the regression model needs.
RAW_INPUT_COLUMNS = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",  # only used to compute the weekday of arrival
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "Meal", "Country", "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "Agent", "Company", "DaysInWaitingList", "CustomerType",
]

# Columns that exist in the CSV but are NOT allowed as regression inputs.
EXCLUDED_COLUMNS = {
    "IsCanceled": "outcome, unknown at booking time",
    "ReservationStatus": "outcome, unknown at booking time",
    "ReservationStatusDate": "date the outcome happened",
    "AssignedRoomType": "assigned at check-in, not known at booking",
    "BookingChanges": "counted up to check-in/cancellation",
    "TotalOfSpecialRequests": "accumulates after booking",
    "RequiredCarParkingSpaces": "accumulates after booking",
    "DepositType": "explains <1% of price; mostly a cancellation marker",
}

NUMERIC = ["lead_time_sqrt", "adults", "total_kids", "weekend_nights",
           "week_nights", "arrival_year", "week_sin", "week_cos",
           "prev_cancel_log", "prev_kept_log"]
BINARY = ["is_repeated_guest", "was_waitlisted", "has_company"]
CATEGORICAL_LOW = ["arrival_month", "arrival_weekday", "meal", "market_segment",
                   "distribution_channel", "reserved_room_type", "customer_type"]
FEATURE_COLUMNS = NUMERIC + BINARY + CATEGORICAL_LOW + ["country", "agent"]


class RegressionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless feature engineering (fixed formulas, nothing learned).

    Takes a DataFrame with the raw source columns and returns the engineered
    features. Because it learns nothing it cannot leak, and because it lives in
    the pipeline the app can feed it raw booking fields.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype(str).str.strip()

        out = pd.DataFrame(index=X.index)

        # Lead time: right-skewed (1.57). sqrt pulls the long tail in without
        # overshooting (log would make it left-skewed).
        out["lead_time_sqrt"] = np.sqrt(X["LeadTime"].astype(float))

        out["adults"] = X["Adults"].astype(float)
        # Babies are 0.7% of rows -> too few to learn alone; merge with children.
        out["total_kids"] = X["Children"].fillna(0).astype(float) + X["Babies"].astype(float)
        out["weekend_nights"] = X["StaysInWeekendNights"].astype(float)
        out["week_nights"] = X["StaysInWeekNights"].astype(float)
        out["arrival_year"] = X["ArrivalDateYear"].astype(float)

        # Week number is circular: week 53 sits next to week 1.
        angle = 2 * np.pi * X["ArrivalDateWeekNumber"].astype(float) / 53.0
        out["week_sin"] = np.sin(angle)
        out["week_cos"] = np.cos(angle)

        # Previous bookings: 97% zeros with a few huge values -> log1p.
        out["prev_cancel_log"] = np.log1p(X["PreviousCancellations"].astype(float))
        out["prev_kept_log"] = np.log1p(X["PreviousBookingsNotCanceled"].astype(float))

        out["is_repeated_guest"] = X["IsRepeatedGuest"].astype(int)
        # Waiting list: 98.7% zeros, so a yes/no flag carries everything usable.
        out["was_waitlisted"] = (X["DaysInWaitingList"] > 0).astype(int)
        # Company is 95% empty and has ~200 ids -> only "was a company involved".
        out["has_company"] = (~X["Company"].astype(str).str.strip().isin(["NULL", "nan", "None", ""])).astype(int)

        month_no = X["ArrivalDateMonth"].map({m: i + 1 for i, m in enumerate(MONTHS)})
        out["arrival_month"] = X["ArrivalDateMonth"].astype(str)
        date = pd.to_datetime(pd.DataFrame({"year": X["ArrivalDateYear"].astype(int),
                                            "month": month_no.astype(int),
                                            "day": X["ArrivalDateDayOfMonth"].astype(int)}))
        out["arrival_weekday"] = date.dt.dayofweek.astype(str)

        for src, dst in [("Meal", "meal"), ("MarketSegment", "market_segment"),
                         ("DistributionChannel", "distribution_channel"),
                         ("ReservedRoomType", "reserved_room_type"),
                         ("CustomerType", "customer_type"), ("Country", "country")]:
            # real NaN (Country 24 rows, Undefined segment/channel) -> explicit 'Unknown'
            out[dst] = X[src].fillna("Unknown").astype(str)

        # 'NULL' agent is real information (booked without an agent) -> own category.
        agent = X["Agent"].astype(str).str.replace(r"\.0$", "", regex=True)
        out["agent"] = agent.replace({"NULL": "No agent", "nan": "No agent", "None": "No agent"})

        return out[FEATURE_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_COLUMNS, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Keep categories that appear in >= min_freq of the TRAINING rows; everything
    else (and anything never seen in training) becomes 'Other'.

    This learns from data, so it must be fitted on training rows only -
    which is automatic when it sits inside the Pipeline.
    """

    def __init__(self, min_freq=0.01, other_label="Other"):
        self.min_freq = min_freq
        self.other_label = other_label

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.keep_ = {}
        for c in self.columns_:
            freq = X[c].astype(str).value_counts(normalize=True)
            self.keep_[c] = set(freq[freq >= self.min_freq].index)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            s = X[c].astype(str)
            X[c] = s.where(s.isin(self.keep_[c]), self.other_label)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)


def build_preprocessor(country_min_freq=0.01, agent_min_freq=0.005):
    """features -> (scale numbers | pass flags | one-hot categories)."""
    def grouped_ohe(min_freq):
        return Pipeline([
            ("group", RareCategoryGrouper(min_freq=min_freq)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

    encode = ColumnTransformer(sparse_threshold=0, transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_LOW),
        ("country", grouped_ohe(country_min_freq), ["country"]),
        ("agent", grouped_ohe(agent_min_freq), ["agent"]),
    ])
    return Pipeline([("features", RegressionFeatureBuilder()), ("encode", encode)])
