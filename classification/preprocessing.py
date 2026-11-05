"""Preprocessing for Task B: predict ``IsCanceled`` on the City Hotel (H2) data.

This module is intentionally self-contained so the Overbook repo does not need the
larger CSE382 package. It follows the same rules as the Overbook EDA script:

* read H2.csv as strings, strip padding, and turn ``NULL``/``N/A`` into real missing;
* drop outcome leakage (``ReservationStatus``, ``ReservationStatusDate``) and
  post-booking mutation fields (``BookingChanges``, ``DaysInWaitingList``,
  ``AssignedRoomType``);
* keep missingness informative (median impute numerics, explicit ``Missing`` level
  for categoricals);
* fit every data-dependent transform on the training split only (the transforms live
  inside the sklearn Pipeline).

Classification only. The regression target ``ADR`` is never used here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "IsCanceled"

SOURCE_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]

NUMERIC_SOURCE = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "IsRepeatedGuest", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "BookingChanges", "Agent", "Company",
    "DaysInWaitingList", "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

NA_MARKERS = {"NULL", "N/A", "NA", "NAN", "NONE", "?", "-", ""}

# Columns that must never enter a cancellation model.
BANNED = {
    "ReservationStatus",          # perfect function of the label
    "ReservationStatusDate",      # records when the outcome was set
    "BookingChanges",             # post-booking mutation
    "DaysInWaitingList",          # post-booking mutation
    "AssignedRoomType",           # assigned at check-in
    "ArrivalDateWeekNumber",      # 53.8% inconsistent with the true arrival date
    "Company",                    # 95.35% missing
    "IsCanceled",                 # the target itself
    "SourceRowNumber",            # row id
}

FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "Meal", "Country", "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "DepositType", "Agent", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ArrivalMonthNum", "TotalNights", "TotalGuests",
    "PreviousBookingsTotal", "ArrivalDayOfWeek", "ArrivalWeekISO",
    "IsWeekendArrival", "ArrivalMonthSin", "ArrivalMonthCos", "HasAgent", "HasCompany",
]

NUMERIC_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ADR",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests", "ArrivalMonthNum",
    "TotalNights", "TotalGuests", "PreviousBookingsTotal", "ArrivalDayOfWeek",
    "ArrivalWeekISO", "IsWeekendArrival", "ArrivalMonthSin", "ArrivalMonthCos",
    "HasAgent", "HasCompany",
]

CATEGORICAL_FEATURES = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "Agent", "CustomerType",
]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_leakage(columns) -> None:
    hit = set(columns) & BANNED
    if hit:
        raise AssertionError(f"leakage: banned column(s) selected -> {sorted(hit)}")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read H2.csv as strings so nothing is silently coerced."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(SOURCE_COLUMNS)
    if missing:
        raise ValueError(f"missing source columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"unexpected extra columns: {sorted(extra)}")
    return df


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Strip padding, convert missing markers, coerce types, parse the arrival date."""
    df = raw.copy()
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip()
    df = df.replace(list(NA_MARKERS), np.nan)

    for c in NUMERIC_SOURCE:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in (
        "IsCanceled", "IsRepeatedGuest", "Adults", "Children", "Babies",
        "ArrivalDateYear", "ArrivalDateDayOfMonth", "ArrivalDateWeekNumber",
    ):
        df[c] = df[c].astype("Int64")

    df["ArrivalMonthNum"] = df["ArrivalDateMonth"].map(MONTHS).astype("Int64")
    df["ArrivalDate"] = pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["ArrivalDateYear"].astype("float"),
                "month": df["ArrivalMonthNum"].astype("float"),
                "day": df["ArrivalDateDayOfMonth"].astype("float"),
            }
        ),
        errors="coerce",
    )
    df["ReservationStatusDate"] = pd.to_datetime(
        df["ReservationStatusDate"], errors="coerce"
    )

    df.insert(0, "SourceRowNumber", np.arange(len(df), dtype=np.int64))
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features alongside the source columns; never overwrite them."""
    out = df.copy()

    out["TotalNights"] = (
        out["StaysInWeekendNights"].fillna(0) + out["StaysInWeekNights"].fillna(0)
    ).astype("Int64")
    out["TotalGuests"] = (
        out["Adults"].fillna(0) + out["Children"].fillna(0) + out["Babies"].fillna(0)
    ).astype("Int64")
    out["PreviousBookingsTotal"] = (
        out["PreviousCancellations"].fillna(0)
        + out["PreviousBookingsNotCanceled"].fillna(0)
    ).astype("Int64")

    out["ArrivalDayOfWeek"] = out["ArrivalDate"].dt.dayofweek.astype("Int64")
    out["ArrivalWeekISO"] = out["ArrivalDate"].dt.isocalendar().week.astype("Int64")
    out["IsWeekendArrival"] = (out["ArrivalDayOfWeek"] >= 5).astype("Int64")

    angle = 2 * np.pi * (out["ArrivalMonthNum"].astype("float") - 1) / 12.0
    out["ArrivalMonthSin"] = np.sin(angle)
    out["ArrivalMonthCos"] = np.cos(angle)

    out["HasAgent"] = out["Agent"].notna().astype("Int64")
    out["HasCompany"] = out["Company"].notna().astype("Int64")
    return out


def load_classification_frame(path: str | Path) -> pd.DataFrame:
    """Return the 34-feature classification frame plus ``IsCanceled``."""
    df = engineer(normalise(load_raw(path)))
    df = df.loc[df[TARGET].notna()].reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"engineered feature(s) missing: {missing}")
    assert_no_leakage(FEATURES)
    return df[FEATURES + [TARGET]].copy()


def prepare_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either engineered feature columns or a raw H2-style booking row.

    The raw path is what a user gets from ``H2.csv``. The target is not used at
    inference, so a missing ``IsCanceled`` column is filled with a placeholder.
    """
    df = pd.DataFrame(frame)
    source_present = set(SOURCE_COLUMNS) - {"IsCanceled"} <= set(df.columns)
    if source_present and TARGET not in df.columns:
        df = df.copy()
        df[TARGET] = 0
        source_present = True

    if set(SOURCE_COLUMNS).issubset(df.columns):
        engineered = engineer(normalise(df))
        missing = [c for c in FEATURES if c not in engineered.columns]
        if missing:
            raise ValueError(f"engineered feature(s) missing: {missing}")
        return engineered[FEATURES]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "input needs either the 31 raw H2 columns or the 34 engineered "
            f"feature columns; missing: {missing}"
        )
    return df[FEATURES].copy()


def _to_string(X: Any) -> pd.DataFrame:
    """Cast categoricals to Python strings with an explicit ``Missing`` level.

    ``astype("string")`` yields ``pd.NA``; scikit-learn's missing-value check compares
    ``X != X`` and cannot evaluate ``pd.NA`` in a boolean context. Filling here also
    preserves the rule that missingness is an informative category.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X.astype("string").fillna("Missing").astype(object)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("to_string", FunctionTransformer(_to_string, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )
