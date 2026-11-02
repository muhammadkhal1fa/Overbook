
"""
Hotel Booking Dataset - Cleaning & Statistical Reporting

Run from project root:
python -m src.data.cleaning --data data/H2.csv --out outputs/01_cleaning
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# 1. CONFIGURATION
# =====================================================

EMPTY_SPELLINGS = {"NULL", "NA", "N/A", "nan", "None", ""}
UNDEFINED_COLUMNS = [
    "MarketSegment",
    "DistributionChannel",
    "Meal"
]


# =====================================================
# 2. LOAD AND FORMAT DATA
# =====================================================

def load_h2(path):

    df = pd.read_csv(path)

    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(df[col]):

            s = df[col].astype(str).str.strip()

            empty = s.isin(EMPTY_SPELLINGS)

            if col in UNDEFINED_COLUMNS:
                empty |= s.eq("Undefined")

            df[col] = s.mask(empty)

    return df


# =====================================================
# 3. COMMON CLEANING
# =====================================================

def clean_common(df):

    report = {
        "rows_raw": len(df),
        "columns_raw": len(df.columns)
    }

    df = df.copy()

    # Children: missing values represent zero children
    missing_children = df["Children"].isna().sum()
    df["Children"] = df["Children"].fillna(0)

    report["children_missing_filled"] = int(missing_children)

    # Remove exact duplicates
    before = len(df)

    df = df.drop_duplicates()

    report["duplicates_removed"] = before - len(df)
    report["rows_after_duplicates"] = len(df)

    # Remove bookings with zero guests
    guests = (
        df["Adults"] +
        df["Children"] +
        df["Babies"]
    )

    zero_guests = guests <= 0

    report["zero_guest_rows_removed"] = int(zero_guests.sum())

    df = df[~zero_guests].copy()

    # Remove bookings with zero nights
    nights = (
        df["StaysInWeekendNights"] +
        df["StaysInWeekNights"]
    )

    zero_nights = nights <= 0

    report["zero_night_rows_removed"] = int(
        zero_nights.sum()
    )

    df = df[~zero_nights].copy()

    report["rows_after_invalid_removed"] = len(df)

    return df.reset_index(drop=True), report


# =====================================================
# 4. REGRESSION-SPECIFIC CLEANING
# =====================================================

def clean_for_regression(df, max_adr=1000):

    report = {
        "rows_before_adr_filter": len(df)
    }

    df = df.copy()

    zero_adr = df["ADR"] <= 0
    extreme_adr = df["ADR"] >= max_adr

    report["zero_or_negative_adr_removed"] = int(
        zero_adr.sum()
    )

    report["extreme_adr_removed"] = int(
        extreme_adr.sum()
    )

    df = df[
        (df["ADR"] > 0) &
        (df["ADR"] < max_adr)
    ].copy()

    report["rows_for_regression"] = len(df)

    return df.reset_index(drop=True), report


# =====================================================
# 5. STATISTICAL ANALYSIS
# =====================================================

def calculate_statistics(df):

    # Numerical statistics
    numeric_stats = df.select_dtypes(
        include=np.number
    ).describe().T

    numeric_stats["missing"] = df[
        numeric_stats.index
    ].isna().sum()

    numeric_stats["missing_percent"] = (
        numeric_stats["missing"] / len(df) * 100
    )

    # Categorical statistics
    categorical_rows = []

    for col in df.select_dtypes(
        include=["object", "category"]
    ).columns:

        categorical_rows.append({
            "Feature": col,
            "Unique Values": df[col].nunique(),
            "Missing Values": df[col].isna().sum(),
            "Missing %": round(
                df[col].isna().mean() * 100, 2
            ),
            "Most Frequent": (
                str(df[col].mode().iloc[0])
                if not df[col].mode().empty
                else "None"
            ),
            "Most Frequent Count": (
                int(df[col].value_counts().iloc[0])
                if not df[col].value_counts().empty
                else 0
            )
        })

    categorical_stats = pd.DataFrame(
        categorical_rows
    )

    # ADR statistics
    adr_stats = df["ADR"].describe()

    return (
        numeric_stats,
        categorical_stats,
        adr_stats
    )


# =====================================================
# 6. VISUALIZATIONS
# =====================================================

def generate_visualizations(df, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # ADR distribution
    plt.figure(figsize=(10, 6))

    plt.hist(
        df["ADR"],
        bins=50,
        edgecolor="black"
    )

    plt.title("ADR Distribution After Cleaning")
    plt.xlabel("ADR")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "adr_distribution.png"),
        dpi=150
    )

    plt.close()

    # ADR boxplot
    plt.figure(figsize=(9, 4))

    plt.boxplot(
        df["ADR"],
        vert=False
    )

    plt.title("ADR Boxplot After Cleaning")
    plt.xlabel("ADR")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "adr_boxplot.png"),
        dpi=150
    )

    plt.close()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    plt.figure(figsize=(13, 10))

    plt.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Numerical Feature Correlation")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "correlation_heatmap.png"),
        dpi=150
    )

    plt.close()


# =====================================================
# 7. GENERATE HTML STATISTICAL REPORT
# =====================================================

def generate_html_report(
    raw_df,
    clean_df,
    cleaning_report,
    numeric_stats,
    categorical_stats,
    adr_stats,
    output_dir
):

    raw_missing = raw_df.isna().sum()

    clean_missing = clean_df.isna().sum()

    missing_comparison = pd.DataFrame({
        "Feature": raw_df.columns,
        "Missing Before": raw_missing.values,
        "Missing After": clean_missing.reindex(
            raw_df.columns
        ).values
    })

    numeric_html = numeric_stats.round(3).to_html(
        classes="data-table",
        border=0
    )

    categorical_html = categorical_stats.to_html(
        index=False,
        classes="data-table",
        border=0
    )

    missing_html = missing_comparison.to_html(
        index=False,
        classes="data-table",
        border=0
    )

    cleaning_html = pd.DataFrame(
        list(cleaning_report.items()),
        columns=["Cleaning Operation", "Result"]
    ).to_html(
        index=False,
        classes="data-table",
        border=0
    )

    adr_html = adr_stats.to_frame(
        name="ADR"
    ).round(3).to_html(
        classes="data-table",
        border=0
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>

    <meta charset="UTF-8">

    <title>Hotel Booking Cleaning Report</title>

    <style>

    body {{
        font-family: Arial, sans-serif;
        margin: 35px;
        background: #f5f6fa;
        color: #252525;
    }}

    h1 {{
        color: #183153;
    }}

    h2 {{
        color: #245b87;
        margin-top: 35px;
    }}

    .card {{
        background: white;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px #ddd;
    }}

    .data-table {{
        border-collapse: collapse;
        width: 100%;
        background: white;
        font-size: 13px;
    }}

    .data-table th,
    .data-table td {{
        border: 1px solid #ddd;
        padding: 9px;
        text-align: left;
    }}

    .data-table th {{
        background: #245b87;
        color: white;
    }}

    .data-table tr:nth-child(even) {{
        background: #f2f5f8;
    }}

    img {{
        max-width: 100%;
        margin: 15px 0;
        border-radius: 8px;
    }}

    </style>
    </head>

    <body>

    <h1>Hotel Booking Dataset</h1>
    <h2>Cleaning and Statistical Analysis Report</h2>

    <div class="card">

    <h2>1. Dataset Overview</h2>

    <p><b>Original rows:</b> {len(raw_df):,}</p>

    <p><b>Cleaned rows:</b> {len(clean_df):,}</p>

    <p><b>Original columns:</b> {len(raw_df.columns)}</p>

    <p><b>Removed rows:</b> {len(raw_df) - len(clean_df):,}</p>

    <p><b>Remaining percentage:</b>
    {len(clean_df) / len(raw_df) * 100:.2f}%</p>

    </div>

    <div class="card">

    <h2>2. Cleaning Operations</h2>

    {cleaning_html}

    </div>

    <div class="card">

    <h2>3. Missing Values Comparison</h2>

    {missing_html}

    </div>

    <div class="card">

    <h2>4. Numerical Statistical Analysis</h2>

    {numeric_html}

    </div>

    <div class="card">

    <h2>5. Categorical Statistical Analysis</h2>

    {categorical_html}

    </div>

    <div class="card">

    <h2>6. ADR Statistical Analysis</h2>

    {adr_html}

    <p>
    Mean: {adr_stats['mean']:.3f}<br>
    Median: {adr_stats['50%']:.3f}<br>
    Standard deviation: {adr_stats['std']:.3f}<br>
    Minimum: {adr_stats['min']:.3f}<br>
    Maximum: {adr_stats['max']:.3f}
    </p>

    </div>

    <div class="card">

    <h2>7. ADR Distribution</h2>

    <img src="adr_distribution.png">

    <h2>8. ADR Outlier Analysis</h2>

    <img src="adr_boxplot.png">

    </div>

    <div class="card">

    <h2>9. Feature Correlations</h2>

    <img src="correlation_heatmap.png">

    </div>

    <p>Generated automatically using Python.</p>

    </body>
    </html>
    """

    report_path = os.path.join(
        output_dir,
        "cleaning_statistical_report.html"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path


# =====================================================
# 8. MAIN EXECUTION
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/H2.csv"
    )

    parser.add_argument(
        "--out",
        default="outputs/01_cleaning"
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\nLoading original dataset...")

    raw_df = load_h2(args.data)

    print("Original shape:", raw_df.shape)

    print("\nCleaning dataset...")

    df, common_report = clean_common(raw_df)

    df, regression_report = clean_for_regression(df)

    cleaning_report = {
        **common_report,
        **regression_report
    }

    print("\nCleaning completed!")

    for operation, result in cleaning_report.items():
        print(f"{operation}: {result}")

    # Save cleaned dataset
    cleaned_path = os.path.join(
        args.out,
        "cleaned_regression_data.csv"
    )

    df.to_csv(cleaned_path, index=False)

    print("\nCleaned CSV saved:", cleaned_path)

    # Calculate statistics
    print("\nCalculating statistical information...")

    numeric_stats, categorical_stats, adr_stats = (
        calculate_statistics(df)
    )

    # Save statistical CSV files
    numeric_stats.to_csv(
        os.path.join(args.out, "numeric_statistics.csv")
    )

    categorical_stats.to_csv(
        os.path.join(args.out, "categorical_statistics.csv"),
        index=False
    )

    pd.DataFrame(
        list(cleaning_report.items()),
        columns=["Operation", "Result"]
    ).to_csv(
        os.path.join(args.out, "cleaning_summary.csv"),
        index=False
    )

    # Generate charts
    print("\nGenerating visualizations...")

    generate_visualizations(df, args.out)

    # Generate HTML report
    report_path = generate_html_report(
        raw_df,
        df,
        cleaning_report,
        numeric_stats,
        categorical_stats,
        adr_stats,
        args.out
    )

    print("\n" + "=" * 55)
    print("CLEANING AND REPORTING COMPLETED")
    print("=" * 55)

    print("Final dataset:", df.shape)
    print("Report:", report_path)
    print("Output folder:", args.out)


if __name__ == "__main__":
    main()
