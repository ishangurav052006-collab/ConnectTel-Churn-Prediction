"""Verify IBM Telco Customer Churn dataset health and integrity."""

import os
import sys
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Telco-Customer-Churn.csv")


def verify_dataset(path: str = DATA_PATH):
    print("=" * 70)
    print("CONNECTTEL CUSTOMER CHURN - DATASET HEALTH CHECK")
    print("=" * 70)

    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at {path}")
        sys.exit(1)

    print(f"File path: {path}")
    print(f"File size: {os.path.getsize(path):,} bytes")

    df = pd.read_csv(path)
    rows, cols = df.shape
    print(f"\nDimensions: {rows} rows x {cols} columns")

    # Verify expected dimensions
    expected_rows, expected_cols = 7043, 21
    if rows == expected_rows and cols == expected_cols:
        print(f" [OK] Dimensions strictly match expectation ({expected_rows} rows, {expected_cols} columns).")
    else:
        print(f" [WARNING] Dimension mismatch! Expected ({expected_rows}, {expected_cols}), got ({rows}, {cols}).")

    print("\n--- Column Overview & Data Types ---")
    for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
        sample_val = df[col].iloc[0]
        print(f"{idx:2d}. {col:25s} | dtype: {str(dtype):10s} | Sample: {repr(sample_val)}")

    print("\n--- Missing / Blank Value Analysis ---")
    # Check explicit nulls
    null_counts = df.isnull().sum()
    # Check for empty strings or whitespace in object columns
    blank_counts = {}
    for col in df.select_dtypes(include=["object"]).columns:
        blanks = (df[col].astype(str).str.strip() == "").sum()
        if blanks > 0:
            blank_counts[col] = blanks

    explicit_nulls = null_counts[null_counts > 0]
    if len(explicit_nulls) == 0 and len(blank_counts) == 0:
        print(" [OK] No explicit nulls or whitespace blanks found.")
    else:
        if len(explicit_nulls) > 0:
            print(" Explicit nulls found:")
            for col, cnt in explicit_nulls.items():
                print(f"   - {col}: {cnt} null values")
        if len(blank_counts) > 0:
            print(" Whitespace blanks (potential hidden nulls, e.g. TotalCharges):")
            for col, cnt in blank_counts.items():
                print(f"   - {col}: {cnt} blank strings")

    print("\n--- Target Variable Distribution ('Churn') ---")
    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts()
        churn_pcts = df["Churn"].value_counts(normalize=True) * 100
        for val in churn_counts.index:
            print(f"   - {val:5s}: {churn_counts[val]:5d} ({churn_pcts[val]:.2f}%)")
    else:
        print(" [WARNING] Target column 'Churn' not found!")

    print("\n" + "=" * 70)
    print("Setup & Data Health Check Status: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    verify_dataset()

