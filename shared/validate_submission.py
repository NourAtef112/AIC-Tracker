"""Validate final competition submission CSV.

From roadmap.
"""

import pandas as pd
import argparse
import sys


def validate_submission(submission_path: str, sample_path: str = None):
    """Validate submission CSV format and content.

    Args:
        submission_path: Path to submission CSV
        sample_path: Path to sample_submission.csv (for format reference)

    Returns:
        True if valid, False otherwise
    """
    print(f"Validating submission: {submission_path}")

    # Load submission
    try:
        sub = pd.read_csv(submission_path)
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return False

    errors = []

    # Expected structure and size
    expected_rows = 74293
    expected_columns = ['id', 'x', 'y', 'w', 'h']

    # Check row count
    if len(sub) != expected_rows:
        errors.append(f"Row count mismatch: {len(sub)} (expected {expected_rows})")

    # Check columns
    missing_cols = set(expected_columns) - set(sub.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    extra_cols = set(sub.columns) - set(expected_columns)
    if extra_cols:
        errors.append(f"Extra columns: {extra_cols}")

    # Check for NaN values
    nan_count = sub[expected_columns].isnull().sum().sum()
    if nan_count > 0:
        errors.append(f"NaN values found: {nan_count}")

    # Check for negative w/h
    if (sub['w'] < 0).any() or (sub['h'] < 0).any():
        errors.append("Negative w or h values found")

    # Check ID format (if sample provided)
    if sample_path:
        try:
            sample = pd.read_csv(sample_path)
            sample_ids = set(sample['id'])
            sub_ids = set(sub['id'])
            missing = sample_ids - sub_ids
            extra = sub_ids - sample_ids
            if missing:
                errors.append(f"Missing IDs: {len(missing)} (e.g. {list(missing)[:3]})")
            if extra:
                errors.append(f"Extra IDs: {len(extra)}")
        except Exception as e:
            print(f"  [warn] Could not load sample for comparison: {e}")

    # Report results
    print()
    if errors:
        print("✗ VALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        print()
        return False
    else:
        print("✓ Submission valid")
        print(f"   Rows:    {len(sub)}")
        print(f"   Columns: {list(sub.columns)}")
        print()
        print("Summary statistics:")
        print(sub[['x', 'y', 'w', 'h']].describe().to_string())
        print()
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate competition submission CSV')
    parser.add_argument('--submission', required=True,
                        help='Path to submission CSV')
    parser.add_argument('--sample', default='data/metadata/sample_submission.csv',
                        help='Path to sample_submission.csv for format reference')
    args = parser.parse_args()

    valid = validate_submission(args.submission, args.sample)
    sys.exit(0 if valid else 1)
