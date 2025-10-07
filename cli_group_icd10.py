import argparse, os
import pandas as pd
from group_rules import categorize

def main():
    parser = argparse.ArgumentParser(description="Group ICD-10-CM codes into project-specific buckets.")
    parser.add_argument("--input", required=True, help="Path to ICD-10 CSV with columns: Code, Description")
    parser.add_argument("--output", required=True, help="Path for grouped CSV output")
    parser.add_argument("--code-col", default="Code", help="Column name containing ICD codes (default: Code)")
    parser.add_argument("--desc-col", default="Description", help="Column name for descriptions (default: Description)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.code_col not in df.columns:
        raise SystemExit(f"Missing column '{args.code_col}' in {args.input}")

    df["Category"] = df[args.code_col].astype(str).map(categorize)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    # Print a quick summary
    counts = df["Category"].value_counts(dropna=False).to_dict()
    print("Saved:", args.output)
    print("Category counts:", counts)

if __name__ == "__main__":
    main()
