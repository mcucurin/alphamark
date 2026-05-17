"""Convert .pkl file(s) to .csv. Accepts a single file or a directory."""
import pandas as pd
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help=".pkl file or directory containing .pkl files")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: same location)")
    args = parser.parse_args()

    inp = Path(args.input)

    if inp.is_file():
        files = [inp]
        out_dir = Path(args.output_dir) if args.output_dir else inp.parent
    elif inp.is_dir():
        files = sorted(inp.glob("*.pkl"))
        out_dir = Path(args.output_dir) if args.output_dir else inp
    else:
        raise FileNotFoundError(f"Not found: {inp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} .pkl file(s)")

    for f in files:
        obj = pd.read_pickle(f)
        if isinstance(obj, pd.DataFrame):
            df = obj
        elif isinstance(obj, dict):
            # Try to make a DataFrame from dict
            try:
                df = pd.DataFrame(obj)
            except Exception:
                df = pd.DataFrame([obj])
        else:
            print(f"  Skipping {f.name} — unsupported type: {type(obj)}")
            continue

        out_path = out_dir / f"{f.stem}.csv"
        df.to_csv(out_path)
        print(f"  {f.name} -> {out_path.name}  ({df.shape})")

    print("Done.")