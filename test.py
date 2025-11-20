import pickle
import pandas as pd
from pprint import pprint

path = "output/DAILY_STATS/stats_20211130.pkl"  # <- change me

with open(path, "rb") as f:
    obj = pickle.load(f)

print("type:", type(obj))

if isinstance(obj, pd.DataFrame):
    print("shape:", obj.shape)
    print("columns:", obj.columns.tolist())
    print(obj.head(5))
    # handy if it's your stats tensor
    for c in ["date","signal","target","qrank","stat_type","bet_size_col"]:
        if c in obj.columns:
            print(f"{c} →", obj[c].dropna().unique()[:10])
    print("\nDtypes:\n", obj.dtypes)

elif isinstance(obj, dict):
    print("keys (sample):")
    for k in list(obj.keys())[:10]:
        v = obj[k]
        print(f"  {k!r}: {type(v).__name__} (len={getattr(v,'__len__',lambda: 'n/a')()})")
    # peek deeper if values are dicts/lists
    first_key = next(iter(obj), None)
    if first_key is not None:
        print("\nFirst item detail:")
        pprint({first_key: obj[first_key]})

elif isinstance(obj, (list, tuple)):
    print("length:", len(obj))
    print("first 3 items types:", [type(x).__name__ for x in obj[:3]])
    pprint(obj[:3])
else:
    pprint(obj)


import pandas as pd
from pathlib import Path

def pkl_to_csv(pkl_path: str, out_name: str | None = None):
    df = pd.read_pickle(pkl_path)          # expects a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)              # fallback if it's a list/dict
    out_path = Path(".") / (out_name or (Path(pkl_path).stem + ".csv"))
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path.resolve()} ({len(df)} rows)")

# example
pkl_to_csv(path)
