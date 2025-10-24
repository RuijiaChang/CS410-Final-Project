# make_eval_negatives.py
# Usage:
#   python make_eval_negatives.py --data data/final/loader_ready.csv --out data/final/negatives.json --k 100
import argparse, json
import numpy as np
import pandas as pd

def leave_one_out_split(df, ts_col="timestamp"):
    df = df.sort_values(["user_id", ts_col]).reset_index(drop=True)
    rid = np.arange(len(df))
    df["_rid"] = rid
    sp = {"train": [], "valid": [], "test": []}
    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values(ts_col)
        r = g["_rid"].to_numpy()
        if len(r) == 1:
            sp["test"].append(int(r[-1]))
        elif len(r) == 2:
            sp["valid"].append(int(r[-2])); sp["test"].append(int(r[-1]))
        else:
            sp["train"].extend([int(x) for x in r[:-2]])
            sp["valid"].append(int(r[-2])); sp["test"].append(int(r[-1]))
    df.drop(columns=["_rid"], inplace=True)
    return sp

def build_eval_negatives(df, splits, k=100, seed=42, ts_col="timestamp"):
    rng = np.random.default_rng(seed)

    items = df["item_id"].astype(str).unique()
    n = items.size

    user_pos = {str(u): set(g["item_id"].astype(str)) for u, g in df.groupby("user_id", sort=False)}
    res = {"valid": [], "test": []}
    def make(name):
        out = []
        rows = splits[name]
        for rid in rows:
            uid = str(df.iloc[rid]["user_id"])
            pos = user_pos.get(uid, set())
            negs, tries = [], 0
            while len(negs) < k and tries < k * 50:
                it = items[int(rng.integers(0, n))]
                if it not in pos:
                    negs.append(it)
                tries += 1
            out.append({"row_id": int(rid), "user_id": uid, "neg_items": negs})
        return out
    res["valid"] = make("valid"); res["test"] = make("test")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)   # data/final/loader_ready.csv 也支持 .parquet
    ap.add_argument("--out", required=True)    # data/final/negatives.json
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--timestamp_col", type=str, default="timestamp")
    args = ap.parse_args()

    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)

    splits = leave_one_out_split(df, ts_col=args.timestamp_col)
    negs = build_eval_negatives(df, splits, k=args.k, ts_col=args.timestamp_col)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(negs, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}: valid={len(negs['valid'])}, test={len(negs['test'])}, k={args.k}")

if __name__ == "__main__":
    main()
