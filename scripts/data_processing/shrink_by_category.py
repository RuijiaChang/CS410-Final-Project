import argparse, glob, os, math
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

def load_dir_parquets(cat_dir: str) -> pd.DataFrame:
    files = sorted(
        glob.glob(os.path.join(cat_dir, "*.parquet")) +
        glob.glob(os.path.join(cat_dir, "**/*.parquet"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(f"No parquet under {cat_dir}")
    return pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True)

def stratified_allocation(sizes: List[int], target_total: int, floor: int) -> List[int]:
    """
    Stratified allocation: each category gets at least `floor`, and the remaining quota
    is distributed proportionally to each category's remaining capacity (Largest Remainder Method).
    sizes: per-category sample counts (after per-category preprocessing)
    target_total: global target number of rows to keep
    floor: per-category minimum quota (will not exceed the actual sample count of that category)
    """
    n = len(sizes)
    total = int(np.sum(sizes))
    if target_total >= total:
        return sizes[:]  # no shrinking needed
    if target_total <= 0:
        return [0] * n

    # Step 1: assign the floor to each category, not exceeding its size
    base = [min(s, floor) for s in sizes]
    used = sum(base)
    remain = target_total - used
    if remain <= 0:
        # If floor already exhausts the quota, proportionally scale down base to target_total (rare path)
        if used == target_total:
            return base
        if used > 0:
            scale = target_total / used
            raw = [b * scale for b in base]
            alloc = [min(s, int(math.floor(r))) for s, r in zip(sizes, raw)]
            need = target_total - sum(alloc)
            rema = [(i, raw[i] - math.floor(raw[i])) for i in range(n) if alloc[i] < sizes[i]]
            rema.sort(key=lambda x: x[1], reverse=True)
            for i, _ in rema:
                if need <= 0:
                    break
                if alloc[i] < sizes[i]:
                    alloc[i] += 1
                    need -= 1
            return alloc
        return [0] * n

    # Step 2: allocate the remaining quota proportionally to remaining capacity
    sizes_remain = [max(0, s - b) for s, b in zip(sizes, base)]
    total_remain = sum(sizes_remain)
    if total_remain == 0:
        return base

    raw = [sr * (remain / total_remain) for sr in sizes_remain]
    add = [min(sr, int(math.floor(r))) for sr, r in zip(sizes_remain, raw)]
    alloc = [b + a for b, a in zip(base, add)]

    # Largest Remainder Method to fill the rest
    need = target_total - sum(alloc)
    if need > 0:
        remainders = [(i, raw[i] - math.floor(raw[i])) for i in range(n) if sizes_remain[i] > add[i]]
        remainders.sort(key=lambda x: x[1], reverse=True)
        for i, _ in remainders:
            if need <= 0:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1
                need -= 1

    return alloc

def main(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged = []
    cat_names = []

    # Expand input category directories (supports globs)
    expanded = []
    for pat in args.cat_dirs:
        expanded += [p for p in glob.glob(pat) if os.path.isdir(p)]
    if not expanded:
        raise SystemExit("No input category dirs.")

    # Per-category pipeline: read -> dedupe -> per-user cap -> per-category cap -> write
    for cat_dir in expanded:
        cat = Path(cat_dir).name
        print(f"=== {cat} ===")
        df = load_dir_parquets(cat_dir)

        need = {"user_id", "item_id", "timestamp"}
        if not need.issubset(df.columns):
            raise ValueError(f"{cat} missing cols. got: {list(df.columns)[:10]}")

        # Keep the most recent record per (user_id, item_id)
        df = (
            df.sort_values(["user_id", "item_id", "timestamp"])
              .drop_duplicates(["user_id", "item_id"], keep="last")
              .reset_index(drop=True)
        )

        # Per-user cap (keep most recent interactions)
        if args.max_per_user > 0:
            df = df.sort_values(["user_id", "timestamp"], ascending=[True, False])
            df["_rk"] = df.groupby("user_id").cumcount()
            df = df[df["_rk"] < args.max_per_user].drop(columns="_rk")

        # Optional per-category cap
        if args.per_category > 0 and len(df) > args.per_category:
            df = df.sample(args.per_category, random_state=args.seed)

        # Write per-category shrunk file
        out = Path(args.out) / f"{cat}_shrunk.parquet"
        df.to_parquet(out, index=False)
        print(f"saved {len(df):,} rows to {out}")

        # Mark category for global stratification
        df = df.copy()
        df["__category__"] = cat

        merged.append(df)
        cat_names.append(cat)

    if not merged:
        print("No data to merge after per-category processing.")
        return

    # Merge all categories
    all_df = pd.concat(merged, ignore_index=True)
    total_rows = len(all_df)
    print(f"\n[merge] total rows after per-category steps: {total_rows:,}")

    # Global stratified downsampling (optional)
    if args.max_total > 0 and total_rows > args.max_total:
        print(f"[global] applying stratified sampling to {args.max_total:,} rows ...")

        # Count per category (keep input order)
        group = all_df.groupby("__category__", sort=False)
        cat_counts = group.size().reindex(cat_names).fillna(0).astype(int).tolist()

        # Allocate per-category quotas (with per-category floor)
        alloc = stratified_allocation(cat_counts, args.max_total, args.per_category_floor)
        alloc_map = dict(zip(cat_names, alloc))

        sampled_list = []
        for cat in cat_names:
            df_cat = group.get_group(cat)
            k = alloc_map[cat]
            if k <= 0:
                continue
            if len(df_cat) <= k:
                sampled = df_cat
            else:
                if args.keep_latest:
                    sampled = df_cat.sort_values("timestamp", ascending=False).head(k)
                else:
                    sampled = df_cat.sample(n=k, random_state=args.seed)
            sampled_list.append(sampled)

        all_df = pd.concat(sampled_list, ignore_index=True)
        # Optional shuffle
        all_df = all_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        print(f"[global] final stratified rows: {len(all_df):,}")

    # Write merged file
    all_out = Path(args.out) / "all_categories_shrunk.parquet"
    if "__category__" in all_df.columns and not args.keep_category_col:
        all_df = all_df.drop(columns="__category__")
    all_df.to_parquet(all_out, index=False)
    print(f"merged {len(all_df):,} rows → {all_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat_dirs", nargs="+", required=True,
                    help="Input category directories (globs supported, e.g., raw_*)")
    ap.add_argument("--per_category", type=int, default=0,
                    help="Per-category upper cap (0 = no cap)")
    ap.add_argument("--max_per_user", type=int, default=30,
                    help="Max interactions to keep per user (most recent)")
    ap.add_argument("--max_total", type=int, default=0,
                    help="Global upper cap (>0 enables stratified sampling)")
    ap.add_argument("--per_category_floor", type=int, default=0,
                    help="Per-category minimum quota for global stratification")
    ap.add_argument("--keep_latest", action="store_true",
                    help="In global stratification, prefer most recent K per category; default is random")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--keep_category_col", action="store_true",
                    help="Keep the internal column __category__ in the output (for debugging)")
    ap.add_argument("--out", type=str, default="data/shrunk", help="Output directory")
    main(ap.parse_args())
