"""
Stream-shrink Amazon Reviews to Parquet shards (keep ALL fields).
- Prefer parent_asin as item id (fallback to asin).
- Keep original fields (title/text/images/verified_purchase/helpful_vote/...).
- Control total rows via --max_rows (e.g., 2,000,000).
"""
import argparse, hashlib, re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

# dynamic field candidates
CAND_USER = ["user_id", "reviewerID", "uid", "reviewer_id"]
CAND_ITEM_P = ["parent_asin"]
CAND_ITEM_A = ["asin", "item_id", "iid", "product_id"]
CAND_RATE = ["rating", "overall", "stars"]
CAND_TIME = ["timestamp", "unixReviewTime", "time", "review_time"]

# keep all these (write if present, else None)
KEEP_ALL = [
  "rating","title","text","images","asin","parent_asin","user_id","reviewerID",
  "timestamp","unixReviewTime","verified_purchase","helpful_vote","small_image_url",
  "medium_image_url","large_image_url","store","categories","details","subtitle"
]

def pick_key(d:Dict, cands:List[str])->Optional[str]:
    for k in cands:
        if k in d: return k
    return None

def year_from_unix(ts)->Optional[int]:
    try:
        t=int(ts)
        if t>1e10: t//=1000
        return pd.Timestamp.utcfromtimestamp(t).year
    except: return None

def stable_hash(s:str)->int:
    import hashlib
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(),16)

def infer_category_from_path(p:str)->str:
    name=Path(p).name
    name=re.sub(r"\.jsonl(\.gz|\.zst)?$","",name,flags=re.I)
    return "raw_"+name.replace(" ","_")

def ensure_writer(schema:pa.Schema, holder:Dict, out_dir:Path, rows_per_shard:int):
    if "writer" not in holder or holder["rows_written"]>=rows_per_shard:
        if "writer" in holder: holder["writer"].close()
        shard=holder.get("shard",0)
        out_path=out_dir/f"full-{shard:05d}-of-99999.parquet"
        holder["writer"]=pq.ParquetWriter(out_path, schema, compression="zstd")
        holder["rows_written"]=0
        holder["shard"]=shard+1

def close_writer(holder:Dict):
    if "writer" in holder: holder["writer"].close()

def stream_one(ds_iter, out_dir:Path, rows_per_shard:int, start_year, end_year,
               sample_user_pct, per_user_max, max_rows, desc:str):
    kept, buf=[], []
    holder={}
    for rec in tqdm(ds_iter, desc=desc):
        if not isinstance(rec,dict): continue

        u_key = pick_key(rec,CAND_USER)
        t_key = pick_key(rec,CAND_TIME)
        p_key = pick_key(rec,CAND_ITEM_P)
        a_key = pick_key(rec,CAND_ITEM_A)
        r_key = pick_key(rec,CAND_RATE)

        if u_key is None or t_key is None: continue
        if (p_key is None) and (a_key is None): continue

        u = str(rec[u_key])
        tm = rec[t_key]
        yr = year_from_unix(tm)
        if yr is None: continue
        if start_year and yr<start_year: continue
        if end_year and yr>end_year: continue

        # deterministic user sampling
        if sample_user_pct is not None:
            if sample_user_pct<=0: continue
            if sample_user_pct<1.0:
                if (stable_hash(u)%10000) >= int(sample_user_pct*10000): continue

        try:
            t_int=int(tm)
        except: 
            continue

        item_id = str(rec[p_key]) if p_key else str(rec[a_key])
        rating = float(rec[r_key]) if (r_key and rec.get(r_key) is not None) else 1.0

        row = {"user_id":u,"item_id":item_id,"rating":rating,"timestamp":t_int}
        # keep original fields
        for k in KEEP_ALL:
            if k not in row:
                row[k] = rec.get(k, None)
        buf.append(row)

        if len(buf)>=200_000:
            df=pd.DataFrame(buf).sort_values(["user_id","timestamp"],ascending=[True,False])
            if per_user_max:
                df["_rk"]=df.groupby("user_id").cumcount()
                df=df[df["_rk"]<per_user_max].drop(columns="_rk")
            table=pa.Table.from_pandas(df, preserve_index=False)
            ensure_writer(table.schema, holder, out_dir, rows_per_shard)
            holder["writer"].write_table(table)
            holder["rows_written"]+=len(df)
            kept.append(len(df)); buf=[]
            if max_rows and sum(kept)>=max_rows: break

    if buf and (not max_rows or sum(kept)<max_rows):
        df=pd.DataFrame(buf).sort_values(["user_id","timestamp"],ascending=[True,False])
        if per_user_max:
            df["_rk"]=df.groupby("user_id").cumcount()
            df=df[df["_rk"]<per_user_max].drop(columns="_rk")
        if max_rows:
            remain=max(0, max_rows-sum(kept))
            if len(df)>remain: df=df.iloc[:remain]
        table=pa.Table.from_pandas(df, preserve_index=False)
        ensure_writer(table.schema, holder, out_dir, rows_per_shard)
        holder["writer"].write_table(table)
        holder["rows_written"]+=len(df)
        kept.append(len(df))
    close_writer(holder)
    return sum(kept)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hf_repo",type=str,default=None)
    ap.add_argument("--hf_paths",nargs="*",default=None)
    ap.add_argument("--urls",nargs="*",default=None)
    ap.add_argument("--out_root",type=str,default=".")
    ap.add_argument("--rows_per_shard",type=int,default=500_000)
    ap.add_argument("--start_year",type=int,default=None)
    ap.add_argument("--end_year",type=int,default=None)
    ap.add_argument("--sample_user_pct",type=float,default=None)
    ap.add_argument("--per_user_max",type=int,default=None)
    ap.add_argument("--max_rows",type=int,default=None)
    args=ap.parse_args()

    if (not args.urls) and (not args.hf_repo or not args.hf_paths):
        raise SystemExit("You must provide either --urls or (--hf_repo and --hf_paths).")

    use_urls=bool(args.urls)
    sources=args.urls if use_urls else args.hf_paths

    total=0
    for src in sources:
        cat_dir = Path(args.out_root)/infer_category_from_path(src)
        cat_dir.mkdir(parents=True, exist_ok=True)
        if use_urls:
            ds=load_dataset("json", data_files=src, split="train", streaming=True)
            desc=f"Streaming {Path(src).name}"
        else:
            uri=f"hf://datasets/{args.hf_repo}/{src}"
            ds=load_dataset("json", data_files=uri, split="train", streaming=True)
            desc=f"Streaming {src}"
        kept=stream_one(ds, cat_dir, args.rows_per_shard, args.start_year, args.end_year,
                        args.sample_user_pct, args.per_user_max, args.max_rows, desc)
        print(f"kept {kept} rows → {cat_dir}"); total+=kept
    print(f"\nALL DONE. Total kept rows: {total}")

if __name__=="__main__":
    main()
