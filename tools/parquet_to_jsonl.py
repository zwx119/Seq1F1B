"""Convert FineWeb-Edu parquet files into a single line-delimited JSONL
suitable for Megatron's `tools/preprocess_data.py`.

Streaming, constant memory, resumable (skips if output already complete).

Usage:
    python3 tools/parquet_to_jsonl.py <in_dir> <out_jsonl>

Example:
    python3 tools/parquet_to_jsonl.py \
        data/fineweb-edu-sample-10BT \
        data/fineweb-edu-sample-10BT.jsonl
"""
import sys
import os
import glob
import json
import time

import pyarrow.parquet as pq


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    in_dir, out_path = sys.argv[1], sys.argv[2]

    files = sorted(glob.glob(os.path.join(in_dir, "**/*.parquet"), recursive=True))
    if not files:
        print(f"!! no parquet files under {in_dir}")
        sys.exit(1)
    print(f"found {len(files)} parquet files -> {out_path}")

    # Resume marker: if .done file exists next to out_path, we've already
    # completed this conversion.
    done_marker = out_path + ".done"
    if os.path.exists(done_marker):
        print(f"✓ {out_path} already converted (marker {done_marker} present)")
        return

    tmp_path = out_path + ".partial"
    t0 = time.time()
    n_docs = 0
    n_chars = 0
    with open(tmp_path, "w", encoding="utf-8") as fout:
        for i, f in enumerate(files):
            try:
                pf = pq.ParquetFile(f)
            except Exception as e:
                print(f"  [{i+1}/{len(files)}] SKIP corrupt {f}: {e!r}")
                continue
            # Stream per row-group to keep memory low (10BT subset rows are large-ish)
            for rg_idx in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg_idx, columns=["text"])
                texts = tbl.column("text").to_pylist()
                for t in texts:
                    if not t:
                        continue
                    fout.write(json.dumps({"text": t}, ensure_ascii=False))
                    fout.write("\n")
                    n_docs += 1
                    n_chars += len(t)
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                elapsed = time.time() - t0
                rate = n_docs / max(1, elapsed)
                print(f"  [{i+1}/{len(files)}]  docs={n_docs:,}  "
                      f"chars={n_chars/1e9:.2f}B  "
                      f"elapsed={elapsed:.0f}s  rate={rate:.0f} docs/s",
                      flush=True)

    os.replace(tmp_path, out_path)
    with open(done_marker, "w") as f:
        f.write(f"docs={n_docs} chars={n_chars}\n")
    print(f"✓ done: {n_docs:,} docs / {n_chars/1e9:.2f}B chars "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
