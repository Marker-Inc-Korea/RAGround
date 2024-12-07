import glob
import os

import pandas as pd


def find_corpus_name(retrieval_gt_id: str, corpus_dir: str):
    if not os.path.isdir(corpus_dir) or not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    all_corpus_files = glob.glob(
        os.path.join(corpus_dir, "**", "*.parquet"), recursive=True
    )
    for corpus_filepath in all_corpus_files:
        corpus_df = pd.read_parquet(corpus_filepath, engine="pyarrow")
        find_row = corpus_df[corpus_df["doc_id"] == retrieval_gt_id]
        if len(find_row) <= 0:
            continue
        return os.path.basename(os.path.dirname(corpus_filepath))
    return None
