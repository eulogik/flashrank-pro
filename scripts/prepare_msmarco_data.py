"""Prepare MS MARCO fine-tune data for FlashRank-Pro.

Streams real MS MARCO queries with pre-mined hard negatives from
`sentence-transformers/msmarbo-hard-negatives` (BM25 + cross-encoder mined),
resolves query texts from `sentence-transformers/msmarco` (queries config),
resolves passage texts from the `BeIR/msmarco` corpus, and writes JSONL:
{"query": str, "pos": [str], "negs": [str]}.

Checkpointed: each phase caches to data/.msmarco_prep/ so a crash resumes.

Run on any machine (CPU fine), then train with:
    python training/05_beir_finetune.py --data_jsonl data/msmarco_train.jsonl ...
"""
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".msmarco_prep"

NEG_SOURCE_PRIORITY = ("ce", "bm25", "ANCE", "trec", "openai")


def _first_split(ds: Any) -> Iterator[dict]:
    """Yield rows whether load_dataset returned a Dataset or a split-keyed dict."""
    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        ds = ds[list(ds.keys())[0]]
    yield from ds


def _id_of(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        item = item.get("corpus_id", item.get("corpus-id", item.get("pid", item.get("_id"))))
    return None if item is None else str(item)


def extract_pos_neg(row: dict, n_negs: int, rng: random.Random) -> Optional[Tuple[list, list]]:
    pos_raw = row.get("pos") or []
    pos: list[str] = []
    for p in pos_raw[:4]:
        pid = _id_of(p)
        if pid:
            pos.append(pid)

    neg_field = row.get("neg") or []
    ordered: list[str] = []
    seen: set[str] = set(pos)

    def push(items: Any) -> None:
        for x in items or []:
            nid = _id_of(x)
            if nid and nid not in seen:
                seen.add(nid)
                ordered.append(nid)

    if isinstance(neg_field, dict):
        for key in NEG_SOURCE_PRIORITY:
            push(neg_field.get(key))
        for key, val in neg_field.items():
            if key not in NEG_SOURCE_PRIORITY:
                push(val)
    else:
        push(neg_field)

    if not pos or len(ordered) < 3:
        return None
    rng.shuffle(ordered)
    return pos[:1], ordered[:n_negs]


def phase_sample(n: int, seed: int, n_negs: int) -> list[dict]:
    """Phase 1: stream hard-negatives dataset, sample n usable rows (ids only)."""
    ckpt = CACHE_DIR / f"rows_n{n}_s{seed}_k{n_negs}.pkl"
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            rows = pickle.load(f)
        print(f"[phase 1] loaded cached sample: {len(rows)} rows")
        return rows

    from datasets import load_dataset

    rng = random.Random(seed)
    print(f"[phase 1] streaming msmarco-hard-negatives, sampling {n} rows...")
    ds = load_dataset(
        "sentence-transformers/msmarco-hard-negatives", split="train", streaming=True
    )
    rows: list[dict] = []
    scanned = 0
    t0 = time.time()
    for row in ds.shuffle(seed=seed, buffer_size=10_000):
        scanned += 1
        got = extract_pos_neg(row, n_negs, rng)
        if got:
            pos, negs = got
            rows.append({"qid": str(row["qid"]), "pos": pos, "negs": negs})
            if len(rows) % 5_000 == 0:
                print(f"  {len(rows)}/{n} rows ({time.time()-t0:.0f}s)", flush=True)
            if len(rows) >= n:
                break

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ckpt, "wb") as f:
        pickle.dump(rows, f)
    print(f"[phase 1] done: {len(rows)} rows from {scanned} scanned ({time.time()-t0:.0f}s)")
    return rows


def phase_queries(rows: list[dict]) -> dict[str, str]:
    """Phase 2: resolve qid -> query text."""
    ckpt = CACHE_DIR / "queries.pkl"
    needed = {r["qid"] for r in rows}
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            queries = pickle.load(f)
        if needed <= set(queries):
            print(f"[phase 2] loaded cached queries: {len(queries)}")
            return queries

    from datasets import load_dataset

    print(f"[phase 2] resolving {len(needed)} query texts...")
    ds = load_dataset("sentence-transformers/msmarco", "queries", split="train", streaming=True)
    queries: dict[str, str] = {}
    t0 = time.time()
    for rec in _first_split(ds):
        qid = str(rec.get("query_id", rec.get("qid", rec.get("_id"))))
        if qid in needed:
            queries[qid] = rec["query"]
        if len(queries) >= len(needed):
            break
    with open(CACHE_DIR / "queries.pkl", "wb") as f:
        pickle.dump(queries, f)
    print(f"[phase 2] resolved {len(queries)}/{len(needed)} ({time.time()-t0:.0f}s)")
    return queries


def phase_docs(rows: list[dict]) -> dict[str, str]:
    """Phase 3: resolve doc ids -> passage text (streams full 8.8M corpus)."""
    ckpt = CACHE_DIR / "docs.pkl"
    needed: set[str] = set()
    for r in rows:
        needed.update(r["pos"])
        needed.update(r["negs"])
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            docs = pickle.load(f)
        if needed <= set(docs):
            print(f"[phase 3] loaded cached docs: {len(docs)}")
            return docs

    from datasets import load_dataset

    print(f"[phase 3] resolving {len(needed)} passages (scanning BeIR/msmarco corpus)...")
    ds = load_dataset("BeIR/msmarco", "corpus", streaming=True)
    docs: dict[str, str] = {}
    t0 = time.time()
    for i, rec in enumerate(_first_split(ds)):
        did = str(rec.get("_id"))
        if did in needed:
            title = rec.get("title") or ""
            text = rec.get("text") or ""
            docs[did] = f"{title} {text}".strip() if title else text
        if i % 2_000_000 == 0:
            print(f"  scanned {i} docs, resolved {len(docs)}/{len(needed)} ({time.time()-t0:.0f}s)", flush=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(ckpt, "wb") as f:
                pickle.dump(docs, f)
    with open(ckpt, "wb") as f:
        pickle.dump(docs, f)
    print(f"[phase 3] resolved {len(docs)}/{len(needed)} ({time.time()-t0:.0f}s)")
    return docs


def main(
    n: int = 100_000,
    n_negs: int = 6,
    seed: int = 0,
    max_doc_chars: int = 2000,
    out: str = "data/msmarco_train.jsonl",
) -> None:
    rng = random.Random(seed)
    rows = phase_sample(n, seed, n_negs)
    queries = phase_queries(rows)
    docs = phase_docs(rows)

    print(f"[phase 4] writing {out}...")
    written = 0
    dropped = 0
    t0 = time.time()
    shuffled = rows[:]
    rng.shuffle(shuffled)
    with open(out, "w") as f:
        for r in shuffled:
            q = queries.get(r["qid"])
            if not q:
                dropped += 1
                continue
            try:
                pos_text = [docs[d][:max_doc_chars] for d in r["pos"]]
                neg_texts = [docs[d][:max_doc_chars] for d in r["negs"]]
            except KeyError:
                dropped += 1
                continue
            if len(neg_texts) < 2:
                dropped += 1
                continue
            f.write(json.dumps({"query": q, "pos": pos_text, "negs": neg_texts}) + "\n")
            written += 1

    size_mb = os.path.getsize(out) / 1e6
    print(f"\nDONE (total session {time.time()-t0:.0f}s for write phase)")
    print(f"  {out}: {written} examples ({size_mb:.0f} MB), dropped {dropped}")
    print("\nTrain with:")
    print(f"  python training/05_beir_finetune.py --data_jsonl {out} --num_epochs 2")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
