"""Prepare MS MARCO fine-tune data for FlashRank-Pro.

Streams a subsample of sentence-transformers/msmarco-hard-negatives (real
queries + BM25/CE-mined hard negative doc ids, free, no API), resolves doc
texts from the BeIR/msmarco corpus, and writes a JSONL in the same schema
as the BEIR-derived training examples: {"query": str, "pos": [str], "negs": [str]}.

Run on any machine (CPU fine), then train with:
    python training/05_beir_finetune.py --data_jsonl data/msmarco_train.jsonl ...
"""
import argparse
import json
import random


def main(
    n: int = 20000,
    seed: int = 0,
    n_negs: int = 10,
    buffer_size: int = 100_000,
    out: str = "data/msmarco_train.jsonl",
):
    from datasets import load_dataset

    rng = random.Random(seed)

    print(f"[1/3] streaming msmarco-hard-negatives (train), sampling {n} rows...")
    ds = load_dataset(
        "sentence-transformers/msmarco-hard-negatives",
        split="train",
        streaming=True,
    )
    sampled = list(ds.shuffle(seed=seed, buffer_size=buffer_size).take(n))
    print(f"  sampled {len(sampled)} queries")

    needed: set = set()
    rows = []
    for row in sampled:
        pos = row.get("pos") or []
        negs = (row.get("ce") or []) + (row.get("neg") or []) + (row.get("bm25") or [])
        negs = [x["corpus_id"] if isinstance(x, dict) else x for x in negs]
        negs = [str(x) for x in negs if x is not None]
        dedup = []
        for x in negs:
            if x not in dedup:
                dedup.append(x)
            if len(dedup) >= n_negs:
                break
        pos = [str(x["corpus_id"]) if isinstance(x, dict) else str(x) for x in pos]
        if not pos or len(dedup) < 2:
            continue
        rows.append({"query": row["query"], "pos": pos[:1], "negs": dedup})
        needed.update(pos[:1])
        needed.update(dedup)
    print(f"  {len(rows)} usable triples; {len(needed)} unique docs to resolve")

    print("[2/3] streaming BeIR/msmarco corpus...")
    corpus = load_dataset("BeIR/msmarco", split="corpus", streaming=True)
    docs = {}
    for i, doc in enumerate(corpus):
        did = doc["_id"]
        if did in needed:
            docs[did] = doc["text"]
        if i % 1_000_000 == 0:
            print(f"  scanned {i} docs, resolved {len(docs)}/{len(needed)}", flush=True)
        if len(docs) >= len(needed):
            break
    missing = needed - set(docs)
    if missing:
        n_before = len(rows)
        rows = [r for r in rows if all(d in docs for d in r["pos"] + r["negs"])]
        print(f"  WARNING: {len(missing)} docs missing; dropped {n_before - len(rows)} triples")

    print(f"[3/3] writing {out}...")
    with open(out, "w") as f:
        for r in rows:
            rec = {
                "query": r["query"],
                "pos": [docs[d] for d in r["pos"]],
                "negs": [docs[d] for d in r["negs"]],
            }
            f.write(json.dumps(rec) + "\n")
    f = open(out)
    print(f"DONE: {out} has {len(rows)} examples")
    print(f"  avg negs per example: {sum(len(r['negs']) for r in rows) / len(rows):.1f}")
    print(f"  example query: {rows[0]['query'][:80]!r}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)