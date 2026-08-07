import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/Users/gautamkishore/Code/reranker/scripts")
from beir.datasets.data_loader import GenericDataLoader
from evaluate_beir_bm25 import build_or_load_index, retrieve
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "scifact"
corpus, queries, qrels = GenericDataLoader(
    data_folder="/Users/gautamkishore/Code/reranker/data/beir/scifact"
).load(split="test")
bm25, doc_ids = build_or_load_index(
    name, corpus, "/Users/gautamkishore/Code/reranker/data/beir_index"
)
res = retrieve(bm25, doc_ids, queries, 100)

t0 = time.time()
model = AutoModelForSequenceClassification.from_pretrained(
    "eulogik/flashrank-pro-base", trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained("eulogik/flashrank-pro-base", trust_remote_code=True)
model.eval()
bos = tok.cls_token_id
sep = tok.sep_token_id
print(f"loaded {time.time()-t0:.0f}s", flush=True)

import random

random.seed(7)
qids = [q for q in res if any(c in res[q] for c in qrels.get(q, {}))]
random.shuffle(qids)
qids = qids[:40]

needed = set()
for q in qids:
    needed.update(res[q])
doc_cache = {
    c: tok.encode(corpus[c]["text"], add_special_tokens=False) for c in needed
}
print(f"tokenized {time.time()-t0:.0f}s", flush=True)

score_buckets = {}
moves_down = moves_up = unchanged = 0
top10_rel_bm25 = top10_rel_rerank = 0
for qi, qid in enumerate(qids):
    rel = set(qrels[qid])
    cands = list(res[qid].keys())
    q_ids = tok.encode(queries[qid], add_special_tokens=False)
    max_doc = 512 - len(q_ids) - 3
    batches = [cands[i : i + 64] for i in range(0, len(cands), 64)]
    scores = []
    for batch in batches:
        input_ids, amask = [], []
        max_len = min(512, max(len(q_ids) + len(doc_cache[c][:max_doc]) + 3 for c in batch))
        for c in batch:
            d = doc_cache[c][:max_doc]
            n_real = len(q_ids) + len(d) + 3
            pad = max_len - n_real
            input_ids.append([bos] + q_ids + [sep] + d + [sep] + [tok.pad_token_id] * pad)
            amask.append([1] * n_real + [0] * pad)
        with torch.inference_mode():
            logits = model(
                input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(amask)
            ).logits
        scores.extend(torch.sigmoid(logits.float()).squeeze(-1).tolist())
    scores = np.array(scores)
    for s in np.round(scores, 1):
        score_buckets[s] = score_buckets.get(s, 0) + 1
    rel_in = [c for c in cands if c in rel]
    bm25_pos = sorted(cands.index(c) for c in rel_in)
    rr_order = np.argsort(-scores)
    rr_pos = sorted(int(np.where(rr_order == cands.index(c))[0][0]) for c in rel_in)
    bm25_rank = bm25_pos[0] + 1
    rr_rank = rr_pos[0] + 1
    if rr_rank > bm25_rank:
        moves_down += 1
    elif rr_rank < bm25_rank:
        moves_up += 1
    else:
        unchanged += 1
    top10_rel_bm25 += sum(1 for p in bm25_pos if p < 10)
    top10_rel_rerank += sum(1 for p in rr_pos if p < 10)
    if qi < 4:
        top = rr_order[:3]
        print(f"=== qid {qid} | first rel: BM25 rank {bm25_rank} -> rerank rank {rr_rank}")
        print(f"  Q: {queries[qid][:110]}")
        for i in top:
            rel_mark = "REL" if cands[i] in rel else "--"
            title = corpus[cands[i]]["title"][:70]
            print("  #%d s=%.3f %s: %s" % (int(i) + 1, scores[i], rel_mark, title))
    if (qi + 1) % 10 == 0:
        print(f"{qi+1}/40 done ({time.time()-t0:.0f}s)", flush=True)
print()
print(f"First-rel moves: down={moves_down} up={moves_up} unchanged={unchanged}")
print(f"REL in top10: BM25={top10_rel_bm25} rerank={top10_rel_rerank}")
print("Score buckets (of 4000):")
for k in sorted(score_buckets):
    print(f"  {k}: {score_buckets[k]}")
