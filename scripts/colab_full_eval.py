"""Full BEIR evaluation for the Colab-trained model. Run as a new cell in Colab.

Loads the trained model from Drive, evaluates all 6 BEIR datasets
(BM25 top-100 + rerank, NDCG@10), saves results to Drive.
"""
import os, sys, json, time, subprocess, re
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rank_bm25 import BM25Okapi

DRIVE = '/content/drive/MyDrive/flashrank_pro'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/'

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def ensure_beir(name):
    d = f'{DRIVE}/data/beir/{name}'
    if os.path.exists(f'{d}/corpus.jsonl'):
        return d
    z = f'{DRIVE}/data/{name}.zip'
    if not os.path.exists(z):
        print(f'  downloading {name}...', flush=True)
        subprocess.run(['wget', '-q', '-O', z, f'{BASE_URL}{name}.zip'], check=True)
    print(f'  extracting {name}...', flush=True)
    subprocess.run(['unzip', '-o', '-q', z, '-d', f'{DRIVE}/data/beir/'], check=True)
    assert os.path.exists(f'{d}/corpus.jsonl'), f'extract failed for {name}'
    return d

def load_split(name, split):
    from beir.datasets.data_loader import GenericDataLoader
    return GenericDataLoader(data_folder=f'{DRIVE}/data/beir/{name}').load(split=split)

def build_index(name, corpus):
    ids = list(corpus.keys())
    bm25 = BM25Okapi([tokenize(corpus[i].get('text', '')) for i in ids], k1=1.5, b=0.75)
    return bm25, ids

def eval_dataset(model, tokenizer, name, top_k=100):
    t0 = time.time()
    ensure_beir(name)
    corpus, queries, qrels = load_split(name, 'test')
    bm25, doc_ids = build_index(name, corpus)
    res = {}
    for qid, q in queries.items():
        scores = bm25.get_scores(tokenize(q))
        order = np.argsort(scores)[::-1][:top_k]
        res[qid] = {doc_ids[i]: float(scores[i]) for i in order}
    pairs, meta = [], []
    for qid, q in queries.items():
        for did in res[qid]:
            meta.append((qid, did))
            pairs.append(q + ' </s> ' + corpus[did]['text'])
    logits = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), 128):
            chunk = pairs[i:i+128]
            t = tokenizer(chunk, truncation=True, max_length=512, padding=True, return_tensors='pt')
            logits.append(model(input_ids=t['input_ids'].to(DEVICE),
                                attention_mask=t['attention_mask'].to(DEVICE)).logits.float().squeeze(-1).cpu())
    logits = torch.cat(logits)
    scores = torch.sigmoid(logits)
    reranked = {}
    for (qid, did), s in zip(meta, scores.tolist()):
        reranked.setdefault(qid, {})[did] = float(s)
    from beir.retrieval.evaluation import EvaluateRetrieval
    metrics = EvaluateRetrieval.evaluate(qrels, reranked, [10])[0]
    ndcg = metrics.get('NDCG@10', 0.0)
    os.makedirs(f'{DRIVE}/eval', exist_ok=True)
    with open(f'{DRIVE}/eval/{name}_ndcg10.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  {name}: NDCG@10 = {ndcg:.4f} ({len(queries)} queries, {time.time()-t0:.0f}s)', flush=True)
    return ndcg

model_path = f'{DRIVE}/final/model'
if not os.path.exists(model_path):
    model_path = f'{DRIVE}/ckpt/model'
print('loading model from', model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(DEVICE)
print('model loaded, evaluating 6 BEIR datasets...')

DATASETS = ['nfcorpus', 'scifact', 'fiqa', 'webis-touche2020', 'arguana', 'scidocs']
results = {}
for name in DATASETS:
    try:
        results[name] = eval_dataset(model, tokenizer, name)
    except Exception as e:
        print(f'  {name}: FAILED ({e})', flush=True)

print()
print('=== BEIR NDCG@10 (BM25 top-100 + rerank) ===')
print('| dataset | fine-tuned | base model | BM25-only | official BM25+CE |')
print('|---------|-----------|-----------|-----------|------------------|')
refs = {'nfcorpus': (0.292, 0.3014, 0.350), 'scifact': (0.567, 0.6367, 0.688),
        'fiqa': (0.221, None, 0.347), 'webis-touche2020': (0.161, None, 0.271),
        'arguana': (None, None, None), 'scidocs': (None, None, None)}
for n in DATASETS:
    r = results.get(n)
    if r is None:
        continue
    base, bm25, official = refs.get(n, (None, None, None))
    f = lambda v: f'{v:.4f}' if v is not None else '—'
    print(f'| {n} | {f(r)} | {f(base)} | {f(bm25)} | {f(official)} |')
avg = np.mean(list(results.values()))
print(f'\nAVERAGE (available datasets): {avg:.4f}')
print('results saved to', f'{DRIVE}/eval/')
