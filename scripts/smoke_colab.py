import os, sys, json, time, pickle, random, re, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from rank_bm25 import BM25Okapi

DRIVE = '.'   # local override; data lives in ./data/beir
CKPT_DIR = '/tmp/ckpt_test'
DEVICE = 'cpu'
MODEL_INIT = 'models/flashrank-pro-beir'
os.makedirs(CKPT_DIR, exist_ok=True)

def tokenize(text): return re.findall(r"[a-z0-9]+", text.lower())
def load_split(name, split):
    from beir.datasets.data_loader import GenericDataLoader
    return GenericDataLoader(data_folder=f'{DRIVE}/data/beir/{name}').load(split=split)
def build_index(name, corpus):
    ids = list(corpus.keys())
    bm25 = BM25Okapi([tokenize(corpus[i].get('text','')) for i in ids], k1=1.5, b=0.75)
    return bm25, ids
def build_examples(datasets, max_q, n_neg, seed=0):
    random.seed(seed); examples = []
    for name in datasets:
        corpus, queries, qrels = load_split(name, 'train')
        bm25, doc_ids = build_index(name, corpus)
        pos_map = {qid: [d for d, s in qrels[qid].items() if s > 0] for qid in qrels}
        qids = [q for q in queries if q in pos_map]; random.shuffle(qids); qids = qids[:max_q]
        n = 0
        for qid in qids:
            pos = [corpus[d]['text'] for d in pos_map[qid] if d in corpus]
            if not pos: continue
            scores = bm25.get_scores(tokenize(queries[qid]))
            order = np.argsort(scores)[::-1]
            negs = [corpus[doc_ids[i]]['text'] for i in order if doc_ids[i] not in pos_map[qid]][:n_neg]
            if len(negs) < 2: continue
            examples.append({'query': queries[qid], 'pos': pos[:2], 'negs': negs}); n += 1
        print(f'  {name}: {n} examples')
    return examples

def evaluate_scifact(model, tokenizer, top_k=100, max_q=None):
    from beir.retrieval.evaluation import EvaluateRetrieval
    corpus, queries, qrels = load_split('scifact', 'test')
    if max_q:
        qids = list(queries.keys())[:max_q]; queries = {q: queries[q] for q in qids}
    bm25, doc_ids = build_index('scifact', corpus)
    res = {}
    for qid, q in queries.items():
        scores = bm25.get_scores(tokenize(q))
        order = np.argsort(scores)[::-1][:top_k]
        res[qid] = {doc_ids[i]: float(scores[i]) for i in order}
    model.eval(); pairs, meta = [], []
    for qid, q in queries.items():
        for did in res[qid]:
            meta.append((qid, did)); pairs.append(q + ' </s> ' + corpus[did]['text'])
    logits = []
    with torch.no_grad():
        for i in range(0, len(pairs), 128):
            chunk = pairs[i:i+128]
            t = tokenizer(chunk, truncation=True, max_length=512, padding=True, return_tensors='pt')
            logits.append(model(input_ids=t['input_ids'].to(DEVICE), attention_mask=t['attention_mask'].to(DEVICE)).logits.float().squeeze(-1).cpu())
    logits = torch.cat(logits); scores = torch.sigmoid(logits)
    reranked = {}
    for (qid, did), s in zip(meta, scores.tolist()):
        reranked.setdefault(qid, {})[did] = float(s)
    return EvaluateRetrieval.evaluate(qrels, reranked, [10])[0]

class RerankDS(Dataset):
    def __init__(self, ex, tok, max_len):
        self.ex, self.tok, self.max_len = ex, tok, max_len
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e = self.ex[i]; q = e['query']
        docs = e['pos'][:1] + e['negs']; labels = [1.0] + [0.0]*len(e['negs'])
        ts = self.tok([q + ' </s> ' + d for d in docs], truncation=True, max_length=self.max_len, padding=False)
        return ts['input_ids'], ts['attention_mask'], torch.tensor(labels, dtype=torch.float)

def collate(batch, tok, max_len):
    ids, masks, labels = zip(*batch)
    n_docs = [len(x) for x in ids]
    flat_ids, flat_masks, flat_labels, bi = [], [], [], []
    for qi, (x, m, l) in enumerate(zip(ids, masks, labels)):
        for j in range(len(x)):
            flat_ids.append(x[j]); flat_masks.append(m[j]); flat_labels.append(l[j]); bi.append(qi)
    L = max(len(x) for x in flat_ids)
    ids = torch.tensor([x + [tok.pad_token_id]*(L-len(x)) for x in flat_ids])
    masks = torch.tensor([m + [0]*(L-len(m)) for m in flat_masks])
    return ids, masks, torch.tensor(flat_labels, dtype=torch.float), torch.tensor(bi), n_docs

def run_training(model, tokenizer, examples, args, resume_state, ckpt_dir):
    ds = RerankDS(examples, tokenizer, args['max_length'])
    loader = DataLoader(ds, batch_size=args['batch_size'], shuffle=True, collate_fn=lambda b: collate(b, tokenizer, args['max_length']), drop_last=True)
    steps_per_epoch = len(loader) // args['grad_accum']
    total_opt_steps = steps_per_epoch * args['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, total_opt_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    step = 0; best = 0.0; ema = None
    if resume_state:
        step = resume_state['step']; best = resume_state.get('best', 0.0)
        optimizer.load_state_dict(resume_state['optimizer']); scheduler.load_state_dict(resume_state['scheduler'])
        print(f'  resumed at step {step}')
    epoch0 = step // steps_per_epoch
    start_t = time.time()
    def save_ckpt():
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step, 'best': best}, os.path.join(ckpt_dir, 'checkpoint.pt'))
        print(f'  [ckpt @ step {step}]')
    for ep in range(epoch0, args['epochs']):
        ep_opt_steps = (ep+1)*steps_per_epoch
        for bi, (ids, mask, labels, bi_map, n_docs) in enumerate(loader):
            if step >= ep_opt_steps: break
            with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
                logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits.float().squeeze(-1)
                scores = torch.sigmoid(logits)
            margin_loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
            bce = torch.zeros((), dtype=torch.float32, device=DEVICE)
            n_pos = 0
            for qi in range(len(n_docs)):
                sel = bi_map == qi; ls = labels[sel]; ss = scores[sel]
                pos_s = ss[ls == 1.0]; neg_s = ss[ls == 0.0]
                if len(pos_s) == 0 or len(neg_s) == 0: continue
                for ps in pos_s:
                    margin_loss += torch.clamp(args['margin'] - (ps - neg_s), min=0.0).sum()
                    bce += -(torch.log(ps + 1e-7)); n_pos += 1
            margin_loss = margin_loss / max(n_pos, 1); bce = bce / max(n_pos, 1)
            loss = margin_loss + args['bce_weight']*bce
            scaler.scale(loss).backward()
            if (bi+1) % args['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                step += 1
                ema = loss.item() if ema is None else 0.95*ema + 0.05*loss.item()
                if step % 50 == 0:
                    print(f'  step {step}/{total_opt_steps} | loss {ema:.4f}')
                if step % 100 == 0: save_ckpt()
    save_ckpt()
    return model, step, best

# ---- smoke ----
print('building tiny examples...')
EXAMPLES = build_examples(['scifact', 'nfcorpus'], 12, 3, seed=0)
print('total:', len(EXAMPLES))
model = AutoModelForSequenceClassification.from_pretrained(MODEL_INIT, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_INIT, trust_remote_code=True)
model.train()
ARGS = {'max_length': 128, 'batch_size': 2, 'grad_accum': 1, 'epochs': 1, 'lr': 2e-5, 'margin': 0.15, 'bce_weight': 0.5, 'eval_every': 999}
model, step, best = run_training(model, tokenizer, EXAMPLES, ARGS, None, CKPT_DIR)
print('train OK, step =', step)
ckpt = torch.load(f'{CKPT_DIR}/checkpoint.pt')
print('ckpt keys:', sorted(ckpt.keys()), '| step', ckpt['step'])
model2 = AutoModelForSequenceClassification.from_pretrained(MODEL_INIT, trust_remote_code=True)
model2.load_state_dict(ckpt['model'])
model2.eval()
m = evaluate_scifact(model2, tokenizer, max_q=3)
print('eval OK, NDCG@10 =', m.get('NDCG@10'))
print('SMOKE TEST PASSED')
