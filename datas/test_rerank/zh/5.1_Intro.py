from datasets import load_dataset
import numpy as np

data = load_dataset("namespace-Pt/msmarco", split="dev")

queries = np.array(data[:100]["query"])
corpus = sum(data[:5000]["positive"], [])

from FlagEmbedding import FlagModel

# get the BGE embedding model
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

# get the embedding of the corpus
corpus_embeddings = model.encode(corpus)

print("shape of the corpus embeddings:", corpus_embeddings.shape)
print("data type of the embeddings: ", corpus_embeddings.dtype)

import faiss

# get the length of our embedding vectors, vectors by bge-base-en-v1.5 have length 768
dim = corpus_embeddings.shape[-1]

# create the faiss index and store the corpus embeddings into the vector space
index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
corpus_embeddings = corpus_embeddings.astype(np.float32)
index.train(corpus_embeddings)
index.add(corpus_embeddings)

print(f"total number of vectors: {index.ntotal}")

query_embeddings = model.encode_queries(queries)
ground_truths = [d["positive"] for d in data]
corpus = np.asarray(corpus)

from tqdm import tqdm

res_scores, res_ids, res_text = [], [], []
query_size = len(query_embeddings)
batch_size = 256
# The cutoffs we will use during evaluation, and set k to be the maximum of the cutoffs.
cut_offs = [1, 10]
k = max(cut_offs)

for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
    q_embedding = query_embeddings[i: min(i+batch_size, query_size)].astype(np.float32)
    # search the top k answers for each of the queries
    score, idx = index.search(q_embedding, k=k)
    res_scores += list(score)
    res_ids += list(idx)
    res_text += list(corpus[idx])

from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) 
# Setting use_fp16 to True speeds up computation with a slight performance degradation

# use the compute_score() function to calculate scores for each input sentence pair
scores = reranker.compute_score([
    ['what is panda?', 'Today is a sunny day'], 
    ['what is panda?', 'The tiger (Panthera tigris) is a member of the genus Panthera and the largest living cat species native to Asia.'],
    ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
    ])
print(scores)

new_ids, new_scores, new_text = [], [], []
for i in range(len(queries)):
    # get the new scores of the previously retrieved results
    new_score = reranker.compute_score([[queries[i], text] for text in res_text[i]])
    # sort the lists of ids and scores by the new scores
    new_id = [tup[1] for tup in sorted(list(zip(new_score, res_ids[i])), reverse=True)]
    new_scores.append(sorted(new_score, reverse=True))
    new_ids.append(new_id)
    new_text.append(corpus[new_id])

def calc_recall(preds, truths, cutoffs):
    recalls = np.zeros(len(cutoffs))
    for text, truth in zip(preds, truths):
        for i, c in enumerate(cutoffs):
            recall = np.intersect1d(truth, text[:c])
            recalls[i] += len(recall) / max(min(len(recall), len(truth)), 1)
    recalls /= len(preds)
    return recalls

recalls_init = calc_recall(res_text, ground_truths, cut_offs)
for i, c in enumerate(cut_offs):
    print(f"recall@{c}:\t{recalls_init[i]}")

recalls_rerank = calc_recall(new_text, ground_truths, cut_offs)
for i, c in enumerate(cut_offs):
    print(f"recall@{c}:\t{recalls_rerank[i]}")

def MRR(preds, truth, cutoffs):
    mrr = [0 for _ in range(len(cutoffs))]
    for pred, t in zip(preds, truth):
        for i, c in enumerate(cutoffs):
            for j, p in enumerate(pred):
                if j < c and p in t:
                    mrr[i] += 1/(j+1)
                    break
    mrr = [k/len(preds) for k in mrr]
    return mrr

mrr_init = MRR(res_text, ground_truths, cut_offs)
for i, c in enumerate(cut_offs):
    print(f"MRR@{c}:\t{mrr_init[i]}")

mrr_rerank = MRR(new_text, ground_truths, cut_offs)
for i, c in enumerate(cut_offs):
    print(f"MRR@{c}:\t{mrr_rerank[i]}")

from sklearn.metrics import ndcg_score

pred_hard_encodings = []
for pred, label in zip(res_text, ground_truths):
    pred_hard_encoding = list(np.isin(pred, label).astype(int))
    pred_hard_encodings.append(pred_hard_encoding)

for i, c in enumerate(cut_offs):
    nDCG = ndcg_score(pred_hard_encodings, res_scores, k=c)
    print(f"nDCG@{c}: {nDCG}")

pred_hard_encodings_rerank = []
for pred, label in zip(new_text, ground_truths):
    pred_hard_encoding = list(np.isin(pred, label).astype(int))
    pred_hard_encodings_rerank.append(pred_hard_encoding)

for i, c in enumerate(cut_offs):
    nDCG = ndcg_score(pred_hard_encodings_rerank, new_scores, k=c)
    print(f"nDCG@{c}: {nDCG}")
