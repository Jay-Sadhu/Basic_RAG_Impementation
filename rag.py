import json
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import torch
import os

# -------------------------------
# CONFIG
# -------------------------------
CHUNKS_FILE = "hotpotqa_chunks_200w_50o.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
NUM_SAMPLES = 5  # number of random questions to test
EMBEDDINGS_FILE = "hotpotqa_embeddings.npy"
FAISS_INDEX_FILE = "hotpotqa_faiss.index"

# -------------------------------
# 1. Load corpus chunks
# -------------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

corpus_texts = [chunk["text"] for chunk in chunks]
corpus_ids = [chunk["chunk_id"] for chunk in chunks]

print(f"Loaded {len(corpus_texts)} chunks.")

# -------------------------------
# 2. Load embedding model on GPU
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)

# -------------------------------
# 3. Load or generate embeddings
# -------------------------------
if os.path.exists(EMBEDDINGS_FILE):
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    corpus_embeddings = np.load(EMBEDDINGS_FILE)
else:
    print("Encoding corpus embeddings...")
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    faiss.normalize_L2(corpus_embeddings)
    np.save(EMBEDDINGS_FILE, corpus_embeddings)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")

# -------------------------------
# 4. Load or build FAISS index
# -------------------------------
embedding_dim = corpus_embeddings.shape[1]

if os.path.exists(FAISS_INDEX_FILE):
    print(f"Loading FAISS index from {FAISS_INDEX_FILE}...")
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embedding_dim)  # Inner Product = cosine sim
    index.add(corpus_embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISS index built and saved with {index.ntotal} vectors.")

# -------------------------------
# 5. Dense retrieval function
# -------------------------------
def retrieve_dense(question, top_k=TOP_K):
    query_embedding = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, top_k)
    return [(corpus_ids[i], corpus_texts[i]) for i in I[0]]

# -------------------------------
# 6. Test retrieval on random HotpotQA questions
# -------------------------------
dataset = load_dataset("hotpot_qa", "fullwiki", split="validation[:1000]")
samples = random.sample(list(dataset), NUM_SAMPLES)

for i, sample in enumerate(samples):
    question = sample["question"]
    ground_truth = sample["answer"]
    retrieved = retrieve_dense(question)

    print(f"\n===== Sample {i+1} =====")
    print(f"Question: {question}")
    print(f"Ground Truth Answer: {ground_truth}")
    print("\nRetrieved Contexts:")
    for j, (chunk_id, ctx) in enumerate(retrieved):
        print(f"[{j+1}] ({chunk_id}) {ctx[:300]}{'...' if len(ctx) > 300 else ''}")


# import json
# import pickle
# import time
# from tqdm import tqdm
# from datasets import load_dataset
# import requests

# from langchain_core.documents import Document
# from langchain_community.retrievers import BM25Retriever

# from ragas import evaluate
# from ragas.metrics.collections import (
#     context_precision,
#     context_recall,
#     answer_relevancy,
#     faithfulness,
# )
# from ragas.dataset_schema import EvaluationDataset

# # -------------------------------
# # CONFIG
# # -------------------------------
# CHUNKS_FILE = "hotpotqa_chunks_200w_50o.json"
# BM25_SAVE_FILE = "bm25_retriever.pkl"
# TOP_K = 5
# NUM_EVAL_SAMPLES = 5  # small number for testing
# GEMINI_API_KEY = "AIzaSyAsGVgGaSdVZAvmwKsotV8q9NwPtI0bqpY"  # replace with your Gemini API key
# GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
# GEMINI_CACHE_FILE = "gemini_responses.json"

# # -------------------------------
# # 1. Load chunks
# # -------------------------------
# with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
#     chunks = json.load(f)

# documents = [
#     Document(page_content=chunk["text"], metadata={"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]})
#     for chunk in chunks
# ]

# # -------------------------------
# # 2. Build BM25 Retriever
# # -------------------------------
# bm25_retriever = BM25Retriever.from_documents(documents=documents, k=TOP_K)

# with open(BM25_SAVE_FILE, "wb") as f:
#     pickle.dump(bm25_retriever, f)

# with open(BM25_SAVE_FILE, "rb") as f:
#     bm25_retriever = pickle.load(f)

# # -------------------------------
# # 3. Retriever Function
# # -------------------------------
# def retrieve_contexts(question, k=TOP_K):
#     docs = bm25_retriever.invoke(question)
#     return [doc.page_content for doc in docs[:k]]

# # -------------------------------
# # 4. Gemini 2.0 Flash LLM Function (Corrected)
# # -------------------------------
# def query_gemini(prompt_text: str, max_retries=25, delay=15) -> str:
#     """
#     Sends prompt to Gemini 2.0 Flash and returns model output.
#     Retries automatically on 429 rate-limit errors.
#     """
#     headers = {
#         "Content-Type": "application/json",
#         "X-goog-api-key": GEMINI_API_KEY
#     }

#     payload = {
#         "contents": [
#             {"parts": [{"text": prompt_text}]}
#         ]
#     }

#     for attempt in range(max_retries):
#         try:
#             response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
#             response.raise_for_status()
#             data = response.json()
#             return data["candidates"][0]["content"]
#         except requests.exceptions.HTTPError as e:
#             if e.response.status_code == 429:  # Rate-limit
#                 print(f"Rate limited, retrying in {delay} seconds... (attempt {attempt+1})")
#                 time.sleep(delay)
#             elif e.response.status_code == 400:
#                 print("Bad request:", e.response.text)
#                 raise e
#             else:
#                 print("HTTP error:", e.response.status_code, e.response.text)
#                 raise e
#     raise RuntimeError(f"Failed after {max_retries} retries due to rate limits")

# # -------------------------------
# # 5. Load Gemini cache
# # -------------------------------
# try:
#     with open(GEMINI_CACHE_FILE, "r", encoding="utf-8") as f:
#         gemini_cache = json.load(f)
# except FileNotFoundError:
#     gemini_cache = {}

# # -------------------------------
# # 6. Load HotpotQA Validation Set
# # -------------------------------
# dataset = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{NUM_EVAL_SAMPLES}]")

# # -------------------------------
# # 7. Prepare RAGAS Evaluation Dataset
# # -------------------------------
# eval_samples = []

# for sample in tqdm(dataset, desc="Preparing RAGAS data"):
#     question = sample["question"]
#     ground_truth = sample["answer"]

#     # Retrieve top-k contexts
#     contexts = retrieve_contexts(question)

#     # Create LLM prompt
#     prompt = f"Question: {question}\n\nContext:\n" + "\n".join(contexts) + "\n\nAnswer the question based on the context above."

#     # Use cached response if available
#     if prompt in gemini_cache:
#         response = gemini_cache[prompt]
#     else:
#         response = query_gemini(prompt)
#         gemini_cache[prompt] = response
#         # Save cache incrementally
#         with open(GEMINI_CACHE_FILE, "w", encoding="utf-8") as f:
#             json.dump(gemini_cache, f, indent=2, ensure_ascii=False)

#     # Prepare RAGAS sample
#     eval_samples.append({
#         "user_input": question,
#         "reference": ground_truth,
#         "retrieved_contexts": contexts,
#         "response": response
#     })

# ragas_dataset = EvaluationDataset.from_list(eval_samples)

# # -------------------------------
# # 8. Run RAGAS Evaluation
# # -------------------------------
# results = evaluate(
#     ragas_dataset,
#     metrics=[
#         context_precision,
#         context_recall,
#         answer_relevancy,
#         faithfulness,
#     ]
# )

# print("\n===== RAGAS RESULTS =====")
# print(results)







# import json
# import pickle
# from tqdm import tqdm
# import os
# import openai

# from datasets import load_dataset

# from langchain_core.documents import Document
# from langchain_community.retrievers import BM25Retriever

# from ragas import evaluate
# from ragas.metrics import (
#     context_precision,
#     context_recall,
#     faithfulness,
#     answer_relevancy,
# )
# from ragas.dataset_schema import EvaluationDataset

# # -------------------------------
# # CONFIGURATION
# # -------------------------------
# CHUNKS_FILE = "hotpotqa_chunks_200w_50o.json"
# BM25_SAVE_FILE = "bm25_retriever.pkl"
# TOP_K = 5
# NUM_EVAL_SAMPLES = 50    # small subset to start, increase later if needed
# OPENAI_MODEL = "gpt-3.5-turbo"
# TEMPERATURE = 0.0        # deterministic answers

# # -------------------------------
# # SET OPENAI API KEY
# # -------------------------------
# # Make sure you set the environment variable first:
# # Windows: setx OPENAI_API_KEY "your_key_here"
# openai.api_key = os.environ["OPENAI_API_KEY"]

# # -------------------------------
# # LOAD CHUNKS
# # -------------------------------
# with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
#     chunks = json.load(f)
# print(f"Loaded {len(chunks)} chunks")

# # -------------------------------
# # CONVERT TO LANGCHAIN DOCUMENTS
# # -------------------------------
# documents = [
#     Document(
#         page_content=chunk["text"],
#         metadata={
#             "chunk_id": chunk["chunk_id"],
#             "doc_id": chunk["doc_id"]
#         }
#     )
#     for chunk in chunks
# ]
# print(f"Converted to {len(documents)} LangChain Documents")

# # -------------------------------
# # BUILD BM25 RETRIEVER
# # -------------------------------
# bm25_retriever = BM25Retriever.from_documents(documents=documents, k=TOP_K)
# print("BM25 retriever built")

# # Save retriever
# with open(BM25_SAVE_FILE, "wb") as f:
#     pickle.dump(bm25_retriever, f)
# print("BM25 retriever saved to disk")

# # Reload retriever
# with open(BM25_SAVE_FILE, "rb") as f:
#     bm25_retriever = pickle.load(f)
# print("BM25 retriever reloaded")

# # -------------------------------
# # RETRIEVAL FUNCTION
# # -------------------------------
# def retrieve_contexts(question, k=TOP_K):
#     docs = bm25_retriever.invoke(question)
#     return [doc.page_content for doc in docs[:k]]

# # -------------------------------
# # LOAD HOTPOTQA VALIDATION
# # -------------------------------
# dataset = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{NUM_EVAL_SAMPLES}]")
# print(f"Loaded {len(dataset)} HotpotQA validation samples")

# # -------------------------------
# # LLM ANSWER GENERATION
# # -------------------------------
# def generate_answer(question, contexts):
#     """
#     Generates answer using GPT-3.5-turbo API
#     """
#     context_text = "\n".join(contexts)
#     prompt = f"Use the following context to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"

#     response = openai.ChatCompletion.create(
#         model=OPENAI_MODEL,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt},
#         ],
#         temperature=TEMPERATURE
#     )

#     answer = response.choices[0].message.content.strip()
#     return answer

# # -------------------------------
# # PREPARE RAGAS EVALUATION DATA
# # -------------------------------
# eval_samples = []

# for sample in tqdm(dataset, desc="Preparing RAGAS data with LLM"):
#     question = sample["question"]
#     ground_truth = sample["answer"]

#     # 1️⃣ Retrieve contexts using BM25
#     contexts = retrieve_contexts(question)

#     # 2️⃣ Generate answer with LLM
#     llm_response = generate_answer(question, contexts)

#     # 3️⃣ Append to evaluation dataset
#     eval_samples.append({
#         "user_input": question,
#         "reference": ground_truth,
#         "retrieved_contexts": contexts,
#         "response": llm_response
#     })

# # Convert to RAGAS EvaluationDataset
# ragas_dataset = EvaluationDataset.from_list(eval_samples)

# # -------------------------------
# # RUN RAGAS EVALUATION
# # -------------------------------
# results = evaluate(
#     ragas_dataset,
#     metrics=[
#         context_precision,
#         context_recall,
#         answer_relevancy,
#         faithfulness,
#     ]
# )

# print("\n===== RAGAS RESULTS =====")
# print(results)
