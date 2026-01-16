# HotpotQA Dense Retrieval Pipeline (RAG-ready)

This repository implements a **complete dense retrieval pipeline** for the **HotpotQA (fullwiki)** dataset, designed for **Retrieval-Augmented Generation (RAG)** experiments and embedding model evaluation.

The pipeline covers:

* Building a Wikipedia-style corpus from HotpotQA
* Chunking long documents with overlap
* Encoding chunks using a sentence embedding model
* Indexing embeddings with FAISS
* Performing dense retrieval for question answering

---

## 📌 Overview of the Pipeline

```
HotpotQA Dataset
      ↓
Corpus Construction (Full Wikipedia Pages)
      ↓
Word-based Chunking (200 words, 50 overlap)
      ↓
Sentence Embedding (all-MiniLM-L6-v2)
      ↓
FAISS Index (Cosine Similarity)
      ↓
Dense Retrieval for Questions
```

This design closely mirrors **production-grade RAG retrievers** and is ideal for:

* Embedding model benchmarking
* Chunking strategy experiments
* Retriever evaluation before LLM integration

---

## 📂 Project Structure

```
.
├── build_corpus.py            # Build full-page corpus from HotpotQA
├── chunk_corpus.py            # Chunk documents with overlap
├── dense_retrieval.py         # Encode, index, and retrieve
├── hotpotqa_corpus_fullpages.json
├── hotpotqa_chunks_200w_50o.json
├── hotpotqa_embeddings.npy
├── hotpotqa_faiss.index
└── README.md
```

---

## 🧱 Step 1: Build the Corpus

Each Wikipedia page from HotpotQA is converted into a **single document**.

**Key points:**

* Titles are prepended to page text
* Each page gets a unique `doc_id`
* Word statistics are computed for analysis

**Output:**

* `hotpotqa_corpus_fullpages.json`

This ensures the retriever works on **realistic long-form documents**, similar to real-world knowledge bases.

---

## ✂️ Step 2: Chunking Strategy

Long documents are split into overlapping chunks to balance:

* Semantic completeness
* Retrieval granularity

**Configuration:**

* Chunk size: **200 words**
* Overlap: **50 words**
* Minimum chunk length: **30 words**

**Why overlap?**
Overlap preserves context across chunk boundaries, reducing information loss during retrieval.

**Output:**

* `hotpotqa_chunks_200w_50o.json`

Each chunk contains:

```json
{
  "chunk_id": "chunk_42",
  "doc_id": "doc_10",
  "text": "...",
  "word_count": 200
}
```

---

## 🔢 Step 3: Embedding Generation

Chunks are embedded using a **Sentence Transformer**:

* Model: `all-MiniLM-L6-v2`
* Framework: `sentence-transformers`
* Device: GPU (if available)

**Important details:**

* Embeddings are **L2-normalized**
* Stored on disk for reuse

**Output:**

* `hotpotqa_embeddings.npy`

This enables fast experimentation without re-encoding the corpus.

---

## ⚡ Step 4: FAISS Indexing

Embeddings are indexed using **FAISS** for efficient similarity search.

* Index type: `IndexFlatIP`
* Similarity: **Inner Product (Cosine Similarity)**

The index is saved and reloadable:

* `hotpotqa_faiss.index`

This setup is simple, fast, and ideal for research-scale experiments.

---

## 🔍 Step 5: Dense Retrieval

Given a question:

1. Encode the query
2. Normalize the embedding
3. Search FAISS index
4. Retrieve top-k chunks

```python
retrieve_dense(question, top_k=5)
```

**Returned:**

* Chunk IDs
* Corresponding chunk text

This retriever can be directly plugged into:

* RAG pipelines
* LLM prompting
* Retriever evaluation frameworks

---

## 🧪 Evaluation Setup

The retriever is tested on **random validation questions** from HotpotQA.

For each sample:

* Question
* Ground-truth answer
* Top-k retrieved chunks

This enables **qualitative inspection** of retrieval quality before quantitative metrics (Recall@K, MRR, etc.).

---

## 🚀 Extensions & Next Steps

You can easily extend this pipeline to:

* ✅ Compare multiple embedding models
* ✅ Add cross-encoder re-ranking
* ✅ Integrate with LLMs (RAG)
* ✅ Compute Recall@K using supporting facts
* ✅ Experiment with different chunk sizes

---

## 📚 Recommended Use Cases

* Embedding model benchmarking (BEIR-style)
* RAG retriever prototyping
* HotpotQA multi-hop QA research
* FAISS-based semantic search demos

---

## 🧠 Notes

* This pipeline intentionally avoids sparse retrieval to isolate **dense embedding behavior**
* Chunking and normalization choices follow best practices from modern RAG systems

---

## 👤 Author

**Jay Vardhan**
Applied AI | Dense Retrieval | RAG Systems

---

If you want, I can also provide:

* 📊 Retrieval metrics code
* 🔁 Multi-embedding fusion retriever
* 🧩 RAG + LLM integration
* 📦 Clean modular refactor
