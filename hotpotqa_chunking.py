import json
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
CHUNK_SIZE = 200        # number of words per chunk
CHUNK_OVERLAP = 50      # overlap between chunks (in words)
MIN_CHUNK_WORDS = 30    # drop chunks smaller than this (optional)

INPUT_CORPUS_FILE = "hotpotqa_corpus_fullpages.json"
OUTPUT_CHUNKS_FILE = "hotpotqa_chunks_200w_50o.json"


# -------------------------------
# Load corpus
# -------------------------------
with open(INPUT_CORPUS_FILE, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Loaded {len(corpus)} documents")


# -------------------------------
# Chunking function
# -------------------------------
def chunk_corpus(
    corpus,
    chunk_size,
    chunk_overlap,
    min_chunk_words=0
):
    chunks = []
    chunk_id = 0
    stride = chunk_size - chunk_overlap

    for doc in tqdm(corpus, desc="Chunking corpus"):
        doc_id = doc["id"]
        text = doc["text"]

        words = text.split()
        num_words = len(words)

        start = 0
        while start < num_words:
            end = start + chunk_size
            chunk_words = words[start:end]

            if len(chunk_words) < min_chunk_words:
                break

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "doc_id": doc_id,
                "text": chunk_text,
                "word_count": len(chunk_words)
            })

            chunk_id += 1
            start += stride

    return chunks


# -------------------------------
# Run chunking
# -------------------------------
chunks = chunk_corpus(
    corpus=corpus,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    min_chunk_words=MIN_CHUNK_WORDS
)

print(f"\nTotal chunks created: {len(chunks)}")


# -------------------------------
# Chunk statistics
# -------------------------------
chunk_lengths = [c["word_count"] for c in chunks]

print("\nChunk Word Statistics:")
print(f"Max words in chunk: {max(chunk_lengths)}")
print(f"Min words in chunk: {min(chunk_lengths)}")
print(f"Avg words per chunk: {sum(chunk_lengths) / len(chunk_lengths):.2f}")


# -------------------------------
# Save chunks to disk
# -------------------------------
with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"\nChunks saved to: {OUTPUT_CHUNKS_FILE}")


# -------------------------------
# Preview sample chunk
# -------------------------------
print("\nExample chunk:")
print(json.dumps(chunks[0], indent=2, ensure_ascii=False))
