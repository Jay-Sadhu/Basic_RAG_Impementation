from datasets import load_dataset
import json
from tqdm import tqdm

# -------------------------------
# 1. Load HotpotQA dataset
# -------------------------------
dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")

# -------------------------------
# 2. Build corpus
# -------------------------------
def build_corpus(dataset):
    corpus = []
    doc_id = 0
    word_counts = []  # Store number of words for each document

    for sample in tqdm(dataset, desc="Building corpus"):
        titles = sample["context"]["title"]
        sentences_list = sample["context"]["sentences"]

        # Each Wikipedia page becomes one document
        for title, sentences in zip(titles, sentences_list):
            doc_text = title + ": " + " ".join(sentences)
            corpus.append({
                "id": f"doc_{doc_id}",
                "text": doc_text
            })
            # Count words in this document
            word_counts.append(len(doc_text.split()))
            doc_id += 1

    return corpus, word_counts

# -------------------------------
# 3. Create the corpus
# -------------------------------
corpus, word_counts = build_corpus(dataset)

print("Total documents in corpus:", len(corpus))
print("Example document:\n", corpus[0])

# -------------------------------
# 4. Compute word statistics
# -------------------------------
max_words = max(word_counts)
min_words = min(word_counts)
avg_words = sum(word_counts) / len(word_counts)

print("\nDocument Word Statistics:")
print(f"Highest number of words in a doc: {max_words}")
print(f"Lowest number of words in a doc: {min_words}")
print(f"Average number of words per doc: {avg_words:.2f}")

# -------------------------------
# 5. (Optional) Save corpus to disk
# -------------------------------
with open("hotpotqa_corpus_fullpages.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)


# from datasets import load_dataset

# dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
# sample = dataset[0]

# print("QUESTION:")
# print(sample["question"])

# print("\nANSWER:")
# print(sample["answer"])


# titles = sample["context"]["title"]
# sentences = sample["context"]["sentences"]

# print("\nNUMBER OF WIKIPEDIA PAGES IN CONTEXT:", len(titles))

# for i, (title, sent_list) in enumerate(zip(titles, sentences)):
#     print("\n" + "=" * 60)
#     print(f"WIKIPEDIA PAGE {i+1}")
#     print("TITLE:", title)
#     print("NUMBER OF SENTENCES:", len(sent_list))
    
#     print("\nSENTENCES:")
#     for j, sent in enumerate(sent_list):
#         print(f"  {j+1}. {sent}")




# sample = dataset[1]

# print("QUESTION:")
# print(sample["question"])

# print("\nANSWER:")
# print(sample["answer"])


# titles = sample["context"]["title"]
# sentences = sample["context"]["sentences"]

# print("\nNUMBER OF WIKIPEDIA PAGES IN CONTEXT:", len(titles))

# for i, (title, sent_list) in enumerate(zip(titles, sentences)):
#     print("\n" + "=" * 60)
#     print(f"WIKIPEDIA PAGE {i+1}")
#     print("TITLE:", title)
#     print("NUMBER OF SENTENCES:", len(sent_list))
    
#     print("\nSENTENCES:")
#     for j, sent in enumerate(sent_list):
#         print(f"  {j+1}. {sent}")
