import chromadb
from chromadb.utils import embedding_functions

# Embedding
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Load DB
client = chromadb.PersistentClient(path="./bis_vector_db")

collection = client.get_collection(
    name="bis_standards",
    embedding_function=hf_ef
)


def clean_sentence(s):
    s = s.strip()
    if len(s) < 40:
        return None

    bad_words = ["annex", "isbn", "figure", "table", "chapter", "reference"]
    if any(b in s.lower() for b in bad_words):
        return None

    return s


def generate_answer(query, docs):

    query_words = set(query.lower().split())

    scored_sentences = []

    for doc in docs:
        sentences = doc.split(".")
        for s in sentences:
            s = clean_sentence(s)
            if not s:
                continue

            score = sum(word in s.lower() for word in query_words)

            if score > 0:
                scored_sentences.append((score, s))

    if not scored_sentences:
        print("\n🤖 Final Answer:\n👉 No clear answer found in document.\n")
        return

    # Sort by relevance
    scored_sentences.sort(reverse=True, key=lambda x: x[0])

    # Take top 3 BEST sentences
    best = [s for _, s in scored_sentences[:3]]

    print("\n🤖 Final Answer:\n")

    for i, line in enumerate(best, 1):
        print(f"{i}. {line.strip()}.")

    print("\n" + "-" * 50)


def query_db(query):

    print("\n🔎 Processing query...\n")

    results = collection.query(
        query_texts=[query],
        n_results=25
    )

    docs = results["documents"][0]

    generate_answer(query, docs)


if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or type 'exit'): ")

        if q.lower() == "exit":
            break

        query_db(q)