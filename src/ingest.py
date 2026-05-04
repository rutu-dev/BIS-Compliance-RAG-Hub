import os
from pypdf import PdfReader
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup Environment
load_dotenv()

# -------------------------------
# 🔧 Optimized Chunking for Accuracy
# -------------------------------
def split_text(text, chunk_size=1000, overlap=200):
    """
    Increases chunk size to 1000 to keep IS codes and their 
    descriptions together in the same context window.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def process_pdf(file_path):
    reader = PdfReader(file_path)
    chunks = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            # Clean text: remove extra whitespace but preserve semantic flow
            text = " ".join(text.split())

            # Skip pages that are mostly empty or just page numbers
            if len(text) < 100:
                continue

            # Split with larger overlap to prevent data loss at boundaries
            splits = split_text(text, chunk_size=1000, overlap=200)

            for j, chunk in enumerate(splits):
                chunks.append({
                    "id": f"page_{i+1}_chunk_{j}",
                    "text": chunk,
                    "metadata": {"page": i + 1, "source": "official_dataset"}
                })

    return chunks

def process_text_file(file_path):
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splits = split_text(text, chunk_size=1000, overlap=200)

    for i, chunk in enumerate(splits):
        chunks.append({
            "id": f"text_chunk_{i}",
            "text": chunk,
            "metadata": {"source": "extra_data"}
        })

    return chunks

def store_in_db(chunks):
    # Use HuggingFace embedding (Sentence Transformers)
    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Path to the persistent database
    client = chromadb.PersistentClient(path="./bis_vector_db")

    # Get or create the collection
    collection = client.get_or_create_collection(
        name="bis_standards",
        embedding_function=hf_ef
    )

    print(f"Starting upload of {len(chunks)} chunks to Vector DB...")
    batch_size = 100 # Increased batch size for faster processing

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        try:
            collection.add(
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
                ids=[c["id"] for c in batch]
            )
            print(f"✅ Indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")
        except Exception as e:
            print(f"❌ Error at batch {i // batch_size + 1}: {e}")

    print("🏁 Indexing finished! Vector DB is ready.")

if __name__ == "__main__":
    all_data = []

    # Process the official rulebook dataset
    if os.path.exists("dataset.pdf"):
        print("Processing dataset.pdf...")
        pdf_data = process_pdf("dataset.pdf")
        all_data.extend(pdf_data)
    else:
        print("⚠️ Warning: dataset.pdf not found!")

    # Process any supplementary text data
    if os.path.exists("data_extra.txt"):
        print("Processing data_extra.txt...")
        extra_data = process_text_file("data_extra.txt")
        all_data.extend(extra_data)

    if all_data:
        store_in_db(all_data)
    else:
        print("❌ No data found to index.")