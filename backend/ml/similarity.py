import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load embedding model once
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")


# -----------------------------
# Load dataset
# -----------------------------
def load_dataset():
    print("Loading text dataset...")

    df = pd.read_csv("backend/data/processed/train_text.csv")

    df["combined_text"] = (
        "Branch Notes: " + df["branch_notes"]
        + "\n\nCall Summary: " + df["call_summary"]
        + "\n\nVerification Notes: " + df["verification_notes"]
    )

    print("Dataset loaded!")
    return df


# -----------------------------
# Build ChromaDB
# -----------------------------
def build_database(limit=None):

    df = load_dataset()

    if limit is None:
        sample_df = df
    else:
        sample_df = df.head(limit)

    print(f"Generating embeddings for {len(sample_df)} borrowers...")

    embeddings = model.encode(
        sample_df["combined_text"].tolist(),
        show_progress_bar=True
    )

    print("Embeddings generated!")

    client = chromadb.PersistentClient(
        path="backend/rag/chroma_db"
    )

    # Delete old collection if it exists
    try:
        client.delete_collection("borrowers")
    except:
        pass

    collection = client.get_or_create_collection(
        name="borrowers"
    )

    print("Saving embeddings in batches...")

    BATCH_SIZE = 5000

    ids = sample_df["LoanID"].tolist()
    documents = sample_df["combined_text"].tolist()
    embedding_list = embeddings.tolist()

    for i in range(0, len(ids), BATCH_SIZE):

        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            documents=documents[i:i + BATCH_SIZE],
            embeddings=embedding_list[i:i + BATCH_SIZE]
        )

        print(f"Stored {min(i + BATCH_SIZE, len(ids))} / {len(ids)} borrowers")

    print(f"\nDatabase created successfully!")
    print(f"Number of borrowers stored: {collection.count()}")

    return collection


# -----------------------------
# Search similar borrowers
# -----------------------------
def search_similar_cases(query_text, top_k=5):

    client = chromadb.PersistentClient(
        path="backend/rag/chroma_db"
    )

    collection = client.get_collection("borrowers")

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    return results


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    print("\nBuilding vector database...\n")

    collection = build_database()

    df = load_dataset()

    query = df["combined_text"].iloc[0]

    results = search_similar_cases(query)

    print("\nTop 5 Similar Borrowers:\n")

    for i in range(len(results["ids"][0])):
        print(f"Rank {i + 1}")
        print("LoanID :", results["ids"][0][i])
        print("Distance:", results["distances"][0][i])
        print("-" * 50)