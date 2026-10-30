import chromadb
import open_clip
import torch


DB_DIR = "chroma_db"

device = "cuda" if torch.cuda.is_available() else "cpu"


model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")


client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name="family_photos",
    embedding_function=None
)

def embed_text(text: str):

    tokens = tokenizer([text]).to(device)

    with torch.no_grad():

        embedding = model.encode_text(tokens)

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

    return embedding.cpu().numpy()[0].tolist()

query = input("Search photos: ")

query_embedding = embed_text(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\nMatches:\n")

for metadata, distance in zip(
    results["metadatas"][0],
    results["distances"][0]
):
    print(metadata["filename"])
    print("Path:", metadata["path"])
    print("Distance:", distance)
    print()