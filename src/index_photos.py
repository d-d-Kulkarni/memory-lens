from pathlib import Path

import chromadb
import open_clip
import torch
from PIL import Image


PHOTO_DIR = Path("photos")
DB_DIR = "chroma_db"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(device)
model.eval()

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name="family_photos",
    embedding_function=None
)

extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_paths = [
    path
    for path in PHOTO_DIR.rglob("*")
    if path.suffix.lower() in extensions
]

print(f"Found {len(image_paths)} images")


for index, image_path in enumerate(image_paths):

    try:
        image = Image.open(image_path).convert("RGB")

        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image_tensor)

            embedding = embedding / embedding.norm(
                dim=-1,
                keepdim=True
            )

        vector = embedding.cpu().numpy()[0].tolist()

        collection.upsert(
            ids=[str(image_path)],
            embeddings=[vector],
            metadatas=[
                {
                    "path": str(image_path),
                    "filename": image_path.name
                }
            ]
        )

        print(
            f"[{index + 1}/{len(image_paths)}] "
            f"Indexed {image_path.name}"
        )

    except Exception as exc:
        print("Failed:", image_path, exc)