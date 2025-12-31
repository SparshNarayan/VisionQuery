import clip
import torch
import os
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = "dataset/person"
image_files = os.listdir(image_folder)

image_embeddings = []
valid_images = []

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(image)
        image_embeddings.append(emb)
        valid_images.append(img_name)

image_embeddings = torch.cat(image_embeddings)

text = input("Describe image: ")
text_tokens = clip.tokenize([text]).to(device)

with torch.no_grad():
    text_emb = model.encode_text(text_tokens)

similarity = (text_emb @ image_embeddings.T).squeeze()
best_idx = similarity.argmax().item()

print("Best matched image:", valid_images[best_idx])
print("Confidence:", round(float(similarity[best_idx]) * 100, 2), "%")
