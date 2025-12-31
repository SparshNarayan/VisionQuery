# 🧠 AI-Based Image Classification & Text-to-Image Semantic Search

**Hackathon Domain:** AI / Machine Learning  
**Hackathon:** GEHU Himtal Hackathon  
**Team Size:** 4  

---

## 📌 Problem Statement

Finding relevant images from large datasets using natural language descriptions is still a challenge in many real-world applications such as surveillance, e-commerce, media management, and smart search engines.

Traditional image search systems rely heavily on tags or metadata, which are often incomplete, inaccurate, or manually generated.

---

## 🚀 Solution Overview

This project builds an intelligent multimodal system by combining two AI pipelines:

- **Image Classification** (Animal vs Person)
- **Text-to-Image Semantic Search** using natural language queries

The system understands both visual content and human language, enabling accurate and meaningful image retrieval.

---

## ✨ Key Features

- Image classification using **MobileNetV2 (Transfer Learning)**
- Text-to-image semantic search using **CLIP**
- Zero-shot learning (no retraining required for text queries)
- Lightweight, modular, and hackathon-ready architecture
- Supports both **CPU and GPU**

---
## 🗂️ Project Structure
NEW FOLDER (2)
│
├── Dataset/
│   ├── animal/                 # Animal images
│   └── person/                 # Person images
│
├── classifier_train.py         # Train image classification model
├── classifier_predict.py       # Predict class from input image
├── classifier_model.h5         # Trained MobileNetV2 model
├── labels.txt                  # Class labels (Animal / Person)
│
├── text_image_search.py        # CLIP-based semantic search
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation



---

## 🔁 System Flow

### 🖼️ Image Classification Flow




---

## 🔁 System Flow

### 🖼️ Image Classification Flow



Input Image
↓
Image Preprocessing (224×224)
↓
MobileNetV2 Feature Extractor
↓
Dense + Softmax Layer
↓
Class Prediction + Confidence


### 📝 Text-to-Image Search Flow


Text Query
↓
CLIP Text Encoder
↓
CLIP Image Encoder
↓
Cosine Similarity
↓
Best Matching Image


---

## 🧩 High-Level Architecture

User
|
|-- Image Input → Image Classifier (TensorFlow)
| → Output: Class + Confidence
|
|-- Text Query → CLIP Model (PyTorch)
→ Output: Best Matching Image


---

## ⚙️ Technologies Used

| Component | Technology |
|---------|------------|
| Image Classification | TensorFlow, Keras |
| Model Architecture | MobileNetV2 |
| Text-to-Image Search | CLIP |
| Programming Language | Python |
| Image Processing | PIL |
| Hardware Support | CPU / GPU |

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

pip install -r requirements.txt


2️⃣ Train the Image Classifier (Optional)
python classifier_train.py

3️⃣ Predict Image Class
python classifier_predict.py

4️⃣ Run Text-to-Image Semantic Search
python text_image_search.py

📈 Scalability & Future Growth

Pre-compute and store image embeddings

Integrate vector databases (FAISS / Pinecone)

Separate inference services for classification and semantic search

Modular architecture for easy extension

⚠️ Current Limitations (Round 1)

Small dataset due to hackathon constraints

Command-line based interaction

No web interface

🔮 Planned Improvements (Round 2)

Web interface using Flask / FastAPI

Multi-class image classification

Advanced text queries (attributes, actions, clothing)

Vector database integration

User upload and search history

Improved evaluation metrics

🏆 Originality & Innovation

Combines supervised image classification with zero-shot semantic search

Uses state-of-the-art CLIP model

Fully original pipeline design

No copied templates or boilerplate projects

👥 Team Information

Team Size: 4
Hackathon: GEHU Himtal Hackathon

📜 License

This project is developed for educational and hackathon purposes only.



