🧠 AI-Based Image Classification & Text-to-Image Search

Hackathon Domain: AI / ML
Team Size: 4
Hackathon: GEHU Himtal Hackathon

📌 Problem Statement

Finding relevant images from large datasets using natural language descriptions is still a challenge in many real-world applications such as surveillance, e-commerce, media management, and smart search engines.

Our project solves this by combining:

Image Classification (Animal vs Person)

Text-to-Image Semantic Search (e.g., “person wearing red dress”)

This creates an intelligent system that understands both visual content and human language.

🚀 Solution Overview

Our system consists of two AI pipelines:

Image Classification Pipeline

Classifies an input image into predefined categories (Animal / Person).

Built using MobileNetV2 (Transfer Learning).

Text-to-Image Search Pipeline

Takes a natural language query.

Finds the most semantically similar image using CLIP (OpenAI) embeddings.

Both pipelines are lightweight, modular, and hackathon-ready.

🗂️ Project Structure
NEW FOLDER (2)
│
├── Dataset/
│   ├── animal/
│   └── person/
│
├── classifier_train.py
├── classifier_predict.py
├── classifier_model.h5
├── labels.txt
├── text_image_search.py
├── requirements.txt
└── README.md

🔁 System Flow (High-Level)
Image Classification Flow
Input Image
     ↓
Image Preprocessing (224x224)
     ↓
MobileNetV2 Feature Extractor
     ↓
Dense Softmax Layer
     ↓
Class Prediction + Confidence

Text-to-Image Search Flow
Text Query
     ↓
CLIP Text Encoder
     ↓
Image Embeddings (CLIP Image Encoder)
     ↓
Cosine Similarity Matching
     ↓
Best Matched Image

🧩 Basic System Diagram (Bird’s Eye View)
User
 │
 ├── Image Input ──▶ Image Classifier (TensorFlow)
 │                   │
 │                   └── Output: Class + Confidence
 │
 └── Text Query ──▶ CLIP Model (PyTorch)
                     │
                     └── Output: Best Matching Image

⚙️ Technologies Used
Component	Technology
Image Classification	TensorFlow, Keras
Model Architecture	MobileNetV2
Text-Image Search	OpenAI CLIP
Backend Logic	Python
Image Processing	PIL
Hardware Support	CPU / GPU

📈 Scalability & Growth Plan
How the system handles more users:

Pre-compute and store image embeddings.

Use vector databases (FAISS / Pinecone) in future.

Separate inference services for classification & search.

Failure Handling:

Input validation for images & text

Graceful fallback to CPU if GPU unavailable

Modular design → failure in one module doesn’t crash system

🧪 Current Limitations (Round 1)

Small dataset (hackathon constraint)

CLI-based interaction

No web interface yet

🔮 Planned Improvements for Round 2 (Mandatory Section)

✔️ Web Interface (Flask / FastAPI)
✔️ Multi-class Classification (beyond Animal/Person)
✔️ Advanced Text Queries (attributes, actions, clothing)
✔️ Vector Database Integration
✔️ User Upload & Search History
✔️ Better Evaluation Metrics

🏆 Originality & Innovation

Combines supervised classification with zero-shot semantic search

Uses state-of-the-art CLIP model

Fully original pipeline design

No copied templates or boilerplate projects

