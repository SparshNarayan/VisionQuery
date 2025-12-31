🧠 AI-Based Image Classification & Text-to-Image Search

Hackathon Domain: AI / Machine Learning
Hackathon: GEHU Himtal Hackathon
Team Size: 4

📌 Problem Statement

Finding relevant images from large datasets using natural language descriptions is still a major challenge in real-world applications such as:

Surveillance systems

E-commerce platforms

Media & asset management

Smart search engines

Traditional image search relies on tags or metadata, which is often incomplete or inaccurate.

🚀 Our Solution

We built an intelligent AI system that understands both visual content and human language by combining:

Image Classification (Animal vs Person)

Text-to-Image Semantic Search (e.g., “person wearing red dress”)

This hybrid approach enables accurate filtering + semantic understanding.

🧠 Core Features
✅ Image Classification Pipeline

Classifies an input image into:

Animal

Person

Built using MobileNetV2 (Transfer Learning)

Fast, lightweight, and efficient

✅ Text-to-Image Semantic Search

Accepts natural language queries

Finds the most relevant image using CLIP embeddings

Zero-shot learning (no retraining required)

🗂️ Project Structure
NEW FOLDER (2)
│
├── Dataset/
│   ├── animal/
│   └── person/
│
├── classifier_train.py        # Train image classifier
├── classifier_predict.py      # Predict class from image
├── classifier_model.h5        # Trained model
├── labels.txt                 # Class labels
├── text_image_search.py       # CLIP-based semantic search
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

🔁 System Flow
🖼️ Image Classification Flow
Input Image
     ↓
Image Preprocessing (224×224)
     ↓
MobileNetV2 Feature Extractor
     ↓
Dense + Softmax Layer
     ↓
Class Prediction + Confidence

📝 Text-to-Image Search Flow
Text Query
     ↓
CLIP Text Encoder
     ↓
CLIP Image Embeddings
     ↓
Cosine Similarity
     ↓
Best Matching Image

🧩 High-Level Architecture
User
 │
 ├── Image Input ──▶ Image Classifier (TensorFlow)
 │                   └── Output: Class + Confidence
 │
 └── Text Query ──▶ CLIP Model (PyTorch)
                     └── Output: Best Matching Image

⚙️ Technologies Used
Component	Technology
Image Classification	TensorFlow, Keras
Model Architecture	MobileNetV2
Text-to-Image Search	CLIP
Backend Logic	Python
Image Processing	PIL
Hardware Support	CPU / GPU
🧪 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train the Classifier (Optional)
python classifier_train.py

3️⃣ Predict Image Class
python classifier_predict.py

4️⃣ Run Text-to-Image Search
python text_image_search.py

📈 Scalability & Growth Plan
🔹 Performance & Scaling

Precompute and store image embeddings

Integrate vector databases:

FAISS

Pinecone

Separate inference services for:

Classification

Semantic search

🔹 Reliability

Input validation for images & text

Automatic CPU fallback if GPU unavailable

Modular architecture → one module failure does not crash system

⚠️ Current Limitations (Round 1)

Small dataset (hackathon constraint)

CLI-based interaction

No web interface

🔮 Planned Improvements (Round 2)

✔️ Web Interface (Flask / FastAPI)
✔️ Multi-class Image Classification
✔️ Advanced Text Queries (attributes, actions, clothing)
✔️ Vector Database Integration
✔️ User Upload & Search History
✔️ Improved Evaluation Metrics

🏆 Originality & Innovation

Combines supervised classification with zero-shot semantic search

Uses state-of-the-art CLIP model

Fully original pipeline design

No copied templates or boilerplate projects

👥 Team

Team Size: 4
Hackathon: GEHU Himtal Hackathon

📜 License

This project is developed for educational and hackathon purposes.
