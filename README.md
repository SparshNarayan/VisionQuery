I-Based Image Classification & Text-to-Image Semantic Search

Hackathon Domain: AI / Machine Learning
Hackathon: GEHU Himtal Hackathon
Team Size: 4

Problem Statement

Finding relevant images from large datasets using natural language descriptions is still a challenge in many real-world applications such as surveillance, e-commerce, media management, and smart search engines.

Traditional systems rely on tags or metadata, which are often incomplete or inaccurate.

Solution Overview

This project combines two AI pipelines to build an intelligent multimodal system:

Image Classification (Animal vs Person)

Text-to-Image Semantic Search using natural language

The system understands both visual content and human language.

Key Features

• Image Classification using MobileNetV2 (Transfer Learning)
• Text-to-Image Semantic Search using CLIP
• Zero-shot learning (no retraining required for text queries)
• Lightweight, modular, and hackathon-ready
• Works on CPU and GPU

Project Structure

NEW FOLDER (2)
|
|-- Dataset/
| |-- animal/
| |-- person/
|
|-- classifier_train.py
|-- classifier_predict.py
|-- classifier_model.h5
|-- labels.txt
|
|-- text_image_search.py
|-- requirements.txt
|-- README.md

System Flow

Image Classification Flow

Input Image
→ Image Preprocessing (224×224)
→ MobileNetV2 Feature Extractor
→ Dense + Softmax Layer
→ Class Prediction with Confidence

Text-to-Image Search Flow

Text Query
→ CLIP Text Encoder
→ CLIP Image Encoder
→ Cosine Similarity
→ Best Matching Image

High-Level Architecture

User
|
|-- Image Input → Image Classifier (TensorFlow)
| → Output: Class + Confidence
|
|-- Text Query → CLIP Model (PyTorch)
→ Output: Best Matching Image

Technologies Used

Image Classification: TensorFlow, Keras
Model Architecture: MobileNetV2
Text-to-Image Search: CLIP
Programming Language: Python
Image Processing: PIL
Hardware Support: CPU / GPU

How to Run the Project

Step 1: Install Dependencies

pip install -r requirements.txt

Step 2: Train the Image Classifier (Optional)

python classifier_train.py

Step 3: Predict Image Class

python classifier_predict.py

Step 4: Run Text-to-Image Semantic Search

python text_image_search.py

Scalability & Future Growth

• Precompute and store image embeddings
• Integrate vector databases (FAISS / Pinecone)
• Separate inference services for classification and search
• Modular architecture for easy expansion

Current Limitations (Round 1)

• Small dataset (hackathon constraint)
• Command-line based interaction
• No web interface

Planned Improvements (Round 2)

• Web Interface using Flask or FastAPI
• Multi-class image classification
• Advanced text queries (attributes, actions, clothing)
• Vector database integration
• User upload and search history
• Improved evaluation metrics

Originality & Innovation

• Combines supervised image classification with zero-shot semantic search
• Uses state-of-the-art CLIP model
• Fully original pipeline design
• No copied boilerplate or templates

Team Information

Team Size: 4
Hackathon: GEHU Himtal Hackathon

License

This project is developed for educational and hackathon purposes only.
