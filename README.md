# Demographic-Aware Graph Neural Network Approach to Personalized Outfit Recommendation for Cold-Start Scenarios

StyleSync AI is a fashion recommendation system designed to address cold-start challenges using a Heterogeneous Graph Neural Network (GATv2). The system models relationships between outfits, items, demographic attributes (age, gender, occasion), and user preference images to generate personalized recommendations without requiring prior user interaction data.

It integrates CLIP-based visual feature extraction to capture item-level representations and combines them with graph-based reasoning for improved recommendation quality. Cold-start is handled through demographic-driven matching for new users and similarity-based graph augmentation for new items.

The system also includes Explainable AI (XAI) for transparent recommendations and a Virtual Try-On module for visualization.

📊 Dataset Information

This project utilizes both large-scale fashion datasets and custom demographic data:

MM-DeepFashion Dataset (Kaggle):
https://www.kaggle.com/datasets/silverstone1903/deep-fashion-multimodal

Includes about 44K images with rich annotations, used for training CLIP-based feature extraction and YOLO-based detection models.

Survey-Based Demographic Dataset:
A custom dataset collected to capture user preferences across age, gender, and occasion for demographic-aware recommendation.
