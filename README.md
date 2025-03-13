# Chatbot Classification

## Overview

The **Chatbot Classification** project is designed to assess the quality and relevance of chatbot responses through machine learning and natural language processing techniques. Using models like BERT and BART, the system performs text classification tasks to evaluate the success of chatbot interactions. The project uses FastAPI for building the backend and provides an API for integrating chatbot evaluation into different applications.

This repository includes multiple models and pipelines for similarity calculation, sentiment analysis, and zero-shot classification to classify chatbot interactions as either "Successful" or "Unsuccessful."

## Features

- **BERT-based Embeddings**: Utilizes BERT (Bidirectional Encoder Representations from Transformers) for calculating semantic similarity between user messages and bot responses.
- **BART-based Zero-Shot Classification**: Implements zero-shot classification with BART to classify chatbot responses as "Successful" or "Unsuccessful."
- **Sentiment Analysis**: Leverages Hugging Face transformers to evaluate the sentiment of user inputs and bot responses.
- **FastAPI Integration**: Exposes an API for interacting with the classification models in real time, making it easier to integrate the solution into chatbots or customer service systems.
- **Similarity Calculation**: Computes semantic similarity scores between user and bot messages using advanced NLP techniques.

## Getting Started

### Prerequisites

- Python 3.7+
- PIP (Python package installer)
- Git (for version control)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ysidm2025/chatbot_classification.git

2. Navigate to the project directory:
  cd chatbot_classification

3. Install the necessary dependencies :
   pip install -r requirements.txt
   
4. Download the required datasets :
   python -m nltk.downloader stopwords punkt

## Running the FastAPI Server

1. To start the FastAPI server, run :
   uvicorn main:app --reload

2. The API will be available at http://127.0.0.1:8000

3. You can access the interactive API documentation at
   http://127.0.0.1:8000/docs


