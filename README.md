# Chatbot Classification

## Overview

The **Chatbot Classification** project is designed to evaluate the success of chatbot interactions by analyzing conversations and classifying them as either "Successful" or "Unsuccessful." The system leverages a combination of advanced NLP models, including **BERT**, **BART**, and **OpenAI's API**, to perform text classification and similarity analysis.

The project is built using **FastAPI**, allowing seamless API integration for real-time evaluation of chatbot interactions. In addition to classifying conversations, the system provides sentiment analysis, semantic similarity scoring, and zero-shot classification capabilities. The OpenAI integration further enhances the evaluation by providing context-aware analysis of chatbot responses.

This solution is ideal for businesses seeking to monitor and enhance the performance of their chatbots, ensuring that customer queries are handled effectively.

## Features

- **BERT-based Embeddings**: Utilizes BERT (Bidirectional Encoder Representations from Transformers) for calculating semantic similarity between user messages and bot responses.
- **BART-based Zero-Shot Classification**: Implements zero-shot classification with BART to classify chatbot responses as "Successful" or "Unsuccessful."
- **Sentiment Analysis**: Leverages Hugging Face transformers to evaluate the sentiment of user inputs and bot responses.
- **FastAPI Integration**: Exposes an API for interacting with the classification models in real time, making it easier to integrate the solution into chatbots or customer service systems.
- **Similarity Calculation**: Computes semantic similarity scores between user and bot messages using advanced NLP techniques.
- **OpenAI-based Classification**: Integrates an OpenAI endpoint for analyzing conversations and classifying them based on contextual understanding.
- **FastAPI Integration**: Exposes an API for interacting with classification models, including the OpenAI endpoint, in real time for seamless chatbot evaluation.

## Getting Started

### Prerequisites

- Python 3.7+
- PIP (Python package installer)
- Git (for version control)
- OpenAI API Key (for OpenAI endpoint usage)  

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ysidm2025/chatbot_classification.git

2. Navigate to the project directory:
  cd chatbot_classification

3. Install the necessary dependencies :
   pip install -r requirements.txt

4. Set Up Environment Variables: Create a .env file with the following values:  
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=your_postgres_database_url  

5. Download the required datasets :
   python -m nltk.downloader stopwords punkt

## Running the FastAPI Server

1. To start the FastAPI server, run :
   uvicorn main:app --reload

2. The API will be available at http://127.0.0.1:8000

3. You can access the interactive API documentation at
   http://127.0.0.1:8000/docs


