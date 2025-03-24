from transformers import pipeline , BartTokenizer, BartForSequenceClassification 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from transformers import BertTokenizer, BertModel
import torch

# # Download necessary nltk data
# nltk.download('stopwords')
# nltk.download('punkt')

# # Load BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Load BART model and tokenizer
# tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
# model_bart = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

# Load BART zero-shot classification pipeline
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels for zero-shot classification
CANDIDATE_LABELS = ["Successful", "Unsuccessful"]

def classify_with_barts(user_message, bot_response):
    """
    Classify a chatbot conversation using BART zero-shot classification.
    """
    try:
        combined_text = f"The user said: '{user_message}'\nThe bot replied: '{bot_response}'.\nWas the bot's response helpful?"
        result = zero_shot_classifier(combined_text, candidate_labels=CANDIDATE_LABELS)

        predicted_label = result["labels"][0]  # Most confident label
        confidence_score = result["scores"][0]  # Confidence score
        
        return predicted_label, float(confidence_score)
    except Exception as e:
        print(f"Error in BART classification: {e}")
        return "Unsuccessful", 0.0  # Default to 'Unsuccessful'

def classify_with_bart(merged_message):
    """
    Classifies the conversation using BART zero-shot classification.
    :param merged_message: Entire structured conversation text
    :return: Predicted label and confidence score
    """
    try:

        combined_text = f"The following conversation occurred:\n{merged_message}\nWas the bot's response satisfactory?"
        # Perform zero-shot classification
        result = zero_shot_classifier(combined_text, candidate_labels=CANDIDATE_LABELS)

        # Extract classification result
        predicted_label = result["labels"][0]  # The highest confidence label
        confidence_score = result["scores"][0]  # Confidence of the classification

        return predicted_label, float(confidence_score)
    except Exception as e:
        print(f"Error in BART classification: {e}")
        return "Unsuccessful", 0.0  # Default to 'Unsuccessful