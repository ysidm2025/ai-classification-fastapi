from transformers import pipeline , BartTokenizer, BartForSequenceClassification 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from transformers import BertTokenizer, BertModel
import torch

# Download necessary nltk data
nltk.download('stopwords')
nltk.download('punkt')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load BART model and tokenizer
tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
model_bart = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

# Load BART zero-shot classification pipeline
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels for zero-shot classification
CANDIDATE_LABELS = ["Successful", "Unsuccessful"]

# Input: A text string (e.g., the user’s or bot’s message).
# Output: A tensor representing the BERT embedding of the input text.
def get_bert_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get the BERT embeddings (we will use the hidden state of the last layer)
    with torch.no_grad():
        outputs = model(**inputs)

    # # The last hidden state represents the token embeddings for each token
    # embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence embedding

    # Extract CLS token embedding (first token representation)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, hidden_dim)

    # return embeddings
    return cls_embedding

def calculate_bert_similarity(user_message, bot_response):
    # Get embeddings for both user message and bot response
    user_embedding = get_bert_embeddings(user_message)
    bot_embedding = get_bert_embeddings(bot_response)
    
    # Calculate cosine similarity between the embeddings
    similarity_score = cosine_similarity(user_embedding, bot_embedding)
    
    return similarity_score[0][0]

# def classify_with_bart(user_message, bot_response):
#     """Perform zero-shot classification using BART for chatbot conversation."""
#     # Combine user and bot messages into a single text
#     combined_text = f"User: {user_message} Bot: {bot_response}"

#     inputs = tokenizer_bart(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model_bart(**inputs)
    
#     # Get probabilities for each class
#     logits = outputs.logits.softmax(dim=-1).squeeze()
#     confidence_scores = {CANDIDATE_LABELS[i]: logits[i].item() for i in range(len(CANDIDATE_LABELS))}
    
#     # Determine the most likely class
#     predicted_label = max(confidence_scores, key=confidence_scores.get)
#     return predicted_label, float(confidence_scores[predicted_label])

def classify_with_bart(user_message, bot_response):
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

# def classify_with_bart(merged_message):
#     """
#     Classifies the conversation using BART zero-shot classification.
#     :param merged_message: Entire structured conversation text
#     :return: Predicted label and confidence score
#     """
#     # Perform zero-shot classification
#     result = zero_shot_classifier(merged_message, candidate_labels=CANDIDATE_LABELS)

#     # Extract classification result
#     predicted_label = result["labels"][0]  # The highest confidence label
#     confidence_score = result["scores"][0]  # Confidence of the classification

#     return predicted_label, float(confidence_score)

# Initialize Hugging Face sentiment-analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    # Analyze sentiment using Hugging Face transformer
    sentiment = sentiment_analyzer(text)[0]
    sentiment_label = sentiment['label']
    sentiment_score = sentiment['score']
    
    return sentiment_label, sentiment_score

def calculate_semantic_similarity(user_message, bot_response):
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    user_tokens = [word for word in word_tokenize(user_message.lower()) if word not in stop_words]
    bot_tokens = [word for word in word_tokenize(bot_response.lower()) if word not in stop_words]

    # Convert the messages to a tf-idf matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(user_tokens), ' '.join(bot_tokens)])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_score = similarity_matrix[0][0]
    
    return similarity_score
