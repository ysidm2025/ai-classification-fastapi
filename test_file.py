from db_connection import get_conversations , create_conversation_review_table , get_conversation 
from model import classify_conversation , store_classification_results , classify_conversations
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from fastapi import FastAPI, HTTPException , Form
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os
from fastapi.responses import HTMLResponse 
import re

load_dotenv()

# FastAPI app initialization
app = FastAPI()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API client
openai.api_key = OPENAI_API_KEY

# Load labeled conversations
def load_labeled_data():
    """
    Load labeled conversations from labeled_conversations.csv.
    """
    try:
        df = pd.read_csv("labeled_conversations.csv")
        print("✅ Loaded labeled conversations from labeled_conversations.csv.")
        return df
    except FileNotFoundError:
        print("❌ File labeled_conversations.csv not found. Creating an empty DataFrame.")
        return pd.DataFrame(columns=["ConversationId", "mergedmessages", "label"])

# Function to classify a conversation using OpenAI
# def classify_conversation(conversation_id, merged_messages):
#     """
#     Send conversation to OpenAI for classification and return classification.
#     """
#     messages = [
#         # System message to guide the model to classify based on chatbot performance
#         {"role": "system", "content": """
#             You are an AI assistant evaluating chatbot performance.
#             Your task is to classify the following conversation based solely on the chatbot's ability to satisfy the user's needs.
#             Focus on whether the chatbot provided relevant, accurate, and helpful responses that helped the user with their questions and concerns.
#             Do not classify the conversation based on the user's mistakes or behavior; only evaluate the chatbot's responses.
#             If user made a mistake but bot answer was correct the conversation is Successful .
#             The classification should be either 'Successful' or 'Unsuccessful'.
#             Please return ONLY 'Successful' or 'Unsuccessful', without any explanations or extra text.
#         """},

#         # User's conversation content
#         {"role": "user", "content": merged_messages},

#         # Assistant message to clarify that only the bot's performance should be evaluated
#         {"role": "assistant", "content": "Classify the conversation based on the bot's performance and whether it satisfied the user's needs."}
#     ]

#     # Send the request to OpenAI model
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-2024-08-06",
#         messages=messages,
#         max_tokens=5,  # Limit to the shortest response
#         temperature=0.0  # Zero temperature for consistent results
#     )

#     # Extract the classification result from the response
#     result = response['choices'][0]['message']['content'].strip()

#     # Ensure that the classification is either 'Successful' or 'Unsuccessful'
#     if result.lower() == "successful":
#         predicted_label = "Successful"
#     elif (result.lower() == "unsuccessful"):
#         predicted_label = "Unsuccessful"
#     else:
#         predicted_label = "Unsuccessful"  # Default to Unsuccessful if result is unexpected

#     return predicted_label

def classify_conversation(conversation_id, merged_messages):
    """
    Send conversation to OpenAI for classification and return classification.
    """
    messages = [
        # System message to guide the model to classify based on chatbot performance
        {"role": "system", "content": """
            You are an AI assistant evaluating chatbot performance.
            Your task is to classify the following conversation based solely on the chatbot's ability to satisfy the user's needs.
            Focus on whether the chatbot provided relevant, accurate, and helpful responses that helped the user with their questions and concerns.
            Do not classify the conversation based on the user's mistakes or behavior; only evaluate the chatbot's responses.
            If the user made a mistake but the bot answered correctly, the conversation is classified as 'Successful.'

            The classification should be either 'Successful' or 'Unsuccessful.'
            Please return ONLY 'Successful' or 'Unsuccessful', without any explanations or extra text.

            One of many example of Successful Conversation:
            - User: "I need directions to a nearby park."
            - Bot: "Here is a map of the nearest parks in your area: [map link]."
            - This conversation is 'Successful' because the bot gave a relevant, accurate response to the user's request.

            Example of Unsuccessful Conversation (where bot is giving too many irrelevant questions and not addressing needs directly):
            - User: "Do you have 2 bedrooms?"
            - Bot: "When would you like to move in?"
            - User: "Mid Oct"
            - Bot: "Here is what I found for 2 beds with a move in date of 10/21/2020."
            - Bot: "Would you like to know more about our apartments?"
            - User: "How about 3 bedrooms?"
            - Bot: "Pet Policy Cats - 1 allowed Max weight 15 lb each, Rent $50.00, Deposit $50.00, Comments: We love our feline companions! Dogs - 1 allowed Max weight 40 lb each, Rent $50.00, Deposit $100.00, Pet Care available."
            - This conversation is 'Unsuccessful' because the bot gave a **non-specific, repetitive response** to each query and when asked about 3 bedrooms he returned pet policy .
        """},

        # User's conversation content
        {"role": "user", "content": merged_messages},

        # Assistant message to clarify that only the bot's performance should be evaluated
        {"role": "assistant", "content": "Classify the conversation based on the bot's performance and whether it satisfied the user's needs."}
    ]

    # Send the request to OpenAI model
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=5,  # Limit to the shortest response
        temperature=0.0  # Zero temperature for consistent results
    )

    # Extract the classification result from the response
    result = response['choices'][0]['message']['content'].strip()

    # Ensure that the classification is either 'Successful' or 'Unsuccessful'
    if result.lower() == "successful":
        predicted_label = "Successful"
    elif result.lower() == "unsuccessful":
        predicted_label = "Unsuccessful"
    else:
        predicted_label = "Unsuccessful"  # Default to Unsuccessful if result is unexpected

    return predicted_label

# Main function to classify conversations and calculate accuracy
def classify_and_update():
    """
    Classify all conversations from labeled_conversations.csv using OpenAI and calculate accuracy.
    """
    # Load labeled data
    df = load_labeled_data()

    # Check if the CSV is empty
    if df.empty or "mergedmessages" not in df.columns:
        print("⚠️ No labeled conversations found in labeled_conversations.csv.")
        return

    # Store predicted labels
    predicted_labels = []

    # Classify each conversation
    for _, row in df.iterrows():
        conversation_id = row["ConversationId"]
        merged_messages = row["mergedmessages"]

        # Get predicted label from OpenAI
        predicted_label = classify_conversation(conversation_id, merged_messages)
        predicted_labels.append(predicted_label)

    # Add predicted_label column to the DataFrame
    df["predicted_label"] = predicted_labels

    # Save updated labeled data
    df.to_csv("labeled_conversations.csv", index=False)
    print("✅ Updated predictions saved to labeled_conversations.csv successfully!")

    # Calculate accuracy
    correct_predictions = (df["label"] == df["predicted_label"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"📊 Classification Accuracy: {round(accuracy * 100, 2)}%")

if __name__ == "__main__":
    classify_and_update()  # Or the name of your main interactive function

