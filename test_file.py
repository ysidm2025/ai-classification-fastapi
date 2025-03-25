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
        # df = pd.read_csv("labeled_conversations.csv")
        df = pd.read_csv("test_set_updated.csv")

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

            example 2 of Successful Conversation :-
            -user: Check Balance Due 
            -bot: Late Fees: $867.00 Late Fees: $1000.00 Late Fees: $1000.00 Late Fees: $1000.00 Late Fees: $1000.00 Late Fees: $1500.00 Late Fees: $1000.00 Rent- Residential: $1000.00 Late Fees: $22.34 Rent- Residential: $468.00 Parking - Industrial: $100.00 Rent- Residential: $1000.00 Late Fees: $31.00 Unapplied Credit: $0.00 Total: $9988.34 
            -bot: Your rent is due on the 10th of the month and you have a grace period of 2 days after which late fees start to accrue. 
            -bot: How else can I help you? 
            -user: late fee 
            -bot: Current late fee charges: $ 7420.34 
            -bot: Your rent is due on the 10th of the month and you have a grace period of 2 days after which late fees start to accrue. 
            -bot: How else can I help you?
            - This conversation is successful because it have the total amount which is the actual balance due .

            One of many example of Unsuccessful Conversation 
            - User: "Do you have 2 bedrooms?"
            - Bot: "When would you like to move in?"
            - User: "Mid Oct"
            - Bot: "Here is what I found for 2 beds with a move in date of 10/21/2020."
            - Bot: "Would you like to know more about our apartments?"
            - User: "How about 3 bedrooms?"
            - Bot: "Pet Policy Cats - 1 allowed Max weight 15 lb each, Rent $50.00, Deposit $50.00, Comments: We love our feline companions! Dogs - 1 allowed Max weight 40 lb each, Rent $50.00, Deposit $100.00, Pet Care available."
            - This conversation is 'Unsuccessful' because when the bot was asked about 3 bedrooms he returned pet policy .

            example 2 of Unsuccessful Conversation :
            "user: Maintenance Request bot: If this is an Emergency Maintenance request, please call (805) 001-2000. 
            bot: Please describe your issue so that I can start the service request. 
            user: water supply  bot: Do we have permission to enter the apartment? 
            user: Yes 
            bot: Do you have pets or any special instructions for our service team? 
            user: yes 
            bot: Your service request has been submitted successfully. Your request I.D. is 8398 bot: How else can I help you? 
            user: pay rent 
            bot: Unable to complete transaction. Please try again later."
            -This conversation is Unsuccessful because bot was Unable to complete transaction .

        """},
        # (where bot is giving too many irrelevant questions and not addressing needs directly):
        # gave a **non-specific, repetitive response** to each query and

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
    # df.to_csv("labeled_conversations.csv", index=False)
    df.to_csv("test_set_updated.csv", index=False)

    print("✅ Updated predictions saved to labeled_conversations.csv successfully!")

    # Calculate accuracy
    correct_predictions = (df["label"] == df["predicted_label"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"📊 Classification Accuracy: {round(accuracy * 100, 2)}%")

def convert_file():
    # Load the CSV file
    file_path = "test_set.csv"  # Replace with your CSV file path
    df = pd.read_csv(file_path)

    # Convert 1 to 'Successful' and 0 to 'Unsuccessful' in the 'label' column
    df['label'] = df['label'].replace({1: 'Successful', 0: 'Unsuccessful'})

    # Save the updated CSV file
    output_path = "test_set.csv"  # Specify the desired output file name
    df.to_csv(output_path, index=False)

    print(f"File updated and saved as '{output_path}'")

# Load test_set.csv
file_path = "test_set.csv"  # Path to your test_set.csv file
df = pd.read_csv(file_path)

# Add 'mergedmessages' column by fetching mergedmessages for each ConversationID
def get_merged_message(conversation_id):
    # Call your existing get_conversations function
    conversations_df = get_conversations(conversation_id)
    
    # Extract mergedmessages from the result
    if not conversations_df.empty:
        return conversations_df['mergedmessages'].values[0]
    return ''  # Return empty string if not found

# Apply get_merged_message to fetch and add the 'mergedmessages' column
df['mergedmessages'] = df['ConversationId'].apply(get_merged_message)

# Save the updated CSV file
output_path = "test_set_updated.csv"  # Name of the updated file
df.to_csv(output_path, index=False)

print(f"✅ File updated successfully and saved as '{output_path}'")

if __name__ == "__main__":
    # get_merged_message()
    # convert_file()
    classify_and_update()  # Or the name of your main interactive function

