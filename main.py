from db_connection import get_conversations , create_conversation_review_table , fetch_conversation_review , fetch_conversation_by_id , get_conversation 
from model import classify_conversation , store_classification_results
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# , store_classification_results
# , create_conversation_review_table

# create_conversation_review_table()

# def run_classification_pipeline():
#     # Get cleaned conversation data from the database
#     df_cleaned = get_conversations()
    
#     # Iterate through each conversation
#     for index, row in df_cleaned.iterrows():
#         status, confidence_score = classify_conversation(row)
#         # store_classification_results(row["ConversationId"], status, confidence_score)
#         print(f"Conversation {row['ConversationId']} classifie`d as {status} with confidence {confidence_score}")
    
# if __name__ == "__main__":
#     run_classification_pipeline()

# FastAPI app initialization
app = FastAPI()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API client
openai.api_key = OPENAI_API_KEY

# Store ground truth and predictions
y_true = []  # Actual labels (manually set or fetched from DB)
y_pred = []  # Model predictions

## This is default original without FASTAPI PIPELINE -> PIPELINES
def run_classification_pipelines():
    # Get cleaned conversation data from the database
    df_cleaned = get_conversations()

     # Only process the first 15 rows trial code
    df_cleaned = df_cleaned.head(10)  # Limit to first 50 entries
    
    for index, row in df_cleaned.iterrows():
        status, confidence_score = classify_conversation(row) #status storeds class result
        
        # Assuming ground truth is available from an existing table
        y_true.append("Successful" if confidence_score > 0.70 else "Unsuccessful")
        y_pred.append(status)

        # Store classification results in the database
        store_classification_results(row["ConversationId"], status, confidence_score)
        
        print(f"Conversation {row['ConversationId']} classified as {status} with confidence {confidence_score}")
    
    # Run evaluation after classification
    # evaluate_classification()

    # I have created this function to check if new table is storing vlues correctly
    # fetch_conversation_review() 

# Request model for conversation ID
class ConversationRequest(BaseModel):
    conversation_id: int

# class ConversationResponse(BaseModel):
#     conversation_id: int
#     classification: str

class ConversationResponse(BaseModel):
    conversation_id: int
    classification: str  # 'successful' or 'unsuccessful'
    conversation: str  # Full conversation text

class Config:
    orm_mode = True

@app.post("/classify-conversation-openai/", response_model=ConversationResponse)
async def classify_conversation(request: ConversationRequest):
    conversation_id = request.conversation_id

    # Retrieve the conversation data using the get_conversations function
    df_conversation = get_conversations(conversation_id)

    if df_conversation.empty or 'mergedmessages' not in df_conversation.columns:
        raise HTTPException(status_code=404, detail="Conversation not found or missing 'mergedmessages' column")

    # Concatenate all merged messages in the conversation into a single string
    conversation_text = " ".join(df_conversation['mergedmessages'].dropna())

    # Prepare a message prompt for OpenAI to classify the conversation
    prompt = f"Classify the following conversation as either successful or unsuccessful: {conversation_text}"

    try:
        # Make an API call to OpenAI using the new chat-completion method
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or choose another model (gpt-3.5-turbo, gpt-4, etc.)
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the classification from OpenAI's response
        classification = response['choices'][0]['message']['content'].strip()

        # Return the classification and full conversation text as part of the response
        return ConversationResponse(
            conversation_id=conversation_id, 
            classification=classification, 
            conversation=conversation_text  # Full conversation included
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

# Endpoint to classify a specific conversation by ID
# Endpoint to fetch and classify conversation
# @app.post("/classify_conversation/BART")
# async def classify_conversation_endpoint(request: ConversationRequest):
#     # Fetch the conversation from the database
#     conversation_data = get_conversation(request.conversation_id)
    
#     if conversation_data is None or len(conversation_data) == 0:
#         raise HTTPException(status_code=404, detail="Conversation not found")

#     # Classify the conversation
#     classification_results = []
#     for index, row in conversation_data.iterrows():
#         user_message = row['UserMessage']
#         bot_response = row['BotMessage']
        
#         # Classify conversation using classify_conversation function from model.py
#         status, confidence_score = classify_conversation({"UserMessage": user_message, "BotMessage": bot_response})
        
#         # Store result in the list
#         classification_results.append({
#             "UserMessage": user_message,
#             "BotMessage": bot_response,
#             "PredictedLabel": status,
#             "ConfidenceScore": confidence_score
#         })

#         # Store classification result in the database (optional)
#         store_classification_results(request.conversation_id, status, confidence_score)
    
#     # Return classification results in the response
#     return {"classification_results": classification_results}

# Existing classification pipeline function to run all conversations
def run_classification_pipeline():
    # Get cleaned conversation data from the database
    df_cleaned = get_conversations()

    # Only process the first 15 rows trial code
    df_cleaned = df_cleaned.head(10)  # Limit to first 10 entries
    
    for index, row in df_cleaned.iterrows():
        status, confidence_score = classify_conversation(row)  # status stores classification result
        
        # Assuming ground truth is available from an existing table
        y_true.append("Successful" if confidence_score > 0.70 else "Unsuccessful")
        y_pred.append(status)

        # Store classification results in the database
        store_classification_results(row["ConversationId"], status, confidence_score)
        
        print(f"Conversation {row['ConversationId']} classified as {status} with confidence {confidence_score}")
    
    # Run evaluation after classification
    # evaluate_classification()

def evaluate_classification():
    """Compute and print evaluation metrics."""
    print("\n==== Classification Report ====")
    print(classification_report(y_true, y_pred, target_names=["Unsuccessful", "Successful"]))
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

if __name__ == "__main__":
    run_classification_pipeline()