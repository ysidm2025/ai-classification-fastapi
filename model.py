import pymysql
import mysql.connector
from text_analysis import analyze_sentiment, calculate_semantic_similarity , calculate_bert_similarity , classify_with_bart
import re

def classify_conversation(conversation):
    # Extract user and bot messages
    user_message = conversation["UserMessage"]
    # print(user_message)
    bot_response = conversation["BotMessage"]
    # print(bot_response)

    # Sentiment analysis
    sentiment_label, sentiment_score = analyze_sentiment(bot_response)

    # Check for fallback responses
    # Define fallback phrases for user and bot
    user_fallback_phrases = [
        "talk to operator", "talk to a human", "speak to a person", 
        "speak with a human", "speak to a human", "talk to a person", 
        "speak to a leasing agent", "talk to a human ?", "talk to operator ?", 
        "speak to a person ?", "speak with a human ?", "speak to a human ?", 
        "talk to a person ?", "speak to a leasing agent ?"
    ]
    
    bot_fallback_phrases = [
        "I don’t understand", "Can you rephrase?", "Sorry, I’m not sure" , "I am unable" , "Sorry, I could not understand.","Sorry, I missed what you just said. Can you say that again?"
    ]

    # Check if user message contains any of the user fallback phrases
    if any(phrase.lower() in user_message.lower() for phrase in user_fallback_phrases):
        return 'Unsuccessful', sentiment_score
    
    # Check if bot response contains any of the bot fallback phrases
    if any(phrase.lower() in bot_response.lower() for phrase in bot_fallback_phrases):
        return 'Unsuccessful', sentiment_score
    

    # Semantic similarity
    similarity_score = calculate_semantic_similarity(user_message, bot_response)

    # # Semantic similarity using BERT
    # similarity_score = calculate_bert_similarity(user_message, bot_response)
    
    # # code for BERT and semantic
    # # Define success criteria
    # if sentiment_label == "POSITIVE" and similarity_score > 0.25:
    #     return 'Successful', similarity_score
    # else:
    #     return 'Unsuccessful', similarity_score

    # uncommeny below for function
    # # Perform classification with BART
    predicted_label, confidence_score = classify_with_bart(user_message, bot_response)

    #code for BART only
    # Define success criteria
    if predicted_label == "Successful" and confidence_score > 0.70:
        return 'Successful', confidence_score
    else:
        return 'Unsuccessful', confidence_score

# def classify_conversation(merged_message):
#     """
#     Classifies the entire merged conversation using BART.
#     :param merged_message: Structured conversation text (User + Bot messages)
#     :return: Classification label and confidence score
#     """
#     predicted_label, confidence_score = classify_with_bart(merged_message)

#     # Define success criteria
#     if predicted_label == "Successful":
#         return 'Successful', confidence_score
#     else:
#         return 'Unsuccessful', confidence_score

# def classify_conversation(df):
#     """
#     Classifies the entire conversation using BART, based on the merged messages.
#     :param df: DataFrame containing merged messages for a conversation
#     :return: Classification label and confidence score
#     """
#     # Assuming that df has a 'mergedmessages' column
#     merged_message = df['mergedmessages'].iloc[0]  # Extract merged messages from the DataFrame

#     # Classify the conversation using the BART model
#     predicted_label, confidence_score = classify_with_bart(merged_message)

#     # Prepare the result in a JSON-serializable format (dictionary)
#     return {
#         "PredictedLabel": predicted_label,
#         "ConfidenceScore": confidence_score
#     }

# def classify_conversation(conversation_data: dict):
#     """
#     Classify a conversation using Zero-Shot classification with BART.
    
#     :param conversation_data: Dictionary containing 'UserMessage' and 'BotMessage'
#     :return: A tuple of (status, confidence_score)
#     """
#     user_message = conversation_data['UserMessage']
#     bot_message = conversation_data['BotMessage']
    
#     # Combine the user message and bot response as input text
#     input_text = f"User: {user_message} Bot: {bot_message}"

#     # Define candidate labels for zero-shot classification
#     candidate_labels = ["Successful", "Unsuccessful"]

#     # Classify the conversation using zero-shot classification
#     result = zero_shot_classifier(input_text, candidate_labels)

#     # Extract the classification result and the confidence score
#     status = result['labels'][0]  # The label with the highest score
#     confidence_score = result['scores'][0]  # Confidence score for the predicted label

#     return status, confidence_score


def store_classification_results(conversation_id, status, confidence_score):

    # Connect to the database
    try:
        connection = mysql.connector.connect(
            host="pcz218dbl23",
            user="prakashd",
            password="TLzWqu8Kyp",
            database="omni_qa_db"
        )
        
        cursor = connection.cursor()
        
        # Convert confidence_score to a Python float
        confidence_score = float(confidence_score) 

        # Insert classification result into ConversationReview table
        cursor.execute("""
            INSERT INTO conversationreview (ConversationId, Status, ConfidenceScore)
            VALUES (%s, %s, %s)
        """, (conversation_id, status, confidence_score))
        
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    finally:
        cursor.close()
        connection.close()

def fetch_conversation_by_id(conversation_id):
    """Fetch a specific conversation by conversation_id."""
    try:
        connection = mysql.connector.connect(
            host="pcz218dbl23",
            user="prakashd",
            password="TLzWqu8Kyp",
            database="omni_qa_db"
        )
        
        cursor = connection.cursor()

        # Fetch conversation data from both incoming and outgoing tables
        cursor.execute("""
            SELECT c.conversationid, c.message, c.conversationincomingtime
            FROM conversationincoming c
            WHERE c.conversationid = %s
        """, (conversation_id,))
        incoming = cursor.fetchall()
        
        cursor.execute("""
            SELECT c.conversationid, c.message, c.conversationoutgoingtime
            FROM conversationoutgoing c
            WHERE c.conversationid = %s
        """, (conversation_id,))
        outgoing = cursor.fetchall()

        # Create DataFrames from the fetched data
        df_incoming = pd.DataFrame(incoming, columns=["ConversationId", "UserMessage", "Timestamp"])
        df_outgoing = pd.DataFrame(outgoing, columns=["ConversationId", "BotMessage", "Timestamp"])

        # Convert timestamps to datetime
        df_incoming["Timestamp"] = pd.to_datetime(df_incoming["Timestamp"])
        df_outgoing["Timestamp"] = pd.to_datetime(df_outgoing["Timestamp"])

        # Merge user and bot messages on ConversationId and Timestamp
        df_combined = pd.merge(df_incoming, df_outgoing, on=["ConversationId", "Timestamp"], how="outer")
        df_combined.sort_values(by=["ConversationId", "Timestamp"], inplace=True)

        # Group by ConversationId to combine messages
        df_grouped = df_combined.groupby("ConversationId").agg({
            "UserMessage": lambda x: " ".join(x.dropna()),  # Combine user messages
            "BotMessage": lambda x: " ".join(x.dropna())    # Combine bot responses
        }).reset_index()

        # Return the processed DataFrame
        return df_grouped
    
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        cursor.close()
        connection.close()