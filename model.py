import pymysql
import mysql.connector
from text_analysis import classify_with_bart , classify_with_barts
import re
from db_connection import get_conversations

def classify_conversations(conversation):

    # Extract user and bot messages
    user_message = conversation["UserMessage"]
    # print(user_message)
    bot_response = conversation["BotMessage"]
    # print(bot_response)

    # Check for fallback responses
    user_fallback_phrases = [
        "talk to operator", "talk to a human", "speak to a person", 
        "speak with a human", "speak to a human", "talk to a person", 
        "speak to a leasing agent", "talk to a human ?", "talk to operator ?", 
        "speak to a person ?", "speak with a human ?", "speak to a human ?", 
        "talk to a person ?", "speak to a leasing agent ?" , "Can I talk to a human?"
    ]
    
    bot_fallback_phrases = [
        "I don’t understand", "Can you rephrase?", "Sorry, I’m not sure", "I am unable", 
        "Sorry, I could not understand.", "Sorry, I missed what you just said. Can you say that again?"
    ]

    # Check if user message contains any of the user fallback phrases
    if any(phrase.lower() in user_message.lower() for phrase in user_fallback_phrases):
        return 'Unsuccessful', 0.0
    
    # Check if bot response contains any of the bot fallback phrases
    if any(phrase.lower() in bot_response.lower() for phrase in bot_fallback_phrases):
        return 'Unsuccessful', 0.0

    predicted_label, confidence_score = classify_with_barts(user_message, bot_response)

    # Define success criteria
    if predicted_label == "Successful" and confidence_score > 0.70:
        return 'Successful', confidence_score
    else:
        return 'Unsuccessful', confidence_score

# Fallback phrases
user_fallback_phrases = [
    "talk to operator", "talk to a human", "speak to a person", 
    "speak with a human", "speak to a human", "talk to a person", 
    "speak to a leasing agent", "talk to a human ?", "talk to operator ?", 
    "speak to a person ?", "speak with a human ?", "speak to a human ?", 
    "talk to a person ?", "speak to a leasing agent ?" , "Can I talk to a human?"
]
bot_fallback_phrases = [
    "I don’t understand", "Can you rephrase?", "Sorry, I’m not sure", "I am unable", 
    "Sorry, I could not understand.", "Sorry, I missed what you just said. Can you say that again?"
]

def check_fallback_phrases(messages, fallback_phrases):
    """
    Checks if any of the given messages contain fallback phrases.
    :param messages: List of messages (either user or bot)
    :param fallback_phrases: List of fallback phrases to check
    :return: True if any fallback phrase is found, otherwise False
    """
    for message in messages:
        message = message.lower()  # Ensure case-insensitive matching
        for phrase in fallback_phrases:
            if phrase.lower() in message:  # Check if fallback phrase is present in message
                print(f"Fallback phrase found: '{phrase}' in message: '{message}'")  # Debugging log
                return True
    return False

def classify_conversation(merged_message):
    """
    Classifies the entire merged conversation using BART with fallback phrase detection.
    :param merged_message: Structured conversation text (User + Bot messages)
    :return: Classification label and confidence score
    """
    # Clean merged message to handle extra spaces and newlines
    # merged_message = merged_message.strip()

    # Updated regex to better capture user and bot messages, including multi-line messages
    user_messages = re.findall(r'user:\s*([^b]+)', merged_message, re.IGNORECASE)
    bot_messages = re.findall(r'bot:\s*([^u]+)', merged_message, re.IGNORECASE)

    # Clean up the messages (strip leading/trailing spaces)
    user_messages = [msg.strip() for msg in user_messages if msg.strip()]
    bot_messages = [msg.strip() for msg in bot_messages if msg.strip()]

    # Check for fallback phrases in user and bot messages
    # if check_fallback_phrases(user_messages, user_fallback_phrases) or check_fallback_phrases(bot_messages, bot_fallback_phrases):
    #     return 'Unsuccessful', 0.9  # If fallback phrase found, mark as unsuccessful with low confidence

    # Proceed with BART classification
    predicted_label, confidence_score = classify_with_bart(merged_message)

    if predicted_label == "Successful":
        return 'Successful', confidence_score
    else:
        return 'Unsuccessful', confidence_score

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