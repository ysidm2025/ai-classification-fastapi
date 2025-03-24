import mysql.connector
import pandas as pd
import gc
import json
import re
from bs4 import BeautifulSoup

########################## cleaning JSON & HTML ##########################

# Function to convert JSON bot messages into plain text
def process_message(df):
    df_processed = df.copy()  # Create a copy of the DataFrame to avoid modifying original
    df_processed["BotMessage"] = df_processed["BotMessage"].apply(lambda x: extract_text_from_json(x))
    return df_processed

# # Helper function to extract plain text from JSON
# def extract_text_from_json(bot_message):
#     if isinstance(bot_message, str):  # Ensure it's a string before processing
#         try:
#             parsed_message = json.loads(bot_message)  # Attempt to parse JSON
#             if isinstance(parsed_message, list) and "text" in parsed_message[0]:  
#                 return parsed_message[0]["text"]  # Extract text field from JSON
#         except json.JSONDecodeError:
#             pass  # If parsing fails, return original message
#     return bot_message  # Return original if not JSON or improperly formatted

# def extract_text_from_json(bot_message):
#     """Extracts text content from a JSON-formatted bot message."""
#     if isinstance(bot_message, str):
#         try:
#             parsed_message = json.loads(bot_message)  # Parse JSON if valid
#             if isinstance(parsed_message, list) and len(parsed_message) > 0 and "text" in parsed_message[0]:
#                 return parsed_message[0]["text"]  # Extract text field
#         except json.JSONDecodeError:
#             pass  # If parsing fails, return original message
#     return bot_message  # Return as-is if not JSON

# def extract_text_from_json(bot_message):
#     """Extracts title, imageUrl, and text from a JSON-formatted bot message."""
#     if isinstance(bot_message, str):
#         try:
#             parsed_message = json.loads(bot_message)  # Parse JSON if valid

#             # Check if parsed data is a list and contains the expected structure
#             if isinstance(parsed_message, list) and len(parsed_message) > 0:
#                 extracted_content = []

#                 # Loop through the list and extract 'title', 'imageUrl', and 'text'
#                 for item in parsed_message:
#                     if 'custom' in item and 'lists' in item['custom']:
#                         for entry in item['custom']['lists']:
#                             title = entry.get('title', '')
#                             image_url = entry.get('imageUrl', '')
#                             if title and image_url:
#                                 extracted_content.append(f"Title: {title}, Image URL: {image_url}")

#                     # Check for 'text' field
#                     if isinstance(item, dict) and 'text' in item:
#                         extracted_content.append(f"Text: {item['text']}")

#                 # If extracted content exists, return it as a formatted string
#                 if extracted_content:
#                     return "\n".join(extracted_content)

#         except json.JSONDecodeError:
#             pass  # If parsing fails, return original message

#     return bot_message  # Return as-is if not JSON or if extraction fails

def extract_text_from_json(bot_message):
    """Extracts content of title, imageUrl, and text from a JSON-formatted bot message."""
    if isinstance(bot_message, str):
        try:
            parsed_message = json.loads(bot_message)  # Parse JSON if valid

            # Check if parsed data is a list and contains the expected structure
            if isinstance(parsed_message, list) and len(parsed_message) > 0:
                extracted_content = []

                # Loop through the list and extract 'title', 'imageUrl', and 'text' content
                for item in parsed_message:
                    # Extract title and imageUrl from custom->lists
                    if 'custom' in item and 'lists' in item['custom']:
                        for entry in item['custom']['lists']:
                            title = entry.get('title', '')
                            image_url = entry.get('imageUrl', '')
                            if title and image_url:
                                extracted_content.append(f"Title: {title}, Image URL: {image_url}")

                    # Check for 'text' field and include without 'Text:' label
                    if isinstance(item, dict) and 'text' in item:
                        extracted_content.append(item['text'])

                # If extracted content exists, return it as a formatted string
                if extracted_content:
                    return "\n".join(extracted_content)

        except json.JSONDecodeError:
            pass  # If parsing fails, return original message

    return bot_message  # Return as-is if not JSON or if extraction fails

def clean_html(html_content):
    #  Remove any JSON-like structures
    html_content = re.sub(r'bot:\s?\[.*?\]', '', html_content)

    # BeautifulSoup to parse and clean the HTML
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove any remaining tags that are not relevant
    # Strip out unwanted tags like <script>, <style>, etc.
    for tag in soup.findAll(["img", "a", "iframe",'script', 'style']):
        tag.decompose()

    #Extract the text (if you need just the plain text)
    cleaned_text = soup.get_text(separator=' ', strip=True)

    return cleaned_text

def clean_df_outgoing(df_outgoing):
    """Cleans the BotMessage column by extracting and sanitizing text."""
    df_outgoing = df_outgoing.copy()  # Avoid modifying the original DataFrame
    
    # Extract text from JSON if applicable and clean HTML
    df_outgoing["BotMessage"] = df_outgoing["BotMessage"].apply(lambda x: clean_html(extract_text_from_json(x)))
    return df_outgoing

########################## cleaning JSON & HTML ##########################

# Original get code
def get_conversation(conversation_id: int):
    # Connect to the database
    connection = mysql.connector.connect(
        host="pcz218dbl23",
        user="prakashd",
        password="TLzWqu8Kyp",
        database="omni_qa_db"
    )

    cursor = connection.cursor()
    
    # Retrieve incoming and outgoing conversation messages
    # cursor.execute("SELECT conversationid, message, conversationincomingtime FROM conversationincoming ")
    # incoming = cursor.fetchall()
    
    # cursor.execute("SELECT conversationid, message, conversationoutgoingtime FROM conversationoutgoing ")
    # outgoing = cursor.fetchall()

    # # Retrieve incoming conversation messages for a specific conversation ID
    cursor.execute("SELECT conversationid, message, conversationincomingtime FROM conversationincoming WHERE conversationid = %s", (conversation_id,))
    incoming = cursor.fetchall()
        
    # Retrieve outgoing conversation messages for a specific conversation ID
    cursor.execute("SELECT conversationid, message, conversationoutgoingtime FROM conversationoutgoing WHERE conversationid = %s", (conversation_id,))
    outgoing = cursor.fetchall()

    # Close connection
    cursor.close()
    connection.close()
    
    # Create dataframes
    df_incoming = pd.DataFrame(incoming, columns=["ConversationId", "UserMessage", "Timestamp"])
    df_outgoing = pd.DataFrame(outgoing, columns=["ConversationId", "BotMessage", "Timestamp"])

    # Convert timestamps to datetime
    df_incoming["Timestamp"] = pd.to_datetime(df_incoming["Timestamp"])
    df_outgoing["Timestamp"] = pd.to_datetime(df_outgoing["Timestamp"])

    df_outgoing = clean_df_outgoing(df_outgoing)

    # Merge user and bot messages on ConversationId and Timestamp
    df_combined = pd.merge(df_incoming, df_outgoing, on=["ConversationId", "Timestamp"], how="outer")

    # Sort by ConversationId and Timestamp
    df_combined.sort_values(by=["ConversationId", "Timestamp"], inplace=True)
   
    # Save to pickle for later use
    df_combined.to_pickle("cleaned_conversations.pkl")
    df_combined.to_csv('cleaned_conversations.csv', index=False)

    # Group by ConversationId to form entire conversations ----------------------
    df_grouped = df_combined.groupby("ConversationId").agg({
        "UserMessage": lambda x: " ".join(x.dropna()),  # Combine user messages
        "BotMessage": lambda x: " ".join(x.dropna())    # Combine bot responses
    }).reset_index()
    # Save grouped conversations
    df_grouped.to_pickle("grouped_conversations.pkl")
    df_grouped.to_csv('grouped_conversations.csv', index=False)
    # Group by ConversationId to form entire conversations ----------------------

    # Drop rows with NaN values in user or bot messages
    df_cleaned = df_combined.dropna(subset=["UserMessage", "BotMessage"])

    # calling function to remove json bot messages ----------------------
    df_processed = process_message(df_grouped)
    df_processed.to_pickle("cleaned_conversations_processed.pkl")
    df_processed.to_csv("cleaned_conversations_processed.csv", index=False)
    # calling function to remove json bot messages ----------------------

    print("Fetched, processed bot messages, and saved to cleaned_conversations.csv")
    print(df_processed)

    # return df_processed
    return df_processed

    # # Convert Timestamp to datetime
    # df_incoming["Timestamp"] = pd.to_datetime(df_incoming["Timestamp"])
    # df_outgoing["Timestamp"] = pd.to_datetime(df_outgoing["Timestamp"])

    # # df_incoming , df_outgoing = clean_incoming_outgoing(df_incoming, df_outgoing)
    # # df_outgoing["BotMessage"] = df_outgoing["BotMessage"].astype(str).apply(extract_text_from_json)
    # df_outgoing = clean_df_outgoing(df_outgoing)

    # # Rename message columns for consistency
    # df_incoming = df_incoming.rename(columns={"UserMessage": "Message"})
    # df_outgoing = df_outgoing.rename(columns={"BotMessage": "Message"})

    # # Add a Speaker column
    # df_incoming["Speaker"] = "User"
    # df_outgoing["Speaker"] = "Bot"

    # # Combine both DataFrames
    # df_combined = pd.concat([df_incoming, df_outgoing], ignore_index=True)

    # # Sort by ConversationId and Timestamp to maintain order
    # df_combined = df_combined.sort_values(by=["ConversationId", "Timestamp"])

    # # Create formatted messages
    # df_combined["MergedMessages"] = df_combined.apply(lambda row: f"{row['Speaker']}: {row['Message']}", axis=1)

    # # Group by ConversationId and concatenate messages
    # df_merged = df_combined.groupby("ConversationId")["MergedMessages"].apply(lambda x: " ".join(x)).reset_index()

    # # df_new = process_messages(df_merged)
    # df_merged.to_csv('merged_conversation.csv', index=False)

    # # Display the final merged DataFrame
    # # print(df_new)

    # return df_merged

def create_conversation_review_table():

    # Connect to the database
    connection = mysql.connector.connect(
        host="pcz218dbl23",
        user="prakashd",
        password="TLzWqu8Kyp",
        database="omni_qa_db"
    )
    
    cursor = connection.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS conversationreview (
        ReviewId INT AUTO_INCREMENT PRIMARY KEY,
        ConversationId INT,
        Status ENUM('Successful', 'Unsuccessful'),
        ConfidenceScore FLOAT,
        FOREIGN KEY (ConversationId) REFERENCES Conversation(Id)
    );
    """
    
    cursor.execute(create_table_query)
    connection.commit()
    cursor.close()
    connection.close()
    # # Connect to the database
    # connection = mysql.connector.connect(
    #     host="pcz218dbl23",
    #     user="prakashd",
    #     password="TLzWqu8Kyp",
    #     database="omni_qa_db"
    # )
    
    # cursor = connection.cursor()
    
    # create_table_query = """
    # CREATE TABLE IF NOT EXISTS ConversationReview (
    #     ReviewId INT AUTO_INCREMENT PRIMARY KEY,
    #     ConversationId INT,
    #     Status ENUM('Successful', 'Unsuccessful'),
    #     ConfidenceScore FLOAT,
    #     FOREIGN KEY (ConversationId) REFERENCES Conversation(Id)
    # );
    # """
    
    # cursor.execute(create_table_query)
    # connection.commit()
    # cursor.close()
    # connection.close()

def get_conversations(conversation_id: int):

    #conversation_id: int
    # Connect to the database
    connection = mysql.connector.connect(
        host="pcz218dbl23",
        user="prakashd",
        password="TLzWqu8Kyp",
        database="omni_qa_db"
    )

    cursor = connection.cursor()

    # cursor.execute("SELECT conversationid, message, conversationincomingtime FROM conversationincoming ")
    # incoming = cursor.fetchall()
    
    # cursor.execute("SELECT conversationid, message, conversationoutgoingtime FROM conversationoutgoing ")
    # outgoing = cursor.fetchall()

    # Retrieve incoming conversation messages for a specific conversation ID
    cursor.execute("SELECT conversationid, message, conversationincomingtime FROM conversationincoming WHERE conversationid = %s", (conversation_id,))
    incoming = cursor.fetchall()

    # Retrieve outgoing conversation messages for a specific conversation ID
    cursor.execute("SELECT conversationid, message, conversationoutgoingtime FROM conversationoutgoing WHERE conversationid = %s", (conversation_id,))
    outgoing = cursor.fetchall()

    # Close connection
    cursor.close()
    connection.close()

    # Create dataframes for incoming and outgoing messages
    df_incoming = pd.DataFrame(incoming, columns=["ConversationId", "UserMessage", "Timestamp"])
    df_outgoing = pd.DataFrame(outgoing, columns=["ConversationId", "BotMessage", "Timestamp"])

    df_outgoing = clean_df_outgoing(df_outgoing)

    # Convert timestamps to datetime
    df_incoming["Timestamp"] = pd.to_datetime(df_incoming["Timestamp"])
    df_outgoing["Timestamp"] = pd.to_datetime(df_outgoing["Timestamp"])

    # Add Speaker column (user for incoming, bot for outgoing)
    df_incoming['Speaker'] = 'user'
    df_outgoing['Speaker'] = 'bot'

    # Rename columns for consistency
    df_incoming.rename(columns={'UserMessage': 'MessageText'}, inplace=True)
    df_outgoing.rename(columns={'BotMessage': 'MessageText'}, inplace=True)

    # Combine both incoming and outgoing messages into one dataframe
    df_combined = pd.concat([df_incoming[['ConversationId', 'Timestamp', 'Speaker', 'MessageText']],
                             df_outgoing[['ConversationId', 'Timestamp', 'Speaker', 'MessageText']]], ignore_index=True)

    # Sort the combined dataframe by ConversationId and Timestamp
    df_combined = df_combined.sort_values(by=['ConversationId', 'Timestamp'])

    # Create a mergedmessages column by concatenating Speaker and MessageText
    df_combined['mergedmessages'] = df_combined['Speaker'] + ': ' + df_combined['MessageText']

    # Group by ConversationId to form the entire conversation as a single text block
    df_final = df_combined.groupby("ConversationId")['mergedmessages'].apply(lambda x: ' '.join(x.astype(str).fillna(''))).reset_index()

    # df_final.to_pickle("final_convo.pkl")
    df_final.to_csv("final_convo.csv", index=False)

    return df_final
