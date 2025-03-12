import mysql.connector
import pandas as pd
import gc
import json
import re
# Function to convert JSON bot messages into plain text
def process_message(df):
    df_processed = df.copy()  # Create a copy of the DataFrame to avoid modifying original
    df_processed["BotMessage"] = df_processed["BotMessage"].apply(lambda x: extract_text_from_json(x))
    return df_processed

# Helper function to extract plain text from JSON
def extract_text_from_json(bot_message):
    if isinstance(bot_message, str):  # Ensure it's a string before processing
        try:
            parsed_message = json.loads(bot_message)  # Attempt to parse JSON
            if isinstance(parsed_message, list) and "text" in parsed_message[0]:  
                return parsed_message[0]["text"]  # Extract text field from JSON
        except json.JSONDecodeError:
            pass  # If parsing fails, return original message
    
    return bot_message  # Return original if not JSON or improperly formatted

# Function to extract the text from JSON structure
################################################
# def extract_text_from_json(json_text):
#     try:
#         # Attempt to parse JSON
#         parsed_json = json.loads(json_text)
#         print(parsed_json)
#         if isinstance(parsed_json, list) and "text" in parsed_json[0]:
#             print('hell_world')
#             return parsed_json[0]["text"]  # Extract and return text field
#     except json.JSONDecodeError:
#         return json_text  # If JSON decoding fails, return original text
#     return json_text  # Return original if no 'text' key is found
#########################################################
def extract_text_from_json(bot_message):
    """Extracts text content from a JSON-formatted bot message."""
    if isinstance(bot_message, str):  # Ensure it's a string
        try:
            parsed_message = json.loads(bot_message)  # Parse JSON
            if isinstance(parsed_message, list) and len(parsed_message) > 0 and "text" in parsed_message[0]:
                return parsed_message[0]["text"]  # Extract text field
        except json.JSONDecodeError:
            pass  # If parsing fails, return original message
    return bot_message  # Return as-is if not JSON

def clean_df_outgoing(df_outgoing):
    """Cleans the BotMessage column in df_outgoing by removing JSON while keeping text."""
    df_outgoing = df_outgoing.copy()  # Avoid modifying original DataFrame
    df_outgoing["BotMessage"] = df_outgoing["BotMessage"].apply(extract_text_from_json)
    return df_outgoing

def get_conversations(conversation_id: int):
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

    # Retrieve incoming conversation messages for a specific conversation ID
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
    # df_cleaned = df_combined.dropna(subset=["UserMessage", "BotMessage"])

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

def fetch_conversation_by_id(conversation_id: 33):
    """Fetch and process a conversation by ConversationId."""
    try:
        connection = mysql.connector.connect(
            host="pcz218dbl23",
            user="prakashd",
            password="TLzWqu8Kyp",
            database="omni_qa_db"
        )
        cursor = connection.cursor(dictionary=True)

        # Retrieve incoming conversation messages for a specific conversation ID
        cursor.execute("SELECT conversationid, message, conversationincomingtime FROM conversationincoming WHERE conversationid = %s", (conversation_id,))
        incoming = cursor.fetchall()
        
        # Retrieve outgoing conversation messages for a specific conversation ID
        cursor.execute("SELECT conversationid, message, conversationoutgoingtime FROM conversationoutgoing WHERE conversationid = %s", (conversation_id,))
        outgoing = cursor.fetchall()

    #     # Create DataFrames for incoming and outgoing messages
    #     df_incoming = pd.DataFrame(incoming, columns=["ConversationId", "UserMessage", "Timestamp"])
    #     df_outgoing = pd.DataFrame(outgoing, columns=["ConversationId", "BotMessage", "Timestamp"])

    #     # Convert timestamps to datetime
    #     df_incoming["Timestamp"] = pd.to_datetime(df_incoming["Timestamp"])
    #     df_outgoing["Timestamp"] = pd.to_datetime(df_outgoing["Timestamp"])

    #     # Merge user and bot messages on ConversationId and Timestamp
    #     df_combined = pd.merge(df_incoming, df_outgoing, on=["ConversationId", "Timestamp"], how="outer")

    #     # Sort by ConversationId and Timestamp
    #     df_combined.sort_values(by=["ConversationId", "Timestamp"], inplace=True)

    #     # Group by ConversationId to form entire conversations ----------------------
    #     df_grouped = df_combined.groupby("ConversationId").agg({
    #         "UserMessage": lambda x: " ".join(x.dropna()),  # Combine user messages
    #         "BotMessage": lambda x: " ".join(x.dropna())    # Combine bot responses
    #     }).reset_index()
    #     # Save grouped conversations

    #     # Process and clean the data (e.g., remove unwanted JSON-like bot responses)
    #     df_processed = process_message(df_combined)

    #     print("Fetched, processed bot messages, and cleaned conversation data.")
    #     print(df_processed)

    #     return df_processed  # Return the processed DataFrame

    # except mysql.connector.Error as e:
    #     print(f"Database error: {e}")
    #     return None
    # finally:
    #     cursor.close()
    #     connection.close()

        # Create dataframes
        df_incoming = pd.DataFrame(incoming, columns=["ConversationId", "UserMessage", "Timestamp"])
        df_outgoing = pd.DataFrame(outgoing, columns=["ConversationId", "BotMessage", "Timestamp"])
        print(df_incoming)
        print(df_outgoing)
        # Convert timestamps to datetime
        df_incoming["Timestamp"] = pd.to_datetime(df_incoming["Timestamp"])
        df_outgoing["Timestamp"] = pd.to_datetime(df_outgoing["Timestamp"])

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
        # df_cleaned = df_combined.dropna(subset=["UserMessage", "BotMessage"])

        # calling function to remove json bot messages ----------------------
        df_processed = process_message(df_grouped)
        df_processed.to_pickle("cleaned_conversations_processed.pkl")
        df_processed.to_csv("cleaned_conversations_processed.csv", index=False)
        # calling function to remove json bot messages ----------------------

        print("Fetched, processed bot messages, and saved to cleaned_conversations.csv")
        print(df_processed)

        # return df_processed
        return df_processed
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        cursor.close()
        connection.close()
fetch_conversation_by_id(33)
# get_conversations()
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

def fetch_conversation_review():
    """Fetch all records from the ConversationReview table and print them."""
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host="pcz218dbl23",
            user="prakashd",
            password="TLzWqu8Kyp",
            database="omni_qa_db"
        )
        cursor = connection.cursor()

        # Execute query to fetch all records from ConversationReview table
        cursor.execute("SELECT * FROM conversationreview;")
        rows = cursor.fetchall()

        if not rows:
            print("⚠️ The ConversationReview table is empty!")
        else:
            print("\n==== ConversationReview Table Contents ====")
            for row in rows:
                print(row)  # Print each row

    except mysql.connector.Error as e:
        print(f"Error fetching ConversationReview data: {e}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()