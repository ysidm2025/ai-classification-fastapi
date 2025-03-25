import mysql.connector
import pandas as pd
import gc
import json
import re
from bs4 import BeautifulSoup

def get_conversation():

    # Connect to the database
    connection = mysql.connector.connect(
        host="pcz218dbl23",
        user="prakashd",
        password="TLzWqu8Kyp",
        database="omni_qa_db"
    )

    cursor = connection.cursor()

    # Retrieve incoming and outgoing conversation messages
    cursor.execute("SELECT conversationid, message, conversationincomingtime FROM conversationincoming ")
    incoming = cursor.fetchall()
    
    cursor.execute("SELECT conversationid, message, conversationoutgoingtime FROM conversationoutgoing ")
    outgoing = cursor.fetchall()

    # # Retrieve incoming conversation messages for a specific conversation ID
    # cursor.execute("SELECT conversationid, message, conversationincomingtime FROM conversationincoming WHERE conversationid = %s", (conversation_id,))
    # incoming = cursor.fetchall()
        
    # # Retrieve outgoing conversation messages for a specific conversation ID
    # cursor.execute("SELECT conversationid, message, conversationoutgoingtime FROM conversationoutgoing WHERE conversationid = %s", (conversation_id,))
    # outgoing = cursor.fetchall()

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
    df_combined.to_csv('cleaned_conversations.csv', index=False)

    return df_combined

get_conversation()