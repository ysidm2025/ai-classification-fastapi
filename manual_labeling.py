import pandas as pd
from db_connection import get_conversations

# New dataframe to store manually labeled conversations
labeled_df = pd.DataFrame(columns=["ConversationId", "mergedmessages", "label"])

# Load labeled data if the file already exists
try:
    labeled_df = pd.read_csv("labeled_conversations.csv")
    print("✅ Loaded existing labeled data from labeled_conversations.csv.")
except FileNotFoundError:
    print("⚠️ No existing labeled data found. Starting fresh.")


# Function to manually label conversations
def label_conversation(conversation_id):
    """
    Fetch conversation using get_conversations and manually label it.
    Add labeled data to labeled_df.
    """
    # Get conversation by conversation_id using your get_conversations function
    df_conversation = get_conversations(conversation_id)
    
    if df_conversation.empty or 'mergedmessages' not in df_conversation.columns:
        print(f"No conversation found with ConversationId: {conversation_id}")
        return

    # Merge messages for the conversation
    conversation_text = " ".join(df_conversation['mergedmessages'].dropna())
    
    # Display the conversation
    print(f"\nConversation ID: {conversation_id}")
    print(f"Conversation: {conversation_text}\n")

    # Input label from the user (1 = Successful, 0 = Unsuccessful, Custom input)
    user_input = input("Enter 1 for Successful, 0 for Unsuccessful, or type your label: ").strip()

    # Automatically label if input is 1 or 0
    if user_input == '1':
        label = "Successful"
    elif user_input == '0':
        label = "Unsuccessful"
    elif user_input.lower() == 'unsuccessful':
        label = "Unsuccessful"
    else:
        # If no input or invalid input, default to Unsuccessful
        label = "Unsuccessful"

    # Create a dictionary with labeled data
    labeled_data = {
        "ConversationId": conversation_id,
        "mergedmessages": conversation_text,
        "label": label
    }
    
    # Append the labeled data to labeled_df
    global labeled_df
    labeled_df = pd.concat([labeled_df, pd.DataFrame([labeled_data])], ignore_index=True)
    print(f"\n✅ Labeled conversation {conversation_id} as {label} and added to labeled_df successfully!")
    
    save_labeled_data()

# Function to delete a conversation from labeled_df
def delete_labeled_conversation(conversation_id):
    """
    Delete a specific conversation by ConversationId.
    If only one conversation is left, ensure it's deleted properly.
    """
    global labeled_df

    # Check if conversation_id exists in the DataFrame
    if conversation_id in labeled_df['ConversationId'].values:
        # Remove the row with the matching ConversationId
        labeled_df = labeled_df[labeled_df['ConversationId'] != conversation_id]
        
        # Reset the index to prevent weird behavior when only 1 row is left
        labeled_df = labeled_df.reset_index(drop=True)

        # Save updated DataFrame
        save_labeled_data()

        print(f"\n✅ Conversation with ID {conversation_id} deleted successfully!")
    else:
        print(f"\n⚠️ Conversation with ID {conversation_id} not found.")

# Function to save the labeled data to CSV
def save_labeled_data():
    """
    Save all labeled conversations to a CSV file.
    Handle case where labeled_df is empty after deletion.
    """
    if labeled_df.empty:
        # Write only the headers if the DataFrame is empty
        pd.DataFrame(columns=["ConversationId", "mergedmessages", "label"]).to_csv("labeled_conversations.csv", index=False)
        print("\n⚠️ All conversations deleted. CSV file cleared but headers retained.")
    else:
        # Save non-empty DataFrame to CSV
        labeled_df.to_csv("labeled_conversations.csv", index=False)
        print("\n✅ All labeled conversations saved to labeled_conversations.csv successfully!")

# Function to display summary of labeled conversations
def display_summary():
    """
    Display total number of conversations, 
    count of successful and unsuccessful conversations, 
    and total count of each label.
    """
    if labeled_df.empty:
        print("\n⚠️ No labeled conversations found.")
        return

    total_conversations = labeled_df['ConversationId'].nunique()
    label_counts = labeled_df['label'].value_counts().to_dict()

    successful_count = label_counts.get('Successful', 0)
    unsuccessful_count = label_counts.get('Unsuccessful', 0)

    print("\n📊 Conversation Summary:")
    print(f"✅ Total conversations: {total_conversations}")
    print(f"🎯 Successful conversations: {successful_count}")
    print(f"⚠️ Unsuccessful conversations: {unsuccessful_count}")

    # Print count of each label
    print("\n📝 Label Breakdown:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

# Main menu for manual operations
def main_menu():
    """
    Main interactive menu for labeling, deleting, and saving.
    """
    while True:
        print("\n📚 Options:")
        print("1. Label a conversation")
        print("2. Delete a conversation by ConversationId")
        print("3. Save labeled conversations")
        print("4. Show conversation summary")
        print("5. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            try:
                conversation_id = int(input("Enter ConversationId to label: "))
                label_conversation(conversation_id)
            except ValueError:
                print("❌ Invalid input. Please enter a valid ConversationId.")
        
        elif choice == "2":
            try:
                conversation_id = int(input("Enter ConversationId to delete: "))
                delete_labeled_conversation(conversation_id)
            except ValueError:
                print("❌ Invalid input. Please enter a valid ConversationId.")
        
        elif choice == "3":
            save_labeled_data()
        
        elif choice == "4":
            display_summary()
        
        elif choice == "5":
            print("🚪 Exiting... Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 4.")


# Uncomment the line below to run the interactive menu
main_menu()
