import openai
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv
import os

# ---- LOAD ENV VARIABLES ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- MODEL CONFIG ----
openai.api_key = OPENAI_API_KEY
# FINE_TUNED_MODEL_ID = "ft:gpt-4-2025-03-25-09-00-00"  # Replace with your fine-tuned model ID
FINE_TUNED_MODEL_ID = "file-32zHbRXEqd8X87mPgYBigz"
TEMPERATURE = 0.0
MAX_TOKENS = 100

# ---- INPUT FILE ----
TEST_SET_FILE = "test_set_updated.csv"  # CSV with ConversationId, label, mergedmessages columns

# ---- CLASSIFY CONVERSATION ----
def classify_conversation(conversation, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Classify the conversation as 'Successful' or 'Unsuccessful'."},
            {"role": "user", "content": conversation},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    result = response['choices'][0]['message']['content'].strip()
    return result

# ---- PROCESS CONVERSATIONS ----
def process_conversations(conversations, model):
    predictions = []
    for i, row in conversations.iterrows():
        merged_message = row['mergedmessages']
        prediction = classify_conversation(merged_message, model)
        predictions.append(prediction)
    return predictions

# ---- LOAD TEST DATA ----
def load_test_data(file):
    return pd.read_csv(file)

# ---- CALCULATE ACCURACY ----
def calculate_accuracy(predictions, ground_truth):
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"✅ Classification Accuracy: {accuracy:.2%}")
    print("\n📊 Classification Report:\n", classification_report(ground_truth, predictions))
    return accuracy

# ---- MAIN FUNCTION ----
def main():
    # Load test data
    conversations = load_test_data(TEST_SET_FILE)

    # Classify conversations using fine-tuned model
    predictions = process_conversations(conversations, FINE_TUNED_MODEL_ID)

    # Store predictions in DataFrame
    conversations['predicted_label'] = predictions

    # Calculate accuracy
    true_labels = conversations['label'].tolist()
    calculate_accuracy(predictions, true_labels)

    # Save results with predictions
    conversations.to_csv("classified_results.csv", index=False)
    print("✅ Results saved to 'classified_results.csv'")

# ---- RUN MAIN ----
if __name__ == "__main__":
    main()
