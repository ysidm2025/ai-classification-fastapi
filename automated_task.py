import pandas as pd
import openai
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv
import os

# ---- Load environment variables ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ---- Configuration ----
MODEL = "gpt-4o"  # GPT-4o or GPT-4-turbo
TEMPERATURE = 0.3  # Lower temp for consistency
MAX_TOKENS = 1000
PROMPT_FILE = "improved_prompt.txt"
TEST_SET_FILE = "test_set_updated.csv"
FEEDBACK_FILE = "feedback_log.csv"

# ---- Base Initial Prompt ----
base_prompt = """
You are an AI assistant evaluating chatbot performance.
Your task is to classify the following conversation based on the chatbot's ability to satisfy the user's needs.
Classify the conversation as either:
- "Successful" if the chatbot resolved the query.
- "Unsuccessful" if the chatbot did not provide relevant information or failed to resolve the issue.
"""

# ---- Load Prompt ----
def load_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as file:
            return file.read().strip()
    return base_prompt

# ---- Save Updated Prompt ----
def save_prompt(prompt):
    with open(PROMPT_FILE, "w") as file:
        file.write(prompt)

# ---- Load Data ----
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# ---- Classify Conversation ----
def classify_conversation(conversation, prompt):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": conversation},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        result = response['choices'][0]['message']['content'].strip()
        return result
    except Exception as e:
        print(f"❌ Error in classification: {e}")
        return "Error"

# ---- Generate Feedback for Misclassified Data ----
def generate_feedback(row):
    return f"Incorrect prediction for conversation {row['ConversationId']}. " \
           f"Expected: {row['label']}, but Predicted: {row['predicted_label']}. " \
           f"Review why the chatbot may have misunderstood or provided irrelevant responses."

# ---- Process Conversations ----
def process_conversations(data, prompt):
    predictions = []
    feedbacks = []

    for i, row in data.iterrows():
        conversation = row['mergedmessages']
        predicted_label = classify_conversation(conversation, prompt)

        # Store predicted label
        predictions.append(predicted_label)

        # Generate feedback if prediction is wrong
        if predicted_label != row['label']:
            feedbacks.append(generate_feedback(row))
        else:
            feedbacks.append("")

    # Store results in dataframe
    data['predicted_label'] = predictions
    data['feedback'] = feedbacks

    return data

# ---- Generate Improved Prompt ----
def generate_improved_prompt(previous_prompt, feedbacks):
    # Summarize feedback into actionable insights
    feedback_summary = "\n".join([f"- {fb}" for fb in feedbacks if fb.strip() != ""])

    # Improve prompt using feedback
    improved_prompt = f"""
{previous_prompt}
Use these new insights to improve classification accuracy:
{feedback_summary}
"""
    return improved_prompt

# ---- Calculate Accuracy ----
def calculate_accuracy(data):
    accuracy = accuracy_score(data['label'], data['predicted_label'])
    print(f"✅ Classification Accuracy: {accuracy:.2%}")
    return accuracy

# ---- Main Loop ----
def main():
    # Load conversation data
    data = load_data(TEST_SET_FILE)

    # Load the last generated prompt or use base prompt
    current_prompt = load_prompt()

    iteration = 0
    max_iterations = 10
    target_accuracy = 0.99

    while iteration < max_iterations:
        print(f"\n🚀 Iteration {iteration + 1} - Using Updated Prompt:\n{current_prompt}\n")

        # Process conversations and classify
        result_data = process_conversations(data, current_prompt)

        # Calculate accuracy and log results
        accuracy = calculate_accuracy(result_data)

        # Save the feedback log
        result_data.to_csv(FEEDBACK_FILE, index=False)

        # Check if target accuracy is reached
        if accuracy >= target_accuracy:
            print("🎯 Target accuracy reached! No further prompt updates needed.")
            break

        # Generate improved prompt with feedback
        feedbacks = result_data['feedback'].tolist()
        improved_prompt = generate_improved_prompt(current_prompt, feedbacks)

        # Save improved prompt for next iteration
        save_prompt(improved_prompt)

        # Update the prompt for next run
        current_prompt = improved_prompt
        iteration += 1

    print("✅ Process Completed.")

# ---- Run the Main Script ----
if __name__ == "__main__":
    main()
