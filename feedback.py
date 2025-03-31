import pandas as pd
import openai
from db_connection import get_conversations, create_conversation_review_table, get_conversation
from model import classify_conversation, store_classification_results, classify_conversations
from sklearn.metrics import classification_report, confusion_matrix
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API client
openai.api_key = OPENAI_API_KEY

# ------------------------------
# Load the updated test set
# ------------------------------
def load_labeled_data():
    """
    Load labeled conversations from test_set_updated.csv.
    """
    try:
        df = pd.read_csv("test_set_updated.csv")
        print("✅ Loaded labeled conversations from test_set_updated.csv.")
        return df
    except FileNotFoundError:
        print("❌ File test_set_updated.csv not found. Creating an empty DataFrame.")
        return pd.DataFrame(columns=["ConversationId", "mergedmessages", "label"])

# ------------------------------
# Generate feedback using GPT
# ------------------------------
def generate_feedback(merged_message, predicted_label, correct_label):
    """
    Generate feedback using GPT to explain why the prediction was wrong.
    """
    prompt = f"""
    The following conversation was classified incorrectly:

    Conversation: {merged_message}
    Predicted Label: {predicted_label}
    Correct Label: {correct_label}

    Analyze the conversation and explain in detail why the classification was wrong. 
    Provide a reason that can help improve future predictions.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert in conversation classification and error analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )

    feedback = response['choices'][0]['message']['content'].strip()
    return feedback


# ------------------------------
# Generate corrective prompt using feedback
# ------------------------------
def generate_corrective_prompt(new_message, incorrect_predictions):
    """
    Generate a corrected prompt using previous feedback for future classifications.
    """
    prompt = f"""
    Consider the following corrected examples of past misclassified conversations:

    {incorrect_predictions[['mergedmessages', 'feedback']].to_string(index=False)}

    Now, classify the following conversation correctly:
    Conversation: {new_message}
    """
    return prompt

# ------------------------------
# Classify conversation with GPT using feedback
# ------------------------------
def classify_conversation(conversation_id, merged_messages, incorrect_predictions=None):
    """
    Classify the conversation using GPT and apply feedback to improve future classifications.
    """
    corrective_prompt = generate_corrective_prompt(merged_messages, incorrect_predictions) if incorrect_predictions is not None else merged_messages

    messages = [
        # {"role": "system", "content": """
        #     You are an AI assistant evaluating chatbot performance.
        #     Your task is to classify the following conversation based solely on the chatbot's ability to satisfy the user's needs.
        #     Focus on whether the chatbot provided relevant, accurate, and helpful responses that helped the user with their questions and concerns.
        #     Do not classify the conversation based on the user's mistakes or behavior; only evaluate the chatbot's responses.
        #     If the user made a mistake but the bot answered correctly, the conversation is classified as 'Successful.'
        #     If the user asks to talk to leasing agent or human or any phrase mark it as 'Unsuccessful'.

        #     The classification should be either 'Successful' or 'Unsuccessful.'
        #     Please return ONLY 'Successful' or 'Unsuccessful', without any explanations or extra text.

        #     One of many example of Successful Conversation:
        #     - User: "I need directions to a nearby park."
        #     - Bot: "Here is a map of the nearest parks in your area: [map link]."
        #     - This conversation is 'Successful' because the bot gave a relevant, accurate response to the user's request.

        #     example 2 of Successful Conversation :-
        #     -user: Check Balance Due 
        #     -bot: Late Fees: $867.00 Late Fees: $1000.00 Late Fees: $1000.00 Late Fees: $1000.00 Late Fees: $1000.00 Late Fees: $1500.00 Late Fees: $1000.00 Rent- Residential: $1000.00 Late Fees: $22.34 Rent- Residential: $468.00 Parking - Industrial: $100.00 Rent- Residential: $1000.00 Late Fees: $31.00 Unapplied Credit: $0.00 Total: $9988.34 
        #     -bot: Your rent is due on the 10th of the month and you have a grace period of 2 days after which late fees start to accrue. 
        #     -bot: How else can I help you? 
        #     -user: late fee 
        #     -bot: Current late fee charges: $ 7420.34 
        #     -bot: Your rent is due on the 10th of the month and you have a grace period of 2 days after which late fees start to accrue. 
        #     -bot: How else can I help you?
        #     - This conversation is successful because it have the total amount which is the actual balance due .

        #     One of many example of Unsuccessful Conversation 
        #     - User: "Do you have 2 bedrooms?"
        #     - Bot: "When would you like to move in?"
        #     - User: "Mid Oct"
        #     - Bot: "Here is what I found for 2 beds with a move in date of 10/21/2020."
        #     - Bot: "Would you like to know more about our apartments?"
        #     - User: "How about 3 bedrooms?"
        #     - Bot: "Pet Policy Cats - 1 allowed Max weight 15 lb each, Rent $50.00, Deposit $50.00, Comments: We love our feline companions! Dogs - 1 allowed Max weight 40 lb each, Rent $50.00, Deposit $100.00, Pet Care available."
        #     - This conversation is 'Unsuccessful' because when the bot was asked about 3 bedrooms he returned pet policy .

        #     example 2 of Unsuccessful Conversation :
        #     "user: Maintenance Request bot: If this is an Emergency Maintenance request, please call (805) 001-2000. 
        #     bot: Please describe your issue so that I can start the service request. 
        #     user: water supply  bot: Do we have permission to enter the apartment? 
        #     user: Yes 
        #     bot: Do you have pets or any special instructions for our service team? 
        #     user: yes 
        #     bot: Your service request has been submitted successfully. Your request I.D. is 8398 bot: How else can I help you? 
        #     user: pay rent 
        #     bot: Unable to complete transaction. Please try again later."
        #     -This conversation is Unsuccessful because bot was Unable to complete transaction .
        # """},

       {"role": "system", "content": """
        You are an AI assistant evaluating chatbot performance.
            Your task is to classify the following conversation based solely on the chatbot's ability to satisfy the user's needs.
            Focus on whether the chatbot provided relevant, accurate, and helpful responses that helped the user with their questions and concerns.
            Do not classify the conversation based on the user's mistakes or behavior; only evaluate the chatbot's responses.
            If the user made a mistake but the bot answered correctly, the conversation is classified as 'Successful.'
            If the user asks to talk to leasing agent or human or any phrase mark it as 'Unsuccessful'.

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

            Use these new insights to improve classification accuracy:
            To understand why the classification of this conversation was incorrect, let's break down the interaction and consider the characteristics that distinguish a "Successful" label from an "Unsuccessful" one:

            1. **Conversation Context**: The user greets, and the bot responds promptly with a greeting and an offer of assistance. It specifically addresses a rental property, indicating that the context is centered around providing information and possibly facilitating a next step related to this property.

            2. **Bot Initiative**: The bot takes initiative by not only acknowledging the user's initial greeting but also steering the conversation towards a potential call to action (scheduling an appointment). This proactive behavior is a positive indicator that the bot is functioning effectively in guiding the user towards achieving a tangible goal.

            3. **
            To analyze the incorrect classification of the conversation, let's break down the interaction and identify the characteristics that led to the mislabeling:

            ### Elements of the Conversation:

            1. **User's Initial Issue:**
            - The user clearly states they have a "payment issue," which the bot acknowledges.

            2. **Bot's Responses:**
            - The bot offers to pass the information along and assures the user someone will get back to them.
            - The bot requests the user's personal information to assist further with the payment issue.
            - After the user provides the necessary details, the bot again assures the information will be forwarded to the appropriate department and the user will be contacted.

            3. **User's Compliance:**
            - The user complies by providing
            The conversation was incorrectly classified as "Unsuccessful" when, in reality, it should have been labeled as "Successful." Let's analyze the conversation step by step to identify why this error occurred and how to improve future predictions.

            ### Analysis of the Conversation:

            1. **User Request to Speak with an Agent:**
            - The user initiates the conversation by asking to speak with an agent, suggesting that they need assistance that the bot either cannot provide or is not preferred at the moment.

            2. **Bot Response:**
            - The bot informs the user that all agents are currently busy but attempts to gather the user’s information to facilitate a callback or future contact. This indicates the bot is actively trying to assist the user despite the lack of immediate availability
            The conversation was classified incorrectly as "Unsuccessful" when it should have been "Successful." To understand why this misclassification occurred, let's analyze the exchange in detail:

            1. **Conflicting Bot Responses**:
            - The bot repeatedly indicated an intention to assist with scheduling ("I would be more than happy to help you schedule a tour! You should see the tour scheduling tool open in a moment."), which suggests progress toward booking the tour.
            - Simultaneously, after these positive messages, the bot follows up with a contradictory statement: "I apologize for the inconvenience, but at the moment we are not scheduling live video tours. However, we do offer guided tours."

            2. **Action Trigger**:
            - Despite the refusal message,
            In analyzing the conversation, we can identify several reasons why the classification as "Successful" was incorrect and should be labeled as "Unsuccessful."

            ### Key Issues in the Conversation

            1. **Initial Misunderstanding:**
            - The user initiated the conversation with the intent to inquire about community amenities, but the bot did not recognize this intent correctly and attempted to verify the communication channel with a request to send a text.

            2. **Repetitive Loop:**
            - The bot repeatedly asked the user for permission to send a text message instead of addressing the initial query about community amenities. The repetition of asking for text message permissions shows a failure in progressing the conversation towards addressing the user's need.

            3. **Irrelevant Focus:**
            -
            The conversation was incorrectly classified as "Successful" despite the correct label being "Unsuccessful." Here’s a detailed analysis of why this misclassification might have occurred and how to improve future predictions:

            ### Analysis of the Conversation:

            1. **Initial Engagement**:
            - The user initiates the conversation with a standard greeting ("Hello").
            - The bot immediately responds with a marketing consent message, which might not directly address the user's initial greeting.

            2. **Consent and Subscription**:
            - After the user replies "YES," the bot confirms subscription to text messages, which does indicate a form of interaction success if the primary goal of the conversation was to obtain consent for marketing messages.

            3. **Continuation of the Conversation**:
            - After subscribing
            The conversation in question has been classified as "Successful," but the correct label is "Unsuccessful." To understand why the classification was inaccurate, we need to closely examine the details of the interaction and identify any indicators of an unsuccessful user experience.

            ### Analysis:

            1. **Objective Recognition:**
            - The conversation is centered around creating a maintenance request. This interaction's goal is to facilitate the user's maintenance request through the bot's assistance.

            2. **Process Execution:**
            - The bot initiates the process by suggesting sending a text message to the user to further the maintenance request. The bot confirms if it can use the provided number for texting, and the user agrees.

            3. **Indicators of Success:**
            - For a conversation about processing
            The classification error in this conversation can be attributed to several key factors:

            1. **Repetition and Lack of Resolution**:
            - The user's query about "office hours" was repeated multiple times without a change in the wording, specifically twice in the conversation. This indicates that the user might not have found the provided information satisfactory or clear.
            - The bot's response, though informative in the second interaction, repeated a similar line in the third response as it used in the first response: "please take prior appointment before visiting our office." This repetition without acknowledging the user’s likely need for clarification or additional information can result in a negative user experience, thus classifying it as "Unsuccessful."

            2. **Absence of Acknowledgment**
            The conversation was classified incorrectly as "Successful" when it should have been "Unsuccessful." Here's a detailed analysis of why the classification was erroneous and how future predictions can be improved:      

            ### Analysis:
            1. **Objective of the Conversation:**
            - The primary goal of the conversation is for the user to obtain information about apartment availability.
            - The bot's role is to facilitate this by confirming contact details for further communication.

            2. **Sequence of Interaction:**
            - The bot offers to gather more information and asks for permission to send a message, to which the user agrees.
            - The bot repeats the phone number it intends to use for sending the information. The user confirms the number is correct.
            - The bot acknowledges the confirmation and
            In analyzing the conversation, we can understand why the classification might have been predicted as "Unsuccessful" despite the actual outcome being "Successful." Here are key points that might have led to the incorrect classification:

            1. **Task Completion vs. User Intent**: The central focus of the conversation is whether the user's request was fulfilled. The user initially requested information about the pet policy, which the bot provided accurately. The interaction proceeded with the user requesting a phone call regarding specific apartment criteria. The user successfully provided all necessary contact information, indicating that this part of the interaction was completed as intended, aligning with a "Successful" outcome.

            2. **Structured Data Gathering**: The bot effectively collected detailed user information (name, email, phone number) required
            The conversation was mislabeled as "Successful" when it should have been labeled as "Unsuccessful." Let's break down the conversation to understand why the classification went wrong and what can be done to improve future predictions.

            ### Conversation Breakdown:

            1. **User's Initial Query**:
            - The user initiates the conversation with a query about "Pricing & Availability."

            2. **Bot's Response: Incomplete Understanding**:
            - The bot responds with a generic fallback message, "Sorry, I missed what you just said. Can you say that again?" indicating a failure to process or understand the initial user request.

            3. **User's Restart**:
            - The user restarts the interaction by saying "/restart" and then saying "hello
            The conversation was classified as "Unsuccessful," but the correct label is "Successful." Let's analyze where the misclassification might have originated and suggest improvements for future predictions.

            ### Analysis:

            1. **Intent Fulfillment**:
            - **User Intent**: The user was looking for an apartment, specifically a 2-bedroom, and wanted information about availability and rent, along with scheduling a tour.
            - **Bot Response**: The bot successfully provided detailed information about the 2-bedroom apartment options, including available floor plans, units, and rental price ranges.

            2. **User Engagement**:
            - The user engaged with the bot by asking for rent details and ultimately expressed interest in scheduling a tour, indicating the interaction was moving toward a successful conclusion
            The conversation was classified incorrectly as "Unsuccessful" when it should have been classified as "Successful". Let’s break down why the misclassification occurred and how it can be improved in future predictions. 

            ### Key Analysis:

            1. **Repetitive Responses:**
            - The bot provided repetitive responses to similar user queries related to nearby parks, dog parks, bars, and schools, directing the user to the same map link for all inquiries.
            - The repetition might have led the classification system to believe the interaction was unsuccessful as it seemed the bot was not providing specific or tailored information for each query.

            2. **Satisfaction of User Intent:**
            - Despite the repetition, the bot fulfilled the overarching user intent by directing the user to a resource
            To properly assess why the conversation was misclassified as "Unsuccessful" instead of "Successful," we need to examine the sequence of interactions and identify where the classification system might have faltered. Below is a detailed analysis:

            ### Key Points of the Conversation

            1. **Restart and Repetition**:
            - The conversation begins with a `/restart` command, and the user initiates a tour scheduling process, making it clear from the start that they have a defined goal.
            - The conversation loops twice through the booking process due to entering "exit" in response to the phone number prompt. However, in both instances, the essential required information (first name, last name, email) appears to be collected correctly.

            2. **Data Collection**
            The conversation was classified incorrectly as "Unsuccessful" when it should have been labeled "Successful." Here's a detailed analysis of why the prediction was incorrect and how future predictions can be improved.  

            ### Key Points of the Conversation:

            1. **Initial Inquiry and Information Gathering:**
            - The user reached out for app fees, and although not directly answered, they were redirected to contact the property, which is a typical task completion step in real estate contexts.

            2. **Tour Scheduling Process:**
            - The conversation smoothly initiated a process to schedule a tour, request preferences, and gather necessary information.

            3. **Handling of Multiple Interactions:**
            - The bot encountered multiple points where further specificity was required (e.g., keys about dates,
            The conversation was incorrectly classified as "Successful" when it should have been labeled "Unsuccessful." Let's analyze the conversation to understand why this misclassification occurred and suggest improvements for future predictions.

            ### Analysis of the Conversation:

            1. **Initial Interaction:**
            - The user inquires about various pieces of information such as lease terms, maps and directions, office hours, online leasing, parking info, pet policy, photo gallery, property email, and property map. The bot responds adequately to each query.

            2. **Communication Breakdown:**
            - The user asks for "property number" and "property phone," but the bot responds with "I am a virtual assistant and did not understand." This suggests a lack of proper handling for synonymous requests for a        
            The conversation was incorrectly classified as "Successful" when it should have been classified as "Unsuccessful." Let's analyze the conversation step by step to understand why this misclassification occurred and to suggest improvements for future predictions.

            ### Analysis:
            1. **User's Initial Query:**
            - The user starts by asking for actual pictures. The bot responds affirmatively, suggesting sending links to pictures via text. This part of the conversation is smooth and indicates successful interaction.

            2. **Text Confirmation:**
            - The user consents to receiving the pictures via text, and the bot confirms the phone number. This exchange is successful, as both parties align on the method of communication.

            3. **Additional Help Inquiry:**
            - The bot asks if the user
            To analyze why the conversation was incorrectly classified as "Successful" instead of "Unsuccessful," we need to examine the criteria typically used to label a conversation in this context. Generally, a "Successful" conversation implies that the user's primary needs or questions are fully addressed, while an "Unsuccessful" conversation suggests that the user's needs were not entirely met or resolved.

            Here are some key issues that likely led to the incorrect classification:

            1. **Incomplete Response on Gym Facilities:**
            - The user asked about the presence of a gym in the building, and the bot responded by indicating it lacked sufficient information and offered to connect the user to the leasing office. This response leaves the user's question unresolved, which is a strong indicator of an "Unsuccessful"

            Use these new insights to improve classification accuracy:
            The conversation was classified as "Unsuccessful," while the correct label should have been "Successful." Let's analyze the conversation step-by-step to understand why it should be considered successful:

            ### Overview of Key Points in the Conversation:

            1. **Repeated Queries and Answers:**
            - The user asks for "Pet Policy" multiple times, and each time, the bot provides a clear and consistent response regarding the pet policy, including details about the number and type of pets allowed, weight limits, and any applicable deposits.
            2. **User Queries about Office Hours:**
            - The user repeatedly inquires about "Office Hours," to which the bot adequately provides the required information.
            3. **Final Confirmation:**
            - The last user query, "do
            In the given conversation, the predicted label was "Unsuccessful," while the correct label is "Successful." Let's analyze the flow of the conversation and identify where the misclassification occurred and how to improve future predictions.

            ### Conversation Analysis

            1. **User Request to Talk with Agent:**
            - The conversation begins with the user's request: "can I talk with agent?" This indicates an intention to connect with a human agent.

            2. **Bot Response:**
            - The bot responds by saying it will attempt to connect the user to a leasing agent and subsequently informs them that all agents are busy. Despite this, it offers an alternative by gathering information to ensure follow-up.

            3. **Information Gathering:**
            - The bot successfully collects the user's
            The conversation you provided was incorrectly classified as "Unsuccessful," while the correct label should have been "Successful." Let's analyze why this misclassification may have occurred and what could be done to improve future predictions.

            ### Analysis of the Incorrect Classification

            1. **Repetitive Patterns:**
            - The conversation shows a pattern where the bot repeatedly attempts to launch the tour scheduling tool but fails to process the user's request appropriately. Despite the bot consistently responding with messages like "I would be more than happy to help you schedule a tour!" and opening the scheduling tool, it always follows with, "I apologize, but at the moment we are not scheduling live video tours."

            2. **Expectation vs. Outcome:**
            - These sequences might confuse the
            The classification of the conversation as "Successful" instead of the correct label "Unsuccessful" might have occurred due to a misunderstanding of the context and the goals in the interaction. Here's a detailed analysis of the conversation that highlights why the classification was incorrect and how future predictions might be improved:

            1. **Intended Outcome**:
            - The primary purpose of this conversation was to initiate a maintenance request. The bot's objective seemed to be getting the user's consent to send a text message to start the process.

            2. **Incomplete Action**:
            - While the conversation successfully establishes that the bot will send a text message to the user, there is no evidence within this excerpt of the conversation that the text was actually sent, nor confirmation from the user that
            The conversation was incorrectly classified as "Successful" when the correct label should have been "Unsuccessful." Let's analyze why the classification was wrong and how it can be improved in the future.

            ### Reasons for Incorrect Classification:

            1. **Incomplete Information Provided:**
            - The bot was unable to provide specific responses to all user queries, such as the application fee and detailed property information. Instead, the bot asked the user to schedule a tour, which suggests that the user's request was not fulfilled online.

            2. **Multiple Unresolved Queries:**
            - Several user inquiries remained unanswered, or the bot deferred providing detailed information. For instance, when the user asked about the application fee and property information, the bot did not give a direct answer, which indicates
            The conversation was incorrectly classified as "Successful" when it should have been labeled "Unsuccessful." Let's break down the elements of the conversation to understand why the prediction was incorrect and how it can be improved in future classifications.

            ### Breakdown of the Conversation:

            1. **Initial Context:**
            - **User Intent:** The user is inquiring about apartment availability.
            - **Bot Response:** The bot offers to send a message to gather more information.

            2. **Subsequent Interaction:**
            - **Bot Action:** The bot requests permission to text the user at a specific phone number.
            - **Reiteration:** The bot repeats the question about sending a text to confirm.
            - **User Confirmation:** The user confirms their consent to receive
            The incorrect classification of the conversation can be attributed to several factors within the interaction between the user and the bot. Let’s break down the conversation and identify key elements that contributed to its misclassification as "Successful" when it should have been labeled "Unsuccessful."

            ### Analysis

            1. **Initial Confusion and Restart:**
            - The conversation starts with a misunderstanding or failed recognition from the bot (“Sorry, I missed what you just said. Can you say that again?”). This indicates a lack of successful communication right at the beginning.

            2. **Use of "/restart":**
            - The user inputs "/restart," which typically signifies that the user is not satisfied with the current state of the conversation or is experiencing issues. This is a critical
            The conversation was classified as "Successful," but the correct label is "Unsuccessful." Let's analyze the conversation to understand where the misclassification occurred and why it is incorrect.

            ### Analysis of the Conversation:

            1. **User Inquiry and Information Exchange:**
            - The conversation starts with the user inquiring about the availability of 2-bedroom and later 3-bedroom apartments. The bot successfully provides information about both types of apartments, including square footage and starting prices.
            - The conversation continues with a discussion about scheduling a tour, which the bot facilitates. The user provides details such as tour preferences, time, and personal information.

            2. **Understanding of "Successful" vs. "Unsuccessful" Labels:**
            - **"Successful"** generally
            The conversation was misclassified as "Unsuccessful" when it should have been labeled as "Successful." Let's analyze the interaction and the potential reasoning behind this misclassification.

            ### Analysis:
            1. **Consistency in Responses:** The bot consistently provided a response directing the user to a map link where information on nearby attractions, including parks, bars, and schools, could potentially be found. While the bot's responses might seem repetitive and not tailored specifically to each query, it successfully directed the user to a resource intended to provide the desired information.

            2. **Fulfillment of User Intent:** The user's inquiries were about finding locations nearby (parks, dog parks, bars, schools). The bot's responses, by directing the user to the location's map and directions page      
            The conversation you provided revolves around a user attempting to schedule a guided tour. The conversation classification model needs to determine whether the entire process leads to a "Successful" or "Unsuccessful" outcome based on the user's objectives. In this case, the model classified the conversation as "Unsuccessful," but the correct label should be "Successful." Here's an analysis of the conversation and the classification error:

            ### Key Components of the Conversation for Classification:

            1. **User Intent and Completion**:
            - The user successfully starts the process of scheduling a guided tour by specifying their preferences (date, time, and personal details) including their name and email.
            - The conversation successfully reaches the point where most typical objectives of scheduling a tour are achieved.

            2. **
            To determine why this conversation was incorrectly classified as "Successful" rather than the correct label, "Unsuccessful," let's break down the conversation flow and identify key indicators that led to this misclassification:

            1. **Indicators of a Successful Conversation:**
            - The user was able to obtain information on various topics such as lease terms, office hours, online leasing, pet policy, etc., without any apparent issues.
            - A tour was successfully scheduled, which usually signifies a key conversion event, indicating a successful interaction.

            2. **Indicators of an Unsuccessful Conversation:**
            - The bot failed to understand some direct user requests, indicating gaps in providing solutions:
                - When the user asked for "property number" and "property phone," the bot
            The conversation was classified as "Successful," whereas the correct label should have been "Unsuccessful." This misclassification likely stems from the initial interactions in the conversation, which appear smooth and adequate, but the latter part reveals a critical issue: a looped error from the virtual assistant.

            Let's break down the conversation to understand the misclassification:

            1. **Initial Interaction:**
            - The user requests pictures.
            - The bot offers to send links via text and verifies the user's number.
            - The user confirms, and the bot thanks the user and inquires if additional help is needed.
            - This part of the conversation is handled well, with clear communication, and could easily be perceived as successful if considered in isolation.

            2. **Critical Failure
            The classification of the conversation as "Successful" rather than "Unsuccessful" appears to be incorrect due to misinterpretation of the conversation's outcome. A successful interaction typically means the user's primary inquiries or requests have been fulfilled in a satisfactory manner. In this analysis, I'll pinpoint the reasons contributing to the wrong classification and suggest improvements.

            ### Breakdown of the Conversation:
            1. **Pet Policy & Parking Spot:**
            - User asked about pets and got a clear answer that dogs are allowed.
            - User inquired about renting a parking spot, and received a satisfactory response.

            2. **Parking Spot Usage:**
            - The bot promised to check the property's policy regarding a friend parking while the user is away but did not provide an answer within this
        """},
        {"role": "user", "content": corrective_prompt},
        {"role": "assistant", "content": "Classify the conversation based on the bot's performance and whether it satisfied the user's needs."}
    ]

    # Send request to OpenAI model
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=5,  # Limit to the shortest response
        temperature=0.0  # Zero temperature for consistent results
    )

    # Extract the classification result
    result = response['choices'][0]['message']['content'].strip()

    # Ensure the classification is either 'Successful' or 'Unsuccessful'
    if result.lower() == "successful":
        predicted_label = "Successful"
    elif result.lower() == "unsuccessful":
        predicted_label = "Unsuccessful"
    else:
        predicted_label = "Unsuccessful"  # Default to Unsuccessful if result is unexpected

    return predicted_label


# ------------------------------
# Classify and update with feedback
# ------------------------------
def classify_and_update():
    """
    Classify all conversations from test_set_updated.csv using OpenAI and calculate accuracy.
    Apply feedback to improve future classification.
    """
    # Load labeled data
    df = load_labeled_data()

    # Check if the CSV is empty
    if df.empty or "mergedmessages" not in df.columns:
        print("⚠️ No labeled conversations found in test_set_updated.csv.")
        return

    # Identify incorrect predictions
    incorrect_predictions = df[df['label'] != df['predicted_label']]
    incorrect_predictions.to_csv('incorrect_predictions.csv', index=False)
    print(f"Identified {len(incorrect_predictions)} incorrect predictions. Saved to 'incorrect_predictions.csv'.")

    # Generate feedback for all incorrect predictions
    incorrect_predictions['feedback'] = incorrect_predictions.apply(
        lambda row: generate_feedback(
            row['mergedmessages'],
            row['predicted_label'],
            row['label']
        ), axis=1
    )

    # Save feedback to CSV
    incorrect_predictions.to_csv('incorrect_predictions_with_feedback.csv', index=False)
    print("Feedback generated and saved to 'incorrect_predictions_with_feedback.csv'.")

    # Store predicted labels
    predicted_labels = []

    # Classify each conversation with feedback
    for _, row in df.iterrows():
        conversation_id = row["ConversationId"]
        merged_messages = row["mergedmessages"]

        # Get predicted label from OpenAI with feedback applied
        predicted_label = classify_conversation(conversation_id, merged_messages, incorrect_predictions)
        predicted_labels.append(predicted_label)

    # Add predicted_label column to the DataFrame
    df["predicted_label"] = predicted_labels

    # Save updated labeled data
    df.to_csv("test_set_updated.csv", index=False)
    print("✅ Updated predictions saved to test_set_updated.csv successfully!")

    # Calculate accuracy
    correct_predictions = (df["label"] == df["predicted_label"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"📊 Classification Accuracy: {round(accuracy * 100, 2)}%")

# ------------------------------
# API Route to Classify Conversations
# ------------------------------
class ConversationRequest(BaseModel):
    conversation_id: int
    merged_messages: str


@app.post("/classify/")
def classify_and_store(request: ConversationRequest):
    """
    API route to classify and store conversation results.
    """
    try:
        # Classify the conversation
        predicted_label = classify_conversation(request.conversation_id, request.merged_messages)

        # Store results in ConversationReview table
        store_classification_results(request.conversation_id, predicted_label)

        return {"conversation_id": request.conversation_id, "predicted_label": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
# Run Classification with Feedback
# ------------------------------
if __name__ == "__main__":
    classify_and_update()
