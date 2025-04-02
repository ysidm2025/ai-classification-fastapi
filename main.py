from db_connection import get_conversations , create_conversation_review_table , get_conversation 
from model import classify_conversation , store_classification_results , classify_conversations
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from fastapi import FastAPI, HTTPException , Form
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os
from fastapi.responses import HTMLResponse 
import re

load_dotenv()

# FastAPI app initialization
app = FastAPI()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API client
openai.api_key = OPENAI_API_KEY

# Store ground truth and predictions
# y_true = []  # Actual labels (manually set or fetched from DB)
# y_pred = []  # Model predictions

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


class ClassificationResult(BaseModel):
    UserMessage: str
    BotMessage: str
    PredictedLabel: str
    ConfidenceScore: float

# Request model for conversation ID
class ConversationRequest(BaseModel):
    conversation_id: int

class ConversationResponse(BaseModel):
    conversation_id: int
    classification: str  # 'successful' or 'unsuccessful'
    conversation: str  # Full conversation text

class Config:
    orm_mode = True

# @app.post("/classify-conversation-openai/", response_model=ConversationResponse)
# async def classify_conversation(request: ConversationRequest):
#     conversation_id = request.conversation_id

#     # Retrieve the conversation data using the get_conversations function
#     df_conversation = get_conversations(conversation_id)

#     if df_conversation.empty or 'mergedmessages' not in df_conversation.columns:
#         raise HTTPException(status_code=404, detail="Conversation not found or missing 'mergedmessages' column")

#     # Concatenate all merged messages in the conversation into a single string
#     conversation_text = " ".join(df_conversation['mergedmessages'].dropna())

#     # Prepare a message prompt for OpenAI to classify the conversation
#     prompt = f"Classify the following conversation as either successful or unsuccessful: {conversation_text}"

#     try:
#         # Make an API call to OpenAI using the new chat-completion method
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-2024-08-06",  
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ]
#             temperature=0.0
#         )

#         # Extract the classification from OpenAI's response
#         classification = response['choices'][0]['message']['content'].strip()

#         # Return the classification and full conversation text as part of the response
#         return ConversationResponse(
#             conversation_id=conversation_id, 
#             classification=classification, 
#             conversation=conversation_text  
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

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
    # prompt = f"Classify the following conversation as either successful or unsuccessful: {conversation_text}"
    messages = [
        {"role":"system","content":""" 
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

            Use the following feedback to improve your classification logic:
            Feedback: The conversation was misclassified as unsuccessful, likely due to the repetitive nature of the bot's responses and the apparent contradiction in the bot's messages. The bot initially indicates that it can help schedule a tour and opens the scheduling tool, but then repeatedly states that live video tours are not being scheduled at the moment. This inconsistency might have led the model to interpret the conversation as unsuccessful because the user's request for a live video tour was not fulfilled.

            However, the conversation should be classified as successful because the bot consistently provides an alternative solution by offering guided tours instead. The user is informed about the unavailability of live video tours and is given an option to proceed with a guided tour, which is a valid resolution to the user's request. The bot's responses, although repetitive, do not leave the user's request unaddressed, as an alternative is provided.

            Therefore, despite the initial confusion, the conversation achieves a resolution by offering an alternative, so this conversation is successful.
            Feedback: The conversation was misclassified as successful, but it should be classified as unsuccessful. Here's why:

            1. **Incomplete Information**: The bot was unable to provide information about the application fee and property information, stating that it needed to look into it and suggesting a tour instead. This indicates that the user's queries were not fully addressed.     

            2. **Callback Request**: The user requested a callback, and while the bot collected the necessary contact information, it did not confirm a specific time for the callback or provide assurance that the callback would address the user's inquiries.

            3. **Repetitive Information**: The user asked for office hours twice, and the bot repeated the same information without addressing the user's other needs or questions.

            4. **Lack of Resolution**: The conversation ended without resolving the user's initial queries about the application fee and property information, which were the primary reasons for the interaction.

            The conversation did not achieve the user's goals of obtaining specific information about the application fee and property details, so this conversation is unsuccessful.
            Feedback: The conversation was misclassified as successful, but it should be classified as unsuccessful. Here's the analysis:

            1. **Initial Confusion**: The conversation starts with the user asking about "Pricing & Availability," but the bot initially fails to understand the request, responding with "Sorry, I missed what you just said. Can you say that again?" This indicates a failure in understanding the user's intent right from the start.

            2. **Restart Command**: The user issues a "/restart" command, which suggests that they were not satisfied with the bot's initial response and wanted to start the conversation over. This is a clear sign of user frustration or confusion.

            3. **Incomplete Information**: Although the bot eventually provides information about a 2-bedroom apartment, it does not confirm the availability for the user's requested move-in date of "tomorrow." Instead, it provides information for a move-in date of 03/18/2021, which may not align with the user's needs.

            4. **Lack of Confirmation**: The conversation ends without the user confirming whether the provided information was satisfactory or if they needed further assistance. The bot's final prompt is open-ended, suggesting more questions can be asked, but it does not address the user's initial request for immediate availability.

            These points indicate that the user's request was not fully understood or addressed, leading to an incomplete and potentially unsatisfactory interaction, so this conversation is unsuccessful.
            Feedback: The conversation was misclassified as successful, but it should be classified as unsuccessful. The user initially inquired about 2-bedroom and 3-bedroom apartments and was provided with information about availability and pricing. The user then requested to schedule a self-guided tour, which the bot facilitated by asking for a preferred date and time. However, after the user provided the time for the tour, the bot asked for the desired move-in date again, even though this information had already been provided earlier in the conversation ("mid oct"). The user repeated the move-in date, but the bot continued to ask for the number of bedrooms needed, which was also previously discussed. This indicates a breakdown in the conversation flow, as the bot failed to recognize that the necessary information had already been collected. Despite the user eventually providing all the requested information, the conversation's repetitive and redundant nature suggests a lack of coherence and efficiency in the interaction, leading to user frustration. Therefore, the conversation did not proceed smoothly and efficiently, so this conversation is unsuccessful.
            Feedback: The model likely misclassified this conversation as unsuccessful due to the repetitive and generic responses provided by the bot. The bot consistently directed the user to a map link without addressing the specific queries about parks, dog parks, bars, and schools within walking distance. This lack of tailored responses might have led the model to interpret the conversation as unhelpful or unsatisfactory.

            However, the user did not express dissatisfaction or ask for further clarification, and the bot consistently offered additional assistance, which indicates that the user might have found the provided information sufficient. The user continued to engage with the bot, suggesting that they were able to navigate the conversation and obtain the information they needed through the provided link.

            Therefore, despite the generic responses, the user did not indicate any dissatisfaction or failure to obtain the desired information, so this conversation is successful.
            Feedback: The model likely misclassified this conversation due to the user's repeated attempts to exit the conversation without providing a phone number. The user typed "exit" and "No" when prompted for a phone number, which might have been interpreted by the model as an indication of an incomplete or unsuccessful interaction. However, the user successfully scheduled a tour and provided all other necessary information, such as the tour type, date, time, first name, last name, and email address. The user's decision not to provide a phone number does not negate the successful scheduling of the tour. The conversation ends with the bot confirming that it will be available for further assistance, indicating a successful interaction.

            Therefore, despite the user's refusal to provide a phone number, the primary goal of scheduling a tour was achieved, so this conversation is successful.
            Feedback: The model might have misclassified this conversation due to a few potential reasons:

            1. **Error Handling**: The bot initially failed to recognize the user's email address and prompted for a valid one. This could have been interpreted as a failure in the conversation flow, even though the user successfully provided a valid email address afterward. 

            2. **Repetitive Prompts**: The bot repeated the question about the preferred tour type after the user had already selected "Guided." This repetition might have been seen as a breakdown in the conversation, even though the user eventually proceeded with the correct selection.

            3. **Date Selection Confusion**: There was a minor confusion with the date selection where the user initially provided an incorrect date ("Monday August 17"), but the bot successfully guided the user to select an available date ("Monday August 16").

            4. **Multiple Confirmations**: The bot asked for confirmation of the contact number and repeated the prompt when the user responded, which might have been perceived as a failure in understanding the user's input.

            Despite these minor issues, the conversation ultimately achieved its goal: the user successfully scheduled a tour. The user was able to provide all necessary information, and the bot confirmed the appointment. Therefore, the conversation achieved its intended outcome, so this conversation is successful.
            Feedback: The conversation involves a user interacting with a bot to gather information about a property, including lease terms, office hours, parking info, pet policy, and more. The user also successfully schedules a guided tour for a specific date and time, providing their contact information. However, the conversation ends with the user asking about "custom," to which the bot responds with "I am a virtual assistant and did not understand." This indicates that the bot failed to address the user's final query, which is a sign of an unsuccessful interaction.

            The model might have misclassified this conversation as successful because the majority of the user's requests were handled correctly, and the tour scheduling process was completed without issues. However, the inability of the bot to understand and respond to the user's final query about "custom" means that the conversation did not fully meet the user's needs, which is a critical factor in determining success.

            Therefore, despite the successful handling of most requests, the conversation ends with an unresolved query, so this conversation is unsuccessful.
            Feedback: The conversation was misclassified as successful, but it should be classified as unsuccessful. Here's why:

            1. **Incomplete Information**: The bot failed to provide information about the presence of a gym in the building. Instead, it suggested connecting the user to the leasing office, which indicates a lack of complete information.

            2. **Inconsistent Responses**: The bot was able to provide detailed information about the pool but not about the gym, which shows inconsistency in the bot's ability to access and provide information.

            3. **Unresolved Inquiry**: The user's inquiry about studio apartment availability was addressed, but the bot did not offer a direct solution or alternative for the user's specific request for a studio apartment. Instead, it suggested other types of apartments or sister properties, which may not meet the user's needs.

            4. **User Experience**: The user had to interact with the bot multiple times to get partial answers, and some of their questions were not fully resolved, leading to a potentially frustrating experience.

            These points indicate that the user's needs were not fully met, and the conversation did not achieve a satisfactory resolution, so this conversation is unsuccessful.
            Feedback: The model might have misclassified this conversation due to the repetitive nature of the user's questions and the bot's responses. The user repeatedly asked the same questions, such as "do you celebrate festivals?" and "what is your name?", and the bot provided consistent and appropriate responses each time. This repetition could have led the model to mistakenly interpret the conversation as unsuccessful, assuming that the user was not satisfied with the bot's answers or that the bot was not effectively addressing the user's needs.

            However, the bot consistently provided relevant and accurate information in response to the user's inquiries, demonstrating its ability to handle repeated questions effectively. The user did not express dissatisfaction or confusion, and the bot maintained a helpful and informative tone throughout the conversation. The bot's responses were appropriate and aligned with the user's questions, indicating that the interaction met the user's needs.

            Therefore, despite the repetitive nature of the conversation, the bot successfully addressed the user's questions and maintained a coherent and helpful dialogue, so this conversation is successful.
            Feedback: The model likely misclassified this conversation due to the presence of an invalid input from the user ("!2w") when asked for the desired move-in date. This could have been interpreted as a failure point in the conversation. However, the bot successfully handled the invalid input by prompting the user again for a valid date, which the user then provided ("april 5"). The conversation continued smoothly after this point, with the bot asking for the user's first name, indicating that the scheduling process was still on track.

            The key factor here is that the bot was able to recover from the invalid input and continue the conversation without any further issues, leading towards the completion of the task. The presence of error handling and the continuation of the conversation towards the intended goal suggest that the interaction was indeed successful.

            Therefore, despite the temporary disruption, the conversation achieved its purpose, so this conversation is successful.
            Feedback: The conversation was misclassified as successful, but it should be classified as unsuccessful. The user repeatedly asks about the "Pet Policy" and "Office Hours," indicating that they are not receiving the information they need or are not satisfied with the responses. The bot provides the same information multiple times without addressing any potential underlying issues or confusion the user might have. Additionally, the bot repeatedly asks for the user's name, which the user does not provide, suggesting a lack of engagement or interest in continuing the conversation. The user's repeated queries and lack of progression in the conversation indicate that their needs were not fully met, so this conversation is unsuccessful.
            Feedback: The model likely misclassified this conversation due to the presence of a clear interaction where the user responds with "YES" to consent to receive messages, which might have been interpreted as a successful completion of a task. However, the primary goal of a successful conversation should be to address the user's initial query or need. In this case, the user starts with "Hello," which typically indicates a desire to initiate a conversation or seek assistance. The bot's response is a subscription consent message, which does not directly address the user's greeting or any potential inquiry. The user then says "Hello again," suggesting that their initial attempt to engage in a conversation was not fulfilled. The conversation does not achieve a meaningful interaction or resolution of the user's potential needs, so this conversation is unsuccessful.
            Feedback: The conversation was misclassified as successful, but it should be classified as unsuccessful. The user attempted to ask a specific question about a pet limit, which was not addressed by the bot. Instead, the bot continued to collect contact information without providing an answer or acknowledging the user's query. The user's main intent was to get information about a pet limit, and since this was not fulfilled, the conversation did not achieve its intended outcome. The bot's response was generic and did not address the user's needs, leading to an unsatisfactory interaction for the user. Therefore, the conversation did not meet the user's expectations or resolve their query, so this conversation is unsuccessful.
            Feedback: The model might have misclassified this conversation due to a few potential reasons:

            1. **Lack of User Intent Clarity**: The user's message is simply "Hello," which does not explicitly state a need or question. The model might have expected a more detailed inquiry or request to classify the conversation as successful.

            2. **Bot's Proactive Response**: The bot provides a proactive and relevant response by offering assistance related to a specific rental property, "Mountain Pine Villas," and suggests scheduling a tour. This indicates the bot is functioning as intended by guiding the user towards a potential next step.

            3. **Absence of User Follow-up**: The conversation is very short, with no follow-up from the user. The model might have been trained to expect a back-and-forth exchange to determine success, but in this case, the bot's response is sufficient to fulfill its role.  

            Despite the brevity and lack of explicit user intent, the bot's response is appropriate and provides a clear path forward for the user, which aligns with the goal of assisting with rental property inquiries, so this conversation is successful.
            Feedback: The model might have misclassified this conversation due to the repetition and lack of acknowledgment of the user's query "can you see this." The bot repeated the previous prompt instead of addressing the user's question, which could be interpreted as a failure in understanding or responding appropriately. However, the conversation still achieved its primary goal of scheduling a tour, as the user provided a date and time range, and the bot guided the user through the process. The user's final question does not impact the overall success of the main task, which was to schedule a tour. Therefore, despite the minor issue, the main objective was accomplished, so this conversation is successful.
            Feedback: The conversation was misclassified as successful, but it should be classified as unsuccessful. Here are the reasons for this:

            1. **Unmet User Needs**: The user initially inquired about the availability of one-bedroom and two-bedroom apartments. The bot responded that there were no floor plans available for the user's requirements, which means the user's primary need was not met.

            2. **Pet Policy Confusion**: When the user asked about the pet policy, the bot failed to provide a clear response initially. It only provided information about cats and dogs after the user specifically asked about dogs, but it did not address the user's question about fish or provide a comprehensive pet policy.

            3. **Tour Scheduling**: Although the bot successfully scheduled a tour, this does not compensate for the fact that the user's primary request regarding apartment availability was not fulfilled.

            4. **Incomplete Information**: The bot did not provide any alternative solutions or suggestions after failing to meet the user's initial requests, such as offering to notify the user when apartments become available or suggesting other properties.

            The conversation did not achieve the user's primary goal of finding an available apartment, and the bot's responses were insufficient in addressing the user's inquiries, so this conversation is unsuccessful.
            Feedback: The model likely misclassified this conversation due to the repetitive nature of the user's request and the bot's response. The user expressed interest in the one-bedroom apartment and requested a callback, but the bot repeatedly provided the same information about available units without addressing the user's request for a callback. This could have led the model to interpret the conversation as unsuccessful because the user's specific request was not directly acknowledged or fulfilled by the bot.

            However, the conversation can still be considered successful because the user received the information they were seeking about the pet policy, apartment amenities, and available one-bedroom units. The primary goal of the conversation was to provide information, which the bot accomplished, even though it did not handle the callback request. The user was able to gather the necessary details to make an informed decision about the apartment.

            Therefore, despite the lack of response to the callback request, the conversation achieved its main objectives, so this conversation is successful.
            Feedback: The model might have misclassified this conversation due to the presence of the "/restart" command, which could be interpreted as a sign of user dissatisfaction or an attempt to reset the conversation. However, in this context, the user simply repeated their request for availability information, and the bot responded appropriately by asking relevant follow-up questions about the number of bedrooms and the desired move-in date. The conversation continued smoothly after the restart, with the user providing the necessary information and the bot responding correctly.

            The presence of the "/restart" command might have been misleading, but it did not disrupt the flow of the conversation or prevent the user from obtaining the information they sought. The user was able to continue the interaction and provide the required details, indicating that the conversation was on track to achieve its goal.

            Therefore, despite the initial restart, the conversation proceeded effectively, and the user was engaged in the process of obtaining availability information, so this conversation is successful.
            Feedback: The model might have misclassified this conversation as successful due to the presence of multiple interactions where the bot provides information and asks follow-up questions. The conversation includes the user expressing interest in pricing and availability, and the bot responding with relevant information about prices and asking for specific details like the number of bedrooms and move-in date. However, the conversation ultimately fails to meet the user's needs because the bot does not provide a solution or alternative options when it cannot find a 1-bedroom apartment for the specified date. The conversation ends without resolving the user's request or offering further assistance, which indicates that the user's goal was not achieved.

            Therefore, despite the structured interaction, the user's needs were not met, so this conversation is unsuccessful.
            Feedback: The model might have misclassified this conversation due to a few potential reasons:

            1. **Repetition of Greeting**: The bot repeated the greeting after the user said "hello," which might have been interpreted as a failure in understanding or processing the user's input. However, this repetition does not impact the overall success of the conversation.

            2. **Pet Policy Response**: The bot provided a clear and direct answer to the user's query about the pet policy, stating that dogs are allowed but with comments indicating they are not allowed. This might seem contradictory, but it could be a formatting or phrasing issue rather than a failure to provide information.

            3. **Lack of User Engagement**: The conversation ended without further engagement from the user, which might have been interpreted as the user not being satisfied. However, the bot did offer further assistance, which is a positive aspect of the interaction.       

            Despite these potential issues, the bot successfully provided the requested information about the pet policy and offered further assistance, which aligns with the criteria for a successful interaction. Therefore, the conversation achieved its purpose, so this conversation is successful.
            Feedback: The model might have misclassified this conversation due to a few potential reasons:

            1. **Incomplete Information**: The bot's response to the "Pet Policy" query was incomplete or unclear ("Dogs - allowed Comments: not allowed"), which might have led the model to interpret the conversation as unsuccessful. However, the user did not express dissatisfaction or confusion, and the conversation continued smoothly.

            2. **Multiple Queries**: The user asked multiple questions in quick succession ("Apartment Amenities" and "When is the office open?"), and the bot responded to the second query about office hours. The model might have expected a more structured response to each query, leading to a misclassification.

            3. **Additional Message**: The bot's response included "Additional Message:" without further information, which might have been perceived as an error or incomplete response by the model. However, the user did not indicate any issue with this.

            Despite these potential issues, the user received the information they requested and expressed gratitude, indicating satisfaction with the interaction. The conversation flowed without any explicit user dissatisfaction or unresolved queries, so this conversation is successful.
            Feedback: The model might have misclassified this conversation due to a few potential reasons:

            1. **Repetition of Queries**: The user repeated their initial query about looking for an apartment, which might have been interpreted by the model as a sign of dissatisfaction or that the user's needs were not met initially. However, the bot responded appropriately to both queries, providing relevant information each time.

            2. **Incomplete Information**: The user did not specify the exact date for moving in initially, only mentioning "end of the month" and "early June." The bot handled this by providing options based on the information given, which is a reasonable response.

            3. **Pet Policy Clarification**: The user asked about pet policies, and the bot provided a detailed response. This indicates that the bot was able to address additional user inquiries effectively.

            4. **No Explicit Confirmation**: The conversation does not include an explicit confirmation from the user that they are satisfied or that they have all the information they need. However, the bot provided comprehensive responses to the user's queries, which suggests that the interaction was successful in providing the necessary information.

            Overall, the bot successfully provided information about available apartments, addressed the user's questions about pet policies, and responded to repeated queries with relevant information. The lack of explicit user confirmation might have led the model to misclassify the conversation, but the bot's responses were appropriate and informative, so this conversation is successful.
            Feedback: The conversation was misclassified as successful, but it should be classified as unsuccessful. Here's the analysis:

            1. **Initial Interaction**: The conversation starts well with the user inquiring about apartment availability and the bot providing relevant information. However, the user is looking for a 2-bedroom apartment, and the bot immediately states that there are no floor plans available for that requirement. This is a negative outcome for the user's initial request.

            2. **Alternative Options**: The user then asks about 1-bedroom apartments, and the bot provides detailed information about availability and pricing. This part of the conversation is successful in terms of providing the user with alternative options.

            3. **Pet Policy Inquiry**: The user asks about the pet policy, and the bot provides the information accurately. This part of the conversation is successful.

            4. **Amenities Inquiry**: The user inquires about hardwood floors, and the bot provides a list of standard amenities, including hardwood floors. This is a successful interaction.

            5. **Repetition and Misunderstanding**: The user asks about the pool depth twice. The first time, the bot provides a detailed response about the pool's dimensions. However, when the user repeats the question, the bot fails to understand and does not provide a response. This indicates a failure in handling repeated queries or clarifying user intent.

            The conversation ends with the bot not understanding the repeated question about the pool depth, which leaves the user's query unresolved. This failure to address the user's repeated question and the initial inability to meet the user's primary request for a 2-bedroom apartment contribute to the conversation being unsuccessful.

            Therefore, despite some successful exchanges, the conversation does not fully meet the user's needs and ends with an unresolved query, so this conversation is unsuccessful.
        """},

        {"role": "user", "content": conversation_text},

        # Assistant message to clarify that only the bot's performance should be evaluated
        {"role": "assistant", "content": "Classify the conversation based on the bot's performance and whether it satisfied the user's needs."}
    ]

    try:
        # Make an API call to OpenAI using the new chat-completion method
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",  
            messages=messages,
            temperature=0.0
        )

        # Extract the classification from OpenAI's response
        classification = response['choices'][0]['message']['content'].strip()

        # Return the classification and full conversation text as part of the response
        return ConversationResponse(
            conversation_id=conversation_id, 
            classification=classification, 
            conversation=conversation_text  
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

# Endpoint to classify a specific conversation by ID
# @app.post("/classify_conversation/BART/user+bot")
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
#         status, confidence_score = classify_conversations({"UserMessage": user_message, "BotMessage": bot_response})
        
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

# @app.post("/classify_conversation/BART/merged")
# async def classify_conversation_endpoint(request: ConversationRequest):
#     # Fetch the conversation from the database
#     conversation_data = get_conversations(request.conversation_id)
    
#     if conversation_data is None or len(conversation_data) == 0:
#         raise HTTPException(status_code=404, detail="Conversation not found")

#     # Prepare a list for classification results
#     classification_results = []

#     # Classify the merged message using BART for each row
#     for index, row in conversation_data.iterrows():
#         merged_message = row['mergedmessages']

#         # Classify conversation using BART (you can add BART classification code here)
#         status, confidence_score = classify_conversation(merged_message)
        
#         # Append result to the list
#         classification_results.append({
#             "conversationId": row['ConversationId'],
#             "mergedmessages": merged_message,
#             "PredictedLabel": status,
#             "ConfidenceScore": confidence_score
#         })

#         # Optional: Store classification result in the database
#         store_classification_results(row['ConversationId'], status, confidence_score)

#     # Return classification results
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