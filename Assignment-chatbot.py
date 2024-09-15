import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import spacy

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Load spaCy model for NLP
nlp = spacy.load('en_core_web_sm')

# FastAPI app initialization
app = FastAPI()

# Predefined price range for negotiation
MIN_PRICE = 80  # Minimum acceptable price
START_PRICE = 120  # Starting price
counter_offer_price = START_PRICE  # Initialize counter offer price

# Initialize memory for conversation history
memory = ConversationBufferMemory(return_messages=True, input_key="user_message")

# Define the prompt template to handle negotiation logic dynamically
prompt_template = PromptTemplate(
    input_variables=["history", "user_message", "logic_response"],  # Handle user_message and logic_response
    template="""
    You are a negotiation assistant. You will evaluate the user's message and respond politely and appropriately.

    Current conversation history:
    {history}

    User said: {user_message}

    Based on the negotiation logic, the appropriate response is: {logic_response}

    Now, respond to the user in a helpful, polite, and engaging way.
    """
)

# Initialize OpenAI Chat model (uses LangChain's ChatOpenAI)
llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

# Define the conversation chain using the LLM and memory
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory
)

# FastAPI request model
class NegotiationRequest(BaseModel):
    message: str

# Enhanced intent recognition from original code
def extract_intent(user_message):
    doc = nlp(user_message.lower())
    offer, quantity = None, 1

    for ent in doc.ents:
        if ent.label_ == "MONEY":
            offer = float(ent.text.replace("$", "").replace(",", ""))
        if ent.label_ == "CARDINAL":
            if ent.text.isdigit():
                quantity = int(ent.text)
            else:
                quantity = {
                    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
                }.get(ent.text.lower(), 1)

    if any(token.lemma_ in ["lower", "reduce", "discount", "better", "cheaper"] for token in doc):
        return "request_discount", quantity

    if offer:
        return "offer", offer, quantity

    return "unknown", 0, 1

# Counter offer logic
def generate_counter_offer(current_offer):
    return max(current_offer - 20, MIN_PRICE)

# Negotiation logic based on user message
def apply_negotiation_logic(user_message):
    global counter_offer_price

    # Extract intent from the message
    intent = extract_intent(user_message)

    # Handle the negotiation logic based on the intent
    if intent[0] == "offer":
        user_offer = intent[1]
        quantity = intent[2]

        # Accept the deal if the offer is at or above the minimum price
        if user_offer >= MIN_PRICE:
            return f"Your offer of ${user_offer} is acceptable. Deal accepted."
        else:
            return f"Your offer of ${user_offer} is too low. The minimum price I can accept is ${MIN_PRICE}."

    elif intent[0] == "request_discount":
        # Handle discount request
        quantity = intent[1]

        if counter_offer_price > MIN_PRICE:
            # Reduce the counter offer by $20 until reaching the minimum price
            counter_offer_price = generate_counter_offer(counter_offer_price)
            if counter_offer_price == MIN_PRICE:
                return f"The best price I can offer is ${counter_offer_price}. Deal accepted."
            else:
                return f"I can offer you a better deal at ${counter_offer_price}."
        else:
            # Accept the deal if it has already reached the minimum price
            return f"The best price I can offer is ${MIN_PRICE}. Deal accepted."

    else:
        # Default response if no valid intent detected
        return "Please provide a specific price offer or ask for a discount."

# FastAPI root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot Negotiation API with ChatGPT-like responses. Use /negotiate to interact."}

# Negotiation route
@app.post("/negotiate")
async def negotiate(req: NegotiationRequest):
    global counter_offer_price
    user_message = req.message

    # Apply the manual negotiation logic to determine the correct response content
    logic_response = apply_negotiation_logic(user_message)

    # Use LangChain to process the conversation, applying negotiation logic and history
    response_with_history = conversation_chain.predict(
        user_message=user_message, logic_response=logic_response
    )

    # Return the final response to the user
    return {"response": response_with_history}

# Run the API with Uvicorn (optional if running via CLI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
