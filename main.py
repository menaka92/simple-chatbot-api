from fastapi import FastAPI
from typing import Annotated, TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv  # Add this import
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

class MessageRequest(BaseModel):
    request_body: str

api_key = os.getenv("OPENAI_API_KEY")

# Add error handling for missing API key
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client_obj = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3.1:free",
    api_key=api_key,
    temperature=0.1,
    max_tokens=50
)


def chatbot(state: AgentState) -> AgentState:
    ai_message = client_obj.invoke(
        [{"role": "system", "content": "Answer very briefly, in 1–2 sentences."}] + state["messages"]
    )
    return {"messages": [ai_message]}


flow = StateGraph(AgentState)
flow.add_node("chatbot", chatbot)
flow.add_edge(START, "chatbot")
flow.add_edge("chatbot", END)
compiled_flow = flow.compile()


@app.post("/chat")
async def chat(request_body: MessageRequest):

    result = compiled_flow.invoke({
        "messages": [
            {"role": "user", "content": request_body.request_body}
        ]
    })
    
    ai_message_content = result["messages"][-1].content
    
    return {"message": ai_message_content}


@app.get("/")
def read_root():
    return {"message": "Chatbot API is running. Use the /chat endpoint to interact."}