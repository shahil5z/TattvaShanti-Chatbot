import os
import uuid
import logging
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, field_validator
import httpx
from fastapi.middleware.cors import CORSMiddleware

from backend.config import N8N_WEBHOOK_URL
from backend.rag_chain import rag_chain_with_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Added CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tattvashanti.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    raise ValueError("Missing required API keys in .env")

sessions = {}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# ping route for Render
@app.get("/ping")
async def ping():
    return {"status": "OK", "message": "Tattva Shanti backend is alive!"}

class QueryRequest(BaseModel):
    question: str
    session_id: str = None

    @field_validator('question')
    def validate_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Question cannot be empty')
        if len(v) > 500:
            raise ValueError('Question too long')
        alpha_ratio = len(re.findall(r'[a-zA-Z]', v)) / len(v)
        if alpha_ratio < 0.3 and len(v) > 5:
            raise ValueError('Nonsense input')
        return v

def get_or_create_session(session_id: str = None):
    now = datetime.utcnow()
    expired_sessions = [sid for sid, data in sessions.items() if now - data["created_at"] > timedelta(hours=1)]
    for sid in expired_sessions:
        del sessions[sid]
    
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "chat_history": [],
            "created_at": datetime.utcnow()
        }
    return session_id

async def log_to_n8n(session_id: str, user_query: str, ai_response: str):
    if not N8N_WEBHOOK_URL:
        return

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "Session_id": session_id,
                "User Query": user_query,
                "AI Response": ai_response,
                "Date and Time": datetime.utcnow().isoformat() + "Z"
            }
            await client.post(N8N_WEBHOOK_URL, json=payload)
            logger.info("Successfully logged chat to n8n")
    except Exception as e:
        logger.warning(f"Failed to send data to n8n: {e}")

@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        question = request.question.strip()
        if not question:
            return {"answer": "I didn't receive a question. Could you please ask something?", "session_id": request.session_id or ""}

        lower_q = question.lower()
        injection_triggers = ["ignore", "forget", "previous", "system prompt", "you are", "act as"]
        if any(trigger in lower_q for trigger in injection_triggers) and len(question.split()) < 8:
            question = "User asked a question that appears to attempt prompt injection."

        session_id = get_or_create_session(request.session_id)
        session = sessions[session_id]

        import asyncio
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: rag_chain_with_history.invoke({
                    "question": question,
                    "chat_history": session["chat_history"]
                })
            ),
            timeout=20.0
        )

        session["chat_history"].append({"role": "user", "content": request.question})
        session["chat_history"].append({"role": "assistant", "content": response})

        if len(session["chat_history"]) > 20:
            session["chat_history"] = session["chat_history"][-20:]

        await log_to_n8n(session_id, request.question, response)

        return {"answer": response, "session_id": session_id}

    except ValueError as ve:
        if "Nonsense input" in str(ve):
            return {"answer": "Sorry, I couldn't understand that. Could you please rephrase your question?", "session_id": request.session_id or ""}
        return {"answer": "Please ask a clear question (max 500 characters).", "session_id": request.session_id or ""}
    except asyncio.TimeoutError:
        logger.warning("RAG timeout")
        return {"answer": "I'm taking a bit longer than expected. Please try again.", "session_id": request.session_id or ""}
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return {"answer": "I'm sorry, I encountered an issue. Could you please rephrase your question?", "session_id": request.session_id or ""}
