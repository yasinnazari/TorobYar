import uuid
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from app.agent.agent_runner import run_agent
from app.services.conversation_memory import load_conversation, save_conversation

router = APIRouter(
   prefix="/api/v1/assistant",
   tags=["Assistant"],
)

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
   message: str = Field(min_length=1, max_length=2000)
   conversation_id: str | None = None

   latitude: float | None = Field(default=None, ge=-90, le=90)
   longitude: float | None = Field(default=None, ge=-180, le=180)
   radius_km: float = Field(default=5.0, gt=0, le=50)


@router.post("/chat")
def chat(request: Request, body: ChatRequest):
   message = body.message.strip()

   if not message:
      raise HTTPException(status_code=422, detail="پیام نمی‌تواند خالی باشد")

   conversation_id = body.conversation_id or str(uuid.uuid4())

   try:
      conversation = load_conversation(conversation_id)

      result = run_agent(
         dataset=request.app.state.dataset,
         user_message=message,
         history=conversation["messages"],
         shopping_state=conversation["shopping_state"],
         latitude=body.latitude,
         longitude=body.longitude,
         radius_km=body.radius_km,
      )

      updated_messages = conversation["messages"] + [
         {"role": "user", "content": message},
         {"role": "assistant", "content": result["reply"]},
      ]

      save_conversation(conversation_id=conversation_id, messages=updated_messages, shopping_state=result["shopping_state"])

      return {
         "conversation_id": conversation_id,
         "reply": result["reply"],
         "tools_used": result["tools_used"],
         "shopping_state": result["shopping_state"],
         "products": result.get("products", []),
         "offers": result.get("offers", []),
         "shops": result.get("shops", []),
      }

   except RuntimeError:
      logger.exception("Assistant Request Failed")
      raise HTTPException(status_code=502, detail="اجرای ترب یار با مشکل مواجه شد")
