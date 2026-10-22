
import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent.agent_runner import run_agent

router = APIRouter(
   prefix="/api/v1/assistant",
   tags=["Assistant"],
)


class ChatRequest(BaseModel):
   message: str = Field(min_length=1, max_length=2000)
   history: list = Field(default_factory=list, max_length=8)


@router.post("/chat")
def chat(request: Request, body: ChatRequest):
   try:
      result = run_agent(dataset=request.app.state.dataset, user_message=body.message.strip(), history=body.history)

      return {
         "reply": result["reply"],
         "tools_used": result["tools_used"],
      }

   except requests.RequestException:
      raise HTTPException(status_code=502, detail="خطایی در برقراری ارتباط با مدل پیش آمد")

   except RuntimeError:
      raise HTTPException(status_code=502, detail="مشکلی در اجرای ترب یار به وجود آمد",)

