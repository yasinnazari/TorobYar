import json
import os
from typing import Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import (
   search_products as search_products_data,
   get_product_offers as get_product_offers_data,
   find_payment_shops as find_payment_shops_data,
)

load_dotenv()

MAX_TOOL_CALLS = 5


def build_agent(dataset: dict):
   @tool
   def search_products(query: str, limit=5) -> dict:
      return search_products_data(dataset, query=query, limit=limit)

   @tool
   def get_product_offers(product_id: str, limit: int = 5) -> dict:
      return get_product_offers_data(dataset, product_id=product_id, limit=limit)

   @tool
   def find_payment_shops(product_id: str, limit=10, channel="in_person") -> dict:
      return find_payment_shops_data(dataset, product_id=product_id, limit=limit, channel=channel)

   api_key = os.getenv("OPENROUTER_API_KEY")
   model_name = os.getenv("OPENROUTER_MODEL")

   if not api_key or not model_name:
      raise RuntimeError("OPENROUTER_API_KEY & OPENROUTER_MODEL must be configured")

   model = ChatOpenRouter(
      model=model_name,
      api_key=api_key,
      temperature=0.2,
   )

   return create_agent(
      model=model,
      tools=[
         search_products,
         get_product_offers,
         find_payment_shops,
      ],
      system_prompt=SYSTEM_PROMPT,
   )


def run_agent(
   dataset: dict[str, Any],
   user_message: str, history: list | None = None) -> dict:
   agent = build_agent(dataset)

   messages = []

   for item in (history or [])[-8:]:
      if (
         isinstance(item, dict)
         and item.get("role") in {"user", "assistant"}
         and isinstance(item.get("content"), str)
      ):
         messages.append({
            "role": item["role"],
            "content": item["content"][:4000],
         })

   messages.append({
      "role": "user",
      "content": user_message.strip(),
   })

   try:
      result = agent.invoke(
         {"messages": messages},
         config={"recursion_limit": 15},
      )

   except Exception as exc:
      raise RuntimeError("LangChain agent execution failed.") from exc

   final_messages = result.get("messages", [])
   reply = ""

   for message in reversed(final_messages):
      if getattr(message, "type", None) == "ai":
         content = message.content
         if isinstance(content, str):
            reply = content
         elif isinstance(content, list):
            reply = "\n".join(
               part["text"]
               for part in content
               if isinstance(part, dict) and isinstance(part.get("text"), str)
            )

         break


   tools_used = []

   for message in final_messages:
      if getattr(message, "type", None) == "ai":
         for call in getattr(message, "tool_calls", []):
            name = call.get("name")
            if name:
               tools_used.append(name)

   return {
      "reply": reply or "نتونستم پاسخ مناسبی تولید کنم. لطفاً دوباره تلاش کن",
      "tools_used": tools_used,
   }
