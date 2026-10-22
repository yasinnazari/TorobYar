import os
import json
import requests
from typing import Any
from dotenv import load_dotenv
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import search_products, get_product_offers, find_payment_shops

load_dotenv('.env')

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TOOLS = [
   {
      "type": "function",
      "function": {
         "name": "search_products",
         "description": "Search the product dataset by product name or keywords",
         "parameters": {
               "type": "object",
               "properties": {
                  "query": {
                     "type": "string",
                     "description": "Product name or search keywords in Persian",
                  },
                  "limit": {
                     "type": "integer",
                     "minimum": 1,
                     "maximum": 10,
                  },
               },
               "required": ["query"],
               "additionalProperties": False,
         },
      },
   },

   {
      "type": "function",
      "function": {
         "name": "get_product_offers",
         "description": "Get recorded prices and store details for a product_id.",
         "parameters": {
               "type": "object",
               "properties": {
                  "product_id": {"type": "string"},
                  "limit": {
                     "type": "integer",
                     "minimum": 1,
                     "maximum": 10,
                  },
               },
               "required": ["product_id"],
               "additionalProperties": False,
         },
      },
   },

   {
      "type": "function",
      "function": {
         "name": "find_payment_shops",
         "description": (
               "Find stores offering a product with recorded payment options. "
               "This does not guarantee active installment eligibility."
         ),
         "parameters": {
               "type": "object",
               "properties": {
                  "product_id": {"type": "string"},
                  "limit": {
                     "type": "integer",
                     "minimum": 1,
                     "maximum": 10,
                  },
                  "channel": {
                     "type": "string",
                     "enum": ["in_person", "unknown", "any"],
                  },
               },
               "required": ["product_id"],
               "additionalProperties": False,
         },
      },
   }
]


def execute_tool(dataset: dict, name: str, arguments: dict[str, Any],) -> dict:
   if name == "search_products":
      return search_products(dataset, query=arguments["query"], limit=arguments.get("limit", 5))

   if name == "get_product_offers":
      return get_product_offers(dataset, product_id=arguments["product_id"], limit=arguments.get("limit", 5))

   if name == "find_payment_shops":
      return find_payment_shops(dataset, product_id=arguments["product_id"], limit=arguments.get("limit", 10), channel=arguments.get("channel", "in_person"))

   raise ValueError(f"Unknown Tool: {name}")


def run_agent(
   dataset: dict, user_message: str, history: list | None = None) -> dict:
   if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
      raise RuntimeError("API_KEY & MODEL are not configured")

   messages = [
      {
         "role": "system", "content": SYSTEM_PROMPT
      }
   ]

   for item in (history or [])[-8:]:
      if item.get("role") in {"user", "assistant"}:
         content = item.get("content")
         if isinstance(content, str):
            messages.append({
               "role": item["role"],
               "content": content[:4000],
            })

   messages.append({"role": "user", "content": user_message})

   tools_used = []

   # -------- Kill tool calling (after 5 attempt) -------
   for _ in range(5):
      response = requests.post(
         OPENROUTER_URL,
         headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
         },
         json={
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.2,
         },
         timeout=60,
      )

      if not response.ok:
         raise RuntimeError(
            f"OpenRouter error {response.status_code}: "
            f"{response.text[:1000]}"
         )

      payload = response.json()
      choices = payload.get("choices", [])

      if not choices:
         raise RuntimeError("OpenRouter returned no choices.")

      message = choices[0]["message"]
      tool_calls = message.get("tool_calls") or []

      if not tool_calls:
         return {
            "reply": message.get("content") or "",
            "tools_used": tools_used,
         }

      messages.append(message)

      for call in tool_calls:
         function = call.get("function", {})
         name = function.get("name", "")

         try:
            arguments = json.loads(function.get("arguments") or "{}")

            if not isinstance(arguments, dict):
               raise ValueError("Tool arguments must be a JSON object.")

            result = execute_tool(dataset, name, arguments)

         except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            result = {
               "error": "invalid_tool_call",
               "message": str(exc),
            }

         tools_used.append(name)

         messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(
               result,
               ensure_ascii=False,
               default=str,
            )
         })

   return {
      "reply": (
         "برای پیدا کردن محصولای مورد نظرت گشتم"
         "اما هنوز پاسخ نهایی آماده نیست. درخواست را کمی دقیق‌تر کن تا بتونم بهتر کمکت کنم که چی میخوای ؟"
      ),
      "tools_used": tools_used,
   }
