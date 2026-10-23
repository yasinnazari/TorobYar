import os
import ast
import json
import logging
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from app.agent.prompts import SYSTEM_PROMPT, MEMORY_PROMPT
from app.agent.tools import (
   search_products as search_products_data,
   get_product_offers as get_product_offers_data,
   find_payment_shops as find_payment_shops_data,
   find_nearby_product_shops as find_nearby_shops_data
)


load_dotenv('.env')

logger = logging.getLogger(__name__)


def build_agent(dataset : dict, shopping_state : dict, latitude, longitude, radius_km):
   @tool
   def search_products(query: str, limit: int = 5) -> dict:
      """Search products by Persian product title or keywords"""

      return search_products_data(dataset, query=query, limit=limit)

   @tool
   def get_product_offers(product_id: str, limit: int = 5) -> dict:
      """Get recorded prices and shop details for a product ID returned by product search"""

      return get_product_offers_data(dataset, product_id=product_id, limit=limit)

   @tool
   def find_payment_shops(product_id: str, limit=10, channel="in_person") -> dict:
      """Find sellers with recorded payment options. This does not guarantee installment eligibility."""
      return find_payment_shops_data(dataset, product_id=product_id, limit=limit, channel=channel)


   @tool
   def find_nearby_product_shops(product_id: str, limit: int = 10) -> dict:
      """Finds sellers of a product within the user’s nearby area"""

      if latitude is None or longitude is None:
         return {
            "error": "location_required",
            "message": "برای پیدا کردن فروشگاه نزدیک، موقعیت کاربر لازم است.",
            "results": [],
            "count": 0,
         }

      return find_nearby_shops_data(
         dataset=dataset,
         product_id=product_id,
         user_lat=latitude,
         user_lon=longitude,
         radius_km=radius_km,
         limit=limit,
      )


   @tool
   def update_shopping_preferences(
      product_type: str | None = None,
      budget_max_toman: int | None = None,
      color: str | None = None,
      usage: str | None = None,
      brand: str | None = None,
      radius_km: float | None = None,
      payment_preference: str | None = None,
      location_label: str | None = None,
      clear_fields: list[str] | None = None,
   ) -> dict:

      """Save or update shopping preferences explicitly stated by the user. Use clear_fields to remove previously saved preferences."""
      allowed_fields = {
         "product_type",
         "budget_max_toman",
         "color",
         "usage",
         "brand",
         "radius_km",
         "payment_preference",
         "location_label",
      }

      updates = {
         "product_type": product_type,
         "budget_max_toman": budget_max_toman,
         "color": color,
         "usage": usage,
         "brand": brand,
         "radius_km": radius_km,
         "payment_preference": payment_preference,
         "location_label": location_label,
      }

      for key in clear_fields or []:
         if key in allowed_fields:
            shopping_state[key] = None

      for key, value in updates.items():
         if value is not None:
            shopping_state[key] = value

      if shopping_state["budget_max_toman"] is not None:
         if shopping_state["budget_max_toman"] <= 0:
            return {"error": "بودجه باید بیشتر از صفر باشد."}

      if shopping_state["radius_km"] is not None:
         if not 0 < shopping_state["radius_km"] <= 50:
            return {"error": "شعاع جست‌وجو باید بین ۰ تا ۵۰ کیلومتر باشد."}

      return {"saved_preferences": shopping_state.copy()}

   state_text = json.dumps(shopping_state, ensure_ascii=False)

   return create_agent(
      model=ChatOpenRouter(
         model=os.environ["OPENROUTER_MODEL"],
         api_key=os.environ["OPENROUTER_API_KEY"],
         temperature=0.2,
      ),
      tools=[
         find_nearby_product_shops,
         search_products,
         get_product_offers,
         find_payment_shops,
         update_shopping_preferences,
      ],
      system_prompt = (
         SYSTEM_PROMPT
         + "\n\n"
         + MEMORY_PROMPT
         + "\n\nوضعیت فعلی ترجیحات خرید کاربر:\n"
         + state_text
      )
   )


def parse_tool_result(message):
   content = getattr(message, "content", None)

   if isinstance(content, dict):
      return content

   if isinstance(content, list):
      content = "\n".join(
         part["text"]
         for part in content
         if isinstance(part, dict)
         and isinstance(part.get("text"), str)
      )

   if isinstance(content, str):
      try:
         result = json.loads(content)
         if isinstance(result, dict):
            return result
      except (json.JSONDecodeError, TypeError):
         pass

      try:
         result = ast.literal_eval(content)
         if isinstance(result, dict):
            return result
      except (ValueError, SyntaxError):
         pass

   return {}


def get_tool_results(payload):
   if not isinstance(payload, dict):
      return []

   for key in ("results", "products", "offers", "shops"):
      value = payload.get(key)

      if isinstance(value, list):
         return value

   return []


def run_agent(dataset, user_message, history=None, shopping_state=None, latitude=None, longitude=None, radius_km=5.0):
   api_key = os.getenv("OPENROUTER_API_KEY")
   model_name = os.getenv("OPENROUTER_MODEL")

   if not api_key or not model_name:
      raise RuntimeError("OPENROUTER_API_KEY & OPENROUTER_MODEL must be configured")

   state = dict(shopping_state or {})

   agent = build_agent(dataset, state, latitude, longitude, radius_km)

   messages = []

   for item in (history or [])[-20:]:
      if (isinstance(item, dict) and item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)):
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
      logger.exception("LangChain agent execution failed")
      raise RuntimeError("LangChain agent execution failed.") from exc

   final_messages = result.get("messages", [])

   reply = ""
   tools_used = []

   products = []
   offers = []
   shops = []

   for message in final_messages:
      if getattr(message, "type", None) == "ai":
         for call in getattr(message, "tool_calls", []):
            if call.get("name"):
               tools_used.append(call["name"])

      if getattr(message, "type", None) != "tool":
         continue

      tool_name = getattr(message, "name", "")
      payload = parse_tool_result(message)
      rows = get_tool_results(payload)

      if tool_name == "search_products":
         products = rows

      elif tool_name == "get_product_offers":
         offers = rows

      elif tool_name in {"find_payment_shops", "find_nearby_product_shops"}:
         shops = rows


   for message in reversed(final_messages):
      if getattr(message, "type", None) == "ai":
         content = message.content

         if isinstance(content, str):
            reply = content

         elif isinstance(content, list):
            reply = "\n".join(
               part["text"]
               for part in content
               if isinstance(part, dict)
               and isinstance(part.get("text"), str)
            )

         if reply:
            break

   return {
      "reply": reply or "نتونستم پاسخ مناسبی تولید کنم",
      "tools_used": tools_used,
      "shopping_state": state,
   }
