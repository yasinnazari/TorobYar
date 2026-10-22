import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("data/conversations.sqlite3")

DEFAULT_SHOPPING_STATE = {
   "product_type": None,
   "budget_max_toman": None,
   "color": None,
   "usage": None,
   "brand": None,
   "radius_km": None,
   "payment_preference": None,
   "location_label": None,
}

def init_memory_db() -> None:
   DB_PATH.parent.mkdir(parents=True, exist_ok=True)

   with sqlite3.connect(DB_PATH) as conn:
      conn.execute("""
         CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            messages_json TEXT NOT NULL,
            shopping_state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
         )
      """)


def load_conversation(conversation_id: str) -> dict[str, Any]:
   with sqlite3.connect(DB_PATH) as conn:
      row = conn.execute(
         """
            SELECT messages_json, shopping_state_json
            FROM conversations
            WHERE conversation_id = ?
         """,
         (conversation_id,),
      ).fetchone()

   if row is None:
      return {
         "messages": [],
         "shopping_state": DEFAULT_SHOPPING_STATE.copy(),
      }

   return {
      "messages": json.loads(row[0]),
      "shopping_state": {
         **DEFAULT_SHOPPING_STATE,
         **json.loads(row[1]),
      },
   }


def save_conversation(conversation_id: str, messages: list, shopping_state: dict) -> None:
   messages = messages[-20:]

   with sqlite3.connect(DB_PATH) as conn:
      conn.execute(
         """
            INSERT INTO conversations (
               conversation_id,
               messages_json,
               shopping_state_json,
               updated_at
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(conversation_id) DO UPDATE SET
               messages_json = excluded.messages_json,
               shopping_state_json = excluded.shopping_state_json,
               updated_at = CURRENT_TIMESTAMP
         """,
         (
            conversation_id,
            json.dumps(messages, ensure_ascii=False),
            json.dumps(shopping_state, ensure_ascii=False),
         ),
      )
