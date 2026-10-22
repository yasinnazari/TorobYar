import json
from typing import Any
import pandas as pd


def search_products(dataset: dict[str, pd.DataFrame], query: str, limit: int = 5) -> dict[str, Any]:
   """Search products by Persian title."""

   query = query.strip()

   if not query:
      return {"query": query, "count": 0, "results": []}

   products = dataset["products"]

   mask = products["persian_title"].str.contains(
      query,
      case=False,
      regex=False,
      na=False,
   )

   matches = products.loc[
      mask,
      [
         "product_id",
         "persian_title",
         "brand_name",
         "category_name",
         "model",
         "image_urls",
      ],
   ].head(max(1, min(limit, 20)))

   matches = matches.astype(object).where(
      matches.notna(), None
   )

   results = matches.to_dict(orient="records")

   return {
      "query": query,
      "count": len(results),
      "results": results,
   }


def get_product_offers(
   dataset: dict[str, pd.DataFrame],
   product_id: str,
   limit: int = 5,
) -> dict[str, Any]:
   """Get price offers and shop details for a product."""

   products = dataset["products"]
   offers = dataset["offers"]
   shops = dataset["shops"]

   product_rows = products[
      products["product_id"] == product_id
   ]

   if product_rows.empty:
      return {
         "product_id": product_id,
         "error": "product_not_found",
         "count": 0,
         "results": [],
      }

   product_offers = offers[
      offers["product_id"] == product_id
   ].copy()

   shop_columns = [
      "shop_id",
      "name",
      "address",
      "location_label",
      "lat",
      "lon",
      "payment_options",
   ]

   product_offers = product_offers.merge(
      shops[shop_columns],
      on="shop_id",
      how="left",
   )

   product_offers = product_offers.sort_values(
      "price_toman",
      ascending=True,
      na_position="last",
   )

   result_columns = [
      "offer_id",
      "shop_id",
      "seller_title",
      "price_toman",
      "warranty",
      "last_price_changed_at",
      "name",
      "address",
      "location_label",
      "lat",
      "lon",
      "payment_options",
   ]

   results = product_offers[result_columns].head(
      max(1, min(limit, 20))
   )

   results = results.astype(object).where(
      results.notna(), None
   )

   return {
      "product_id": product_id,
      "product_title": product_rows.iloc[0]["persian_title"],
      "count": len(results),
      "results": results.to_dict(orient="records"),
      "price_is_live": False,
   }



def find_payment_shops(
   dataset: dict[str, pd.DataFrame],
   product_id: str,
   limit: int = 10,
   channel: str = "in_person",
) -> dict[str, Any]:
   """
   Find product offers from shops with declared payment options.

   channel:
      in_person -> explicitly declared in-person payment
      unknown   -> payment channel is unknown
      any       -> either of the above
   """
   if channel not in {"in_person", "unknown", "any"}:
      return {
         "error": "invalid_channel",
         "allowed_channels": ["in_person", "unknown", "any"],
         "results": [],
         "count": 0,
      }

   products = dataset["products"]
   offers = dataset["offers"]
   shops = dataset["shops"]

   product_rows = products[
      products["product_id"] == product_id
   ]

   if product_rows.empty:
      return {
         "product_id": product_id,
         "error": "product_not_found",
         "count": 0,
         "results": [],
      }

   # Get all offers for this product, not just the first 20.
   product_offers = offers[
      offers["product_id"] == product_id
   ].copy()

   shop_columns = [
      "shop_id",
      "name",
      "address",
      "location_label",
      "lat",
      "lon",
      "payment_options",
   ]

   product_offers = product_offers.merge(
      shops[shop_columns],
      on="shop_id",
      how="left",
   )

   results = []

   for _, offer in product_offers.iterrows():
      raw_options = offer.get("payment_options")

      if not isinstance(raw_options, str):
         continue

      try:
         options = json.loads(raw_options)
      except (json.JSONDecodeError, TypeError):
         continue

      if not isinstance(options, list) or not options:
         continue

      matching_options = []

      for option in options:
         if not isinstance(option, dict):
               continue

         option_channel = option.get("channel")

         if channel == "any":
               matches = option_channel in {
                  "in_person",
                  "unknown",
               }
         else:
               matches = option_channel == channel

         if matches:
               matching_options.append(option)

      if not matching_options:
         continue

      item = offer.to_dict()
      item["payment_options"] = matching_options
      item["payment_channel_status"] = (
         "in_person_declared"
         if any(
               option.get("channel") == "in_person"
               for option in matching_options
         )
         else "unknown"
      )

      # Avoid NaN values in the JSON response.
      item = {
         key: None if pd.isna(value) else value
         for key, value in item.items()
         if key != "payment_options"
      } | {
         "payment_options": matching_options,
         "payment_channel_status": item[
               "payment_channel_status"
         ],
      }

      results.append(item)

   results.sort(
      key=lambda item: (
         item["price_toman"] is None,
         item["price_toman"]
         if item["price_toman"] is not None
         else float("inf"),
      )
   )

   results = results[:max(1, min(limit, 50))]

   return {
      "product_id": product_id,
      "product_title": product_rows.iloc[0]["persian_title"],
      "channel_filter": channel,
      "count": len(results),
      "results": results,
      "price_is_live": False,
      "note": (
         "گزینه پرداخت از داده‌های ثبت‌شده فروشگاه استخراج شده است؛ "
         "فعال بودن اعتبار یا امکان خرید اقساطی باید تأیید شود."
      ),
   }