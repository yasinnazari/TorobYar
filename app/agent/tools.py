import re
import json
import unicodedata
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from app.services.product_service import search_products_in_dataset


def search_products(dataset, query, limit=5):
   products = dataset["products"]

   return search_products_in_dataset(products=products, query=query, limit=limit)


def get_product_offers(dataset: dict, product_id: str, limit=5) -> dict:
   products = dataset["products"]
   offers = dataset["offers"]
   shops = dataset["shops"]

   product_rows = products[products["product_id"] == product_id]

   if product_rows.empty:
      return {
         "product_id": product_id,
         "error": "product_not_found",
         "count": 0,
         "results": [],
      }

   product_offers = offers[offers["product_id"] == product_id].copy()

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

   results = product_offers[result_columns].head(max(1, min(limit, 20)))

   results = results.astype(object).where(results.notna(), None)

   return {
      "product_id": product_id,
      "product_title": product_rows.iloc[0]["persian_title"],
      "count": len(results),
      "results": results.to_dict(orient="records"),
      "price_is_live": False,
   }


def find_payment_shops(dataset: dict, product_id: str, limit=10, channel="in_person") -> dict:
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

   product_offers = offers[offers["product_id"] == product_id].copy()

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
         "گزینه پرداخت از داده‌های ثبت‌شده فروشگاه استخراج شده است"
         "فعال بودن اعتبار یا امکان خرید اقساطی باید تأیید شود"
      ),
   }



def calculate_distance_km(
   lat1: float,
   lon1: float,
   lat2: float,
   lon2: float,
) -> float:
   earth_radius_km = 6371.0

   dlat = radians(lat2 - lat1)
   dlon = radians(lon2 - lon1)

   a = (
      sin(dlat / 2) ** 2
      + cos(radians(lat1))
      * cos(radians(lat2))
      * sin(dlon / 2) ** 2
   )

   a = min(1.0, max(0.0, a))

   return earth_radius_km * 2 * atan2(sqrt(a), sqrt(1 - a))


# ------------------------------------------

def find_nearby_product_shops(
   dataset: dict, product_id: str, user_lat: float, user_lon: float, radius_km: float = 5.0, limit: int = 10) -> dict:

   products = dataset["products"]
   offers = dataset["offers"]
   shops = dataset["shops"]

   product = products[products["product_id"] == product_id]

   if product.empty:
      return {
         "error": "product_not_found",
         "count": 0,
         "results": [],
      }

   product_offers = offers[offers["product_id"] == product_id].copy()

   shop_columns = [
      "shop_id",
      "name",
      "address",
      "location_label",
      "lat",
      "lon",
      "payment_options",
   ]

   results = product_offers.merge(shops[shop_columns], on="shop_id", how="left")

   results["lat"] = pd.to_numeric(
      results["lat"], errors="coerce"
   )
   results["lon"] = pd.to_numeric(
      results["lon"], errors="coerce"
   )
   results["price_toman"] = pd.to_numeric(
      results["price_toman"], errors="coerce"
   )

   results = results.dropna(subset=["lat", "lon"]).copy()

   if results.empty:
      return {
         "product_id": product_id,
         "product_title": product.iloc[0]["persian_title"],
         "count": 0,
         "results": [],
         "note": "برای این محصول مختصات فروشگاه معتبری پیدا نشد.",
      }

   results["distance_km"] = results.apply(
      lambda row: calculate_distance_km(
         user_lat,
         user_lon,
         row["lat"],
         row["lon"],
      ),
      axis=1,
   )

   results = results[results["distance_km"] <= max(0, radius_km)]

   results = results.sort_values(
      ["distance_km", "price_toman"],
      ascending=[True, True],
      na_position="last",
   ).head(max(1, min(limit, 50)))

   results = results.astype(object).where(results.notna(), None)

   return {
      "product_id": product_id,
      "product_title": product.iloc[0]["persian_title"],
      "radius_km": radius_km,
      "count": len(results),
      "results": results[
         [
            "offer_id",
            "shop_id",
            "seller_title",
            "price_toman",
            "name",
            "address",
            "location_label",
            "lat",
            "lon",
            "distance_km",
            "payment_options",
         ]
      ].to_dict(orient="records"),
      "price_is_live": False,
      "distance_type": "straight_line",
   }


def normalize_fa(text: str) -> str:
   text = unicodedata.normalize("NFKC", str(text or ""))
   text = text.replace("ي", "ی").replace("ك", "ک")
   text = text.replace("\u200c", " ")
   text = re.sub(r"\s+", " ", text)
   return text.strip().lower()


def is_relevant_product(title: str, query: str) -> bool:
   title = normalize_fa(title)
   query = normalize_fa(query)

   generic_words = {
      "میخوام", "میخوام بخرم", "خرید", "قیمت",
      "لطفا", "برای", "مناسب", "اصل",
   }

   tokens = [token for token in re.findall(r"[\w]+", query) if token not in generic_words]

   if not tokens:
      return True

   return all(token in title for token in tokens)
