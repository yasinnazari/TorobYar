import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from app.services.product_service import search_products_in_dataset

router = APIRouter(
   prefix="/api/v1/products",
   tags=["Products"],
)


@router.get("/{product_id}/similar")
def get_similar_products(product_id, request, limit: int = Query(default=10, ge=1, le=30)):
   products = request.app.state.dataset["products"]

   product_rows = products[
      products["product_id"].astype(str) == product_id
   ]

   if product_rows.empty:
      raise HTTPException(status_code=404, detail="محصول موردنظر پیدا نشد")

   product = product_rows.iloc[0]
   category = str(product.get("category_name") or "")
   brand = str(product.get("brand_name") or "")
   model = str(product.get("model") or "")

   candidates = products[products["category_name"].fillna("").astype(str) == category].copy()

   candidates = candidates[candidates["product_id"].astype(str) != product_id]

   if candidates.empty:
      return {
         "product_id": product_id,
         "count": 0,
         "results": [],
      }

   output_columns = [
      col for col in [
         "product_id",
         "persian_title",
         "brand_name",
         "category_name",
         "model",
         "image_urls",
      ]
      if col in candidates.columns
   ]

   results = candidates[output_columns].head(limit)
   results = results.astype(object).where(results.notna(), None)

   return {
      "product_id": product_id,
      "product_title": str(product.get("persian_title") or ""),
      "count": len(results),
      "results": results.to_dict(orient="records"),
   }


@router.get("/search")
def search_products(request, q=Query(min_length=1, max_length=100), limit=Query(default=10, ge=1, le=50)):
   products = request.app.state.dataset["products"]

   return search_products_in_dataset(products=products, query=q, limit=limit)



@router.get("/{product_id}/offers")
def get_product_offers(request, product_id, limit = Query(default=20, ge=1, le=100)):
   dataset = request.app.state.dataset

   products = dataset["products"]
   offers = dataset["offers"]
   shops = dataset["shops"]

   product_rows = products[
      products["product_id"] == product_id
   ]

   if product_rows.empty:
      return {
         "product_id": product_id,
         "count": 0,
         "results": [],
      }

   product_offers = offers[offers["product_id"] == product_id].copy()

   if product_offers.empty:
      return {
         "product_id": product_id,
         "product_title": product_rows.iloc[0]["persian_title"],
         "count": 0,
         "results": [],
      }

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

   results = product_offers[result_columns].head(limit)

   results = results.astype(object).where(
      results.notna(), None
   )

   records = results.to_dict(orient="records")

   return {
      "product_id": product_id,
      "product_title": product_rows.iloc[0]["persian_title"],
      "count": len(records),
      "results": records,
      "price_source": "dataset",
      "price_is_live": False,
   }


@router.get("/offers/search")
def search_products_with_offers(request, q=Query(min_length=1, max_length=100), limit = Query(default=10, ge=1, le=20)):
   dataset = request.app.state.dataset
   products = dataset["products"]
   offers = dataset["offers"]
   shops = dataset["shops"]

   mask = products["persian_title"].str.contains(
      q.strip(),
      case=False,
      regex=False,
      na=False,
   )

   matches = products.loc[mask].head(limit)

   if matches.empty:
      return {
         "query": q,
         "count": 0,
         "results": [],
      }

   product_ids = matches["product_id"].tolist()

   matched_offers = offers[offers["product_id"].isin(product_ids)].copy()

   shop_columns = [
      "shop_id",
      "name",
      "address",
      "location_label",
      "lat",
      "lon",
      "payment_options",
   ]

   matched_offers = matched_offers.merge(
      shops[shop_columns],
      on="shop_id",
      how="left",
   )

   matched_offers = matched_offers.sort_values(
      "price_toman",
      ascending=True,
      na_position="last",
   )

   offers_by_product = {}

   for product_id, group in matched_offers.groupby("product_id"):
      group = group.astype(object).where(group.notna(), None)

      offers_by_product[product_id] = (
         group[
            [
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
         ]
         .head(5)
         .to_dict(orient="records")
      )

   results = []

   for _, product in matches.iterrows():
      product_id = product["product_id"]

      results.append({
         "product_id": product_id,
         "persian_title": product["persian_title"],
         "brand_name": (
               None if pd.isna(product["brand_name"])
               else product["brand_name"]
         ),
         "category_name": (
               None if pd.isna(product["category_name"])
               else product["category_name"]
         ),
         "image_urls": product["image_urls"],
         "offers": offers_by_product.get(product_id, []),
      })

   return {
      "query": q,
      "count": len(results),
      "results": results,
      "price_is_live": False,
   }
