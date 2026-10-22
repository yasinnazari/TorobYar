from fastapi import APIRouter, Query, Request
import pandas as pd

router = APIRouter(
   prefix="/api/v1/products",
   tags=["Products"],
)

@router.get("/search")
def search_products(request: Request, q: str = Query(min_length=1, max_length=100), limit: int = Query(default=10, ge=1, le=50)):
   products = request.app.state.dataset["products"]

   query = q.strip()

   if not query:
      return {
         "query": q,
         "count": 0,
         "results": [],
      }

   mask = products["persian_title"].str.contains(
      query,
      case=False,
      regex=False,
      na=False,
   )

   results = products.loc[
      mask,
      [
         "product_id",
         "persian_title",
         "brand_name",
         "category_name",
         "model",
         "image_urls",
      ],
   ].head(limit)

   results = results.astype(object).where(
      results.notna(),
      None,
   )

   records = results.to_dict(orient="records")

   return {
      "query": q,
      "count": len(records),
      "results": records,
   }

@router.get("/{product_id}/offers")
def get_product_offers(
   request: Request,
   product_id: str,
   limit: int = Query(default=20, ge=1, le=100),
):
   dataset = request.app.state.dataset

   products = dataset["products"]
   offers = dataset["offers"]
   shops = dataset["shops"]

   # بررسی وجود محصول
   product_rows = products[
      products["product_id"] == product_id
   ]

   if product_rows.empty:
      return {
         "product_id": product_id,
         "count": 0,
         "results": [],
      }

   # پیدا کردن پیشنهادهای فروش این محصول
   product_offers = offers[
      offers["product_id"] == product_id
   ].copy()

   if product_offers.empty:
      return {
         "product_id": product_id,
         "product_title": product_rows.iloc[0]["persian_title"],
         "count": 0,
         "results": [],
      }

   # اتصال پیشنهادهای فروش به اطلاعات فروشگاه
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

   # مرتب‌سازی از ارزان‌ترین پیشنهاد
   product_offers = product_offers.sort_values(
      "price_toman",
      ascending=True,
      na_position="last",
   )

   # انتخاب فیلدهای موردنیاز API
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

   # تبدیل مقادیر خالی به None برای JSON معتبر
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


@router.get("/search-with-offers")
def search_products_with_offers(
   request: Request,
   q: str = Query(min_length=1, max_length=100),
   limit: int = Query(default=10, ge=1, le=20),
):
   dataset = request.app.state.dataset
   products = dataset["products"]
   offers = dataset["offers"]
   shops = dataset["shops"]

   # جست‌وجوی عنوان محصول
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

   # گرفتن پیشنهادهای فروش همین محصولات
   matched_offers = offers[
      offers["product_id"].isin(product_ids)
   ].copy()

   # اتصال اطلاعات فروشگاه
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

   # تبدیل پیشنهادها به ساختار قابل‌نمایش برای هر محصول
   offers_by_product = {}

   for product_id, group in matched_offers.groupby("product_id"):
      group = group.astype(object).where(
         group.notna(), None
      )
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
