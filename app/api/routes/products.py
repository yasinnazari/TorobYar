from fastapi import APIRouter, Query, Request

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
