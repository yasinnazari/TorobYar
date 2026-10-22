
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def load_dataset() -> dict[str, pd.DataFrame]:
   products = pd.read_csv(
      DATA_DIR / "products.csv",
      usecols=[
         "product_id",
         "persian_title",
         "brand_name",
         "category_name",
         "model",
         "image_urls",
      ],
      dtype="string",
   )

   offers = pd.read_csv(
      DATA_DIR / "offers.csv",
      usecols=[
         "offer_id",
         "product_id",
         "shop_id",
         "seller_title",
         "price_toman",
         "warranty",
         "last_price_changed_at",
      ],
      dtype="string",
   )

   shops = pd.read_csv(
      DATA_DIR / "shops.csv",
      usecols=[
         "shop_id",
         "name",
         "address",
         "category_names",
         "lat",
         "lon",
         "location_label",
         "payment_options",
         "supports_fast_delivery",
         "weekly_hours",
      ],
      dtype="string",
   )

   # Convert prices to numeric values.
   offers["price_toman"] = pd.to_numeric(
      offers["price_toman"],
      errors="coerce",
   )

   print(f"Loaded {len(products):,} products")
   print(f"Loaded {len(offers):,} offers")
   print(f"Loaded {len(shops):,} shops")

   return {
      "products": products,
      "offers": offers,
      "shops": shops,
   }
