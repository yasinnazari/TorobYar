import json
from app.repositories.dataset import load_dataset
from app.agent.tools import search_products, get_product_offers, find_payment_shops, find_nearby_product_shops

def main():
   dataset = load_dataset()

   shops = dataset["shops"]

   print("\n-------------------------- PAYMENT OPTIONS: RAW VALUES ---")
   print(
      shops["payment_options"]
      .value_counts(dropna=False)
      .head(20)
      .to_string()
   )

   print("\n---~!!!!!!!!!!!!--------- NON-EMPTY PAYMENT OPTIONS ---")
   non_empty = shops[
      shops["payment_options"].notna()
      & shops["payment_options"].str.strip().ne("")
      & shops["payment_options"].str.strip().ne("[]")
   ]

   print("Non Empty Shops:", len(non_empty))

   for _, shop in non_empty.head(10).iterrows():
      print(f"Shop: {shop["name"]} | Payment:", repr(shop["payment_options"]))

   search_result = search_products(dataset, query="کابل", limit=3)

   print("\n--- PRODUCT SEARCH ---")
   for product in search_result["results"]:
      print(
         product["product_id"],
         product["persian_title"],
      )

   if not search_result["results"]:
      print("No products found.")
      return

   product_id = search_result["results"][0]["product_id"]

   offers_result = get_product_offers(
      dataset,
      product_id,
      limit=5,
   )

   nearby = find_nearby_product_shops(
      dataset=dataset,
      product_id=product_id,
      user_lat=35.7000,
      user_lon=51.4200,
      radius_km=10,
      limit=10,
   )

   print("\n--- NEARBY SHOPS ---")
   print("Count:", nearby["count"])

   for shop in nearby["results"]:
      print(
         shop["distance_km"],
         "km |",
         shop["price_toman"],
         "Toman |",
         shop["name"],
         "|",
         shop["address"],
      )

   print("\n--- PRODUCT OFFERS ---")
   for offer in offers_result["results"]:
      print(
         offer["price_toman"],
         offer["name"],
         offer["address"],
      )

   payment_result = find_payment_shops(
      dataset,
      product_id,
      limit=10,
      channel="in_person",
   )

   print("--------------------- IN-PERSON PAYMENT OPTIONS ---")
   print("Count:", payment_result["count"])

   for offer in payment_result["results"]:
      providers = [
         option.get("provider_name")
         for option in offer["payment_options"]
      ]

      print(f"Price: {offer["price_toman"]} | Shop: {offer["name"]} | Providers: {providers} | Address:", offer["address"])

   print("\n--- Payment Result Counts ---")
   print(payment_result["count"])


if __name__ == "__main__":
   main()
