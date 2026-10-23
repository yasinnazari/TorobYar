
import re
import pandas as pd


STOP_WORDS = {
   "میخوام", "میخواهم", "میخوام", "می‌خوام",
   "می‌خواهم", "میخری", "میخرم", "لطفا", "لطفاً",
   "یه", "یک", "برای", "با", "از", "تا", "توی",
   "در", "رو", "را", "من", "میلیون", "هزار",
   "تومان", "تومن", "قیمت", "حدود", "حداکثر",
   "زیر", "مناسب", "میخوامش",
}

TOKEN_ALIASES = {
   "مردونه": ["مردونه", "مردانه", "مردا", "آقایان", "آقا"],
   "مردانه": ["مردانه", "مردونه", "مردا", "آقایان", "آقا"],
   "زنونه": ["زنونه", "زنانه", "زنا", "خانم‌ها", "خانمها", "خانم"],
   "زنانه": ["زنانه", "زنونه", "زنا", "خانم‌ها", "خانمها", "خانم"],
   "بچگونه": ["بچگونه", "بچه گانه", "بچگانه", "بچه‌گانه", "کودک", "کودکانه", "بچه"],
   "بچه گانه": ["بچگونه", "بچه گانه", "بچگانه", "بچه‌گانه", "کودک", "کودکانه", "بچه"],
   "بچگانه": ["بچگونه", "بچه گانه", "بچگانه", "بچه‌گانه", "کودک", "کودکانه", "بچه"],
   "کودک": ["بچگونه", "بچه گانه", "بچگانه", "بچه‌گانه", "کودک", "کودکانه", "بچه"],
   "کودکانه": ["بچگونه", "بچه گانه", "بچگانه", "بچه‌گانه", "کودک", "کودکانه", "بچه"],
}


def normalize_text(value):
   text = str(value or "").lower()

   text = text.replace("ي", "ی").replace("ك", "ک")
   text = text.replace("ۀ", "ه").replace("ة", "ه")
   text = text.replace("ـ", "")

   text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
   text = text.replace("\u200c", " ")

   text = re.sub(r"[^\w\s]", " ", text)
   text = re.sub(r"\s+", " ", text).strip()

   return text


def clean_search_query(query):
   query = normalize_text(query)

   query = re.sub(
      r"(تا|زیر|حداکثر|حدود)\s*"
      r"[\d۰-۹٠-٩,.]+\s*"
      r"(میلیون|هزار)?\s*"
      r"(تومان|تومن)?",
      " ",
      query,
   )

   tokens = query.split()
   tokens = [
      token for token in tokens
      if token not in STOP_WORDS
   ]
   return tokens


def prepare_product_search(products):
   search_columns = {
      "_search_title": "persian_title",
      "_search_brand": "brand_name",
      "_search_category": "category_name",
      "_search_model": "model",
      "_search_specs": "specs",
   }

   for search_column, source_column in search_columns.items():
      if search_column in products.columns:
         continue

      if source_column not in products.columns:
         products[search_column] = ""
         continue

      values = products[source_column].fillna("").astype(str).str.lower()
      values = values.str.replace("ي", "ی", regex=False)
      values = values.str.replace("ك", "ک", regex=False)
      values = values.str.replace("ۀ", "ه", regex=False)
      values = values.str.replace("ة", "ه", regex=False)
      values = values.str.replace("ـ", "", regex=False)
      values = values.str.replace(
         r"[\u064B-\u065F\u0670]", "", regex=True
      )
      values = values.str.replace("\u200c", " ", regex=False)
      values = values.str.replace(r"[^\w\s]", " ", regex=True)
      values = values.str.replace(r"\s+", " ", regex=True).str.strip()

      products[search_column] = values

   return products



def search_products_in_dataset(products, query, limit=10):
   query = str(query or "").strip()

   if not query or products.empty:
      return {
         "query": query,
         "count": 0,
         "total_matches": 0,
         "search_mode": "empty",
         "results": [],
         "similar_results": [],
         "needs_clarification": False,
      }

   products = prepare_product_search(products)
   tokens = clean_search_query(query)

   if not tokens:
      return {
         "query": query,
         "count": 0,
         "total_matches": 0,
         "search_mode": "empty",
         "results": [],
         "similar_results": [],
         "needs_clarification": False,
      }

   fields = {
      "_search_title": 5,
      "_search_category": 3,
      "_search_specs": 2,
      "_search_brand": 2,
      "_search_model": 1,
   }

   scores = pd.Series(0, index=products.index, dtype="int32")
   matched_terms = pd.Series(0, index=products.index, dtype="int32")

   for token in tokens:
      aliases = TOKEN_ALIASES.get(token, [token])
      term_matched = pd.Series(False, index=products.index)

      for column, weight in fields.items():
         if column not in products.columns:
            continue

         field_matched = pd.Series(False, index=products.index)

         for alias in aliases:
            alias = normalize_text(alias)

            if alias:
               field_matched |= products[column].str.contains(
                  re.escape(alias),
                  regex=True,
                  na=False,
               )

         scores += field_matched.astype("int32") * weight
         term_matched |= field_matched

      matched_terms += term_matched.astype("int32")

   normalized_query = normalize_text(query)

   scores += (
      products["_search_title"]
      .str.contains(re.escape(normalized_query), regex=True, na=False)
      .astype("int32") * 5
   )

   minimum_matches = len(tokens)
   exact_mask = matched_terms >= minimum_matches

   ranked = products.loc[exact_mask].copy()
   search_mode = "exact"

   if ranked.empty:
      return {
         "query": query,
         "count": 0,
         "total_matches": 0,
         "search_mode": "no_exact_results",
         "results": [],
         "similar_results": [],
         "needs_clarification": True,
         "clarification_message": (
            "محصول دقیقی پیدا نکردم. "
            "می‌خواهی محصولات مشابه را هم بررسی کنم؟"
         ),
      }

   ranked["_match_score"] = scores.loc[ranked.index]
   ranked["_matched_terms"] = matched_terms.loc[ranked.index]

   ranked = ranked.sort_values(
      ["_match_score", "_matched_terms"],
      ascending=False,
      kind="stable",
   )

   total_matches = len(ranked)

   output_columns = [
      column
      for column in [
         "product_id",
         "persian_title",
         "brand_name",
         "category_name",
         "model",
         "image_urls",
      ]
      if column in ranked.columns
   ]

   results = ranked.head(max(1, min(limit, 50)))[output_columns].copy()
   results = results.astype(object).where(results.notna(), None)

   return {
      "query": query,
      "count": len(results),
      "total_matches": total_matches,
      "search_mode": search_mode,
      "results": results.to_dict(orient="records"),
      "similar_results": [],
      "needs_clarification": False,
   }
