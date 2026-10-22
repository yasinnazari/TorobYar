from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.repositories.dataset import load_dataset
from app.api.routes.products import router as products_router


@asynccontextmanager
async def lifespan(app: FastAPI):
   app.state.dataset = load_dataset()
   yield


app = FastAPI(
   title="Torob Yar | Your Shopping Assistant",
   version="0.0.1",
   lifespan=lifespan,
)


@app.get("/health")
def health_check():
   return {"status": "ok"}


app.include_router(products_router)
