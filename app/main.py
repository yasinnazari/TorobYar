from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.repositories.dataset import load_dataset
from app.services.conversation_memory import init_memory_db
from app.api.routes.products import router as products_router
from app.api.routes.assistant import router as assistant_router

@asynccontextmanager
async def lifespan(app: FastAPI):
   init_memory_db()
   app.state.dataset = load_dataset()
   yield

app = FastAPI(
   title="TorobYar | Dastyar Kharid Shoma",
   version="0.0.1",
   lifespan=lifespan,
)

app.include_router(products_router)
app.include_router(assistant_router)
