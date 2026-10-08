from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv(override=True)

from rag.pipeline import RagPipeline  # noqa: E402  (환경변수 로드 후 import)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 한 번만 로드 (BM25 색인, 리랭커 모델은 무거워서 요청마다 만들지 않는다)
    state["rag"] = RagPipeline()
    yield
    state.clear()


app = FastAPI(title="AI 기본법 QA", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["고영향 인공지능이란 무엇인가요?"])


class Source(BaseModel):
    article: str = Field(description="근거 조문 (예: 제2조(정의))")
    unit: str | None = Field(default=None, description="검색에 걸린 항/호 (예: 제4호)")
    chapter: str | None = None
    content: str = Field(description="검색된 법률 내용")
    score: float = Field(description="리랭킹 점수 (0~1, 높을수록 질문과 관련)")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {"message": "RAG API"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = state["rag"].ask(req.question.strip())
    return {"answer": result["answer"], "sources": result["sources"]}
