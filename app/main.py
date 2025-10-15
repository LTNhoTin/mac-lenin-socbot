import json
import os
import time
from pathlib import Path
from typing import Optional
import re
import argparse
import sys
import subprocess

# Lưu ý: Trì hoãn import các thư viện bên thứ ba (fastapi, uvicorn, dotenv, sentence_transformers, sklearn)
# cho tới khi đảm bảo đã cài đặt đầy đủ phụ thuộc. Nhờ đó, chỉ cần chạy:
#   python app/main.py
# là có thể host dự án, kể cả lần đầu chưa cài dependencies.

def ensure_dependencies(config: dict | None = None):
    """Đảm bảo các package bắt buộc đã có. Nếu thiếu sẽ tự động cài bằng pip.

    Cài gói theo từng module để tránh thất bại toàn bộ khi một package (ví dụ faiss-cpu) không khả dụng.
    """
    # Map module -> pip package
    required = {
        "fastapi": "fastapi",
        "pydantic": "pydantic",
        "dotenv": "python-dotenv",
        "uvicorn": "uvicorn",
        "sentence_transformers": "sentence-transformers",
        "sklearn": "scikit-learn",
        # scipy cần thiết cho lưu/load ma trận TF-IDF dạng sparse
        "scipy": "scipy",
        "numpy": "numpy",
        "google.generativeai": "google-generativeai",
        "torch": "torch",
    }

    # Nếu config bật use_faiss, cố gắng cài đặt faiss-cpu
    use_faiss = False
    try:
        use_faiss = bool(config.get("use_faiss", False)) if config else False
    except Exception:
        use_faiss = False

    # Helper: import thử một module
    def _try_import(modname: str) -> bool:
        try:
            __import__(modname)
            return True
        except Exception:
            return False

    def _install(pkg: str):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError:
            # Không dừng hẳn server nếu một package cài thất bại; sẽ báo lỗi rõ ràng sau.
            pass

    # Cài từng package nếu thiếu
    for mod, pkg in required.items():
        if not _try_import(mod):
            _install(pkg)

    # faiss chỉ cài nếu thực sự cần và chưa có
    if use_faiss and (not _try_import("faiss")):
        _install("faiss-cpu")

    # Kiểm tra lại các module tối quan trọng
    critical = ["fastapi", "pydantic", "uvicorn", "sentence_transformers", "sklearn", "numpy"]
    missing = [m for m in critical if not _try_import(m)]
    if missing:
        raise RuntimeError(
            "Thiếu các thư viện bắt buộc: " + ", ".join(missing) +
            "\nVui lòng chạy: pip install -r requirements.txt hoặc để chương trình tự cài đặt có kết nối Internet."
        )

# Đảm bảo thư mục gốc dự án có trong sys.path khi chạy trực tiếp: python app/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đọc config sớm để quyết định có cần cài faiss hay không
def load_config() -> dict:
    """Đọc config/settings.json luôn theo đường dẫn tuyệt đối của project root,
    tránh phụ thuộc thư mục làm việc hiện tại (CWD)."""
    project_root = Path(__file__).resolve().parent.parent
    cfg_path = project_root / "config/settings.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# Đảm bảo dependencies trước khi import các module bên thứ ba
ensure_dependencies(config)

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from app.rag_service import RagService
from app.gemini_client import GeminiClient
from app.chunking import chunk_text
from app.preprocess import preprocess_text


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class RebuildRequest(BaseModel):
    backend: Optional[str] = None  # gemma | sbert | tfidf
load_dotenv()

app = FastAPI(title="MLN131 RAG Chatbot", version="1.0.0")

rag: RagService = RagService(config)
gemini: Optional[GeminiClient] = None


@app.on_event("startup")
def startup_event():
    global gemini
    # Khởi tạo Gemini
    api_key = os.getenv("GEMINI_API_KEY", "")
    gemini = GeminiClient(
        api_key=api_key,
        model_name=config.get("gemini_model_name", "gemini-1.5-flash"),
        response_language=config.get("response_language", "vi"),
        max_output_tokens=int(config.get("max_output_tokens", 150)),
        temperature=float(config.get("temperature", 0.2)),
    )

    # Load index; nếu chưa có thì build nhanh từ data hiện có
    try:
        rag.load_index()
    except Exception:
        # Build index tại runtime (chỉ lần đầu) – đọc data và tạo chunks
        project_root = Path(__file__).resolve().parent.parent
        data_path_cfg = config.get("data_path", "data/data.txt")
        data_path = (project_root / data_path_cfg) if not Path(data_path_cfg).is_absolute() else Path(data_path_cfg)
        if not data_path.exists():
            raise FileNotFoundError(f"Không thấy file dữ liệu: {data_path}")
        text = data_path.read_text(encoding="utf-8")
        # Tiền xử lý toàn bộ dữ liệu trước khi chunk để tối ưu ranh giới đoạn/câu
        text = preprocess_text(text)
        chunks = chunk_text(
            text,
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 120)),
            separators=config.get("separators", None),
            source=str(data_path)
        )
        rag.build_index(chunks)
        rag.load_index()


@app.get("/health")
def health():
    chunk_count = 0
    try:
        chunk_count = len(rag.docstore)
    except Exception:
        pass
    return {"status": "ok", "index_ready": rag.is_ready(), "chunk_count": chunk_count}


@app.post("/query")
def query(req: QueryRequest):
    start = time.perf_counter()
    # Retrieve trước
    results = rag.search(req.question, top_k=req.top_k)
    # Lọc theo ngưỡng tương đồng để tránh nhiễu
    similarity_threshold = float(config.get("similarity_threshold", 0.6))
    filtered = [r for r in results if float(r.get("score", 0.0)) >= similarity_threshold]
    # Giới hạn số ngữ cảnh đưa vào model để tránh verbose
    contexts_max = int(config.get("contexts_max", 3))
    contexts_for_llm = filtered[:contexts_max]

    # Gọi LLM cả khi không có ngữ cảnh (open-domain) để trả lời thân thiện theo kiến thức chung.
    answer, meta = gemini.answer(req.question, contexts_for_llm)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "question": req.question,
        "answer": answer,
        "contexts": contexts_for_llm,
        "meta": meta,
        "latency_ms": elapsed_ms
    }


def _wc(s: str) -> int:
    return len(re.findall(r"\w+", s))


@app.get("/chunks")
def chunks(limit: int = 3, preview_chars: int = 300):
    """Xem nhanh các chunk đã build (preview)."""
    limit = max(1, min(limit, 50))
    pcs = []
    for i, c in enumerate(rag.docstore[:limit]):
        txt = c.get("text", "")
        pcs.append({
            "id": i,
            "source": c.get("source", "unknown"),
            "word_count": _wc(txt),
            "preview": txt[:preview_chars]
        })
    return {"chunk_count": len(rag.docstore), "preview_count": len(pcs), "chunks": pcs}


@app.post("/admin/rebuild_index")
def rebuild_index(req: RebuildRequest):
    global rag
    # Cho phép override backend qua API
    cfg = load_config()
    if req.backend:
        cfg["backend"] = req.backend
    # Build lại index
    project_root = Path(__file__).resolve().parent.parent
    data_path_cfg = cfg.get("data_path", "data/data.txt")
    data_path = (project_root / data_path_cfg) if not Path(data_path_cfg).is_absolute() else Path(data_path_cfg)
    text = data_path.read_text(encoding="utf-8")
    # Tiền xử lý toàn bộ dữ liệu trước khi chunk
    text = preprocess_text(text)
    chunks = chunk_text(
        text,
        chunk_size=int(cfg.get("chunk_size", 800)),
        chunk_overlap=int(cfg.get("chunk_overlap", 120)),
        separators=cfg.get("separators", None),
        source=str(data_path)
    )
    new_rag = RagService(cfg)
    new_rag.build_index(chunks)
    new_rag.load_index()
    rag = new_rag
    return {"status": "rebuilt", "backend": cfg.get("backend"), "index_ready": rag.is_ready(), "chunks": len(chunks)}


def _set_runtime_env_for_mac():
    """Thiết lập biến môi trường để server ổn định hơn trên macOS."""
    os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")
    os.environ.setdefault("TORCH_MPS_ENABLED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


def _parse_args():
    parser = argparse.ArgumentParser(description="Chạy MLN131 FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host (mặc định 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (mặc định 8000)")
    parser.add_argument("--reload", action="store_true", help="Bật reload khi phát triển")
    return parser.parse_args()


if __name__ == "__main__":
    _set_runtime_env_for_mac()
    args = _parse_args()
    # Khi bật reload, uvicorn yêu cầu truyền app dưới dạng import string
    if args.reload:
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True, log_level="info")
    else:
        uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="info")