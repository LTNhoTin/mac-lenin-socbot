import json
import os
import time
from pathlib import Path
from typing import Optional
import re
import argparse
import sys
import subprocess
import logging

def ensure_dependencies(config: dict | None = None):
    """Đảm bảo các package bắt buộc đã có. Nếu thiếu sẽ tự động cài bằng pip.

    Cài gói theo từng module để tránh thất bại toàn bộ khi một package (ví dụ faiss-cpu) không khả dụng.
    """
    required = {
        "fastapi": "fastapi",
        "pydantic": "pydantic",
        "dotenv": "python-dotenv",
        "uvicorn": "uvicorn",
        "sentence_transformers": "sentence-transformers",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "numpy": "numpy",
        "torch": "torch",
    }

    use_faiss = False
    try:
        use_faiss = bool(config.get("use_faiss", False)) if config else False
    except Exception:
        use_faiss = False

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
            pass

    for mod, pkg in required.items():
        if not _try_import(mod):
            _install(pkg)

    if use_faiss and (not _try_import("faiss")):
        _install("faiss-cpu")

    critical = ["fastapi", "pydantic", "uvicorn", "sentence_transformers", "sklearn", "numpy"]
    missing = [m for m in critical if not _try_import(m)]
    if missing:
        raise RuntimeError(
            "Thiếu các thư viện bắt buộc: " + ", ".join(missing) +
            "\nVui lòng chạy: pip install -r requirements.txt hoặc để chương trình tự cài đặt có kết nối Internet."
        )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_config() -> dict:
    """Đọc config/settings.json luôn theo đường dẫn tuyệt đối của project root."""
    src_dir = Path(__file__).resolve().parent
    cfg_path = src_dir / "config" / "settings.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

ensure_dependencies(config)

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Union

from src.rag_service import RagService
from src.clients.openai_client import OpenAIClient
from src.clients.ollama_client import OllamaClient
from src.utils.chunking import chunk_text
from src.utils.preprocess import preprocess_text

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLN131 RAG Chatbot", version="2.0.0")

# Cấu hình CORS để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (có thể thay bằng danh sách cụ thể trong production)
    allow_credentials=False,  # Phải False khi dùng allow_origins=["*"]
    allow_methods=["*"],  # Cho phép tất cả methods
    allow_headers=["*"],  # Cho phép tất cả headers
)

rag: RagService = RagService(config)
llm_client: Union[OpenAIClient, OllamaClient, None] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    image_urls: Optional[List[str]] = None
    file_urls: Optional[List[str]] = None
    use_websearch: Optional[bool] = False


class RebuildRequest(BaseModel):
    backend: Optional[str] = None


@app.on_event("startup")
def startup_event():
    global llm_client
    model_type = os.getenv("MODEL_TYPE", "openai").lower()
    response_language = config.get("response_language", "vi")
    max_output_tokens = int(config.get("max_output_tokens", 400))
    temperature = float(config.get("temperature", 0.2))
    
    if model_type == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://server.nhotin.space:11434")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "gpt-oss:20b")
        llm_client = OllamaClient(
            base_url=base_url,
            model_name=model_name,
            response_language=response_language,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        if not api_key:
            raise ValueError("OPENAI_API_KEY không được tìm thấy trong environment variables")
        llm_client = OpenAIClient(
            api_key=api_key,
            model_name=model_name,
            response_language=response_language,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    
    try:
        rag.load_index()
    except Exception:
        project_root = Path(__file__).resolve().parent.parent
        data_path_cfg = config.get("data_path", "data/data.txt")
        data_path = (project_root / data_path_cfg) if not Path(data_path_cfg).is_absolute() else Path(data_path_cfg)
        if not data_path.exists():
            raise FileNotFoundError(f"Không thấy file dữ liệu: {data_path}")
        text = data_path.read_text(encoding="utf-8")
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


def _is_about_maclenin(question: str) -> bool:
    """Kiểm tra xem câu hỏi có liên quan đến maclenin (cấu hình chatbot) không."""
    question_lower = question.lower()
    keywords = [
        "maclenin", "máclênin", 
        "cấu hình", "cấu hình chatbot", 
        "chatbot của bạn", 
        "bạn là ai", "who are you", "what is your name",
        "giới thiệu về bạn", "tell me about",
        "bot này", "bot tên", "tên gì", "tên bạn",
        "bạn làm gì", "what do you do"
    ]
    return any(keyword in question_lower for keyword in keywords)


def _get_bot_config_info() -> str:
    """Trả về thông tin cấu hình chatbot."""
    model_type = os.getenv("MODEL_TYPE", "openai").lower()
    
    info_parts = [
        "👋 Xin chào! Tôi là Maclenin, một chatbot hỗ trợ thông tin dựa trên RAG (Retrieval-Augmented Generation).",
        "",
        "📋 **Cấu hình hiện tại:**",
        f"- **Loại model:** {model_type.upper()}",
    ]
    
    if model_type == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://server.nhotin.space:11434")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "gpt-oss:20b")
        info_parts.extend([
            f"- **Server Ollama:** {base_url}",
            f"- **Model:** {model_name}",
        ])
    else:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        info_parts.extend([
            f"- **Model:** {model_name}",
        ])
    
    info_parts.extend([
        "",
        "🔧 **Tính năng:**",
        "- Tìm kiếm thông tin từ database vector",
        "- Trả lời câu hỏi dựa trên ngữ cảnh RAG",
    ])
    
    if model_type == "openai" and model_name.startswith("gpt-4.1"):
        info_parts.extend([
            "- Hỗ trợ web search",
            "- Hỗ trợ xử lý hình ảnh và file",
        ])
    
    info_parts.append("")
    info_parts.append("💡 Bạn có thể hỏi tôi bất kỳ câu hỏi nào liên quan đến dữ liệu đã được lưu trữ!")
    
    return "\n".join(info_parts)


@app.post("/query")
def query(req: QueryRequest):
    start = time.perf_counter()
    
    logger.info("=" * 80)
    logger.info(f"📝 NHẬN CÂU HỎI: {req.question}")
    logger.info(f"🔧 Tham số: top_k={req.top_k}, use_websearch={req.use_websearch}")
    
    # Kiểm tra nếu câu hỏi về maclenin thì trả về thông tin cấu hình
    if _is_about_maclenin(req.question):
        logger.info("ℹ️  Câu hỏi về bot config, trả về thông tin cấu hình")
        answer = _get_bot_config_info()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"⏱️  Thời gian xử lý: {elapsed_ms}ms")
        logger.info("=" * 80)
        return {
            "question": req.question,
            "answer": answer,
            "contexts": [],
            "meta": {"type": "bot_info"},
            "latency_ms": elapsed_ms
        }
    
    contexts_for_llm = []
    
    # RAG Search
    logger.info("🔍 BẮT ĐẦU RAG SEARCH...")
    results = rag.search(req.question, top_k=req.top_k)
    logger.info(f"📊 Tìm thấy {len(results)} kết quả từ RAG search")
    
    # Log từng kết quả
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        source = r.get("source", "unknown")
        text_preview = r.get("text", "")[:100] + "..." if len(r.get("text", "")) > 100 else r.get("text", "")
        logger.info(f"  [{i}] Score: {score:.4f} | Source: {source}")
        logger.info(f"      Preview: {text_preview}")
    
    # Filter theo similarity threshold
    similarity_threshold = float(config.get("similarity_threshold", 0.6))
    logger.info(f"🎯 Lọc theo similarity threshold: {similarity_threshold}")
    filtered = [r for r in results if float(r.get("score", 0.0)) >= similarity_threshold]
    logger.info(f"✅ Sau khi lọc: {len(filtered)} kết quả đạt ngưỡng")
    
    # Giới hạn số lượng contexts
    contexts_max = int(config.get("contexts_max", 3))
    contexts_for_llm = filtered[:contexts_max]
    logger.info(f"📦 Chọn {len(contexts_for_llm)} contexts để gửi đến LLM (max: {contexts_max})")
    
    # Log contexts được chọn
    for i, ctx in enumerate(contexts_for_llm, 1):
        logger.info(f"  Context {i}: Score={ctx.get('score', 0.0):.4f}, Source={ctx.get('source', 'unknown')}")
        logger.info(f"    Text: {ctx.get('text', '')[:200]}...")
    
    # Build prompt và gọi LLM
    model_type = os.getenv("MODEL_TYPE", "openai").lower()
    logger.info(f"🤖 Gọi LLM: {model_type.upper()}")
    
    if model_type == "ollama":
        prompt = llm_client.build_prompt(req.question, contexts_for_llm)
        logger.info("📄 PROMPT ĐƯỢC XÂY DỰNG:")
        logger.info("-" * 80)
        logger.info(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        logger.info("-" * 80)
        
        answer, meta = llm_client.answer(
            req.question, 
            contexts_for_llm
        )
    else:
        image_urls = req.image_urls or []
        file_urls = req.file_urls or []
        use_websearch = req.use_websearch or False
        
        prompt = llm_client.build_prompt(req.question, contexts_for_llm)
        logger.info("📄 PROMPT ĐƯỢC XÂY DỰNG:")
        logger.info("-" * 80)
        logger.info(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        logger.info("-" * 80)
        
        if image_urls:
            logger.info(f"🖼️  Có {len(image_urls)} ảnh được gửi kèm")
        if file_urls:
            logger.info(f"📎 Có {len(file_urls)} file được gửi kèm")
        if use_websearch:
            logger.info("🌐 Web search được bật")
        
        answer, meta = llm_client.answer(
            req.question, 
            contexts_for_llm, 
            image_urls=image_urls if image_urls else None,
            file_urls=file_urls if file_urls else None,
            use_websearch=use_websearch
        )
    
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(f"💬 Câu trả lời nhận được (độ dài: {len(answer)} ký tự)")
    logger.info(f"⏱️  Tổng thời gian xử lý: {elapsed_ms}ms")
    logger.info("=" * 80)
    
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
    cfg = load_config()
    if req.backend:
        cfg["backend"] = req.backend
    project_root = Path(__file__).resolve().parent.parent
    data_path_cfg = cfg.get("data_path", "data/data.txt")
    data_path = (project_root / data_path_cfg) if not Path(data_path_cfg).is_absolute() else Path(data_path_cfg)
    text = data_path.read_text(encoding="utf-8")
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


@app.post("/query/upload")
async def query_with_upload(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None),
    top_k: Optional[int] = Form(None),
    use_websearch: Optional[bool] = Form(False)
):
    """Query với hỗ trợ upload file (text/ảnh)."""
    start = time.perf_counter()
    
    logger.info("=" * 80)
    logger.info(f"📝 NHẬN CÂU HỎI VỚI FILE UPLOAD: {question}")
    logger.info(f"🔧 Tham số: top_k={top_k}, use_websearch={use_websearch}")
    
    # Kiểm tra nếu câu hỏi về maclenin thì trả về thông tin cấu hình
    if _is_about_maclenin(question):
        logger.info("ℹ️  Câu hỏi về bot config, trả về thông tin cấu hình")
        answer = _get_bot_config_info()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"⏱️  Thời gian xử lý: {elapsed_ms}ms")
        logger.info("=" * 80)
        return {
            "question": question,
            "answer": answer,
            "contexts": [],
            "meta": {"type": "bot_info"},
            "latency_ms": elapsed_ms
        }
    
    contexts_for_llm = []
    image_urls = []
    
    if file:
        logger.info(f"📎 XỬ LÝ FILE: {file.filename} (size: {file.size if hasattr(file, 'size') else 'unknown'} bytes)")
        content = await file.read()
        file_ext = file.filename.split('.')[-1].lower() if file.filename else ''
        logger.info(f"   Loại file: {file_ext}")
        
        if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            logger.info("   → Đây là file ảnh, encode thành base64")
            # Encode ảnh thành base64 để gửi đến OpenAI API
            import base64
            base64_image = base64.b64encode(content).decode('utf-8')
            mime_type = f"image/{file_ext}" if file_ext != 'jpg' else "image/jpeg"
            image_data_url = f"data:{mime_type};base64,{base64_image}"
            image_urls.append(image_data_url)
            logger.info(f"   ✅ Đã encode ảnh thành base64 (length: {len(base64_image)} chars)")
        else:
            logger.info("   → Đây là file text, đọc và chunk")
            try:
                text_content = content.decode('utf-8')
                logger.info(f"   Đọc được {len(text_content)} ký tự")
                processed = preprocess_text(text_content)
                file_chunks = chunk_text(
                    processed,
                    chunk_size=int(config.get("chunk_size", 800)),
                    chunk_overlap=int(config.get("chunk_overlap", 120)),
                    source=f"uploaded:{file.filename}"
                )
                logger.info(f"   Chunk thành {len(file_chunks)} chunks, chọn {min(2, len(file_chunks))} chunks đầu tiên")
                contexts_for_llm.extend([{"text": c["text"], "source": c["source"], "score": 1.0} for c in file_chunks[:2]])
                for i, ctx in enumerate(contexts_for_llm, 1):
                    logger.info(f"   File Context {i}: {ctx.get('text', '')[:100]}...")
            except Exception as e:
                logger.error(f"   ❌ Lỗi đọc file: {str(e)}")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Không thể đọc file: {str(e)}"}
                )
    else:
        logger.info("   Không có file được upload")
    
    # RAG Search
    logger.info("🔍 BẮT ĐẦU RAG SEARCH...")
    results = rag.search(question, top_k=top_k)
    logger.info(f"📊 Tìm thấy {len(results)} kết quả từ RAG search")
    
    # Log từng kết quả
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        source = r.get("source", "unknown")
        text_preview = r.get("text", "")[:100] + "..." if len(r.get("text", "")) > 100 else r.get("text", "")
        logger.info(f"  [{i}] Score: {score:.4f} | Source: {source}")
        logger.info(f"      Preview: {text_preview}")
    
    # Filter theo similarity threshold
    similarity_threshold = float(config.get("similarity_threshold", 0.6))
    logger.info(f"🎯 Lọc theo similarity threshold: {similarity_threshold}")
    filtered = [r for r in results if float(r.get("score", 0.0)) >= similarity_threshold]
    logger.info(f"✅ Sau khi lọc: {len(filtered)} kết quả đạt ngưỡng")
    
    # Giới hạn số lượng contexts
    contexts_max = int(config.get("contexts_max", 3))
    contexts_for_llm.extend(filtered[:contexts_max])
    logger.info(f"📦 Tổng cộng {len(contexts_for_llm)} contexts để gửi đến LLM (RAG: {len(filtered[:contexts_max])}, File: {len([c for c in contexts_for_llm if c.get('source', '').startswith('uploaded:')])})")
    
    # Log contexts được chọn
    for i, ctx in enumerate(contexts_for_llm, 1):
        logger.info(f"  Context {i}: Score={ctx.get('score', 0.0):.4f}, Source={ctx.get('source', 'unknown')}")
        logger.info(f"    Text: {ctx.get('text', '')[:200]}...")
    
    # Build prompt và gọi LLM
    model_type = os.getenv("MODEL_TYPE", "openai").lower()
    logger.info(f"🤖 Gọi LLM: {model_type.upper()}")
    
    if model_type == "ollama":
        prompt = llm_client.build_prompt(question, contexts_for_llm)
        logger.info("📄 PROMPT ĐƯỢC XÂY DỰNG:")
        logger.info("-" * 80)
        logger.info(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        logger.info("-" * 80)
        
        answer, meta = llm_client.answer(
            question, 
            contexts_for_llm
        )
    else:
        prompt = llm_client.build_prompt(question, contexts_for_llm)
        logger.info("📄 PROMPT ĐƯỢC XÂY DỰNG:")
        logger.info("-" * 80)
        logger.info(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        logger.info("-" * 80)
        
        if image_urls:
            logger.info(f"🖼️  Có {len(image_urls)} ảnh được gửi kèm")
        if use_websearch:
            logger.info("🌐 Web search được bật")
        
        answer, meta = llm_client.answer(
            question, 
            contexts_for_llm, 
            image_urls=image_urls if image_urls else None,
            file_urls=None,
            use_websearch=use_websearch
        )
    
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(f"💬 Câu trả lời nhận được (độ dài: {len(answer)} ký tự)")
    logger.info(f"⏱️  Tổng thời gian xử lý: {elapsed_ms}ms")
    logger.info("=" * 80)
    
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts_for_llm,
        "meta": meta,
        "latency_ms": elapsed_ms
    }


class StoryRequest(BaseModel):
    topic: Optional[str] = None  # Chủ đề câu chuyện (ví dụ: "giai cấp công nhân", "liên minh giai cấp")
    character: Optional[str] = "Hồ Chí Minh"  # Nhân vật chính (mặc định là Bác Hồ)
    length: Optional[str] = "medium"  # "short" (3-5 câu), "medium" (5-8 câu), "long" (8-12 câu)


@app.post("/story")
def generate_story(req: StoryRequest):
    """Tự động tạo một câu chuyện về lịch sử cách mạng."""
    start = time.perf_counter()
    
    logger.info("=" * 80)
    logger.info(f"📖 YÊU CẦU TẠO CÂU CHUYỆN")
    logger.info(f"🔧 Tham số: topic={req.topic}, character={req.character}, length={req.length}")
    
    # Xây dựng prompt cho việc kể chuyện
    topic_text = f"về chủ đề '{req.topic}'" if req.topic else "về chủ nghĩa Mác-Lênin, kinh tế chính trị hoặc lịch sử cách mạng"
    
    length_instruction = {
        "short": "3-5 câu",
        "medium": "5-8 câu", 
        "long": "8-12 câu"
    }.get(req.length, "5-8 câu")
    
    story_prompt = (
        f"Bạn là trợ lý tên ViVi. Hãy kể một câu chuyện {length_instruction} "
        f"về {req.character} {topic_text}. "
        "Câu chuyện phải sống động, có cảm xúc, mang tính giáo dục và truyền cảm hứng. "
        "Hãy kể như một người kể chuyện chân thực, không cần mở đầu hay kết thúc trang trọng, "
        "chỉ cần kể câu chuyện một cách tự nhiên và hấp dẫn."
    )
    
    # Gọi LLM để tạo câu chuyện
    model_type = os.getenv("MODEL_TYPE", "openai").lower()
    
    if model_type == "ollama":
        answer, meta = llm_client.answer(story_prompt, [])
    else:
        answer, meta = llm_client.answer(story_prompt, [])
    
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(f"📖 Câu chuyện đã được tạo (độ dài: {len(answer)} ký tự)")
    logger.info(f"⏱️  Thời gian xử lý: {elapsed_ms}ms")
    logger.info("=" * 80)
    
    return {
        "story": answer,
        "topic": req.topic,
        "character": req.character,
        "length": req.length,
        "meta": meta,
        "latency_ms": elapsed_ms
    }


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "nova"  # "alloy", "echo", "fable", "onyx", "nova", "shimmer"
    model: Optional[str] = "tts-1"  # "tts-1" hoặc "tts-1-hd"


@app.post("/tts")
def text_to_speech(req: TTSRequest):
    """
    Chuyển đổi text thành speech sử dụng OpenAI TTS API.
    Trả về file audio MP3.
    """
    try:
        model_type = os.getenv("MODEL_TYPE", "openai").lower()
        
        if model_type != "openai":
            return JSONResponse(
                status_code=400,
                content={"error": "TTS chỉ hỗ trợ khi sử dụng OpenAI model"}
            )
        
        if not isinstance(llm_client, OpenAIClient):
            return JSONResponse(
                status_code=400,
                content={"error": "LLM client không phải OpenAI client"}
            )
        
        # Giới hạn độ dài text để tránh tốn phí
        max_length = 5000
        if len(req.text) > max_length:
            req.text = req.text[:max_length] + "..."
        
        audio_data = llm_client.text_to_speech(
            text=req.text,
            voice=req.voice,
            model=req.model
        )
        
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3"
            }
        )
    except Exception as e:
        logger.error(f"Lỗi TTS: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Lỗi tạo TTS: {str(e)}"}
        )


def _set_runtime_env_for_mac():
    """Thiết lập biến môi trường để server ổn định."""
    os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")
    os.environ.setdefault("TORCH_MPS_ENABLED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


def _parse_args():
    parser = argparse.ArgumentParser(description="Chạy MLN131 FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host (mặc định 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2000, help="Port (mặc định 2000)")
    parser.add_argument("--reload", action="store_true", help="Bật reload khi phát triển")
    return parser.parse_args()


if __name__ == "__main__":
    _set_runtime_env_for_mac()
    args = _parse_args()
    if args.reload:
        uvicorn.run("src.main:app", host=args.host, port=args.port, reload=True, log_level="info")
    else:
        uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="info")
