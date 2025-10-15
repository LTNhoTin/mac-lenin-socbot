import json
import sys
from pathlib import Path

# Bổ sung sys.path để có thể import package app khi chạy từ scripts
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunking import chunk_text, preview_chunks
from app.preprocess import preprocess_text
from app.rag_service import RagService


def load_config() -> dict:
    cfg_path = Path("config/settings.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = load_config()
    rag = RagService(cfg)

    data_path = Path(cfg.get("data_path", "data/data.txt"))
    text = data_path.read_text(encoding="utf-8")
    # Tiền xử lý toàn bộ văn bản trước khi chunk để tối ưu chất lượng phân đoạn
    text = preprocess_text(text)
    chunks = chunk_text(
        text,
        chunk_size=int(cfg.get("chunk_size", 800)),
        chunk_overlap=int(cfg.get("chunk_overlap", 120)),
        separators=cfg.get("separators", None),
        source=str(data_path)
    )

    # Xem thử vài chunk đầu
    for c in preview_chunks(chunks, 3):
        print("--- PREVIEW CHUNK ---")
        print(c["text"][:400])
        print("[source]", c.get("source"))

    # Build index theo backend (sbert/tfidf) và lưu
    rag.build_index(chunks)
    print("Index đã được build và lưu.")


if __name__ == "__main__":
    main()