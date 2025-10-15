MLN131 RAG Chatbot (FastAPI + Gemini)

Hướng dẫn nhanh
- Cấu trúc thư mục đã tạo:
  - app/: mã nguồn FastAPI, RAG, Gemini client
  - config/settings.json: cấu hình chunking, model, đường dẫn
  - scripts/build_index.py: script tạo FAISS index từ data
  - storage/: lưu index và docstore
  - data/data.txt: dữ liệu nguồn
  - final_model/: model embedding (SentenceTransformers) đã fine-tune

Thiết lập môi trường
1) Tạo conda env Python 3.10:
   conda create -n mln131-rag -y python=3.10

2) Cài package:
   conda run -n mln131-rag pip install -r requirements.txt

3) Cấu hình API key Gemini (đã có trong .env), nếu cần sửa:
   mở file .env và thay GEMINI_API_KEY=...

Build index (lần đầu)
   conda run -n mln131-rag python scripts/build_index.py

Chạy server FastAPI
   conda run -n mln131-rag uvicorn app.main:app --reload --port 8000 --host 127.0.0.1

API
- GET /health: kiểm tra trạng thái
- POST /query {"question": "...", "top_k": 5}

Note về chunking (điểm a) hợp lý – gợi ý thêm)
- Chunk size ~800 từ, overlap ~120 từ (10-15%).
- Tôn trọng ranh giới câu/đoạn, tránh cắt giữa câu.
- Loại bỏ chunk quá ngắn (< 40 từ) để giảm nhiễu.
- Có thể tăng overlap nếu nội dung có nhiều tham chiếu chéo.
- Nếu tài liệu có tiêu đề/sections, cân nhắc tách theo heading trước, sau đó chia câu.