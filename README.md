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
python 3.10

API
- GET /health: kiểm tra trạng thái
- POST /query {"question": "...", "top_k": 5}

Note về chunking (điểm a) hợp lý – gợi ý thêm)
- Chunk size ~800 từ, overlap ~120 từ (10-15%).
- Tôn trọng ranh giới câu/đoạn, tránh cắt giữa câu.
- Loại bỏ chunk quá ngắn (< 40 từ) để giảm nhiễu.
- Có thể tăng overlap nếu nội dung có nhiều tham chiếu chéo.
- Nếu tài liệu có tiêu đề/sections, cân nhắc tách theo heading trước, sau đó chia câu.