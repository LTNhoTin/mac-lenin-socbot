# MLN131 RAG Chatbot API Documentation

## Tổng quan

API này cung cấp dịch vụ RAG (Retrieval-Augmented Generation) chatbot với hỗ trợ:
- RAG retrieval từ database vector (Gemma embedding model)
- LLM generation (OpenAI GPT-4o-mini/gpt-4.1 hoặc Ollama gpt-oss20b)
- Web search (OpenAI gpt-4.1 với tools: web_search_preview)
- File upload và image inputs (text và ảnh)

## Cấu hình môi trường

Tạo file `.env` trong thư mục `backend/` với các biến sau:

```env
# Model Type: "openai" hoặc "ollama"
MODEL_TYPE=openai

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini  # hoặc gpt-4.1 để sử dụng endpoint /v1/responses với web search, file/image inputs

# Ollama Configuration
OLLAMA_BASE_URL=http://server.nhotin.space:11434
OLLAMA_MODEL_NAME=gpt-oss20b
```

**Lưu ý:** Các cấu hình khác như `response_language`, `max_output_tokens`, `temperature` được đọc từ file `src/config/settings.json`.

## Endpoints

### 1. Health Check

**GET** `/health`

Kiểm tra trạng thái của server và index.

**Response:**
```json
{
  "status": "ok",
  "index_ready": true,
  "chunk_count": 1234
}
```

---

### 2. Query (RAG)

**POST** `/query`

Gửi câu hỏi và nhận câu trả lời dựa trên RAG.

**Request Body:**
```json
{
  "question": "Vốn điều lệ của công ty cổ phần là gì?",
  "top_k": 5,
  "image_urls": [],
  "file_urls": [],
  "use_websearch": false
}
```

**Parameters:**
- `question` (string, required): Câu hỏi cần trả lời
- `top_k` (integer, optional): Số lượng chunks top-K để retrieve (mặc định: 5)
- `image_urls` (array of strings, optional): Danh sách URL ảnh để đưa vào context (hỗ trợ với OpenAI gpt-4.1)
- `file_urls` (array of strings, optional): Danh sách URL file để đưa vào context (hỗ trợ với OpenAI gpt-4.1)
- `use_websearch` (boolean, optional): Bật web search (chỉ hỗ trợ với OpenAI gpt-4.1, mặc định: false)

**Lưu ý về OpenAI gpt-4.1:**
- Khi sử dụng model `gpt-4.1`, API sẽ tự động sử dụng endpoint `/v1/responses` với các tính năng:
  - Web search: Thêm `"tools": [{"type": "web_search_preview"}]` khi `use_websearch=true`
  - File inputs: Sử dụng format `{"type": "input_file", "file_url": "..."}`
  - Image inputs: Sử dụng format `{"type": "input_image", "image_url": "..."}`

**Response:**
```json
{
  "question": "Vốn điều lệ của công ty cổ phần là gì?",
  "answer": "Vốn điều lệ của công ty cổ phần là tổng mệnh giá cổ phần đã bán hoặc được đăng ký mua khi thành lập công ty cổ phần...",
  "contexts": [
    {
      "text": "...",
      "source": "data/data.txt",
      "score": 0.85
    }
  ],
  "meta": {
    "model": "gpt-4o-mini"
  },
  "latency_ms": 1234
}
```

---

### 3. Query with File Upload

**POST** `/query/upload`

Gửi câu hỏi kèm file upload (text hoặc ảnh).

**Request (multipart/form-data):**
- `question` (string, required): Câu hỏi
- `file` (file, optional): File upload (text hoặc ảnh)
- `top_k` (integer, optional): Số lượng chunks top-K
- `use_websearch` (boolean, optional): Bật web search (chỉ hỗ trợ với OpenAI gpt-4.1)

**Response:** Tương tự như `/query`

**Ví dụ với curl:**
```bash
# Upload file text
curl -X POST "http://localhost:8000/query/upload" \
  -F "question=Vốn điều lệ là gì?" \
  -F "file=@document.txt" \
  -F "top_k=5"

# Upload ảnh với web search (yêu cầu OpenAI gpt-4.1)
curl -X POST "http://localhost:8000/query/upload" \
  -F "question=Phân tích ảnh này và tìm thông tin liên quan trên web" \
  -F "file=@image.jpg" \
  -F "use_websearch=true"
```

**Ví dụ với OpenAI gpt-4.1:**
```bash
# Query với web search
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tin tức tích cực hôm nay là gì?",
    "use_websearch": true
  }'

# Query với file URL
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Nội dung trong file này là gì?",
    "file_urls": ["https://www.example.com/document.pdf"]
  }'

# Query với image URL
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Ảnh này có gì?",
    "image_urls": ["https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"]
  }'
```

---

### 4. View Chunks

**GET** `/chunks`

Xem preview các chunks đã được build.

**Query Parameters:**
- `limit` (integer, optional): Số lượng chunks (mặc định: 3, tối đa: 50)
- `preview_chars` (integer, optional): Số ký tự preview (mặc định: 300)

**Response:**
```json
{
  "chunk_count": 1234,
  "preview_count": 3,
  "chunks": [
    {
      "id": 0,
      "source": "data/data.txt",
      "word_count": 156,
      "preview": "..."
    }
  ]
}
```

---

### 5. Rebuild Index

**POST** `/admin/rebuild_index`

Xây dựng lại index từ dữ liệu.

**Request Body:**
```json
{
  "backend": "gemma"
}
```

**Parameters:**
- `backend` (string, optional): Backend để sử dụng ("gemma", "sbert", "tfidf")

**Response:**
```json
{
  "status": "rebuilt",
  "backend": "gemma",
  "index_ready": true,
  "chunks": 1234
}
```

---

## Cấu trúc dự án

```
backend/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── rag_service.py          # RAG service với Gemma embedding
│   ├── clients/
│   │   ├── openai_client.py    # OpenAI client
│   │   └── ollama_client.py    # Ollama client
│   ├── utils/
│   │   ├── chunking.py         # Text chunking utilities
│   │   └── preprocess.py       # Text preprocessing
│   ├── config/
│   │   └── settings.json       # Configuration
│   └── final_model/            # Gemma embedding model (fine-tuned)
├── data/                        # Dữ liệu nguồn và database files
│   ├── data.txt                # Dữ liệu nguồn
│   ├── docstore.json           # Document store
│   ├── vectors.npy             # Vector embeddings
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   └── tfidf.npz               # TF-IDF matrix
└── .env                         # Environment variables
```

## Chạy server

```bash
# Từ thư mục backend/
python src/main.py --host 127.0.0.1 --port 8000

# Hoặc với uvicorn
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

## Lưu ý

1. **Model Type**: Chọn `MODEL_TYPE=openai` hoặc `MODEL_TYPE=ollama` trong `.env`
2. **API Keys**: Cần cung cấp `OPENAI_API_KEY` nếu dùng OpenAI
3. **OpenAI gpt-4.1**: 
   - Sử dụng endpoint `/v1/responses` thay vì `/chat/completions`
   - Hỗ trợ web search với `tools: [{"type": "web_search_preview"}]`
   - Hỗ trợ file và image inputs với format mới
   - Tự động detect khi model name bắt đầu bằng "gpt-4.1"
4. **Web Search**: Chỉ hoạt động với OpenAI gpt-4.1, không hỗ trợ với Ollama hoặc các model OpenAI khác
5. **File/Image URLs**: Phải là URLs công khai (public URLs), không hỗ trợ local files
6. **Index**: Index sẽ tự động build lần đầu tiên nếu chưa có

