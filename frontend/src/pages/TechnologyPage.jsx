function TechnologyPage() {
  return (
    <div className="w-full h-full overflow-y-auto bg-gradient-to-br from-orange-50 to-orange-100">
      <div className="w-full flex justify-center px-4 py-8">
        <div className="max-w-4xl w-full">
        <h1 className="text-4xl text-center font-bold mb-8 bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-orange-600">
          Công Nghệ & Công Cụ AI
        </h1>
        
        <div className="space-y-6">
          {/* RAG Model Section */}
          <div className="card bg-white shadow-lg rounded-xl">
            <div className="card-body">
              <h2 className="card-title text-2xl text-orange-600 mb-4">
                <span className="text-3xl mr-2">🔍</span>
                RAG Model - Gemma 300M
              </h2>
              <div className="prose max-w-none">
                <p className="text-gray-700 leading-relaxed mb-4">
                  Hệ thống ViVi sử dụng mô hình <strong>Gemma 300M</strong> của Google đã được fine-tune 
                  để làm <strong>embedding encoder</strong> trong kiến trúc RAG (Retrieval-Augmented Generation). 
                  Mô hình này được chuyển đổi từ mô hình ngôn ngữ sang encoder chuyên dụng để tạo vector embeddings 
                  cho các đoạn văn bản pháp lý.
                </p>
                <div className="bg-orange-50 p-4 rounded-lg mb-4">
                  <h3 className="font-semibold text-lg mb-2 text-orange-800">Quá trình Fine-tuning:</h3>
                  <ul className="list-disc list-inside space-y-2 text-gray-700">
                    <li>Dataset chuyên biệt: Hơn <strong>5,000 mẫu dữ liệu</strong> về luật pháp Việt Nam</li>
                    <li>Mục tiêu: Fine-tune Gemma 300M thành embedding encoder cho tác vụ retrieval</li>
                    <li>Xử lý tiếng Việt: Tối ưu hóa cho ngôn ngữ tiếng Việt với các đặc thù về dấu, ngữ pháp</li>
                    <li>Domain-specific: Tập trung vào lĩnh vực MLN131 và pháp luật kinh tế</li>
                  </ul>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  Mô hình embedding này đóng vai trò quan trọng trong việc <strong>chuyển đổi văn bản thành vector</strong> 
                  và <strong>tìm kiếm semantic</strong> trong cơ sở dữ liệu vector, giúp hệ thống tìm được 
                  các đoạn văn bản pháp lý liên quan nhất với câu hỏi của người dùng.
                </p>
              </div>
            </div>
          </div>

          {/* GPT-OSS Section */}
          <div className="card bg-white shadow-lg rounded-xl">
            <div className="card-body">
              <h2 className="card-title text-2xl text-orange-600 mb-4">
                <span className="text-3xl mr-2">🤖</span>
                GPT-OSS (Open Source) - Fine-tuned Model
              </h2>
              <div className="prose max-w-none">
                <p className="text-gray-700 leading-relaxed mb-4">
                  Để tạo ra câu trả lời chính xác và tự nhiên, ViVi sử dụng mô hình <strong>GPT-OSS</strong> 
                  (GPT Open Source) đã được fine-tune chuyên sâu trên dữ liệu pháp luật Việt Nam.
                </p>
                <div className="bg-orange-50 p-4 rounded-lg mb-4">
                  <h3 className="font-semibold text-lg mb-2 text-orange-800">Nguồn dữ liệu training:</h3>
                  <ul className="list-disc list-inside space-y-2 text-gray-700">
                    <li>Thu thập từ <strong>Thư viện Pháp luật</strong> - nguồn tài liệu chính thức của Nhà nước</li>
                    <li>Dataset <strong>Harmony</strong> với định dạng <strong>analysis + final</strong> cho fine-tuning</li>
                    <li>Bao gồm các văn bản: Luật, Nghị định, Thông tư, Quyết định</li>
                    <li>Chủ đề tập trung: Kinh tế thị trường, doanh nghiệp, đầu tư, thương mại</li>
                  </ul>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg mb-4">
                  <h3 className="font-semibold text-lg mb-2 text-blue-800">Triển khai với Ollama:</h3>
                  <ul className="list-disc list-inside space-y-2 text-gray-700">
                    <li>Chuyển đổi mô hình đã fine-tune sang định dạng tương thích với <strong>Ollama</strong></li>
                    <li>Hosting trên server riêng để đảm bảo bảo mật và tốc độ phản hồi</li>
                    <li>Tối ưu hóa inference time và memory usage</li>
                    <li>Hỗ trợ xử lý ngữ cảnh dài (long context) cho các câu hỏi phức tạp</li>
                  </ul>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  Mô hình này chịu trách nhiệm <strong>tổng hợp và diễn đạt</strong> thông tin từ các đoạn 
                  văn bản được RAG model tìm thấy, tạo ra câu trả lời tự nhiên, dễ hiểu và chính xác về mặt pháp lý.
                </p>
              </div>
            </div>
          </div>

          {/* Vector Database Section */}
          <div className="card bg-white shadow-lg rounded-xl">
            <div className="card-body">
              <h2 className="card-title text-2xl text-orange-600 mb-4">
                <span className="text-3xl mr-2">💾</span>
                Vector Database & Embedding
              </h2>
              <div className="prose max-w-none">
                <p className="text-gray-700 leading-relaxed mb-4">
                  Hệ thống sử dụng <strong>Vector Database</strong> để lưu trữ và tìm kiếm semantic các 
                  đoạn văn bản pháp lý một cách hiệu quả.
                </p>
                <div className="bg-orange-50 p-4 rounded-lg mb-4">
                  <h3 className="font-semibold text-lg mb-2 text-orange-800">Công nghệ:</h3>
                  <ul className="list-disc list-inside space-y-2 text-gray-700">
                    <li>Embedding model: Mô hình embedding đã được fine-tune đặc thù cho tiếng Việt và văn bản pháp luật</li>
                    <li>Vector similarity search: Tìm kiếm dựa trên độ tương đồng ngữ nghĩa (cosine similarity)</li>
                    <li>Chunking strategy: Chia nhỏ văn bản pháp lý thành các đoạn có ý nghĩa</li>
                    <li>Indexing: Tối ưu hóa tốc độ truy vấn với index vector hiệu suất cao (FAISS hoặc scikit-learn)</li>
                  </ul>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  Khi người dùng đặt câu hỏi, hệ thống sẽ chuyển đổi câu hỏi thành vector embedding, 
                  sau đó tìm kiếm các đoạn văn bản có độ tương đồng cao nhất trong cơ sở dữ liệu.
                </p>
              </div>
            </div>
          </div>

          {/* Frontend Section */}
          <div className="card bg-white shadow-lg rounded-xl">
            <div className="card-body">
              <h2 className="card-title text-2xl text-orange-600 mb-4">
                <span className="text-3xl mr-2">🎨</span>
                Frontend Technology Stack
              </h2>
              <div className="prose max-w-none">
                <p className="text-gray-700 leading-relaxed mb-4">
                  Giao diện người dùng của ViVi được xây dựng với các công nghệ web hiện đại, 
                  đảm bảo trải nghiệm mượt mà và responsive trên mọi thiết bị.
                </p>
                <div className="grid md:grid-cols-2 gap-4 mb-4">
                  <div className="bg-orange-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-lg mb-2 text-orange-800">Core Framework:</h3>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
                      <li><strong>React 18</strong> - UI library hiện đại</li>
                      <li><strong>Vite</strong> - Build tool siêu nhanh</li>
                      <li><strong>React Router</strong> - Client-side routing</li>
                    </ul>
                  </div>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-lg mb-2 text-blue-800">Styling & UI:</h3>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
                      <li><strong>Tailwind CSS</strong> - Utility-first CSS</li>
                      <li><strong>DaisyUI</strong> - Component library</li>
                      <li><strong>React Markdown</strong> - Render markdown</li>
                    </ul>
                  </div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg mb-4">
                  <h3 className="font-semibold text-lg mb-2 text-green-800">Tính năng nổi bật:</h3>
                  <ul className="list-disc list-inside space-y-2 text-gray-700">
                    <li>Real-time chat interface với streaming response</li>
                    <li>Upload và xử lý hình ảnh trực tiếp trong chat</li>
                    <li>Lưu trữ lịch sử chat trên localStorage</li>
                    <li>Responsive design cho mobile và desktop</li>
                    <li>Dark/Light mode support (tùy chọn)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Backend Section */}
          <div className="card bg-white shadow-lg rounded-xl">
            <div className="card-body">
              <h2 className="card-title text-2xl text-orange-600 mb-4">
                <span className="text-3xl mr-2">⚙️</span>
                Backend Technology Stack
              </h2>
              <div className="prose max-w-none">
                <p className="text-gray-700 leading-relaxed mb-4">
                  Backend của ViVi được xây dựng với <strong>FastAPI</strong> - framework Python hiện đại, 
                  nhanh chóng và dễ mở rộng.
                </p>
                <div className="grid md:grid-cols-2 gap-4 mb-4">
                  <div className="bg-orange-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-lg mb-2 text-orange-800">API & Server:</h3>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
                      <li><strong>FastAPI</strong> - High-performance API framework</li>
                      <li><strong>Uvicorn</strong> - ASGI server</li>
                      <li><strong>Pydantic</strong> - Data validation</li>
                      <li><strong>CORS</strong> - Cross-origin support</li>
                    </ul>
                  </div>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-lg mb-2 text-blue-800">AI Integration:</h3>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
                      <li><strong>Ollama Client</strong> - GPT-OSS inference</li>
                      <li><strong>OpenAI API</strong> - GPT-4.1 nano cho vision</li>
                      <li><strong>Embedding Service</strong> - Vector generation</li>
                      <li><strong>RAG Pipeline</strong> - Retrieval & generation</li>
                    </ul>
                  </div>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg mb-4">
                  <h3 className="font-semibold text-lg mb-2 text-purple-800">Xử lý dữ liệu:</h3>
                  <ul className="list-disc list-inside space-y-2 text-gray-700">
                    <li>Document parsing và preprocessing</li>
                    <li>Text chunking và embedding generation</li>
                    <li>Similarity search với threshold filtering</li>
                    <li>Context ranking và selection</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Architecture Flow */}
          <div className="card bg-white shadow-lg rounded-xl">
            <div className="card-body">
              <h2 className="card-title text-2xl text-orange-600 mb-4">
                <span className="text-3xl mr-2">🔄</span>
                Kiến Trúc Hệ Thống
              </h2>
              <div className="prose max-w-none">
                <div className="bg-gradient-to-r from-orange-50 to-blue-50 p-6 rounded-lg">
                  <div className="space-y-4">
                    <div className="flex items-start">
                      <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                      <div className="ml-4">
                        <h4 className="font-semibold text-lg">Người dùng gửi câu hỏi</h4>
                        <p className="text-gray-700 text-sm">Câu hỏi được gửi từ frontend đến backend API</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                      <div className="ml-4">
                        <h4 className="font-semibold text-lg">Embedding & Vector Search</h4>
                        <p className="text-gray-700 text-sm">Câu hỏi được chuyển thành vector và tìm kiếm trong Vector DB</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                      <div className="ml-4">
                        <h4 className="font-semibold text-lg">Context Retrieval</h4>
                        <p className="text-gray-700 text-sm">Embedding model (Gemma 300M fine-tuned encoder) tìm kiếm và trích xuất các đoạn văn bản liên quan từ Vector DB</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                      <div className="ml-4">
                        <h4 className="font-semibold text-lg">Answer Generation</h4>
                        <p className="text-gray-700 text-sm">GPT-OSS (Ollama) tạo câu trả lời dựa trên context và câu hỏi</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                      <div className="ml-4">
                        <h4 className="font-semibold text-lg">Response & Streaming</h4>
                        <p className="text-gray-700 text-sm">Câu trả lời được stream về frontend và hiển thị real-time</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
    </div>
  );
}

export default TechnologyPage;

