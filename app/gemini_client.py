import os
import re
from typing import List, Dict, Tuple

import google.generativeai as genai


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash-latest",
        response_language: str = "vi",
        max_output_tokens: int = 150,
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.response_language = response_language
        self.generation_config = {
            "temperature": float(temperature),
            "max_output_tokens": int(max_output_tokens),
        }
        genai.configure(api_key=api_key)
        # Chuẩn bị danh sách model fallback để đảm bảo tương thích phiên bản API
        self._fallback_models = [
            model_name,
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-pro",
        ]
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception:
            # Nếu tạo model thất bại, thử fallback nhanh
            self.model = genai.GenerativeModel("gemini-1.5-flash-latest")
            self.model_name = "gemini-1.5-flash-latest"

    def build_prompt(self, question: str, contexts: List[Dict]) -> List[str]:
        """Xây dựng prompt theo hai chế độ: có ngữ cảnh (RAG) và không có ngữ cảnh (open-domain)."""
        # Có ngữ cảnh RAG: ưu tiên trả lời theo ngữ cảnh, ngắn gọn
        if contexts and len(contexts) > 0:
            instruction = (
                f"Bạn là trợ lý AI trả lời bằng tiếng {self.response_language}, giọng tự nhiên và mạch lạc. "
                "Trả lời trực tiếp, rõ ràng và ngắn gọn dựa trên các ngữ cảnh được cung cấp. "
                "Không phân tích hay suy đoán ngoài ngữ cảnh, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'. "
                "Không nhắc tới 'Context 1/2' hay 'Source: Context X' trong câu trả lời, và không trích nguồn trừ khi người dùng yêu cầu. "
                "Chỉ dùng bullet khi người dùng yêu cầu; mặc định hãy viết thành một đoạn hoặc vài câu liên kết."
            )
            context_texts = []
            for c in contexts:
                txt = c.get("text", "")
                if not txt:
                    continue
                context_texts.append(txt)
            context_block = "\n\n".join(context_texts)
            return [instruction, "Ngữ cảnh:\n" + context_block, f"\n\nCâu hỏi: {question}"]

        # Không có ngữ cảnh RAG: trả lời ngắn gọn theo kiến thức chung (chit-chat, open-domain)
        instruction = (
            f"Bạn là trợ lý AI trả lời bằng tiếng {self.response_language}, giọng thân thiện và ngắn gọn. "
            "Trả lời trực tiếp dựa trên kiến thức chung của bạn. "
            "Nếu câu hỏi yêu cầu thông tin hoặc trích dẫn từ tài liệu cụ thể, hãy nói rằng hiện không có dữ liệu tài liệu để trích dẫn, nhưng vẫn giải thích ngắn gọn theo hiểu biết chung. "
            "Không trích nguồn, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'."
        )
        return [instruction, f"Câu hỏi: {question}"]

    def _post_process(self, text: str) -> str:
        """Làm sạch các cụm tham chiếu không tự nhiên nếu mô hình vẫn lỡ chèn.

        Xoá các chuỗi như: "Source: Context 2", "Nguồn: Context 3", các dấu [Context ...].
        """
        if not text:
            return text
        patterns = [
            r"Source:\s*Context\s*\d+",
            r"Nguồn:\s*Context\s*\d+",
            r"\[\s*Context\s*\d+\s*\|[^\]]*\]",
            r"\[\s*Context\s*\d+\s*\]",
            r"^\s*(Theo\s+ngữ\s+cảnh\s+được\s+cung\s+cấp[^\n]*|Dựa\s+trên\s+ngữ\s+cảnh[^\n]*)",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)
        # Thu gọn khoảng trắng thừa sau khi xoá
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def answer(self, question: str, contexts: List[Dict]) -> Tuple[str, Dict]:
        prompt = self.build_prompt(question, contexts)
        last_error = None
        # Thử gọi lần lượt theo danh sách fallback để tránh lỗi "model not found"
        for name in self._fallback_models:
            try:
                model = self.model if name == self.model_name else genai.GenerativeModel(name)
                resp = model.generate_content(prompt, generation_config=self.generation_config)
                text = resp.text if hasattr(resp, "text") else str(resp)
                text = self._post_process(text)
                # Nếu thành công, cập nhật model hiện dùng
                self.model = model
                self.model_name = name
                return text, {"model": name}
            except Exception as e:
                last_error = str(e)
                continue
        # Nếu vẫn lỗi, thử động lấy danh sách model khả dụng từ API và chọn model có hỗ trợ generateContent
        try:
            available = []
            for m in genai.list_models():
                methods = getattr(m, "supported_generation_methods", []) or []
                name = getattr(m, "name", None)
                if name and ("generateContent" in methods or "generate_text" in methods):
                    available.append(name)
            for name in available:
                try:
                    model = genai.GenerativeModel(name)
                    resp = model.generate_content(prompt, generation_config=self.generation_config)
                    text = resp.text if hasattr(resp, "text") else str(resp)
                    text = self._post_process(text)
                    self.model = model
                    self.model_name = name
                    return text, {"model": name}
                except Exception as e:
                    last_error = str(e)
                    continue
        except Exception as e:
            last_error = str(e)
        return f"Lỗi gọi Gemini: {last_error}", {"model": self.model_name, "error": last_error}