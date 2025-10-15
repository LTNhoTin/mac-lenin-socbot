import json
import sys
import time
from pathlib import Path

import requests


DEFAULT_URL = "http://127.0.0.1:8000"


def ask(url: str, question: str, top_k: int = 5) -> dict:
    api = url.rstrip("/") + "/query"
    payload = {"question": question, "top_k": top_k}
    r = requests.post(api, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    url = DEFAULT_URL
    if len(sys.argv) > 1:
        # Cho phép truyền URL server: python chat.py http://127.0.0.1:8000
        url = sys.argv[1]

    print("MLN131 RAG Chatbot (CLI) — nhập câu hỏi, gõ /exit để thoát.")
    print(f"Server URL: {url}")
    # Ping health
    try:
        health = requests.get(url.rstrip("/") + "/health", timeout=10).json()
        if not health.get("index_ready"):
            print("[Cảnh báo] Index chưa sẵn sàng, lần gọi đầu có thể chậm do server sẽ tự build.")
    except Exception as e:
        print(f"[Lỗi] Không thể kết nối server tại {url} — {e}")
        print("Vui lòng khởi chạy server trước: uvicorn app.main:app --reload --port 8000 --host 127.0.0.1")
        return

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in {"/exit", "exit", ":q", "quit"}:
            print("Bye!")
            break
        try:
            t0 = time.perf_counter()
            resp = ask(url, q)
            dt = int((time.perf_counter() - t0) * 1000)
            print("-----------------------")
            print(resp.get("answer", "(không có trả lời)"))
            print(f"\n[latency] {dt} ms")
            ctxs = resp.get("contexts", [])
            if ctxs:
                print("[contexts]")
                for i, c in enumerate(ctxs, start=1):
                    src = c.get("source", "unknown")
                    score = c.get("score")
                    print(f"  {i}. {src} (score={score:.3f})")
            print("-----------------------")
        except Exception as e:
            print(f"[Lỗi] {e}")


if __name__ == "__main__":
    main()