// chatbotService.js
const API_BASE_URL = 'http://127.0.0.1:2000';

// 发送消息（不带文件）
export const sendMessageChatService = async (promptInput, model, topK = null) => {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      body: JSON.stringify({
        question: promptInput,
        top_k: topK,
        image_urls: null,
        file_urls: null,
        use_websearch: false
      }),
      headers: {
        "Content-Type": "application/json"
      },
    });
    
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    
    const result = await response.json();
    // 转换响应格式以匹配前端期望的格式
    return {
      result: result.answer,
      source_documents: result.contexts || [],
      references: result.contexts?.map(ctx => ctx.source) || [],
      meta: result.meta,
      latency_ms: result.latency_ms
    };
};

// 发送消息（带文件/图片上传）
export const sendMessageWithFileService = async (promptInput, file, topK = null, useWebsearch = false) => {
    const formData = new FormData();
    formData.append('question', promptInput);
    if (file) {
      formData.append('file', file);
    }
    if (topK !== null) {
      formData.append('top_k', topK.toString());
    }
    formData.append('use_websearch', useWebsearch.toString());
    
    const response = await fetch(`${API_BASE_URL}/query/upload`, {
      method: "POST",
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    
    const result = await response.json();
    // 转换响应格式以匹配前端期望的格式
    return {
      result: result.answer,
      source_documents: result.contexts || [],
      references: result.contexts?.map(ctx => ctx.source) || [],
      meta: result.meta,
      latency_ms: result.latency_ms
    };
};