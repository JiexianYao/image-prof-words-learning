from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
import httpx
import uvicorn
from loguru import logger

from .config import get_settings
from .services.tts_provider import build_tts_provider

# 获取配置
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动和关闭时的资源清理"""
    logger.info("服务启动中...")
    yield
    logger.info("服务关闭中，清理资源...")

app = FastAPI(
    title="AudioLex AI后端服务",
    description="本地AI对话服务，为Android APK提供单词学习对话和TTS能力",
    version="2.0.0",
    lifespan=lifespan,
)

# 添加CORS中间件，允许局域网访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化TTS提供方
tts_provider = build_tts_provider(settings)


class ChatMessage(BaseModel):
    """对话消息"""
    role: str  # "user" 或 "assistant"
    content: str


class ChatRequest(BaseModel):
    """单词对话请求模型"""
    word: str
    chinese_def: str
    pos: str = ""  # 词性，如 "n." "v." "adj."
    messages: List[ChatMessage]  # 对话历史


class ChatResponse(BaseModel):
    """单词对话响应模型"""
    reply: str


class TTSRequest(BaseModel):
    """TTS请求模型"""
    text: str
    voice: Optional[str] = None  # alloy, echo, fable, onyx, nova, shimmer
    speed: float = 1.0  # 0.25-4.0


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy", 
        "service": "AudioLex AI Backend",
        "tts_provider": tts_provider.name
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    单词学习对话接口
    
    用户可以围绕一个单词持续提问，支持多轮对话。
    """
    try:
        logger.info(f"收到对话请求: {request.word} - {request.chinese_def}, 消息数: {len(request.messages)}")
        
        # 构建系统提示词
        system_prompt = (
            f"你是一个专业的英语学习助手，擅长为单词造句和解释用法。\n\n"
            f"当前正在学习的单词信息：\n"
            f"- 单词：{request.word}\n"
            f"- 词性：{request.pos or '未指定'}\n"
            f"- 中文释义：{request.chinese_def}\n\n"
            f"请围绕这个单词回答学生的问题。要求：\n"
            f"1. 用自然、地道的英语回答，示例句子要符合真实语境。\n"
            f"2. 如果学生请求造句，请构造具体、场景化的句子，不要使用模板化的泛泛表达。\n"
            f"3. 可以使用相关领域的词汇和搭配来丰富语境。\n"
            f"4. 除非学生明确要求中文解释，否则请用英语回答。\n"
            f"5. 句子要展示该单词的常见搭配和真实用法。\n"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # 调用LLM
        chat_url = settings.llm_base_url.rstrip("/")
        if not chat_url.endswith("/chat/completions"):
            chat_url = f"{chat_url}/chat/completions"
        
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            resp = await client.post(
                chat_url,
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.llm_provider_model,
                    "messages": messages,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            reply = payload["choices"][0]["message"]["content"]
        
        logger.info(f"对话回复成功: {request.word}, 回复长度: {len(reply)}")
        return ChatResponse(reply=reply)
        
    except httpx.TimeoutException:
        logger.error(f"LLM请求超时: {request.word}")
        raise HTTPException(status_code=504, detail="LLM请求超时，请稍后重试")
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM请求HTTP错误: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail=f"LLM服务返回错误: {e.response.status_code}")
    except Exception as e:
        logger.error(f"对话生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    文本转语音接口
    
    根据文本生成语音，返回MP3音频数据
    """
    try:
        logger.info(f"收到TTS请求: {request.text[:50]}...")
        
        audio_data = await tts_provider.generate(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        )
        
        logger.info(f"成功生成TTS音频，大小: {len(audio_data)} 字节")
        media_type = "audio/wav"
        file_ext = "wav"
        
        return Response(
            content=audio_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"tts_audio.{file_ext}\""
            }
        )
        
    except NotImplementedError as e:
        logger.warning(f"TTS未配置: {str(e)}")
        raise HTTPException(status_code=503, detail=f"TTS服务不可用: {str(e)}")
    except Exception as e:
        logger.error(f"TTS生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS生成失败: {str(e)}")


@app.get("/tts/voices")
async def list_tts_voices():
    """
    获取可用的TTS语音列表
    """
    if tts_provider.name == "openai":
        voices = [
            {
                "id": settings.tts_provider_voice,
                "name": settings.tts_provider_voice,
                "description": "当前配置的默认音色（完整音色库见MiMo文档）",
            },
        ]
    else:
        voices = []

    return {"voices": voices, "provider": tts_provider.name}

def run():
    """启动服务器"""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )

if __name__ == "__main__":
    run()
