from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from loguru import logger

from .config import get_settings
from .services.example_provider import build_provider, SentenceAIProvider
from .services.tts_provider import build_tts_provider

# 获取配置
settings = get_settings()

app = FastAPI(
    title="AudioLex AI后端服务",
    description="本地AI例句生成服务，为Android APK提供例句生成和TTS能力",
    version="1.0.0"
)

# 添加CORS中间件，允许局域网访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化例句生成provider
llm_provider: SentenceAIProvider = build_provider(settings)

# 初始化TTS提供方
tts_provider = build_tts_provider(settings)

class SentenceRequest(BaseModel):
    """例句生成请求模型"""
    word: str
    chinese_def: str
    scene_type: str = "Academic"  # Academic, Mythology, Daily
    count: int = 3

class SentenceResponse(BaseModel):
    """例句生成响应模型"""
    sentences: List[str]

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

@app.post("/generate-sentences", response_model=SentenceResponse)
async def generate_sentences(request: SentenceRequest):
    """
    生成例句接口
    
    根据单词、中文释义、场景类型生成指定数量的英文例句
    """
    try:
        logger.info(f"收到例句生成请求: {request.word} - {request.chinese_def}")
        
        # 尝试使用LLM生成
        try:
            sentences = await llm_provider.generate(
                word=request.word,
                chinese_def=request.chinese_def,
                pos="",
                scene_type=request.scene_type,
                count=request.count
            )
            logger.info(f"成功生成 {len(sentences)} 个例句 (来源: {llm_provider.name})")
        except Exception as e:
            logger.error(f"AI生成失败: {str(e)}，降级使用模板兜底")
            # 降级到模板兜底
            from .services.example_provider import TemplateProvider
            fallback = TemplateProvider()
            sentences = await fallback.generate(
                word=request.word,
                chinese_def=request.chinese_def,
                pos="",
                scene_type=request.scene_type,
                count=request.count
            )
            logger.info(f"模板兜底生成 {len(sentences)} 个例句")
        
        return SentenceResponse(sentences=sentences)
        
    except Exception as e:
        logger.error(f"例句生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"例句生成失败: {str(e)}")

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
        # 根据 TTS 提供方设置正确的媒体类型和文件扩展名
        media_type = "audio/mpeg"
        file_ext = "mp3"
        
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
            {"id": "alloy", "name": "Alloy", "description": "中性声音"},
            {"id": "echo", "name": "Echo", "description": "男性声音"},
            {"id": "fable", "name": "Fable", "description": "男性声音"},
            {"id": "onyx", "name": "Onyx", "description": "男性声音"},
            {"id": "nova", "name": "Nova", "description": "女性声音"},
            {"id": "shimmer", "name": "Shimmer", "description": "女性声音"},
        ]
    else:
        voices = []
    
    return {"voices": voices, "provider": tts_provider.name}

def run():
    """启动服务器"""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run()
