"""
TTS提供方抽象。

支持：
- OpenAI TTS：通过OpenAI Audio Speech API生成语音
- 模板兜底：返回错误提示，不消耗API额度

设计目的：
1. TTS_PROVIDER 未配置时，返回错误提示，不消耗API额度
2. 真调用云端模型时，用强约束prompt只要求输出JSON数组，生成后仍做一次结构+内容校验
3. 任何一步失败（网络错误/超时/解析失败）都不让整个请求报错，而是返回错误提示
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from loguru import logger

from ..config import Settings


class TTSProvider(ABC):
    """TTS提供方的统一接口"""

    name: str = "base"

    @abstractmethod
    async def generate(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """
        生成语音
        
        Args:
            text: 要转换为语音的文本
            voice: 语音类型（可选，覆盖默认设置）
            speed: 语速（0.25-4.0）
            
        Returns:
            音频数据（MP3格式）
        """
        ...


class OpenAITTSProvider(TTSProvider):
    """通过OpenAI Audio Speech API生成语音"""

    name = "openai"

    def __init__(self, settings: Settings):
        self._settings = settings

    async def generate(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """
        通过OpenAI API生成语音
        
        Args:
            text: 要转换为语音的文本
            voice: 语音类型（覆盖默认设置）
            speed: 语速（0.25-4.0）
            
        Returns:
            音频数据（MP3格式）
        """
        voice = voice or self._settings.tts_provider_voice
        speed = max(0.25, min(4.0, speed))  # 限制速度范围
        
        # 使用connect_timeout + read_timeout分离，避免建立连接后长时间无响应
        timeout = httpx.Timeout(
            connect=10.0,           # 连接超时10秒
            read=self._settings.tts_timeout,  # 读取超时（从配置读取，默认60秒）
            write=10.0,
            pool=10.0,
        )
        
        # 最多重试2次
        last_error = None
        # 拼接TTS端点路径：如果base_url已包含 /audio/speech 则不重复拼接
        tts_url = self._settings.tts_provider_base_url.rstrip("/")
        if not tts_url.endswith("/audio/speech"):
            tts_url = f"{tts_url}/audio/speech"
        
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        tts_url,
                        headers={
                            "Authorization": f"Bearer {self._settings.llm_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self._settings.tts_provider_model,
                            "input": text,
                            "voice": voice,
                            "speed": speed,
                            "response_format": "mp3",
                        },
                    )
                    resp.raise_for_status()
                    return resp.content
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"TTS请求超时 (尝试 {attempt + 1}/3): {str(e)}")
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1)  # 等待1秒后重试
            except httpx.HTTPStatusError as e:
                logger.error(f"TTS API返回错误: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"TTS请求异常: {str(e)}")
                raise
        
        raise last_error or Exception("TTS请求失败")


class FallbackTTSProvider(TTSProvider):
    """兜底TTS：返回错误提示，不联网"""
    
    name = "fallback"

    async def generate(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """返回错误提示音频"""
        # 这里可以返回一个预置的错误提示音频，或者抛出异常
        raise NotImplementedError("TTS未配置或不可用，请在设置中配置TTS服务商")


def build_tts_provider(settings: Settings) -> TTSProvider:
    """根据配置选择TTS提供方"""
    logger.info(f"TTS配置: provider={settings.tts_provider}, model={settings.tts_provider_model}, voice={settings.tts_provider_voice}")
    logger.info(f"TTS base_url: {settings.tts_provider_base_url}")
    
    if settings.tts_provider == "openai":
        # 检查API密钥是否配置且不是默认值
        if settings.llm_api_key and settings.llm_api_key != "default":
            logger.info("使用OpenAI TTS提供方")
            return OpenAITTSProvider(settings)
        else:
            logger.warning("TTS_PROVIDER=openai 但 LLM_API_KEY 未配置或为默认值，无法使用TTS")
    
    logger.warning("TTS未配置或配置错误，使用兜底TTS（将返回错误提示）")
    return FallbackTTSProvider()
