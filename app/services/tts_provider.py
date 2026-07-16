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
        
        async with httpx.AsyncClient(timeout=self._settings.tts_timeout) as client:
            resp = await client.post(
                self._settings.tts_provider_base_url,
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


class FallbackTTSProvider(TTSProvider):
    """兜底TTS：返回错误提示，不联网"""
    
    name = "fallback"

    async def generate(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """返回错误提示音频"""
        # 这里可以返回一个预置的错误提示音频，或者抛出异常
        raise NotImplementedError("TTS未配置或不可用，请在设置中配置TTS服务商")


def build_tts_provider(settings: Settings) -> TTSProvider:
    """根据配置选择TTS提供方"""
    if settings.tts_provider == "openai":
        if settings.llm_api_key:
            return OpenAITTSProvider(settings)
        logger.warning("TTS_PROVIDER=openai 但 LLM_API_KEY 未配置")
    
    return FallbackTTSProvider()
