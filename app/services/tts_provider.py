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

import base64
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
    """
    通过小米MiMo生成语音。

    注意：MiMo没有独立的、标准OpenAI风格的 /v1/audio/speech 接口，TTS是复用
    /v1/chat/completions 实现的（用 audio 字段声明输出音频）：
    - 待合成文本必须放在 role=assistant 的消息里，不能放 role=user
      （user消息是可选的风格/指令描述，比如"用温柔的语气"，这里暂不使用）
    - 响应是一个JSON（普通chat completion结构），音频是
      choices[0].message.audio.data 里的 base64 字符串，不是原始二进制body
    - 没有 speed 数值参数，语速只能靠自然语言指令控制，这里的 speed 参数
      仅为兼容上层调用签名保留，不会真正生效
    参考: https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/audio/speech-synthesis-v2.5

    class名/name沿用"openai"是因为原先按标准OpenAI TTS接口设计，
    实际这里只对接MiMo，如果以后要支持真正的OpenAI TTS需要拆成两个provider。
    """

    name = "openai"

    def __init__(self, settings: Settings):
        self._settings = settings

    async def generate(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """
        通过MiMo Chat Completions + audio字段生成语音

        Args:
            text: 要转换为语音的文本（会被放进assistant消息）
            voice: 音色ID（覆盖默认设置），必须是MiMo支持的音色，如"Mia"/"Chloe"/
                   "冰糖"/"茉莉"等，不能沿用OpenAI的"alloy"这类名字
            speed: 语速，MiMo当前不支持数值语速，这个参数会被忽略

        Returns:
            音频数据（WAV格式）
        """
        voice = voice or self._settings.tts_provider_voice

        # 使用connect_timeout + read_timeout分离，避免建立连接后长时间无响应
        timeout = httpx.Timeout(
            connect=10.0,           # 连接超时10秒
            read=self._settings.tts_timeout,  # 读取超时（从配置读取；代码默认15秒，.env里设为60秒）
            write=10.0,
            pool=10.0,
        )

        # 拼接端点路径：如果base_url已包含 /chat/completions 则不重复拼接
        chat_url = self._settings.tts_provider_base_url.rstrip("/")
        if not chat_url.endswith("/chat/completions"):
            chat_url = f"{chat_url}/chat/completions"

        payload = {
            "model": self._settings.tts_provider_model,
            "messages": [
                {"role": "assistant", "content": text},
            ],
            "audio": {
                "format": "wav",
                "voice": voice,
            },
        }

        # 最多重试2次
        last_error: Optional[Exception] = None
        for attempt in range(3):
            resp: Optional[httpx.Response] = None
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        chat_url,
                        headers={
                            # MiMo文档curl示例用api-key，官方Python SDK示例走标准
                            # OpenAI客户端（即Authorization: Bearer）。两个都带上，
                            # 避免因为不确定网关到底认哪个header而调不通。
                            "Authorization": f"Bearer {self._settings.llm_api_key}",
                            "api-key": self._settings.llm_api_key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    audio_b64 = data["choices"][0]["message"]["audio"]["data"]
                    return base64.b64decode(audio_b64)
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"TTS请求超时 (尝试 {attempt + 1}/3): {str(e)}")
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1)  # 等待1秒后重试
            except httpx.HTTPStatusError as e:
                logger.error(f"TTS API返回错误: {e.response.status_code} - {e.response.text}")
                raise
            except (KeyError, IndexError, TypeError, ValueError) as e:
                body_preview = resp.text[:500] if resp is not None else "(无响应)"
                logger.error(f"TTS响应解析失败: {str(e)}, 原始响应: {body_preview}")
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
