"""
应用配置

集中管理需要跨模块使用的配置项：AI 提供方选择、API Key、局域网监听地址等。
此前这些配置分散硬编码在 main.py / Dockerfile / docker-compose.yml 里，改一处要跟着改好几处；
现在统一从环境变量 / image/.env 文件读取，具体可配置项参见 .env.example。
"""
from functools import lru_cache
from typing import List, Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 服务监听
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "*"  # 逗号分隔；局域网demo默认放开，需要收紧时改成具体网段
    reload: bool = False  # 生产环境应设为False，开发环境可设为True

    # LLM配置
    llm_provider_model: str = "mimo-v2.5-pro"
    llm_api_key: str = "default"
    llm_base_url: str = "https://api.xiaomimimo.com/v1"
    llm_timeout: float = 15.0

    # TTS配置
    tts_provider: Literal["none", "openai"] = "none"
    tts_provider_model: str = "mimo-v2.5-tts"
    tts_provider_voice: str = "mimo_default"
    tts_provider_base_url: str = "https://api.xiaomimimo.com/v1"
    tts_timeout: float = 15.0

    # 腾讯云COS配置
    cos_secret_id: Optional[str] = None
    cos_secret_key: Optional[str] = None
    cos_bucket_name: Optional[str] = None
    cos_region: str = "ap-guangzhou"  # 默认区域

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
