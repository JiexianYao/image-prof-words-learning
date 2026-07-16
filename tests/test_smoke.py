"""冒烟测试 - 验证基本功能"""
import pytest
from app.config import Settings


def test_settings_load():
    """测试配置加载"""
    settings = Settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000


def test_llm_config():
    """测试LLM配置"""
    settings = Settings()
    assert settings.llm_provider_model == "mimo-v2.5-pro"
    assert settings.llm_api_key == "default"
    assert settings.llm_base_url == "https://api.xiaomimimo.com/v1"
    assert settings.llm_timeout == 15.0


def test_tts_config():
    """测试TTS配置"""
    settings = Settings()
    assert settings.tts_provider_model == "mimo-v2.5-tts"
    assert settings.tts_provider_voice == "mimo_default"
    assert settings.tts_provider_base_url == "https://api.xiaomimimo.com/v1"
    assert settings.tts_timeout == 15.0