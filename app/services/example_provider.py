"""
例句生成提供方抽象层。

本模块为 sentence_generator 提供底层AI调用能力，支持：
- TemplateProvider：零配置可用的模板兜底，不联网、不消耗任何 API 额度。
- OpenAIProvider：调用 OpenAI Chat Completions 兼容接口。

设计目的：
1. LLM_PROVIDER 未配置 / 对应 API Key 缺失时，自动退回模板，保证服务零配置也能跑通、
   不会因为没填key就直接报错。
2. 真调用云端模型时，用强约束 prompt 只要求输出 JSON 数组，生成后仍做一次结构+内容校验——
   "约束生成能大幅降低幻觉率"不等于"约束生成=零幻觉"，格式对了不代表内容一定对，
   所以这里至少做最基本的兜底检查（数量对不对、句子里有没有出现目标单词）。
3. 任何一步失败（网络错误/超时/解析失败/校验不通过）都不让整个请求报错，而是降级回模板，
   并把 source 字段如实标注给调用方，让上层知道这批例句到底是不是真AI产出。
"""
from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import httpx
from loguru import logger

from ..config import Settings


class SentenceAIProvider(ABC):
    """例句生成提供方的统一接口"""

    name: str = "base"

    @abstractmethod
    async def generate(self, word: str, chinese_def: str, pos: str, scene_type: str, count: int) -> List[str]:
        ...


class TemplateProvider(SentenceAIProvider):
    """模板兜底：不联网、零配置、结果可预测，用于占位或AI调用失败时的降级路径"""

    name = "template"

    SCENE_TEMPLATES = {
        "Academic": [
            "The {word} is commonly used in academic research.",
            "In scholarly literature, {word} refers to {chinese_def}.",
            "Academic papers frequently employ the term {word}.",
            "The concept of {word} is fundamental in {chinese_def}.",
            "Researchers often encounter {word} in their studies.",
            "The definition of {word} varies across disciplines.",
            "In the context of {chinese_def}, {word} plays a crucial role.",
            "Academic discourse often revolves around {word}.",
            "The term {word} has evolved to mean {chinese_def}.",
            "Understanding {word} is essential for academic success.",
        ],
        "Mythology": [
            "In Greek mythology, {word} is associated with {chinese_def}.",
            "The ancient Greeks believed in the power of {word}.",
            "Mythological stories often feature {word} as a central theme.",
            "The concept of {word} appears in Homer's epics.",
            "Ancient myths describe {word} as {chinese_def}.",
            "The mythology surrounding {word} is complex and fascinating.",
            "In epic poetry, {word} symbolizes {chinese_def}.",
            "The gods and heroes of mythology often invoked {word}.",
            "Ancient texts reference {word} in the context of {chinese_def}.",
            "The cultural significance of {word} extends beyond mythology.",
        ],
        "Daily": [
            "In everyday conversation, {word} is often used to describe {chinese_def}.",
            "People commonly use {word} when discussing {chinese_def}.",
            "The word {word} is frequently encountered in daily life.",
            "When we talk about {chinese_def}, we often use {word}.",
            "In modern usage, {word} means {chinese_def}.",
            "The term {word} has become part of everyday language.",
            "We often encounter {word} in various contexts related to {chinese_def}.",
            "The practical application of {word} involves {chinese_def}.",
            "In contemporary English, {word} refers to {chinese_def}.",
            "The word {word} is versatile and can mean {chinese_def}.",
        ],
    }

    async def generate(self, word: str, chinese_def: str, pos: str, scene_type: str, count: int) -> List[str]:
        templates = self.SCENE_TEMPLATES.get(scene_type, self.SCENE_TEMPLATES["Academic"])
        # 用可重复抽样的 random.choices 代替 random.sample：
        # 此前 random.sample(templates, min(count, len(templates))) 在 count 超过模板池大小时
        # 会不报错地静默返回更少的句子，与接口"返回count条"的约定不符。
        chosen = random.choices(templates, k=count)
        return [t.format(word=word, chinese_def=chinese_def) for t in chosen]


def _extract_json_array(raw_text: str) -> List[str]:
    """从模型输出里稳健地摘出JSON数组，容忍模型偶尔在外面包一层```json代码块"""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("模型输出不是JSON数组")
    if not all(isinstance(item, str) for item in data):
        raise ValueError("JSON数组里存在非字符串元素")
    return data


def _sanity_check(sentences: List[str], word: str, count: int) -> Tuple[bool, str]:
    """
    对AI输出做最基本的内容校验。格式约束只能防"结构错"（是不是合法JSON），
    防不了"内容错"（是不是真的围绕目标单词/释义生成的）。这不是幻觉检测，
    只是一个便宜的兜底闸门：数量不对、或者句子里完全没出现目标单词，就判定不通过。
    """
    if len(sentences) != count:
        return False, f"期望{count}句，实际{len(sentences)}句"
    word_stem = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if word_stem.endswith(suffix) and len(word_stem) > len(suffix) + 2:
            word_stem = word_stem[: -len(suffix)]
            break
    for s in sentences:
        if not s.strip():
            return False, "存在空句子"
        if word_stem and word_stem not in s.lower():
            return False, f"句子疑似没有围绕目标单词生成: {s!r}"
    return True, "ok"


def _build_prompt(word: str, chinese_def: str, pos: str, scene_type: str, count: int) -> str:
    # 从TXT文件加载提示词模板
    prompt_file = Path(__file__).parent.parent / "templates" / "sentence_generation_prompt.txt"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        # 如果文件不存在，使用默认提示词
        template = (
            "你是一个严格遵守输出格式的英语例句生成器。\n"
            "目标单词：{word}（词性：{pos or '未指定'}）\n"
            "目标释义（只围绕这一个释义造句，不要涉及这个单词的其他含义）：{chinese_def}\n"
            "场景标签：{scene_type}\n"
            "请生成恰好 {count} 个英文例句，每句都必须包含单词「{word}」本身或其常见词形变化，"
            "且语境要匹配上面给出的释义和场景标签。\n"
            "只输出一个JSON数组，数组元素是字符串，不要输出任何解释文字、前后缀或Markdown代码块标记。\n"
            '输出格式例如：["First sentence.", "Second sentence."]'
        )
    
    # 使用模板生成提示词
    return template.format(
        word=word,
        chinese_def=chinese_def,
        pos=pos or '未指定',
        scene_type=scene_type,
        count=count
    )


class OpenAIProvider(SentenceAIProvider):
    """通过 OpenAI Chat Completions 兼容接口生成例句"""

    name = "openai"

    def __init__(self, settings: Settings):
        self._settings = settings

    async def generate(self, word: str, chinese_def: str, pos: str, scene_type: str, count: int) -> List[str]:
        prompt = _build_prompt(word, chinese_def, pos, scene_type, count)
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout) as client:
            resp = await client.post(
                self._settings.llm_base_url,
                headers={
                    "Authorization": f"Bearer {self._settings.llm_api_key}",
                    "content-type": "application/json",
                },
                json={
                    "model": self._settings.llm_provider_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            raw_text = payload["choices"][0]["message"]["content"]

        sentences = _extract_json_array(raw_text)
        ok, reason = _sanity_check(sentences, word, count)
        if not ok:
            raise ValueError(f"OpenAI输出未通过内容校验: {reason}")
        return sentences


def build_provider(settings: Settings) -> SentenceAIProvider:
    """根据配置选择provider；缺key或未配置时明确告警并退回模板，而不是静默失败"""
    logger.info(f"LLM配置: model={settings.llm_provider_model}, base_url={settings.llm_base_url}")
    
    # 检查API密钥是否配置且不是默认值
    if settings.llm_api_key and settings.llm_api_key != "default" and settings.llm_provider_model:
        logger.info("使用OpenAI LLM提供方")
        return OpenAIProvider(settings)
    else:
        logger.warning("LLM_API_KEY/LLM_PROVIDER_MODEL 未配置完整或为默认值，退回模板兜底")
        return TemplateProvider()