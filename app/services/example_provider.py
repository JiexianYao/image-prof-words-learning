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
from typing import List, Optional, Tuple

import httpx
from loguru import logger

from ..config import Settings


class SentenceAIProvider(ABC):
    """例句生成提供方的统一接口"""

    name: str = "base"

    @abstractmethod
    async def generate(self, word: str, chinese_def: str, pos: str, scene_type: str, count: int, custom_prompt: Optional[str] = None) -> List[str]:
        ...


class TemplateProvider(SentenceAIProvider):
    """模板兜底：不联网、零配置、结果可预测，用于占位或AI调用失败时的降级路径。

    改进：根据词性（POS）选择合适的句法模板，使兜底例句也符合词性约束。
    名词模板用作主语/宾语，动词模板用作谓语，形容词模板用作定语/表语，副词模板修饰动词等。
    """

    name = "template"

    # ── 按词性分类的模板池 ──
    # 每个模板只使用 {word} 和 {chinese_def}，{pos} 仅用于选择模板组
    POS_TEMPLATES = {
        "n.": [
            # 名词：作主语或宾语
            "The {word} of the ancient city puzzled archaeologists who sought {chinese_def}.",
            "Without sufficient {word}, the project could not achieve {chinese_def}.",
            "Her research on {word} revealed surprising connections to {chinese_def}.",
            "The professor explained how {word} relates to {chinese_def} in modern society.",
            "After years of study, he finally understood the true nature of {word} and its link to {chinese_def}.",
            "A lack of {word} often leads to unexpected challenges in achieving {chinese_def}.",
            "The committee debated the role of {word} in promoting {chinese_def}.",
            "New findings suggest that {word} plays a critical part in {chinese_def}.",
            "The conference highlighted the growing importance of {word} for {chinese_def}.",
            "Every expert in the field of {chinese_def} acknowledges the significance of {word}.",
            "The discovery of {word} changed how researchers approach {chinese_def}.",
            "Investing in {word} proved essential for long-term progress toward {chinese_def}.",
            "The concept of {word} has evolved considerably in the context of {chinese_def}.",
            "Children benefit greatly when {word} is introduced alongside {chinese_def}.",
            "The relationship between {word} and {chinese_def} remains an active area of research.",
        ],
        "v.": [
            # 动词：作谓语
            "The researchers {word} the data until they uncovered evidence of {chinese_def}.",
            "She decided to {word} the challenge despite the obstacles standing in the way of {chinese_def}.",
            "When the team began to {word} the problem, solutions for {chinese_def} became clear.",
            "He tried to {word} the system but struggled with the complexities of {chinese_def}.",
            "To achieve {chinese_def}, the engineer had to carefully {word} each component.",
            "The policy was designed to {word} the growing demand for {chinese_def}.",
            "Students learn to {word} complex issues when they explore {chinese_def}.",
            "The ability to {word} effectively is crucial for anyone pursuing {chinese_def}.",
            "Leaders who {word} with integrity inspire others to work toward {chinese_def}.",
            "In order to {word} the goal, the organization invested heavily in {chinese_def}.",
            "The experiment showed that participants could {word} the task more efficiently with {chinese_def}.",
            "She continued to {word} despite setbacks, driven by her commitment to {chinese_def}.",
            "The new method allows practitioners to {word} results faster in the domain of {chinese_def}.",
            "Critics argue that the approach fails to {word} the root causes of {chinese_def}.",
            "With proper training, anyone can learn to {word} situations involving {chinese_def}.",
        ],
        "adj.": [
            # 形容词：作定语或表语
            "The {word} landscape stretched before them, embodying the spirit of {chinese_def}.",
            "His {word} approach to the problem impressed everyone working on {chinese_def}.",
            "The result was unexpectedly {word}, far exceeding expectations for {chinese_def}.",
            "A {word} atmosphere filled the room as the team celebrated progress toward {chinese_def}.",
            "She described the {word} conditions that made {chinese_def} so challenging.",
            "The {word} nature of the phenomenon captivated researchers studying {chinese_def}.",
            "Despite the {word} circumstances, they persisted in their pursuit of {chinese_def}.",
            "The evidence appeared {word} at first, yet pointed clearly toward {chinese_def}.",
            "His writing style is remarkably {word}, especially when addressing topics like {chinese_def}.",
            "The {word} beauty of the coastline reminded visitors of {chinese_def}.",
            "The team felt {word} about the prospect of achieving {chinese_def}.",
            "The {word} design of the building reflects principles of {chinese_def}.",
            "Critics called the proposal {word}, noting it overlooked key aspects of {chinese_def}.",
            "What makes this approach particularly {word} is its direct connection to {chinese_def}.",
            "The {word} contrast between the two theories deepened the debate on {chinese_def}.",
        ],
        "adv.": [
            # 副词：修饰动词、形容词或其他副词
            "She {word} approached the task, demonstrating true mastery of {chinese_def}.",
            "The team {word} tackled the issue, producing innovative solutions for {chinese_def}.",
            "He {word} explained the connection between the concept and {chinese_def}.",
            "The system {word} improved after the updates related to {chinese_def}.",
            "They {word} debated the merits of each approach to {chinese_def}.",
            "The results {word} exceeded all predictions about {chinese_def}.",
            "The researcher {word} analyzed the data, uncovering new insights about {chinese_def}.",
            "Participants {word} responded to the stimuli, confirming the theory of {chinese_def}.",
            "The policy {word} addressed the concerns surrounding {chinese_def}.",
            "Progress was {word} slower than anticipated in the area of {chinese_def}.",
            "The audience {word} appreciated the insights on {chinese_def}.",
            "The experiment was {word} designed to test hypotheses about {chinese_def}.",
            "She {word} considered every angle before reaching conclusions about {chinese_def}.",
            "The data {word} supported the hypothesis linking the variable to {chinese_def}.",
            "He {word} navigated the complexities inherent in {chinese_def}.",
        ],
        "prep.": [
            # 介词：构成介词短语
            "The agreement was reached {word} the two parties involved in {chinese_def}.",
            "She placed the report {word} the desk before discussing {chinese_def}.",
            "The key insight emerged {word} careful observation of {chinese_def}.",
            "He stood {word} the audience and presented his findings on {chinese_def}.",
            "The results were published {word} leading journals focused on {chinese_def}.",
            "She walked {word} the library searching for resources on {chinese_def}.",
            "The decision was made {word} consideration of all factors related to {chinese_def}.",
            "The ceremony was held {word} the main hall in honor of {chinese_def}.",
            "He traveled {word} several countries to study {chinese_def}.",
            "The evidence pointed {word} a fundamental flaw in the approach to {chinese_def}.",
            "She found inspiration {word} the works of scholars who studied {chinese_def}.",
            "The project was completed {word} the deadline set for {chinese_def}.",
            "The debate continued {word} the evening, covering various aspects of {chinese_def}.",
            "He placed great emphasis {word} the practical applications of {chinese_def}.",
            "The framework was developed {word} collaboration with experts in {chinese_def}.",
        ],
        "conj.": [
            # 连词：连接分句或短语
            "She studied hard, {word} her efforts alone were not enough for {chinese_def}.",
            "The theory was elegant, {word} it failed to account for {chinese_def}.",
            "He wanted to help, {word} he lacked the expertise needed for {chinese_def}.",
            "The results were promising, {word} further research on {chinese_def} was still necessary.",
            "She pursued the goal relentlessly, {word} others doubted the feasibility of {chinese_def}.",
            "The system was efficient, {word} it still struggled with aspects of {chinese_def}.",
            "He understood the risks, {word} he proceeded with the plan for {chinese_def}.",
            "The policy aimed to reduce costs, {word} it also addressed issues of {chinese_def}.",
            "Some argued for caution, {word} others championed bold action on {chinese_def}.",
            "The experiment yielded mixed results, {word} the team remained optimistic about {chinese_def}.",
            "She recognized the opportunity, {word} she hesitated to commit fully to {chinese_def}.",
            "The model predicted growth, {word} real-world data on {chinese_def} told a different story.",
            "He had the talent, {word} he needed guidance to master {chinese_def}.",
            "The organization expanded rapidly, {word} it never lost sight of {chinese_def}.",
            "Technology advanced quickly, {word} ethical questions about {chinese_def} remained unresolved.",
        ],
        "pron.": [
            # 代词：代替名词
            "Everyone in the room understood {word} and its connection to {chinese_def}.",
            "Among the candidates, {word} demonstrated the deepest knowledge of {chinese_def}.",
            "Nobody expected {word} to have such a profound impact on {chinese_def}.",
            "Something about the way {word} related to {chinese_def} caught the researcher's attention.",
            "Each of the participants brought a unique perspective on {word} and {chinese_def}.",
            "Whatever {word} may imply, it remains central to discussions of {chinese_def}.",
            "Whoever studies {word} must also grapple with the broader context of {chinese_def}.",
            "The survey asked respondents what {word} meant to them in terms of {chinese_def}.",
            "Both researchers agreed that {word} held the key to understanding {chinese_def}.",
            "Neither side could deny the importance of {word} when discussing {chinese_def}.",
            "Someone who truly comprehends {word} will appreciate its role in {chinese_def}.",
            "Anything connected to {word} demands careful consideration in the realm of {chinese_def}.",
            "This report examines how {word} influences outcomes related to {chinese_def}.",
            "That {word} matters so much speaks volumes about the nature of {chinese_def}.",
            "Others may disagree, but {word} clearly shapes the discourse around {chinese_def}.",
        ],
    }

    # 按首字母匹配的回退映射
    _POS_KEY_MAP = {
        "n": "n.", "v": "v.", "a": "adj.", "j": "adj.",
        "d": "adv.", "r": "adv.", "p": "prep.",
        "c": "conj.", "i": "pron.", "r": "pron.",
    }

    @staticmethod
    def _resolve_pos_key(pos: str) -> str:
        """从原始词性字符串中提取最核心的词性键，用于选择模板组"""
        pos_lower = pos.strip().lower()
        if not pos_lower:
            return "n."  # 默认回退到名词
        # 复合词性如 "adj./n." → 取第一个
        first = pos_lower.split("/")[0].split(".")[0].strip()
        if first in TemplateProvider.POS_TEMPLATES:
            return first + "."
        return TemplateProvider._POS_KEY_MAP.get(first, "n.")

    async def generate(self, word: str, chinese_def: str, pos: str, scene_type: str, count: int, custom_prompt: Optional[str] = None) -> List[str]:
        pos_key = self._resolve_pos_key(pos)
        templates = self.POS_TEMPLATES.get(pos_key, self.POS_TEMPLATES["n."])
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


def _build_prompt(word: str, chinese_def: str, pos: str, scene_type: str, count: int, custom_prompt: Optional[str] = None) -> str:
    # 从TXT文件加载默认提示词模板
    prompt_file = Path(__file__).parent.parent / "templates" / "sentence_generation_prompt.txt"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            default_template = f.read()
    except FileNotFoundError:
        # 如果文件不存在，使用默认提示词
        default_template = (
            "你是一个专业的英语例句生成器，能够为每个单词构建真实、贴切的语境。\n"
            "目标单词：{word}（词性：{pos}）\n"
            "目标释义（只围绕这一个释义造句，不要涉及这个单词的其他含义）：{chinese_def}\n"
            "场景类型：{scene_type}\n"
            "请生成恰好 {count} 个英文例句，每个例句必须满足以下要求：\n"
            "1. 严禁使用任何中文，包括中文单词、中文标点、中文引号等。\n"
            "2. 严禁出现 \"the word {word}\" / \"the term {word}\" 这类讨论单词本身的句式。\n"
            "3. 严禁使用定义句式：禁止 {word} means / {word} refers to / {word} is defined as。\n"
            "4. 句子要求：构建真实场景、矛盾、目标或客观现象，将单词融入现实逻辑中，不要空洞空谈。\n"
            "5. 场景适配规则：\n"
            "   - 如果scene_type包含politics/law/sociology/philosophy：使用社科现实议题、社会治理、制度相关场景。\n"
            "   - 如果scene_type包含physics/engineering：使用自然现象、工程、客观物理场景。\n"
            "   - 如果scene_type=general：使用普通人日常生活真实场景。\n"
            "6. 严格使用给定词性{pos}，不要随意改变单词词性。\n"
            "7. 每个例句应该是一个完整的、自然的英文句子，能够帮助学习者理解单词的用法和含义。\n"
            "8. 尽量使用具体的、生动的场景，而不是抽象的概念。\n"
            "只输出一个JSON数组，数组元素是字符串，不要输出任何解释文字、前后缀或Markdown代码块标记。\n"
            '输出格式例如：["First sentence.", "Second sentence."]'
        )
    
    # 检查自定义模板是否包含必要的占位符
    required_placeholders = ['{word}', '{chinese_def}']
    if custom_prompt and custom_prompt.strip():
        template = custom_prompt.strip()
        # 检查是否包含必要的占位符
        missing_placeholders = [p for p in required_placeholders if p not in template]
        if missing_placeholders:
            logger.warning(f"自定义提示词模板缺少必要占位符 {missing_placeholders}，使用默认模板")
            template = default_template
        else:
            logger.info("使用用户自定义的提示词模板")
    else:
        template = default_template
    
    # 使用模板生成提示词
    try:
        return template.format(
            word=word,
            chinese_def=chinese_def,
            pos=pos or '未指定',
            scene_type=scene_type,
            count=count
        )
    except KeyError as e:
        # 如果模板中有其他占位符导致KeyError，使用默认模板
        logger.warning(f"自定义提示词模板格式错误: {e}，使用默认模板")
        try:
            return default_template.format(
                word=word,
                chinese_def=chinese_def,
                pos=pos or '未指定',
                scene_type=scene_type,
                count=count
            )
        except Exception:
            # 如果默认模板也失败，返回原始模板
            return template


class OpenAIProvider(SentenceAIProvider):
    """通过 OpenAI Chat Completions 兼容接口生成例句"""

    name = "openai"

    def __init__(self, settings: Settings):
        self._settings = settings

    async def generate(self, word: str, chinese_def: str, pos: str, scene_type: str, count: int, custom_prompt: Optional[str] = None) -> List[str]:
        prompt = _build_prompt(word, chinese_def, pos, scene_type, count, custom_prompt)
        # 拼接端点路径：base_url 只是 .../v1，真正的对话补全接口是 /v1/chat/completions。
        # 直接 POST 到裸 base_url 会 404（和 tts_provider 里的处理保持一致）。
        chat_url = self._settings.llm_base_url.rstrip("/")
        if not chat_url.endswith("/chat/completions"):
            chat_url = f"{chat_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout) as client:
            resp = await client.post(
                chat_url,
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