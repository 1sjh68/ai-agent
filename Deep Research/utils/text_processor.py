# utils/text_processor.py

import logging
import re
import hashlib
import json
from collections import OrderedDict

# 从重构后的模块中导入依赖
from config import Config
from services.llm_interaction import call_ai

def consolidate_document_structure(final_markdown_content: str) -> str:
    """
    (V3 - 健壮修复版) 对最终生成的Markdown文档进行结构性整合和清理。
    """
    logging.info("--- 开始执行最终文档结构整合 (V3 - 健壮修复版) ---")

    if not final_markdown_content or not final_markdown_content.strip():
        return ""

    sections = re.split(r'(?m)(^#+ .*)', final_markdown_content)
    
    intro_content = sections[0].strip()
    
    consolidated_sections = OrderedDict()

    for i in range(1, len(sections), 2):
        title_line = sections[i].strip()
        content = sections[i+1].strip() if (i + 1) < len(sections) else ""
        
        title_text_only = re.sub(r'^#+\s*', '', title_line).strip()
        clean_title = re.sub(r'\s*\s*$', '', title_text_only).strip()

        if clean_title in consolidated_sections:
            logging.warning(f"发现重复章节: '{clean_title}'。将使用最新版本覆盖旧版本。")
        
        consolidated_sections[clean_title] = (title_line, content)

    final_content_parts = []
    if intro_content:
        final_content_parts.append(intro_content)

    for original_title, content in consolidated_sections.values():
        final_content_parts.append(original_title)
        final_content_parts.append(content)

    final_document = "\n\n".join(part for part in final_content_parts if part)
    logging.info("--- 文档结构整合完成 ---")
    
    return final_document

def truncate_text_for_context(config: Config, text: str, max_tokens: int, truncation_style: str = "middle") -> str:
    """
    根据 token 数量安全地截断文本，以适应模型的上下文窗口。
    """
    if not text: 
        return ""
        
    if not config.encoder:
        logging.warning("Tiktoken 编码器不可用，将使用基于字符的近似截断。")
        char_limit = max_tokens * 3
        if len(text) <= char_limit:
            return text
        logging.info(f"    - 正在截断文本: {len(text)} chars -> {char_limit} chars (方式: {truncation_style})")
        if truncation_style == "head": return text[:char_limit] + "\n... [内容已截断] ..."
        if truncation_style == "tail": return "... [内容已截断] ...\n" + text[-char_limit:]
        half = char_limit // 2
        return text[:half] + "\n... [中间内容已截断] ...\n" + text[-half:]

    tokens = config.encoder.encode(text)
    if len(tokens) <= max_tokens: 
        return text
    
    logging.info(f"    - 正在截断文本: {len(tokens)} tokens -> {max_tokens} tokens (方式: {truncation_style})")
    
    decode_fn = config.encoder.decode
    if truncation_style == "head":
        truncated_tokens = tokens[:max_tokens]
        return decode_fn(truncated_tokens) + "\n... [内容已截断，只显示开头部分] ..."
    elif truncation_style == "tail":
        truncated_tokens = tokens[-max_tokens:]
        return "... [内容已截断，只显示结尾部分] ...\n" + decode_fn(truncated_tokens)
    else:  # middle
        h_len = max_tokens // 2
        t_len = max_tokens - h_len
        head_part = decode_fn(tokens[:h_len])
        tail_part = decode_fn(tokens[-t_len:])
        return head_part + "\n... [中间内容已截断] ...\n" + tail_part

def calculate_checksum(data: str) -> str:
    """计算字符串的 SHA256 校验和，用于比较内容是否有变化。"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def preprocess_json_string(json_string: str) -> str:
    """
    应用一系列正则表达式修复常见的 LLM 生成的 JSON 错误。
    """
    if not json_string or json_string.isspace():
        return ""
    processed_string = json_string.strip()
    processed_string = re.sub(r"//.*", "", processed_string)
    processed_string = re.sub(r"/\*[\s\S]*?\*/", "", processed_string, flags=re.MULTILINE)
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', processed_string, re.DOTALL)
    if match:
        processed_string = match.group(1).strip()
    processed_string = re.sub(r",\s*([}\]])", r"\1", processed_string)
    processed_string = re.sub(r"\bTrue\b", "true", processed_string)
    processed_string = re.sub(r"\bFalse\b", "false", processed_string)
    processed_string = re.sub(r"\bNone\b", "null", processed_string)
    return processed_string

def extract_json_from_ai_response(config: Config, response_text: str, context_for_error_log: str = "AI response") -> str | None:
    """
    使用“三振出局”策略从 AI 的文本响应中稳健地提取 JSON 字符串。
    策略: 1. 直接解析 -> 2. 正则表达式预处理后解析 -> 3. 调用 AI 修复后解析
    """
    logging.debug(f"尝试从以下内容提取JSON: {response_text[:300]}... 上下文: {context_for_error_log}")

    def _try_parse(s_to_parse, stage_msg):
        if not s_to_parse or s_to_parse.isspace():
            return None
        try:
            json.loads(s_to_parse)
            logging.info(f"  JSON 在 {stage_msg} 阶段解析成功。")
            return s_to_parse
        except json.JSONDecodeError:
            logging.debug(f"  JSON 在 {stage_msg} 阶段解析失败。")
            return None

    if (parsed_str := _try_parse(response_text, "直接解析")) is not None:
        return parsed_str

    pre_repaired_str = preprocess_json_string(response_text)
    if (parsed_str := _try_parse(pre_repaired_str, "正则预处理")) is not None:
        return parsed_str

    logging.info(f"  JSON 解析在预处理后仍然失败，尝试调用 AI 修复...")
    fixer_prompt = (
        "The following text is supposed to be a valid JSON string, but it's malformed. "
        "Please fix it and return ONLY the corrected, valid JSON string. "
        "Do not add any explanations, apologies, or markdown formatting like ```json ... ```.\n\n"
        f"Malformed JSON attempt:\n```\n{pre_repaired_str}\n```\n\nCorrected JSON string:"
    )
    
    ai_fixed_str = call_ai(
        config, 
        config.json_fixer_model_name,
        [{"role": "user", "content": fixer_prompt}],
        max_tokens_output=max(2048, int(len(pre_repaired_str) * 1.5)),
        temperature=0.0
    )

    if "AI模型调用失败" in ai_fixed_str or not ai_fixed_str.strip():
        logging.error(f"  AI JSON 修复调用失败或返回空。")
        return None

    final_attempt_str = preprocess_json_string(ai_fixed_str)
    if (parsed_str := _try_parse(final_attempt_str, "AI 修复后")) is not None:
        return parsed_str

    logging.error(f"在所有三个阶段（直接、预处理、AI修复）后，都无法从响应中解析出有效的 JSON。")
    return None

def extract_knowledge_gaps(feedback: str) -> list[str]:
    """从审稿人的反馈中提取知识空白列表。"""
    match = re.search(
        r'###?\s*(KNOWLEDGE GAPS|知识鸿沟)\s*###?\s*\n(.*?)(?=\n###?|\Z)', 
        feedback, 
        re.DOTALL | re.IGNORECASE
    )
    if not match:
        logging.info("反馈中未找到 'KNOWLEDGE GAPS' 或 '知识鸿沟' 部分。")
        return []
        
    content = match.group(2).strip()
    gaps = [g.strip() for g in re.split(r'\n\s*(?:\d+\.|\-|\*)\s*', content) if g.strip()]
    
    logging.info(f"从反馈中提取了 {len(gaps)} 个知识空白。")
    return gaps


def chunk_document_for_rag(config: Config, document_text: str, doc_id: str) -> tuple[list[str], list[dict]]:
    """
    (V5 - 双重保险版) 为 RAG 对原始文本文档进行分块。
    此版本在滑动窗口基础上，增加了对每个最终块的token数进行重新编码验证的步骤，
    100%确保不会有任何块超出API限制。
    """
    logging.info(f"  正在为 RAG 对文档 (doc_id: {doc_id}) 进行分块 (双重保险版)...")
    chunks, metadatas = [], []

    if not document_text or not document_text.strip() or not config.encoder:
        if not config.encoder:
            logging.error("  chunk_document_for_rag: Tiktoken 编码器不可用，无法进行精确分块。")
        else:
            logging.warning("  chunk_document_for_rag: 文档文本为空/无效。返回空块。")
        return [], []

    # 留出足够的安全边际
    safety_margin = 300 
    max_tokens = config.embedding_model_max_tokens - safety_margin
    
    all_tokens = config.encoder.encode(document_text)
    total_token_count = len(all_tokens)
    
    if total_token_count == 0:
        return [], []

    overlap_tokens = config.count_tokens(" " * config.overlap_chars)
    step = max(1, max_tokens - overlap_tokens)

    chunk_index = 0
    for i in range(0, total_token_count, step):
        start_index = i
        end_index = min(i + max_tokens, total_token_count)
        
        token_chunk = all_tokens[start_index:end_index]
        text_chunk = config.encoder.decode(token_chunk)

        # [核心修正] 双重保险：重新编码验证，确保万无一失
        final_token_count = config.count_tokens(text_chunk)
        if final_token_count > config.embedding_model_max_tokens:
            logging.warning(f"  - 检测到解码/编码偏差导致token超限 (解码后 {final_token_count} > {config.embedding_model_max_tokens})。正在截断...")
            # 如果超限，则从token层面进行截断
            safe_token_chunk = token_chunk[:config.embedding_model_max_tokens]
            text_chunk = config.encoder.decode(safe_token_chunk)
        
        if not text_chunk or text_chunk.isspace():
            continue

        chunks.append(text_chunk)
        metadatas.append({
            "doc_id": doc_id,
            "chunk_index": chunk_index
        })
        chunk_index += 1
        
        if end_index == total_token_count:
            break

    logging.info(f"  文档 RAG 分块完成。共生成 {len(chunks)} 个块。")
    return chunks, metadatas
def final_post_processing(text: str) -> str:
    """
    对最终文档进行后处理修复。
    """
    logging.info("\n--- 正在对最终文档进行后处理修复 ---")
    
    processed_text = text
    rules = [
        ("修复不规范的section_id", r'section_\{i\}d', 'section_id'),
        ("修复代码/文本中的错误下标", r'(\w+)_\{i\}(\w+)', r'\1_\2'),
        ("修复多余的闭合大括号", r'\}\}', '}'),
        ("为悬空的 \\right 命令配对", r'(\\right)(?!\s*[\)\}\]])', r'\1)'),
        ("为单个字符的上下标自动加括号", r'([_^])([a-zA-Z0-9])(?!{)', r'\1{\2}'),
        ("为多个字符的上下标补充缺失的括号", r'([_^])\{([a-zA-Z0-9\\]+)\s', r'\1{\2} '),
        ("为命令作为指数时自动加括号", r'\^(\\[a-zA-Z]+(?:_{\w+})?)', r'^{\1}'),
        ("移除公式中无效的分号转义", r'\\;', ';'),
        ("修复 dM_{w}{dt} 这类常见错误", r'(\{)(\w+)\}(\{dt\})', r'{\1\2}\3'),
        ("合并3个及以上的换行符", r'\n{3,}', '\n\n'),
    ]

    for description, pattern, replacement in rules:
        original_text = processed_text
        processed_text = re.sub(pattern, replacement, processed_text)
        if original_text != processed_text:
            logging.info(f"  - (规则) {description}")

    lines = processed_text.splitlines()
    cleaned_lines = [line.strip() for line in lines]
    processed_text = "\n".join(cleaned_lines)
    logging.info("  - (规则) 已清理所有行首/行尾的空白字符。")
    
    logging.info("--- 后处理完成 ---")
    return processed_text

def quality_check(config: Config, content: str) -> str:
    """
    对最终内容进行质量评估。
    """
    content_for_review = truncate_text_for_context(config, content, 10000)
    prompt = f"请深入评估以下内容的质量。为以下方面提供评分(0-10分): 深度、细节、结构、连贯性、问题契合度。并列出主要优缺点。\n\n内容:\n{content_for_review}"
    return call_ai(config, config.secondary_ai_model, [{"role": "user", "content": prompt}], temperature=config.temperature_factual)
