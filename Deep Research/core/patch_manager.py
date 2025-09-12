# core/patch_manager.py

import json
import logging
import re
# [核心修改] 导入新的库来实现模糊字符串匹配
from thefuzz import fuzz

def apply_patch(current_solution: str, patches_json: str) -> str:
    """
    (V2 - Section-Aware) 应用一个或多个补丁到当前的解决方案文本上。
    此版本能够识别Markdown的章节结构，并对整个章节块进行操作。
    """
    try:
        data = json.loads(patches_json)
        patches = data.get("patches", [])
        if not isinstance(patches, list):
            logging.error(f"补丁数据格式错误：'patches' 字段不是一个列表。内容: {patches_json}")
            return current_solution
    except json.JSONDecodeError as e:
        logging.error(f"无法解析补丁JSON字符串: {e}")
        return current_solution

    modified_solution = current_solution
    applied_patches_count = 0

    for i, patch in enumerate(patches):
        logging.info(f"--- 正在处理补丁 {i+1}/{len(patches)} ---")
        
        action = patch.get("action", "").upper()
        target_id_str = str(patch.get("target_id"))
        new_content = patch.get("new_content", "")
        
        if not target_id_str:
            logging.warning(f"提供的补丁缺少'target_id'，跳过此补丁。")
            continue

        escaped_id = re.escape(target_id_str)
        # 修正正则表达式以更好地处理章节末尾
        pattern = re.compile(
            rf"(^#+.*?\s*.*?)(?=^#+ |\Z)",
            re.MULTILINE | re.DOTALL
        )
        
        match = pattern.search(modified_solution)

        if not match:
            logging.error(f"未能找到锚点ID '{target_id_str}' 对应的完整章节块，无法应用补丁: {patch}")
            continue

        section_to_modify = match.group(1)

        try:
            if action == 'REPLACE':
                logging.info(f"执行 REPLACE 操作于锚点 '{target_id_str}'")
                modified_solution = modified_solution.replace(section_to_modify, new_content)
                applied_patches_count += 1
            elif action == 'DELETE':
                logging.info(f"执行 DELETE 操作于锚点 '{target_id_str}'")
                modified_solution = modified_solution.replace(section_to_modify, "")
                applied_patches_count += 1
            elif action == 'INSERT_AFTER':
                logging.info(f"执行 INSERT_AFTER 操作于锚点 '{target_id_str}'")
                modified_solution = modified_solution.replace(
                    section_to_modify, f"{section_to_modify}\n{new_content}"
                )
                applied_patches_count += 1
            elif action == 'INSERT_BEFORE':
                logging.info(f"执行 INSERT_BEFORE 操作于锚点 '{target_id_str}'")
                modified_solution = modified_solution.replace(
                    section_to_modify, f"{new_content}\n\n{section_to_modify}"
                )
                applied_patches_count += 1
            else:
                logging.warning(f"未知的补丁操作类型 '{action}'，跳过此操作: {patch}")

        except Exception as e:
            logging.error(f"应用补丁时发生意外错误: {e}", exc_info=True)

    logging.info(f"--- 所有补丁处理完毕，共成功应用 {applied_patches_count}/{len(patches)} 个补丁 ---")
    return modified_solution

# --- [核心修改] 升级 apply_fine_grained_edits 函数以使用模糊匹配 ---
def apply_fine_grained_edits(current_solution: str, changes_list: list) -> str:
    """
    (V2 - 模糊匹配版) 应用细粒度的、句子级别的修订。
    """
    modified_solution = current_solution
    logging.info(f"--- 开始应用 {len(changes_list)} 个章节的细粒度修订 (模糊匹配模式) ---")

    # 定义一个简单的文本清理函数，用于提高匹配准确率
    def clean_text_for_matching(text):
        # 移除非字母数字字符，保留基本文本内容
        return re.sub(r'[\W_]+', '', text).lower()

    for change in changes_list:
        if not isinstance(change, dict):
             change = change.model_dump(mode='json')

        target_id_str = str(change.get('target_id'))
        edits = change.get('edits', [])

        if not edits:
            logging.info(f"  - 章节 '{target_id_str}' 无需修订，跳过。")
            continue

        escaped_id = re.escape(target_id_str)
        pattern = re.compile(rf"(^#+.*?\s*.*?)(?=^#+ |\Z)", re.MULTILINE | re.DOTALL)
        match = pattern.search(modified_solution)
        
        if not match:
            logging.warning(f"  - 未能找到ID为 '{target_id_str}' 的章节块，跳过此修订。")
            continue
            
        original_section_content = match.group(1)
        modified_section_content = original_section_content
        applied_count = 0
        
        # 将章节内容按句子分割（一个更健壮的实现，处理多种结束符和换行）
        sentences_in_section = re.split(r'(?<=[。？！.!?\n])\s+', original_section_content)
        # 过滤掉空的字符串
        sentences_in_section = [s.strip() for s in sentences_in_section if s and s.strip()]

        for edit in edits:
            original_sentence_from_ai = edit.get('original_sentence')
            revised_sentence_from_ai = edit.get('revised_sentence')

            if not original_sentence_from_ai or not revised_sentence_from_ai:
                continue
            
            best_match_sentence = None
            highest_ratio = 0
            
            # 清理AI提供的句子以进行匹配
            cleaned_ai_sentence = clean_text_for_matching(original_sentence_from_ai)

            # 在章节的实际句子中寻找最佳匹配
            for sentence_in_doc in sentences_in_section:
                cleaned_doc_sentence = clean_text_for_matching(sentence_in_doc)
                ratio = fuzz.ratio(cleaned_ai_sentence, cleaned_doc_sentence)
                
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match_sentence = sentence_in_doc

            # [核心修改] 匹配率设置为95
            SIMILARITY_THRESHOLD = 95 
            if highest_ratio >= SIMILARITY_THRESHOLD:
                # 使用函数式替换以避免替换错误的部分
                modified_section_content = modified_section_content.replace(best_match_sentence, revised_sentence_from_ai, 1)
                applied_count += 1
                # 从待匹配列表中移除已使用的句子，防止重复匹配
                sentences_in_section.remove(best_match_sentence)
            else:
                logging.warning(f"  - 在章节 '{target_id_str}' 中未找到足够相似的句子 (最高相似度: {highest_ratio}%)，跳过替换: '{original_sentence_from_ai[:50]}...'")

        if applied_count > 0:
            modified_solution = modified_solution.replace(original_section_content, modified_section_content)
            logging.info(f"  - 成功向章节 '{target_id_str}' 应用了 {applied_count}/{len(edits)} 条句子级修订。")
        
    logging.info("--- 所有细粒度修订应用完毕 ---")
    return modified_solution