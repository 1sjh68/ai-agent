# planning/tool_definitions.py

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# [核心修复] 为“最终润色”流程 (Final Polish) 恢复专用的 Pydantic 模型
# --------------------------------------------------------------------------

class SentenceEdit(BaseModel):
    original_sentence: str = Field(..., description="文本中需要被修改的、完整的原始句子。")
    revised_sentence: str = Field(..., description="经过语法、风格或流畅度优化后的新句子。")

class EditList(BaseModel):
    """定义一个包含多条句子修改建议的列表，这是润色AI需要返回的顶层对象。"""
    edits: List[SentenceEdit] = Field(..., description="一个句子修改建议的列表。如果某段文本无需修改，则返回一个空列表。")


# --------------------------------------------------------------------------
# [核心标准] 为“迭代优化”流程 (Iterative Refinement) 使用的 Pydantic 模型
# --------------------------------------------------------------------------

class FineGrainedPatch(BaseModel):
    """
    定义针对单个章节的、细粒度的句子级修订操作。
    """
    target_id: uuid.UUID = Field(..., description="需要进行句子级修订的目标章节的精确锚点ID。")
    edits: List[SentenceEdit] = Field(..., description="一个针对该章节的句子修改建议的列表。")

class FineGrainedPatchList(BaseModel):
    """
    定义一个包含多个章节细粒度修订操作的列表。
    这是Patcher AI必须返回的、且是唯一有效的顶层JSON对象。
    """
    patches: List[FineGrainedPatch] = Field(..., description="一个补丁操作的列表，每个补丁包含一个目标ID和一系列句子级修订。")


# --------------------------------------------------------------------------
# 用于“大纲生成”的工具定义函数 (保持不变)
# --------------------------------------------------------------------------

def get_outline_review_tool_definition():
    # ... 此函数内容保持不变 ...
    chapter_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "章节的标题，必须简洁明了。"
            },
            "description": {
                "type": "string",
                "description": "对本章节核心内容的2-3句话简要描述。"
            },
            "target_chars_ratio": {
                "type": "number",
                "description": "本章节预计占剩余总字数的比例（一个0到1之间的小数）。"
            },
            "sections": {
                "type": "array",
                "description": "子章节列表（可选），其结构与父章节相同。",
                "items": {"$ref": "#/definitions/chapter_definition"}
            }
        },
        "required": ["title", "description"]
    }

    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "update_document_outline",
                "description": "根据评审意见和已完成的工作，更新或修正文档的剩余大纲。如果原始计划依然完美，则必须提交与原始计划完全相同的计划。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "revised_plan": {
                            "type": "array",
                            "description": "一个包含所有剩余章节对象的列表。",
                            "items": {"$ref": "#/definitions/chapter_definition"}
                        }
                    },
                    "required": ["revised_plan"],
                    "definitions": {
                        "chapter_definition": chapter_schema
                    }
                }
            }
        }
    ]
    return tools_definition

def get_initial_outline_tool_definition():
    # ... 此函数内容保持不变 ...
    chapter_schema_for_initial_outline = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "本章节或小节的标题。"
            },
            "description": {
                "type": "string",
                "description": "对本章节内容和目的的简要描述（2-3句话）。"
            },
            "target_chars_ratio": {
                "type": "number",
                "description": "本章节预计占文档总长度的比例（例如，0.1代表10%）。顶层章节的比例总和应约为1.0。"
            },
            "sections": {
                "type": "array",
                "description": "可选的子章节或小节列表，每个都遵循此相同结构。",
                "items": {"$ref": "#/definitions/chapter_definition_initial"}
            }
        },
        "required": ["title", "description"]
    }

    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "create_initial_document_outline",
                "description": "根据用户的问题陈述生成结构化的文档大纲。大纲应包括一个主标题和章节列表，每个章节都有标题、描述以及可选的子小节。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "整个文档/报告的主标题。"
                        },
                        "outline": {
                            "type": "array",
                            "description": "一个对象列表，其中每个对象代表文档的一个主要章节。",
                            "items": {"$ref": "#/definitions/chapter_definition_initial"}
                        }
                    },
                    "required": ["title", "outline"],
                    "definitions": {
                        "chapter_definition_initial": chapter_schema_for_initial_outline
                    }
                }
            }
        }
    ]
    return tools_definition