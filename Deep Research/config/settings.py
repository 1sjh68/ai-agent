# config/settings.py

import os
import sys
import logging
from datetime import datetime
import tiktoken
import openai

# --- [新增] 辅助函数，用于安全地获取和清理环境变量 ---
def get_env_variable(key: str, default: str = "") -> str:
    """
    获取环境变量，并自动去除值两端的空格和引号。
    """
    value = os.getenv(key, default).strip()
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value
# --- [修改结束] ---

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    if load_dotenv():
        logging.info(".env 文件加载成功。")
    else:
        logging.info(".env 文件未找到或为空，将依赖于系统环境变量。")
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logging.info("dotenv 库未安装，将依赖于系统环境变量。")
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logging.warning(f"加载 .env 文件时发生错误: {e}")


class Config:
    """
    一个集中的配置类，用于管理项目的所有设置。
    """
    def __init__(self):
        # --- API 配置 (使用新的辅助函数加载) ---
        self.deepseek_api_key = get_env_variable("DEEPSEEK_API_KEY")
        self.deepseek_base_url = get_env_variable("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        self.embedding_api_base_url = get_env_variable("EMBEDDING_API_BASE_URL")
        self.embedding_api_key = get_env_variable("EMBEDDING_API_KEY")
        self.embedding_model_name = get_env_variable("EMBEDDING_MODEL_NAME", "bge-m3")

        keys_str = get_env_variable("GOOGLE_API_KEYS", "")
        self.google_api_keys = [k.strip() for k in keys_str.split(',') if k.strip()]
        ids_str = get_env_variable("GOOGLE_CSE_IDS", "")
        self.google_cse_ids = [i.strip() for i in ids_str.split(',') if i.strip()]
        self.google_service_account_path = get_env_variable("GOOGLE_APPLICATION_CREDENTIALS")
        self.current_google_api_key_index = 0

        if self.google_api_keys and self.google_cse_ids and len(self.google_api_keys) != len(self.google_cse_ids):
            logging.warning("警告：GOOGLE_API_KEYS 和 GOOGLE_CSE_IDS 的数量不匹配！程序将只使用数量较少者对应的配对。")

        # --- 向量数据库配置 ---
        self.vector_db_path = get_env_variable("VECTOR_DB_PATH", "./chroma_db")
        self.vector_db_collection_name = get_env_variable("VECTOR_DB_COLLECTION_NAME", "experience_store")
        self.embedding_batch_size = int(get_env_variable("EMBEDDING_BATCH_SIZE", "25"))
        self.num_retrieved_experiences = int(get_env_variable("NUM_RETRIEVED_EXPERIENCES", "3"))

        # --- 模型名称配置 ---
        self.main_ai_model = get_env_variable("MAIN_AI_MODEL", "deepseek-chat")
        self.main_ai_model_heavy = get_env_variable("MAIN_AI_MODEL_HEAVY", "deepseek-reasoner")
        self.secondary_ai_model = get_env_variable("SECONDARY_AI_MODEL", "deepseek-reasoner")
        self.summary_model_name = get_env_variable("SUMMARY_MODEL_NAME", "deepseek-coder")
        self.researcher_model_name = get_env_variable("RESEARCHER_MODEL_NAME", "deepseek-reasoner")
        self.outline_model_name = get_env_variable("OUTLINE_MODEL_NAME", "deepseek-coder")
        self.planning_review_model_name = get_env_variable("PLANNING_REVIEW_MODEL_NAME", "deepseek-coder")
        self.editorial_model_name = get_env_variable("EDITORIAL_MODEL_NAME", self.main_ai_model)
        self.json_fixer_model_name = get_env_variable("JSON_FIXER_MODEL_NAME", "deepseek-coder")
        # --- [核心修改] 为 Patcher 添加专用模型配置 ---
        self.patcher_model_name = get_env_variable("PATCHER_MODEL_NAME", self.json_fixer_model_name)

        # --- LLM 调用参数 ---
        self.temperature_factual = float(get_env_variable("LLM_TEMPERATURE_FACTUAL", "0.1"))
        self.top_p_factual = float(get_env_variable("LLM_TOP_P_FACTUAL", "0.95"))
        self.temperature_creative = float(get_env_variable("LLM_TEMPERATURE_CREATIVE", "0.3"))
        self.top_p_creative = float(get_env_variable("LLM_TOP_P_CREATIVE", "0.95"))
        self.frequency_penalty = float(get_env_variable("LLM_FREQUENCY_PENALTY", "0.2"))
        self.presence_penalty = float(get_env_variable("LLM_PRESENCE_PENALTY", "0.0"))

        # --- 核心工具与客户端 ---
        self.client = None
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logging.error(f"严重错误：初始化 tiktoken 编码器失败: {e}")
            self.encoder = None

        # --- 路径与目录 ---
        self.checkpoint_file_name = "optimization_checkpoint.json"
        self.session_base_dir = "C:/Users/oo/PycharmProjects/output"
        self.session_dir = ""
        self.log_file_path = ""

        # --- 核心运行参数 ---
        self.api_request_timeout_seconds = int(get_env_variable("API_TIMEOUT_SECONDS", "900"))
        self.max_iterations = int(get_env_variable("MAX_ITERATIONS", "4"))
        self.initial_solution_target_chars = int(get_env_variable("INITIAL_SOLUTION_TARGET_CHARS", "15000"))

        # --- Token 与分块参数 ---
        self.embedding_model_max_tokens = int(get_env_variable("EMBEDDING_MODEL_MAX_TOKENS", "1024"))
        self.max_context_for_long_text_review_tokens = int(get_env_variable("MAX_CONTEXT_TOKENS_REVIEW", "30000"))
        self.intermediate_edit_max_tokens = int(get_env_variable("INTERMEDIATE_EDIT_MAX_TOKENS", "8192"))
        self.max_chunk_tokens = int(get_env_variable("MAX_CHUNK_TOKENS", "4096"))
        self.overlap_chars = int(get_env_variable("OVERLAP_CHARS", "800"))
        self.max_chunks_per_section = int(get_env_variable("MAX_CHUNKS_PER_SECTION", "20"))

        # --- API 重试参数 ---
        self.api_retry_max_attempts = int(get_env_variable("API_RETRY_MAX_ATTEMPTS", "3"))
        self.api_retry_wait_multiplier = int(get_env_variable("API_RETRY_WAIT_MULTIPLIER", "1"))
        self.api_retry_max_wait = int(get_env_variable("API_RETRY_MAX_WAIT", "60"))

        # --- 研究模块参数 ---
        self.num_search_results = int(get_env_variable("NUM_SEARCH_RESULTS", "3"))
        self.max_queries_per_gap = int(get_env_variable("MAX_QUERIES_PER_GAP", "5"))

        # --- 布尔标志 ---
        self.interactive_mode = get_env_variable("INTERACTIVE_MODE", "False").lower() == "true"
        self.use_async_research = get_env_variable("USE_ASYNC_RESEARCH", "True").lower() == "true"
        self.enable_dynamic_outline_correction = get_env_variable("ENABLE_DYNAMIC_OUTLINE_CORRECTION", "True").lower() == "true"

        # --- [新增] 读取回放模式配置 ---
        self.replay_from_file = get_env_variable("REPLAY_FROM_FILE", "")

        # --- 用户输入占位符 ---
        self.user_problem = ""
        self.external_data_files = []

        # --- 内容生成参数 ---
        self.min_allocated_chars_for_section = int(get_env_variable("MIN_ALLOCATED_CHARS_SECTION", "100"))

    def setup_logging(self, logging_level=logging.INFO):
        """为当前运行会话配置日志记录器。"""
        now = datetime.now()
        session_timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.session_base_dir, f"session_{session_timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.session_dir, "session.log")

        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                handler.close()

        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"日志记录已初始化。会话目录: {self.session_dir}")
        logging.info(f"日志文件: {self.log_file_path}")

    def _initialize_deepseek_client(self):
        """初始化与 DeepSeek API 通信的客户端。"""
        if not self.deepseek_api_key:
            logging.critical("DEEPSEEK_API_KEY 环境变量未设置。")
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。")
        try:
            self.client = openai.OpenAI(
                api_key=self.deepseek_api_key,
                base_url=self.deepseek_base_url,
                timeout=float(self.api_request_timeout_seconds)
            )
            logging.info(f"DeepSeek 客户端初始化成功，连接至 {self.deepseek_base_url}")
        except Exception as e:
            logging.critical(f"DeepSeek 客户端初始化期间出错: {e}。请检查 API 密钥/URL 和网络连接。")
            raise RuntimeError(f"DeepSeek 客户端初始化失败: {e}")

    def save_to_env(self, env_file_path=".env"):
        """
        (V2 - 健壮版) 将当前配置安全地保存到.env文件。
        此版本会读取现有文件，仅更新UI管理的字段，并保留所有其他行（包括注释和未管理的变量）。
        """
        logging.info(f"正在尝试将配置安全地保存到 {env_file_path}...")
        try:
            # 定义一个UI可以修改的键的集合，以便精确更新
            managed_keys = {
                'DEEPSEEK_API_KEY', 'EMBEDDING_API_KEY', 'GOOGLE_API_KEYS',
                'GOOGLE_CSE_IDS', 'MAX_CHUNKS_PER_SECTION', 'MAX_ITERATIONS',
                'INITIAL_SOLUTION_TARGET_CHARS', 'NUM_RETRIEVED_EXPERIENCES',
                'MAIN_AI_MODEL', 'MAIN_AI_MODEL_HEAVY', 'EMBEDDING_MODEL_NAME'
            }

            # 从当前配置对象获取新值
            new_values = {
                'DEEPSEEK_API_KEY': self.deepseek_api_key,
                'EMBEDDING_API_KEY': self.embedding_api_key,
                'GOOGLE_API_KEYS': ','.join(self.google_api_keys),
                'GOOGLE_CSE_IDS': ','.join(self.google_cse_ids),
                'MAX_CHUNKS_PER_SECTION': str(self.max_chunks_per_section),
                'MAX_ITERATIONS': str(self.max_iterations),
                'INITIAL_SOLUTION_TARGET_CHARS': str(self.initial_solution_target_chars),
                'NUM_RETRIEVED_EXPERIENCES': str(self.num_retrieved_experiences),
                'MAIN_AI_MODEL': self.main_ai_model,
                'MAIN_AI_MODEL_HEAVY': self.main_ai_model_heavy,
                'EMBEDDING_MODEL_NAME': self.embedding_model_name,
            }

            updated_lines = []
            keys_to_update = set(managed_keys) # 创建一个副本，用于追踪是否已更新

            # 如果文件存在，则读取并更新
            if os.path.exists(env_file_path):
                with open(env_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line_stripped = line.strip()
                        # 跳过注释和空行
                        if not line_stripped or line_stripped.startswith('#') or '=' not in line_stripped:
                            updated_lines.append(line)
                            continue

                        key, _ = line_stripped.split('=', 1)
                        key = key.strip()

                        # 如果当前行是我们需要管理的键，则用新值替换
                        if key in managed_keys:
                            new_value = new_values.get(key, "")
                            # 为包含特殊字符的值加上引号
                            if any(c in new_value for c in [',', ' ', '(', ')']):
                                new_value = f'"{new_value}"'
                            updated_lines.append(f'{key}={new_value}\n')
                            keys_to_update.discard(key) # 标记为已更新
                        else:
                            # 否则，保留原始行
                            updated_lines.append(line)
            
            # 如果.env文件中原本没有某些受管的键，则将它们追加到文件末尾
            for key in keys_to_update:
                new_value = new_values.get(key, "")
                if any(c in new_value for c in [',', ' ', '(', ')']):
                    new_value = f'"{new_value}"'
                updated_lines.append(f'{key}={new_value}\n')
                logging.info(f"  - 在.env文件中新增了配置项: {key}")

            # 将更新后的所有内容写回文件
            with open(env_file_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            logging.info(f"配置已成功保存到 {env_file_path}")
            return True
        except Exception as e:
            logging.error(f"保存配置到.env文件时发生严重错误: {e}", exc_info=True)
            return False

    def count_tokens(self, text: str) -> int:
        """使用 tiktoken 计算文本的 token 数量，如果失败则回退到近似计算。"""
        if not text:
            return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                logging.warning(f"Tiktoken 编码失败: {e}。回退到近似计算。")
                return len(text) // 3 # 中文近似
        logging.warning("Tiktoken 编码器不可用，Token 计数使用近似值。")
        return len(text) // 3
