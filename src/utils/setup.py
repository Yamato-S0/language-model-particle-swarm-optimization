from datetime import datetime
import os
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


RESULTS_DIRECTORY = os.path.join("../results")
LOGS_DIRECTORY = os.path.join("../logs")

DEFAULT_MODEL_PATH = "microsoft/Phi-3.5-mini-instruct"


def create_unique_id():
    """
    ユニークなIDを生成する（日時に基づくID）。
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time


def setup_directories(problem_type, approach, unique_id, identifier, prefix="dim"):
    """
    問題、ID、次元ごとに結果とログを保存するディレクトリを作成し、パスを返す。
    """
    if prefix != None:
        results_path = os.path.join(
            RESULTS_DIRECTORY,
            approach,
            problem_type,
            f"id_{unique_id}",
            f"{prefix}_{identifier}",
        )
        logs_path = os.path.join(
            LOGS_DIRECTORY,
            approach,
            problem_type,
            f"id_{unique_id}",
            f"{prefix}_{identifier}",
        )
    else:
        results_path = os.path.join(
            RESULTS_DIRECTORY,
            approach,
            problem_type,
            f"id_{unique_id}",
            f"{identifier}",
        )
        logs_path = os.path.join(
            LOGS_DIRECTORY, approach, problem_type, f"id_{unique_id}", f"{identifier}"
        )

    # ディレクトリを作成
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    return results_path, logs_path


def setup_logger(log_file):
    """各シードごとに異なるロガーをセットアップする"""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # ハンドラが既に設定されている場合はスキップ
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def setup_large_language_model(model_path=DEFAULT_MODEL_PATH):
    """
    大規模言語モデルをセットアップする。
    """
    logging.info("Setting up model...")
    logging.info("Model path: %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_PATH,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    logging.info("Setup complete.")
    return pipe
