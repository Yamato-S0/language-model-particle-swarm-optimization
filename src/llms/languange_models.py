import logging
import torch


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from abc import abstractmethod


class LargeLanguageModel:
    @abstractmethod
    def generate_output(self, messages, max_new_tokens, temperature):
        """入力プロンプトに対するLLMの出力を生成"""
        pass


class LLAMA3_2_3B_Instruct(LargeLanguageModel):
    def __init__(self):
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        logging.info("Setting up model...")
        logging.info("Model path: %s", model_name)
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logging.info("Setup complete.")

    def generate_output(self, messages, max_new_tokens, temperature):
        """
        入力プロンプトに対するLLMの出力を生成
        """
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        logging.info("Generation Args: %s", generation_args)
        outputs = self.pipe(messages, **generation_args)
        return outputs[0]["generated_text"][-1]


class LLAMA3_1_8B_Instruct(LargeLanguageModel):
    def __init__(self):
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        logging.info("Setting up model...")
        logging.info("Model path: %s", model_name)
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        logging.info("Setup complete.")

    def generate_output(self, messages, max_new_tokens, temperature):
        """
        入力プロンプトに対するLLMの出力を生成
        """
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        logging.info("Generation Args: %s", generation_args)
        outputs = self.pipe(messages, **generation_args)
        return outputs[0]["generated_text"][-1]["content"]
