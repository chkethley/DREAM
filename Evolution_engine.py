```python
import os
import sys
import yaml
import json
import sqlite3
import logging
import torch
import transformers
import openai
import anthropic
import numpy as np
import wandb
from skopt import gp_minimize
from typing import Optional, Any, Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ConfigManager
class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config: Dict[str, Any] = {}
        self.config_path: Optional[str] = config_path
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        _, ext = os.path.splitext(config_path)
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file) if 'yaml' in ext else json.load(file)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config file: {e}")
            raise  # Re-raise to halt execution if config loading fails
        self.config_path = config_path
        return self.config

    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False) if 'yaml' in self.config_path else json.dump(self.config, file, indent=2)
        logger.info(f"Config saved to {self.config_path}")

    def set(self, key: str, value: Any):
        keys = key.split('.')
        cfg = self.config
        for k in keys[:-1]:
            cfg = cfg.setdefault(k, {})
        cfg[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        cfg = self.config
        for k in keys[:-1]:
            if k not in cfg:
                return default
            cfg = cfg[k]
        return cfg.get(keys[-1], default)

# PromptLibrary
class PromptLibrary:
    def __init__(self, db_path: str = 'prompt_library.db'):
        self.db_path = self.get_db_path(db_path)  # Use a helper function
        self._initialize_db()

    def get_db_path(self, db_path):
        if getattr(sys, 'frozen', False):
            # Running in a PyInstaller bundle
            application_path = sys._MEIPASS
        else:
            # Running in a normal Python environment
            application_path = os.path.dirname(os.path.abspath(__file__))

        return os.path.join(application_path, db_path)

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript('''
            CREATE TABLE IF NOT EXISTS prompts (id INTEGER PRIMARY KEY, task_type TEXT, prompt_text TEXT, parameters TEXT);
            CREATE TABLE IF NOT EXISTS evaluations (id INTEGER PRIMARY KEY, prompt_id INTEGER, metric_name TEXT, metric_value REAL);
            ''')

    def add_prompt(self, task_type: str, prompt_text: str, parameters: Optional[dict] = None) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO prompts (task_type, prompt_text, parameters) VALUES (?, ?, ?)',
                               (task_type, prompt_text, json.dumps(parameters)))
                return cursor.lastrowid  # Return the ID of the new prompt
        except sqlite3.Error as e:
            logger.error(f"Error adding prompt: {e}")
            raise

    def get_prompt(self, prompt_id: int) -> Optional[dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM prompts WHERE id = ?', (prompt_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "task_type": row[1],
                        "prompt_text": row[2],
                        "parameters": json.loads(row[3]) if row[3] else None
                    }
                else:
                    return None
        except sqlite3.Error as e:
            logger.error(f"Error getting prompt: {e}")
            raise

    def update_prompt(self, prompt_id: int, task_type: Optional[str] = None, prompt_text: Optional[str] = None, parameters: Optional[dict] = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                updates = []
                if task_type:
                    updates.append(('task_type', task_type))
                if prompt_text:
                    updates.append(('prompt_text', prompt_text))
                if parameters:
                    updates.append(('parameters', json.dumps(parameters)))
                if updates:
                    set_clause = ', '.join(f"{k} = ?" for k, _ in updates)
                    values = [v for _, v in updates]
                    values.append(prompt_id)
                    cursor.execute(f'UPDATE prompts SET {set_clause} WHERE id = ?', values)
            except sqlite3.Error as e:
                logger.error(f"Error updating prompt: {e}")
                raise

    def delete_prompt(self, prompt_id: int):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM prompts WHERE id = ?', (prompt_id,))
            except sqlite3.Error as e:
                logger.error(f"Error deleting prompt: {e}")
                raise

# ModelAdapter (Abstract Base Class)
class ModelAdapter(ABC):
    def __init__(self, model_type: str, model_config: Optional[dict] = None):
        self.model_type = model_type.lower()
        self.model_config = model_config or {}
        self._verify_dependencies()

    @abstractmethod
    def _verify_dependencies(self):
        pass

    @abstractmethod
    def load_model(self, model_name: str, **kwargs):
        pass

    @abstractmethod
    def predict(self, inputs: str, **kwargs) -> dict:
        pass

# HuggingFaceModelAdapter
class HuggingFaceModelAdapter(ModelAdapter):
    def __init__(self, model_config: Optional[dict] = None):
        super().__init__("huggingface", model_config)
        self.tokenizer = None
        self.model = None

    def _verify_dependencies(self):
        import transformers

    def load_model(self, model_name: str, **kwargs):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        return self.model

    def predict(self, inputs: str, **kwargs) -> dict:
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model.generate(**tokenized_inputs, **kwargs)
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"prediction": decoded_output}

# OpenAiModelAdapter
class OpenAiModelAdapter(ModelAdapter):
    def __init__(self, model_config: Optional[dict] = None):
        super().__init__("openai", model_config)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = self.api_key
        self.model = None

    def _verify_dependencies(self):
        import openai

    def load_model(self, model_name: str, **kwargs):
        self.model = model_name
        return self.model

    def predict(self, inputs: str, **kwargs) -> dict:
        response = openai.Completion.create(model=self.model, prompt=inputs, **kwargs)
        return {"prediction": response.choices[0].text}

# AnthropicModelAdapter
class AnthropicModelAdapter(ModelAdapter):
    def __init__(self, model_config: Optional[dict] = None):
        super().__init__("anthropic", model_config)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Client(api_key=self.api_key)
        self.model = None

    def _verify_dependencies(self):
        import anthropic

    def load_model(self, model_name: str, **kwargs):
        self.model = model_name
        return self.model

    def predict(self, inputs: str, **kwargs) -> dict:
        response = self.client.completions.create(prompt=inputs, model=self.model, **kwargs)
        return {"prediction": response.completion}

# Factory function to create the correct ModelAdapter
def create_model_adapter(model_type: str, model_config: Optional[dict] = None) -> ModelAdapter:
    if model_type == "huggingface":
        return HuggingFaceModelAdapter(model_config)
    elif model_type == "openai":
        return OpenAiModelAdapter(model_config)
    elif model_type == "anthropic":
        return AnthropicModelAdapter(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# ExperimentTracker
class ExperimentTracker:
    def __init__(self, project_name: str, use_wandb: bool = True):
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=project_name)

    def log_metrics(self, metrics: dict):
        if self.use_wandb:
            wandb.log(metrics)
        else:
            logger.info(f"Metrics (wandb disabled): {metrics}")

# DataProcessor
class DataProcessor:
    def preprocess(self, data: str, tokenizer, max_length: int = 512):
        return tokenizer(data, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

# RegressionTester
class RegressionTester:
    def validate(self, function_a, function_b, test_cases: List[tuple]):
        for case in test_cases:
            result_a = function_a(*case)
            result_b = function_b(*case)
            assert result_a == result_b, f"Regression test failed for input {case}: {result_a} != {result_b}"

# HyperparameterOptimizer
class HyperparameterOptimizer:
    def __init__(self, model_adapter: ModelAdapter, param_space: Dict[str, List], evaluation_data: List[Dict[str,str]], n_calls: int = 20):
        self.model_adapter = model_adapter
        self.param_space = param_space
        self.n_calls = n_calls
        self.evaluation_data = evaluation_data # Store the evaluation data

    def objective(self, params):
        config = {key: val for key, val in zip(self.param_space.keys(), params)}
        # Assuming model_name is part of param_space, we still need to load for each iteration.
        model = self.model_adapter.load_model(**config)  # Pass the entire config
        score = self.evaluate_model(model, self.evaluation_data) # Use self.evaluation_data
        return -score  # Minimize negative accuracy

    def evaluate_model(self, model: ModelAdapter, evaluation_data: List[Dict[str, str]]) -> float:
        """
        Evaluates the model on the provided data using exact match accuracy.

        Args:
            model: The ModelAdapter instance to evaluate.
            evaluation_data: A list of dictionaries, each with "input" and "expected_output" keys.

        Returns:
            The exact match accuracy (a float between 0 and 1).
        """
        correct_predictions = 0
        total_predictions = len(evaluation_data)

        for example in evaluation_data:
            input_text = example["input"]
            expected_output = example["expected_output"]
            prediction = model.predict(input_text)["prediction"]  # Use the standardized prediction format

            if prediction.strip() == expected_output.strip():  # Remove leading/trailing whitespace
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy
# EvolutionEngine
class EvolutionEngine:
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.prompt_library = PromptLibrary(db_path=self.config_manager.get("prompt_db_path", "prompt_library.db"))
        self.model_adapter = create_model_adapter(
            self.config_manager.get("model_type", "huggingface"),
            self.config_manager.get("model_config", {})
        )
        self.experiment_tracker = ExperimentTracker(
            "evolution_engine",
            use_wandb=self.config_manager.get("use_wandb", True)
        )
        self.data_processor = DataProcessor()
        self.regression_tester = RegressionTester()

        # --- Evaluation Data (Example) ---
        self.evaluation_data: List[Dict[str, str]] = [
            {"input": "What is the capital of France?", "expected_output": "Paris"},
            {"input": "What is 2 + 2?", "expected_output": "4"},
            {"input": "Who painted the Mona Lisa?", "expected_output": "Leonardo da Vinci"},
        ]

        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.model_adapter,
            param_space=self.config_manager.get("hyperparameter_space", {"model_name": ["gpt2", "bert-base-uncased"]}), #Example of hyperparameter
            evaluation_data = self.evaluation_data
        )

    def run_optimization(self):
        return self.hyperparameter_optimizer.optimize()

    def cleanup(self):
        if self.experiment_tracker.use_wandb:
            wandb.finish()
```