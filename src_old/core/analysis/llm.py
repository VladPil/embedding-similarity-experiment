"""
LLM-based text similarity analysis.
Uses large language models to evaluate text similarity with understanding of context, plot, and style.
"""

import torch
from typing import Dict, Optional, List
import re
import json
from loguru import logger

from server.config import settings


class LLMAnalyzer:
    """Analyzes text similarity using large language models."""

    # Available models (sorted by size)
    AVAILABLE_MODELS = {
        'qwen2.5-0.5b': {
            'name': 'Qwen/Qwen2.5-0.5B-Instruct',
            'size_gb': 1,
            'description': 'Qwen 2.5 0.5B - Fast, lightweight'
        },
        'qwen2.5-1.5b': {
            'name': 'Qwen/Qwen2.5-1.5B-Instruct',
            'size_gb': 3,
            'description': 'Qwen 2.5 1.5B - Balanced speed/quality'
        },
        'qwen2.5-3b': {
            'name': 'Qwen/Qwen2.5-3B-Instruct',
            'size_gb': 6,
            'description': 'Qwen 2.5 3B - Good quality'
        },
        'qwen2.5-7b': {
            'name': 'Qwen/Qwen2.5-7B-Instruct',
            'size_gb': 14,
            'description': 'Qwen 2.5 7B - High quality'
        },
        'qwen2.5-14b': {
            'name': 'Qwen/Qwen2.5-14B-Instruct',
            'size_gb': 28,
            'description': 'Qwen 2.5 14B - Very high quality'
        },
        'mistral-7b': {
            'name': 'mistralai/Mistral-7B-Instruct-v0.3',
            'size_gb': 14,
            'description': 'Mistral 7B - Excellent reasoning'
        }
    }

    def __init__(self, model_key: str = 'qwen2.5-1.5b', device: Optional[str] = None):
        """
        Initialize LLM analyzer.

        Args:
            model_key: Model identifier
            device: Device to use ('cuda', 'cpu', or None for settings)
        """
        self.model_key = model_key
        self.model = None
        self.tokenizer = None

        # Determine device
        if device is None:
            device = settings.llm_device

        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = 'cpu'

        self.device = device
        logger.info(f"LLM Analyzer initialized with model {model_key} on {self.device}")

    def check_gpu_memory(self) -> Dict[str, float]:
        """
        Check GPU memory usage.

        Returns:
            Dictionary with memory stats (GB)
        """
        if not torch.cuda.is_available():
            return {
                'available': False,
                'total_gb': 0,
                'allocated_gb': 0,
                'free_gb': 0
            }

        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        free = total - allocated

        return {
            'available': True,
            'total_gb': round(total, 2),
            'allocated_gb': round(allocated, 2),
            'free_gb': round(free, 2),
            'device_name': torch.cuda.get_device_name(0)
        }

    def load_model(self):
        """Load the LLM model."""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        if self.model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {self.model_key}")

        model_info = self.AVAILABLE_MODELS[self.model_key]
        model_name = model_info['name']

        # Check GPU memory before loading
        mem_info = self.check_gpu_memory()
        if self.device == 'cuda' and mem_info['available']:
            required_gb = model_info['size_gb']
            max_memory_gb = settings.llm_max_memory_gb

            if mem_info['free_gb'] < required_gb:
                logger.warning(
                    f"Insufficient GPU memory. Required: {required_gb}GB, "
                    f"Available: {mem_info['free_gb']}GB. Switching to CPU."
                )
                self.device = 'cpu'
            elif required_gb > max_memory_gb:
                logger.warning(
                    f"Model size ({required_gb}GB) exceeds max_memory_gb setting "
                    f"({max_memory_gb}GB). Switching to CPU."
                )
                self.device = 'cpu'

        logger.info(f"Loading model {model_name} on {self.device}...")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=settings.models_cache_dir
            )

            # Load model
            if self.device == 'cuda':
                # Load with bfloat16 for efficiency
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                    cache_dir=settings.models_cache_dir
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    cache_dir=settings.models_cache_dir
                )
                self.model.to(self.device)

            logger.info(f"Model loaded successfully. Memory: {self.check_gpu_memory()}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded")

    def _create_comparison_prompt(self, text1: str, text2: str, max_chars: int = 4000) -> str:
        """
        Create prompt for text comparison.

        Args:
            text1: First text
            text2: Second text
            max_chars: Maximum characters per text

        Returns:
            Formatted prompt
        """
        # Truncate texts if too long
        if len(text1) > max_chars:
            text1 = text1[:max_chars] + "..."
        if len(text2) > max_chars:
            text2 = text2[:max_chars] + "..."

        prompt = f"""Проанализируй два текста и оцени их сходство по различным аспектам.

ТЕКСТ 1:
{text1}

ТЕКСТ 2:
{text2}

Оцени сходство текстов по следующим критериям (от 0 до 1, где 0 - полностью различные, 1 - почти идентичные):

1. СЮЖЕТ И СОДЕРЖАНИЕ: Насколько похожи сюжет, события, основные идеи?
2. СТИЛЬ НАПИСАНИЯ: Насколько похожа манера изложения, структура предложений?
3. ЖАНР И АТМОСФЕРА: Насколько похожи жанр, настроение, атмосфера?
4. ПЕРСОНАЖИ И ОБРАЗЫ: Насколько похожи персонажи, их характеры, взаимоотношения?
5. ЯЗЫК И ЛЕКСИКА: Насколько похож используемый язык, словарный запас?

Ответь СТРОГО в формате JSON:
{{
    "plot_similarity": <число от 0 до 1>,
    "style_similarity": <число от 0 до 1>,
    "genre_similarity": <число от 0 до 1>,
    "characters_similarity": <число от 0 до 1>,
    "language_similarity": <число от 0 до 1>,
    "overall_similarity": <число от 0 до 1>,
    "explanation": "<краткое объяснение на русском>"
}}"""

        return prompt

    def generate_text(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()

        # Format for chat models
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for consistency
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response (remove prompt)
        response = generated_text.split("assistant")[-1].strip()
        if response.startswith('\n'):
            response = response[1:]

        return response

    def compare_texts(self, text1: str, text2: str) -> Dict:
        """
        Compare two texts using LLM.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary with comparison results
        """
        try:
            # Create prompt
            prompt = self._create_comparison_prompt(text1, text2)

            # Generate response
            response = self.generate_text(prompt)

            # Parse JSON response
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', response, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                raise ValueError("Could not find JSON in response")

            # Validate and ensure all fields exist
            required_fields = [
                'plot_similarity', 'style_similarity', 'genre_similarity',
                'characters_similarity', 'language_similarity', 'overall_similarity'
            ]

            for field in required_fields:
                if field not in result:
                    result[field] = 0.5  # Default value

            # Add memory info
            result['memory_info'] = self.check_gpu_memory()
            result['model'] = self.model_key

            return result

        except Exception as e:
            logger.error(f"LLM comparison failed: {e}")
            return {
                'error': str(e),
                'plot_similarity': 0,
                'style_similarity': 0,
                'genre_similarity': 0,
                'characters_similarity': 0,
                'language_similarity': 0,
                'overall_similarity': 0,
                'explanation': f'Ошибка анализа: {str(e)}',
                'memory_info': self.check_gpu_memory(),
                'model': self.model_key
            }

    def quick_summary(self, text: str, max_words: int = 100) -> Dict:
        """
        Generate a quick summary of a text.

        Args:
            text: Text to summarize
            max_words: Maximum words in summary

        Returns:
            Dictionary with summary
        """
        try:
            # Truncate text if too long
            max_chars = 6000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            prompt = f"""Создай краткое резюме следующего текста ({max_words} слов или меньше).

ТЕКСТ:
{text}

Ответь СТРОГО в формате JSON:
{{
    "summary": "<краткое резюме на русском>",
    "key_points": ["<ключевой пункт 1>", "<ключевой пункт 2>", "<ключевой пункт 3>"]
}}"""

            response = self.generate_text(prompt, max_new_tokens=300)

            # Parse JSON
            json_match = re.search(r'\{[^{}]*\[[^\[\]]*\][^{}]*\}|\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not find JSON in response")

            result['model'] = self.model_key
            return result

        except Exception as e:
            logger.error(f"Quick summary failed: {e}")
            return {
                'error': str(e),
                'summary': f'Ошибка: {str(e)}',
                'key_points': [],
                'model': self.model_key
            }

    def extract_key_themes(self, text: str) -> Dict:
        """
        Extract main themes from a text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with themes
        """
        try:
            # Truncate text if too long
            max_chars = 6000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            prompt = f"""Извлеки основные темы из следующего текста.

ТЕКСТ:
{text}

Ответь СТРОГО в формате JSON:
{{
    "themes": ["<тема 1>", "<тема 2>", "<тема 3>"],
    "main_topic": "<основная тема>"
}}"""

            response = self.generate_text(prompt, max_new_tokens=200)

            # Parse JSON
            json_match = re.search(r'\{[^{}]*\[[^\[\]]*\][^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not find JSON in response")

            result['model'] = self.model_key
            return result

        except Exception as e:
            logger.error(f"Extract themes failed: {e}")
            return {
                'error': str(e),
                'themes': [],
                'main_topic': f'Ошибка: {str(e)}',
                'model': self.model_key
            }

    def extract_key_differences(self, text1: str, text2: str) -> Dict:
        """
        Quickly identify key differences between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary with key differences
        """
        try:
            # Truncate texts if too long
            max_chars = 3000
            if len(text1) > max_chars:
                text1 = text1[:max_chars] + "..."
            if len(text2) > max_chars:
                text2 = text2[:max_chars] + "..."

            prompt = f"""Найди ключевые различия между двумя текстами.

ТЕКСТ 1:
{text1}

ТЕКСТ 2:
{text2}

Ответь СТРОГО в формате JSON:
{{
    "differences": ["<различие 1>", "<различие 2>", "<различие 3>"],
    "similarities": ["<сходство 1>", "<сходство 2>"],
    "main_difference": "<основное различие>"
}}"""

            response = self.generate_text(prompt, max_new_tokens=300)

            # Parse JSON
            json_match = re.search(r'\{[^{}]*\[[^\[\]]*\][^{}]*\[[^\[\]]*\][^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not find JSON in response")

            result['model'] = self.model_key
            return result

        except Exception as e:
            logger.error(f"Extract differences failed: {e}")
            return {
                'error': str(e),
                'differences': [],
                'similarities': [],
                'main_difference': f'Ошибка: {str(e)}',
                'model': self.model_key
            }

    def quick_sentiment(self, text: str) -> Dict:
        """
        Fast sentiment analysis.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment
        """
        try:
            # Truncate text if too long
            max_chars = 4000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            prompt = f"""Проанализируй эмоциональный тон следующего текста.

ТЕКСТ:
{text}

Ответь СТРОГО в формате JSON:
{{
    "sentiment": "<positive/negative/neutral>",
    "confidence": <число от 0 до 1>,
    "emotions": ["<эмоция 1>", "<эмоция 2>"],
    "explanation": "<краткое объяснение>"
}}"""

            response = self.generate_text(prompt, max_new_tokens=200)

            # Parse JSON
            json_match = re.search(r'\{[^{}]*\[[^\[\]]*\][^{}]*\}|\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not find JSON in response")

            result['model'] = self.model_key
            return result

        except Exception as e:
            logger.error(f"Quick sentiment failed: {e}")
            return {
                'error': str(e),
                'sentiment': 'neutral',
                'confidence': 0,
                'emotions': [],
                'explanation': f'Ошибка: {str(e)}',
                'model': self.model_key
            }

    def compare_quick(self, text1: str, text2: str) -> Dict:
        """
        Faster comparison with less detail.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary with quick comparison
        """
        try:
            # Truncate texts more aggressively for speed
            max_chars = 2000
            if len(text1) > max_chars:
                text1 = text1[:max_chars] + "..."
            if len(text2) > max_chars:
                text2 = text2[:max_chars] + "..."

            prompt = f"""Быстро сравни два текста и оцени их общее сходство.

ТЕКСТ 1:
{text1}

ТЕКСТ 2:
{text2}

Ответь СТРОГО в формате JSON:
{{
    "similarity": <число от 0 до 1>,
    "similar": <true/false>,
    "reason": "<краткая причина на русском>"
}}"""

            response = self.generate_text(prompt, max_new_tokens=150)

            # Parse JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not find JSON in response")

            result['model'] = self.model_key
            return result

        except Exception as e:
            logger.error(f"Quick comparison failed: {e}")
            return {
                'error': str(e),
                'similarity': 0,
                'similar': False,
                'reason': f'Ошибка: {str(e)}',
                'model': self.model_key
            }

    def generate_report(self, text1: str, text2: str, text1_title: str = "Текст 1", text2_title: str = "Текст 2") -> Dict:
        """
        Generate a comprehensive comparison report.

        Args:
            text1: First text
            text2: Second text
            text1_title: Title of first text
            text2_title: Title of second text

        Returns:
            Dictionary with detailed report
        """
        try:
            # Truncate texts if too long
            max_chars = 4000
            if len(text1) > max_chars:
                text1 = text1[:max_chars] + "..."
            if len(text2) > max_chars:
                text2 = text2[:max_chars] + "..."

            prompt = f"""Создай подробный аналитический отчёт о сравнении двух текстов.

ТЕКСТ 1 ({text1_title}):
{text1}

ТЕКСТ 2 ({text2_title}):
{text2}

Создай структурированный отчёт со следующими разделами:

1. КРАТКОЕ РЕЗЮМЕ (2-3 предложения о главных находках)
2. СХОДСТВА (детальное описание общих черт)
3. РАЗЛИЧИЯ (детальное описание отличий)
4. СТИЛИСТИЧЕСКИЙ АНАЛИЗ (манера письма, тон, структура)
5. КОНТЕНТНЫЙ АНАЛИЗ (темы, идеи, сюжет)
6. ВЫВОДЫ И РЕКОМЕНДАЦИИ

Ответь СТРОГО в формате JSON:
{{
    "summary": "<краткое резюме>",
    "similarities": {{
        "content": "<описание сходств в содержании>",
        "style": "<описание сходств в стиле>",
        "tone": "<описание сходств в тоне>"
    }},
    "differences": {{
        "content": "<описание различий в содержании>",
        "style": "<описание различий в стиле>",
        "tone": "<описание различий в тоне>"
    }},
    "stylistic_analysis": "<подробный стилистический анализ>",
    "content_analysis": "<подробный контентный анализ>",
    "conclusions": "<выводы и рекомендации>",
    "overall_similarity": <число от 0 до 1>
}}"""

            response = self.generate_text(prompt, max_new_tokens=1024)

            # Parse JSON
            json_match = re.search(r'\{[^\{\}]*\{[^\{\}]*\{[^\{\}]*\}[^\{\}]*\}[^\{\}]*\}', response, re.DOTALL)
            if not json_match:
                # Try simpler pattern
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not find JSON in response")

            result['model'] = self.model_key
            result['text1_title'] = text1_title
            result['text2_title'] = text2_title
            return result

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                'error': str(e),
                'summary': f'Ошибка генерации отчёта: {str(e)}',
                'similarities': {'content': '', 'style': '', 'tone': ''},
                'differences': {'content': '', 'style': '', 'tone': ''},
                'stylistic_analysis': '',
                'content_analysis': '',
                'conclusions': '',
                'overall_similarity': 0,
                'model': self.model_key,
                'text1_title': text1_title,
                'text2_title': text2_title
            }

    def generate_single_text_report(self, text: str, text_title: str = "Текст") -> Dict:
        """
        Generate a comprehensive analysis report for a single text.

        Args:
            text: Text to analyze
            text_title: Title of the text

        Returns:
            Dictionary with detailed report
        """
        try:
            # Truncate text if too long
            max_chars = 6000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            prompt = f"""Создай подробный аналитический отчёт о тексте.

ТЕКСТ ({text_title}):
{text}

Создай структурированный отчёт со следующими разделами:

1. КРАТКОЕ РЕЗЮМЕ (основная идея текста)
2. ТЕМЫ И ИДЕИ (ключевые темы, концепции)
3. СТИЛИСТИЧЕСКИЙ АНАЛИЗ (манера письма, тон, структура)
4. ХАРАКТЕРИСТИКИ (жанр, настроение, целевая аудитория)
5. КЛЮЧЕВЫЕ ФРАГМЕНТЫ (важные цитаты или моменты)
6. ОБЩАЯ ОЦЕНКА И РЕКОМЕНДАЦИИ

Ответь СТРОГО в формате JSON:
{{
    "summary": "<краткое резюме>",
    "themes": ["<тема 1>", "<тема 2>", "<тема 3>"],
    "stylistic_analysis": {{
        "writing_style": "<описание стиля>",
        "tone": "<описание тона>",
        "structure": "<описание структуры>"
    }},
    "characteristics": {{
        "genre": "<жанр>",
        "mood": "<настроение>",
        "target_audience": "<целевая аудитория>"
    }},
    "key_quotes": ["<цитата 1>", "<цитата 2>"],
    "evaluation": "<общая оценка и рекомендации>"
}}"""

            response = self.generate_text(prompt, max_new_tokens=1024)

            # Parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not find JSON in response")

            result['model'] = self.model_key
            result['text_title'] = text_title
            return result

        except Exception as e:
            logger.error(f"Single text report generation failed: {e}")
            return {
                'error': str(e),
                'summary': f'Ошибка генерации отчёта: {str(e)}',
                'themes': [],
                'stylistic_analysis': {'writing_style': '', 'tone': '', 'structure': ''},
                'characteristics': {'genre': '', 'mood': '', 'target_audience': ''},
                'key_quotes': [],
                'evaluation': '',
                'model': self.model_key,
                'text_title': text_title
            }

    @staticmethod
    def list_models() -> List[Dict]:
        """List all available models."""
        models = []
        for key, info in LLMAnalyzer.AVAILABLE_MODELS.items():
            models.append({
                'key': key,
                'name': info['name'],
                'size_gb': info['size_gb'],
                'description': info['description']
            })
        return models
