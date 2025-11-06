"""
Мониторинг GPU ресурсов
"""
import subprocess
from typing import Optional, List, Dict
from loguru import logger


class GPUMonitor:
    """
    Мониторинг использования GPU

    Использует nvidia-smi для получения информации о GPU
    """

    def __init__(self, device_id: int = 0):
        """
        Инициализация монитора

        Args:
            device_id: ID GPU устройства
        """
        self.device_id = device_id
        self._nvidia_smi_available = self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> bool:
        """Проверить доступность nvidia-smi"""
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5
            )
            return True
        except:
            logger.warning("nvidia-smi недоступен, GPU мониторинг отключен")
            return False

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Получить использование памяти GPU

        Returns:
            Dict: Словарь с информацией о памяти
                - used_mb: Использованная память в МБ
                - total_mb: Общая память в МБ
                - free_mb: Свободная память в МБ
                - usage_percent: Процент использования
        """
        if not self._nvidia_smi_available:
            return {
                "used_mb": 0.0,
                "total_mb": 24000.0,  # Заглушка для RTX 4090
                "free_mb": 24000.0,
                "usage_percent": 0.0
            }

        try:
            # Запрос информации о памяти
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                logger.error(f"nvidia-smi ошибка: {result.stderr}")
                return self._get_fallback_memory()

            # Парсинг вывода
            output = result.stdout.strip()
            used_str, total_str = output.split(",")
            used_mb = float(used_str.strip())
            total_mb = float(total_str.strip())
            free_mb = total_mb - used_mb
            usage_percent = (used_mb / total_mb) * 100

            return {
                "used_mb": used_mb,
                "total_mb": total_mb,
                "free_mb": free_mb,
                "usage_percent": usage_percent
            }

        except Exception as e:
            logger.error(f"Ошибка получения памяти GPU: {e}")
            return self._get_fallback_memory()

    def get_utilization(self) -> float:
        """
        Получить утилизацию GPU в процентах

        Returns:
            float: Утилизация 0-100
        """
        if not self._nvidia_smi_available:
            return 0.0

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return float(result.stdout.strip())

        except Exception as e:
            logger.error(f"Ошибка получения утилизации GPU: {e}")

        return 0.0

    def get_temperature(self) -> float:
        """
        Получить температуру GPU в градусах Цельсия

        Returns:
            float: Температура
        """
        if not self._nvidia_smi_available:
            return 0.0

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return float(result.stdout.strip())

        except Exception as e:
            logger.error(f"Ошибка получения температуры GPU: {e}")

        return 0.0

    def get_full_stats(self) -> Dict:
        """
        Получить полную статистику GPU

        Returns:
            Dict: Полная статистика
        """
        memory = self.get_memory_usage()
        utilization = self.get_utilization()
        temperature = self.get_temperature()

        return {
            "device_id": self.device_id,
            "memory": memory,
            "utilization_percent": utilization,
            "temperature_celsius": temperature,
            "available": self._nvidia_smi_available,
        }

    def _get_fallback_memory(self) -> Dict[str, float]:
        """Получить заглушку для памяти"""
        return {
            "used_mb": 0.0,
            "total_mb": 24000.0,
            "free_mb": 24000.0,
            "usage_percent": 0.0
        }

    def is_memory_critical(self, threshold: float = 90.0) -> bool:
        """
        Проверить критическое использование памяти

        Args:
            threshold: Пороговое значение в процентах

        Returns:
            bool: True если критическое
        """
        memory = self.get_memory_usage()
        return memory["usage_percent"] >= threshold

    def __str__(self) -> str:
        stats = self.get_full_stats()
        return (
            f"GPUMonitor(device={self.device_id}, "
            f"memory={stats['memory']['usage_percent']:.1f}%, "
            f"util={stats['utilization_percent']:.1f}%)"
        )
