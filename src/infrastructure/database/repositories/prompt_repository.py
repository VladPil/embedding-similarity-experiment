"""
Репозиторий для работы с промпт-шаблонами
"""
from typing import Optional, List
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from ..models import PromptTemplateModel
from src.common.exceptions import DatabaseOperationError
from src.common.utils import generate_id


class PromptTemplateRepository:
    """Репозиторий для работы с промпт-шаблонами"""

    def __init__(self, session: AsyncSession):
        """
        Args:
            session: Сессия SQLAlchemy
        """
        self.session = session

    async def create(
        self,
        name: str,
        analyzer_type: str,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        output_schema: Optional[dict] = None,
        is_default: bool = False
    ) -> PromptTemplateModel:
        """
        Создать промпт-шаблон

        Args:
            name: Название шаблона
            analyzer_type: Тип анализатора
            system_prompt: Системный промпт
            user_prompt_template: Шаблон пользовательского промпта
            temperature: Температура генерации
            max_tokens: Максимум токенов
            output_schema: JSON-схема выходных данных
            is_default: Является ли дефолтным

        Returns:
            Созданный шаблон
        """
        try:
            # Если делаем дефолтным, снимаем флаг с других
            if is_default:
                await self._unset_default_for_analyzer(analyzer_type)

            template = PromptTemplateModel(
                id=generate_id("prompt"),
                name=name,
                analyzer_type=analyzer_type,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                temperature=temperature,
                max_tokens=max_tokens,
                output_schema=output_schema,
                is_default=is_default
            )

            self.session.add(template)
            await self.session.commit()
            await self.session.refresh(template)

            logger.info(f"Создан промпт-шаблон: {template.id} - {name}")
            return template

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка создания промпт-шаблона: {e}")
            raise DatabaseOperationError(
                message=f"Failed to create prompt template: {e}",
                details={"name": name, "analyzer_type": analyzer_type}
            )

    async def get_by_id(self, template_id: str) -> Optional[PromptTemplateModel]:
        """
        Получить шаблон по ID

        Args:
            template_id: ID шаблона

        Returns:
            Шаблон или None
        """
        try:
            stmt = select(PromptTemplateModel).where(
                PromptTemplateModel.id == template_id
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Ошибка получения шаблона {template_id}: {e}")
            return None

    async def get_default_for_analyzer(
        self,
        analyzer_type: str
    ) -> Optional[PromptTemplateModel]:
        """
        Получить дефолтный шаблон для анализатора

        Args:
            analyzer_type: Тип анализатора

        Returns:
            Шаблон или None
        """
        try:
            stmt = select(PromptTemplateModel).where(
                and_(
                    PromptTemplateModel.analyzer_type == analyzer_type,
                    PromptTemplateModel.is_default == True
                )
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Ошибка получения дефолтного шаблона: {e}")
            return None

    async def list_for_analyzer(
        self,
        analyzer_type: str
    ) -> List[PromptTemplateModel]:
        """
        Получить все шаблоны для анализатора

        Args:
            analyzer_type: Тип анализатора

        Returns:
            Список шаблонов
        """
        try:
            stmt = select(PromptTemplateModel).where(
                PromptTemplateModel.analyzer_type == analyzer_type
            ).order_by(
                PromptTemplateModel.is_default.desc(),
                PromptTemplateModel.created_at.desc()
            )

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Ошибка получения шаблонов для {analyzer_type}: {e}")
            return []

    async def list_all(self) -> List[PromptTemplateModel]:
        """
        Получить все шаблоны

        Returns:
            Список всех шаблонов
        """
        try:
            stmt = select(PromptTemplateModel).order_by(
                PromptTemplateModel.analyzer_type,
                PromptTemplateModel.is_default.desc(),
                PromptTemplateModel.created_at.desc()
            )

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Ошибка получения всех шаблонов: {e}")
            return []

    async def update(
        self,
        template_id: str,
        **kwargs
    ) -> Optional[PromptTemplateModel]:
        """
        Обновить шаблон

        Args:
            template_id: ID шаблона
            **kwargs: Поля для обновления

        Returns:
            Обновлённый шаблон или None
        """
        try:
            template = await self.get_by_id(template_id)
            if not template:
                return None

            # Если меняем на дефолтный, снимаем флаг с других
            if kwargs.get('is_default') and not template.is_default:
                await self._unset_default_for_analyzer(template.analyzer_type)

            for key, value in kwargs.items():
                if hasattr(template, key):
                    setattr(template, key, value)

            await self.session.commit()
            await self.session.refresh(template)

            logger.info(f"Обновлён промпт-шаблон: {template_id}")
            return template

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка обновления шаблона {template_id}: {e}")
            return None

    async def delete(self, template_id: str) -> bool:
        """
        Удалить шаблон

        Args:
            template_id: ID шаблона

        Returns:
            True если удалён
        """
        try:
            template = await self.get_by_id(template_id)
            if not template:
                return False

            await self.session.delete(template)
            await self.session.commit()

            logger.info(f"Удалён промпт-шаблон: {template_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка удаления шаблона {template_id}: {e}")
            return False

    async def _unset_default_for_analyzer(self, analyzer_type: str) -> None:
        """
        Снять флаг is_default со всех шаблонов анализатора

        Args:
            analyzer_type: Тип анализатора
        """
        try:
            stmt = select(PromptTemplateModel).where(
                and_(
                    PromptTemplateModel.analyzer_type == analyzer_type,
                    PromptTemplateModel.is_default == True
                )
            )
            result = await self.session.execute(stmt)
            templates = result.scalars().all()

            for template in templates:
                template.is_default = False

            if templates:
                await self.session.commit()

        except Exception as e:
            logger.error(f"Ошибка снятия флага default: {e}")
