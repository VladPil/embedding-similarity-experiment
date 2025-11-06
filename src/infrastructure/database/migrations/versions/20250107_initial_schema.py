"""Initial schema

Revision ID: 001_initial
Revises:
Create Date: 2025-01-07 12:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Создание всех таблиц схемы
    """

    # ===== ТЕКСТЫ =====
    op.create_table(
        'texts',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('text_type', sa.String(20), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('language', sa.String(10), nullable=True),
        sa.Column('storage_type', sa.String(20), nullable=False),
        sa.Column('content', sa.Text, nullable=True),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('length', sa.Integer, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=False, server_default='{}'),
        sa.Column('author', sa.String(200), nullable=True),
        sa.Column('genre', postgresql.JSONB, nullable=True),
        sa.Column('year', sa.Integer, nullable=True),
        sa.Column('publisher', sa.String(200), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_texts_created_at', 'texts', ['created_at'])
    op.create_index('idx_texts_title', 'texts', ['title'])
    op.create_index('idx_texts_type', 'texts', ['text_type'])

    # ===== СТРАТЕГИИ ЧАНКИНГА =====
    op.create_table(
        'chunking_strategies',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('base_chunk_size', sa.Integer, nullable=False, server_default='2000'),
        sa.Column('min_chunk_size', sa.Integer, nullable=False, server_default='500'),
        sa.Column('max_chunk_size', sa.Integer, nullable=False, server_default='4000'),
        sa.Column('overlap_percentage', sa.Float, nullable=False, server_default='0.1'),
        sa.Column('use_sentence_boundaries', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('use_paragraph_boundaries', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('balance_chunks', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('is_default', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # ===== СЕССИИ АНАЛИЗА =====
    op.create_table(
        'analysis_sessions',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('mode', sa.String(20), nullable=False, server_default='full_text'),
        sa.Column('chunking_strategy_id', sa.String(64), sa.ForeignKey('chunking_strategies.id'), nullable=True),
        sa.Column('chunked_comparison_strategy', sa.String(20), nullable=True),
        sa.Column('use_faiss_search', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('faiss_index_id', sa.String(64), nullable=True),
        sa.Column('similarity_top_k', sa.Integer, nullable=False, server_default='10'),
        sa.Column('similarity_threshold', sa.Float, nullable=False, server_default='0.7'),
        sa.Column('progress', sa.Integer, nullable=False, server_default='0'),
        sa.Column('progress_message', sa.Text, nullable=True),
        sa.Column('result', postgresql.JSONB, nullable=True),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('queued_at', sa.DateTime, nullable=True),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('user_id', sa.String(64), nullable=True),
    )
    op.create_index('idx_session_status', 'analysis_sessions', ['status'])
    op.create_index('idx_session_created_at', 'analysis_sessions', ['created_at'])
    op.create_index('idx_session_user', 'analysis_sessions', ['user_id'])

    # ===== СВЯЗИ СЕССИЙ С ТЕКСТАМИ =====
    op.create_table(
        'session_texts',
        sa.Column('session_id', sa.String(64), sa.ForeignKey('analysis_sessions.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('text_id', sa.String(64), sa.ForeignKey('texts.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('position', sa.Integer, nullable=False),
    )

    # ===== СВЯЗИ СЕССИЙ С АНАЛИЗАТОРАМИ =====
    op.create_table(
        'session_analyzers',
        sa.Column('session_id', sa.String(64), sa.ForeignKey('analysis_sessions.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('analyzer_name', sa.String(50), primary_key=True),
        sa.Column('position', sa.Integer, nullable=False),
    )

    # ===== РЕЗУЛЬТАТЫ АНАЛИЗОВ =====
    op.create_table(
        'analysis_results',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('session_id', sa.String(64), sa.ForeignKey('analysis_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('text_id', sa.String(64), sa.ForeignKey('texts.id', ondelete='CASCADE'), nullable=False),
        sa.Column('analyzer_name', sa.String(50), nullable=False),
        sa.Column('result_data', postgresql.JSONB, nullable=False),
        sa.Column('interpretation', sa.Text, nullable=True),
        sa.Column('execution_time_ms', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index('idx_result_session', 'analysis_results', ['session_id'])
    op.create_index('idx_result_text_analyzer', 'analysis_results', ['text_id', 'analyzer_name'])

    # ===== МАТРИЦЫ СРАВНЕНИЙ =====
    op.create_table(
        'comparison_matrices',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('session_id', sa.String(64), sa.ForeignKey('analysis_sessions.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('matrix_data', postgresql.JSONB, nullable=False),
        sa.Column('aggregated_scores', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # ===== ПРОМПТ-ШАБЛОНЫ =====
    op.create_table(
        'prompt_templates',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('analyzer_type', sa.String(50), nullable=False),
        sa.Column('system_prompt', sa.Text, nullable=False),
        sa.Column('user_prompt_template', sa.Text, nullable=False),
        sa.Column('temperature', sa.Float, nullable=False, server_default='0.7'),
        sa.Column('max_tokens', sa.Integer, nullable=False, server_default='1000'),
        sa.Column('output_schema', postgresql.JSONB, nullable=True),
        sa.Column('is_default', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index('idx_prompt_analyzer_type', 'prompt_templates', ['analyzer_type'])

    # ===== КОНФИГУРАЦИИ МОДЕЛЕЙ =====
    op.create_table(
        'model_configs',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('model_type', sa.String(20), nullable=False),
        sa.Column('model_name', sa.String(200), nullable=False),
        sa.Column('model_path', sa.Text, nullable=True),
        sa.Column('quantization', sa.String(20), nullable=True),
        sa.Column('max_memory_gb', sa.Float, nullable=True),
        sa.Column('dimensions', sa.Integer, nullable=True),
        sa.Column('batch_size', sa.Integer, nullable=False, server_default='32'),
        sa.Column('device', sa.String(20), nullable=False, server_default='cuda'),
        sa.Column('priority', sa.Integer, nullable=False, server_default='0'),
        sa.Column('is_enabled', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('usage_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('last_used_at', sa.DateTime, nullable=True),
        sa.Column('avg_inference_time_ms', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_unique_constraint('uq_model_type_name', 'model_configs', ['model_type', 'model_name'])
    op.create_index('idx_model_type', 'model_configs', ['model_type'])
    op.create_index('idx_model_enabled', 'model_configs', ['is_enabled'])

    # ===== МЕТРИКИ МОДЕЛЕЙ =====
    op.create_table(
        'model_metrics',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('model_config_id', sa.String(64), sa.ForeignKey('model_configs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('gpu_memory_used_mb', sa.Float, nullable=True),
        sa.Column('gpu_utilization_percent', sa.Float, nullable=True),
        sa.Column('inference_time_ms', sa.Float, nullable=True),
        sa.Column('task_id', sa.String(64), nullable=True),
        sa.Column('task_type', sa.String(50), nullable=True),
        sa.Column('input_tokens', sa.Integer, nullable=True),
        sa.Column('output_tokens', sa.Integer, nullable=True),
        sa.Column('success', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('error', sa.Text, nullable=True),
    )
    op.create_index('idx_metrics_model_timestamp', 'model_metrics', ['model_config_id', 'timestamp'])
    op.create_index('idx_metrics_task', 'model_metrics', ['task_id'])

    # ===== FAISS ИНДЕКСЫ =====
    op.create_table(
        'faiss_indexes',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('model_name', sa.String(200), nullable=False),
        sa.Column('index_type', sa.String(20), nullable=False),
        sa.Column('nlist', sa.Integer, nullable=True),
        sa.Column('nprobe', sa.Integer, nullable=True),
        sa.Column('hnsw_m', sa.Integer, nullable=True),
        sa.Column('pq_m', sa.Integer, nullable=True),
        sa.Column('pq_nbits', sa.Integer, nullable=True),
        sa.Column('dimension', sa.Integer, nullable=False),
        sa.Column('total_vectors', sa.Integer, nullable=False, server_default='0'),
        sa.Column('file_path', sa.Text, nullable=True),
        sa.Column('use_gpu', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('gpu_id', sa.Integer, nullable=False, server_default='0'),
        sa.Column('last_rebuilt', sa.DateTime, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # ===== FAISS МАППИНГИ =====
    op.create_table(
        'faiss_vector_mappings',
        sa.Column('index_id', sa.String(64), sa.ForeignKey('faiss_indexes.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('position', sa.Integer, primary_key=True),
        sa.Column('text_id', sa.String(64), sa.ForeignKey('texts.id', ondelete='CASCADE'), nullable=False),
    )
    op.create_index('idx_vector_mapping_text', 'faiss_vector_mappings', ['index_id', 'text_id'])

    # ===== КЭШ EMBEDDINGS =====
    op.create_table(
        'embedding_cache',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('text_id', sa.String(64), sa.ForeignKey('texts.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_name', sa.String(200), nullable=False),
        sa.Column('embedding', postgresql.JSONB, nullable=False),
        sa.Column('dimensions', sa.Integer, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_unique_constraint('uq_text_model', 'embedding_cache', ['text_id', 'model_name'])
    op.create_index('idx_embedding_text', 'embedding_cache', ['text_id'])
    op.create_index('idx_embedding_model', 'embedding_cache', ['model_name'])

    # ===== СИСТЕМНЫЕ НАСТРОЙКИ =====
    op.create_table(
        'system_settings',
        sa.Column('id', sa.String(64), primary_key=True, server_default='system'),
        sa.Column('default_llm_model', sa.String(200), nullable=True),
        sa.Column('default_embedding_model', sa.String(200), nullable=True),
        sa.Column('default_chunking_strategy_id', sa.String(64), nullable=True),
        sa.Column('max_concurrent_llm_tasks', sa.Integer, nullable=False, server_default='2'),
        sa.Column('max_concurrent_embedding_tasks', sa.Integer, nullable=False, server_default='4'),
        sa.Column('max_texts_per_session', sa.Integer, nullable=False, server_default='5'),
        sa.Column('redis_ttl_embeddings', sa.Integer, nullable=False, server_default='86400'),
        sa.Column('redis_ttl_analysis', sa.Integer, nullable=False, server_default='3600'),
        sa.Column('task_timeout_seconds', sa.Integer, nullable=False, server_default='3600'),
        sa.Column('max_retries', sa.Integer, nullable=False, server_default='3'),
        sa.Column('ui_settings', postgresql.JSONB, nullable=False, server_default='{}'),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("id = 'system'", name='check_singleton'),
    )

    # ===== ИСТОРИЯ ЗАДАЧ =====
    op.create_table(
        'task_history',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('session_id', sa.String(64), sa.ForeignKey('analysis_sessions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('payload', postgresql.JSONB, nullable=True),
        sa.Column('result', postgresql.JSONB, nullable=True),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('queued_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('execution_time_ms', sa.Float, nullable=True),
    )
    op.create_index('idx_task_status_queued', 'task_history', ['status', 'queued_at'])
    op.create_index('idx_task_session', 'task_history', ['session_id'])
    op.create_index('idx_task_type', 'task_history', ['task_type'])


def downgrade() -> None:
    """
    Удаление всех таблиц
    """
    op.drop_table('task_history')
    op.drop_table('system_settings')
    op.drop_table('embedding_cache')
    op.drop_table('faiss_vector_mappings')
    op.drop_table('faiss_indexes')
    op.drop_table('model_metrics')
    op.drop_table('model_configs')
    op.drop_table('prompt_templates')
    op.drop_table('comparison_matrices')
    op.drop_table('analysis_results')
    op.drop_table('session_analyzers')
    op.drop_table('session_texts')
    op.drop_table('analysis_sessions')
    op.drop_table('chunking_strategies')
    op.drop_table('texts')
