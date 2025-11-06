"""
Тесты для /analysis API
"""
import pytest
import asyncio
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_analysis_session(client: AsyncClient):
    """Тест создания сессии анализа"""
    # Сначала создаем текст
    text_response = await client.post(
        "/texts/",
        json={
            "title": "Текст для анализа",
            "content": "Это интересный текст с различными эмоциями и стилями написания.",
            "storage_type": "database"
        }
    )

    text_id = text_response.json()["id"]

    # Создаем сессию
    response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Тестовый анализ",
            "text_ids": [text_id],
            "analyzer_types": ["GenreAnalyzer", "StyleAnalyzer"],
            "mode": "full_text"
        }
    )

    assert response.status_code == 201
    data = response.json()

    assert data["name"] == "Тестовый анализ"
    assert data["status"] == "draft"
    assert data["mode"] == "full_text"
    assert len(data["text_ids"]) == 1
    assert len(data["analyzer_types"]) == 2
    assert "id" in data


@pytest.mark.asyncio
async def test_list_sessions(client: AsyncClient):
    """Тест получения списка сессий"""
    # Создаем текст
    text_response = await client.post(
        "/texts/",
        json={
            "title": "Текст",
            "content": "Содержимое",
            "storage_type": "database"
        }
    )

    text_id = text_response.json()["id"]

    # Создаем несколько сессий
    for i in range(3):
        await client.post(
            "/analysis/sessions",
            json={
                "name": f"Сессия {i+1}",
                "text_ids": [text_id],
                "analyzer_types": ["GenreAnalyzer"],
                "mode": "full_text"
            }
        )

    # Получаем список
    response = await client.get("/analysis/sessions?offset=0&limit=10")

    assert response.status_code == 200
    data = response.json()

    assert "sessions" in data
    assert "total" in data
    assert data["total"] >= 3


@pytest.mark.asyncio
async def test_get_session(client: AsyncClient):
    """Тест получения сессии по ID"""
    # Создаем текст
    text_response = await client.post(
        "/texts/",
        json={
            "title": "Текст",
            "content": "Содержимое",
            "storage_type": "database"
        }
    )

    text_id = text_response.json()["id"]

    # Создаем сессию
    create_response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Сессия для получения",
            "text_ids": [text_id],
            "analyzer_types": ["GenreAnalyzer"],
            "mode": "full_text"
        }
    )

    session_id = create_response.json()["id"]

    # Получаем сессию
    response = await client.get(f"/analysis/sessions/{session_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == session_id
    assert data["name"] == "Сессия для получения"
    assert "results" in data


@pytest.mark.asyncio
async def test_run_session_async(client: AsyncClient):
    """Тест асинхронного запуска сессии"""
    # Создаем текст
    text_response = await client.post(
        "/texts/",
        json={
            "title": "Текст для анализа",
            "content": "Это текст с различными стилями и жанрами написания.",
            "storage_type": "database"
        }
    )

    text_id = text_response.json()["id"]

    # Создаем сессию
    create_response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Асинхронный анализ",
            "text_ids": [text_id],
            "analyzer_types": ["PaceAnalyzer"],  # Быстрый анализатор без LLM
            "mode": "full_text"
        }
    )

    session_id = create_response.json()["id"]

    # Запускаем асинхронно
    response = await client.post(
        f"/analysis/sessions/{session_id}/run?async_mode=true"
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "queued"
    assert data["session_id"] == session_id


@pytest.mark.asyncio
async def test_delete_session(client: AsyncClient):
    """Тест удаления сессии"""
    # Создаем текст
    text_response = await client.post(
        "/texts/",
        json={
            "title": "Текст",
            "content": "Содержимое",
            "storage_type": "database"
        }
    )

    text_id = text_response.json()["id"]

    # Создаем сессию
    create_response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Сессия для удаления",
            "text_ids": [text_id],
            "analyzer_types": ["GenreAnalyzer"],
            "mode": "full_text"
        }
    )

    session_id = create_response.json()["id"]

    # Удаляем
    response = await client.delete(f"/analysis/sessions/{session_id}")

    assert response.status_code == 204

    # Проверяем что удалена
    get_response = await client.get(f"/analysis/sessions/{session_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_session_validation(client: AsyncClient):
    """Тест валидации при создании сессии"""
    # Без текстов
    response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Сессия без текстов",
            "text_ids": [],
            "analyzer_types": ["GenreAnalyzer"],
            "mode": "full_text"
        }
    )

    assert response.status_code == 400

    # Без анализаторов
    text_response = await client.post(
        "/texts/",
        json={
            "title": "Текст",
            "content": "Содержимое",
            "storage_type": "database"
        }
    )

    text_id = text_response.json()["id"]

    response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Сессия без анализаторов",
            "text_ids": [text_id],
            "analyzer_types": [],
            "mode": "full_text"
        }
    )

    assert response.status_code == 400

    # Более 5 текстов
    response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Слишком много текстов",
            "text_ids": ["id1", "id2", "id3", "id4", "id5", "id6"],
            "analyzer_types": ["GenreAnalyzer"],
            "mode": "full_text"
        }
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_full_analysis_workflow(client: AsyncClient, fb2_file_path_1: str):
    """Полный тест workflow анализа с реальным FB2 файлом"""
    # 1. Создаем FB2 текст
    text_response = await client.post(
        "/texts/fb2",
        json={
            "title": "Эволюция Хакайна",
            "file_path": fb2_file_path_1,
            "parse_metadata": False
        }
    )

    assert text_response.status_code == 201
    text_id = text_response.json()["id"]

    # 2. Создаем сессию анализа
    session_response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Полный анализ книги",
            "text_ids": [text_id],
            "analyzer_types": [
                "PaceAnalyzer",  # Быстрый, без LLM
                "ReadabilityAnalyzer"  # Быстрый, без LLM
            ],
            "mode": "full_text"
        }
    )

    assert session_response.status_code == 201
    session_id = session_response.json()["id"]

    # 3. Запускаем анализ
    run_response = await client.post(
        f"/analysis/sessions/{session_id}/run?async_mode=true"
    )

    assert run_response.status_code == 200

    # 4. Ждем немного (в реальном тесте использовали бы WebSocket или polling)
    await asyncio.sleep(2)

    # 5. Получаем результаты
    result_response = await client.get(f"/analysis/sessions/{session_id}")

    assert result_response.status_code == 200
    data = result_response.json()

    # Проверяем что сессия выполнена или в процессе
    assert data["status"] in ["queued", "running", "completed"]


@pytest.mark.asyncio
async def test_multi_text_analysis(client: AsyncClient, fb2_file_path_1: str, fb2_file_path_2: str):
    """Тест анализа нескольких текстов"""
    # Создаем два текста
    text1_response = await client.post(
        "/texts/fb2",
        json={
            "title": "Эволюция Хакайна",
            "file_path": fb2_file_path_1,
            "parse_metadata": False
        }
    )

    text2_response = await client.post(
        "/texts/fb2",
        json={
            "title": "Зона дремлет",
            "file_path": fb2_file_path_2,
            "parse_metadata": False
        }
    )

    text_id_1 = text1_response.json()["id"]
    text_id_2 = text2_response.json()["id"]

    # Создаем сессию для обоих текстов
    session_response = await client.post(
        "/analysis/sessions",
        json={
            "name": "Сравнительный анализ двух книг",
            "text_ids": [text_id_1, text_id_2],
            "analyzer_types": ["PaceAnalyzer", "StructureAnalyzer"],
            "mode": "full_text"
        }
    )

    assert session_response.status_code == 201
    data = session_response.json()

    assert len(data["text_ids"]) == 2
    assert len(data["analyzer_types"]) == 2
