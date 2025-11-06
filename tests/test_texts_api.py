"""
Тесты для /texts API
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_text(client: AsyncClient):
    """Тест создания текста"""
    response = await client.post(
        "/texts/",
        json={
            "title": "Тестовый текст",
            "content": "Это содержимое тестового текста для проверки API.",
            "storage_type": "database",
            "metadata": {"author": "Test Author"}
        }
    )

    assert response.status_code == 201
    data = response.json()

    assert data["title"] == "Тестовый текст"
    assert data["storage_type"] == "database"
    assert data["content_length"] > 0
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_fb2_text(client: AsyncClient, fb2_file_path_1: str):
    """Тест создания FB2 текста"""
    response = await client.post(
        "/texts/fb2",
        json={
            "title": "Эволюция Хакайна",
            "file_path": fb2_file_path_1,
            "parse_metadata": True
        }
    )

    assert response.status_code == 201
    data = response.json()

    assert data["title"] == "Эволюция Хакайна"
    assert data["storage_type"] == "file"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_texts(client: AsyncClient):
    """Тест получения списка текстов"""
    # Создаем несколько текстов
    for i in range(3):
        await client.post(
            "/texts/",
            json={
                "title": f"Текст {i+1}",
                "content": f"Содержимое текста {i+1}",
                "storage_type": "database"
            }
        )

    # Получаем список
    response = await client.get("/texts/?offset=0&limit=10")

    assert response.status_code == 200
    data = response.json()

    assert "texts" in data
    assert "total" in data
    assert data["total"] >= 3
    assert len(data["texts"]) >= 3


@pytest.mark.asyncio
async def test_get_text(client: AsyncClient):
    """Тест получения текста по ID"""
    # Создаем текст
    create_response = await client.post(
        "/texts/",
        json={
            "title": "Текст для получения",
            "content": "Содержимое",
            "storage_type": "database"
        }
    )

    text_id = create_response.json()["id"]

    # Получаем текст
    response = await client.get(f"/texts/{text_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == text_id
    assert data["title"] == "Текст для получения"


@pytest.mark.asyncio
async def test_get_text_content(client: AsyncClient):
    """Тест получения содержимого текста"""
    # Создаем текст
    create_response = await client.post(
        "/texts/",
        json={
            "title": "Текст с содержимым",
            "content": "Это тестовое содержимое текста",
            "storage_type": "database"
        }
    )

    text_id = create_response.json()["id"]

    # Получаем содержимое
    response = await client.get(f"/texts/{text_id}/content")

    assert response.status_code == 200
    data = response.json()

    assert data["text_id"] == text_id
    assert data["content"] == "Это тестовое содержимое текста"


@pytest.mark.asyncio
async def test_chunk_text(client: AsyncClient):
    """Тест чанкинга текста"""
    # Создаем текст
    long_text = " ".join(["Предложение номер {}.".format(i) for i in range(100)])

    create_response = await client.post(
        "/texts/",
        json={
            "title": "Длинный текст",
            "content": long_text,
            "storage_type": "database"
        }
    )

    text_id = create_response.json()["id"]

    # Чанкуем
    response = await client.post(
        "/texts/chunk",
        json={
            "text_id": text_id,
            "chunk_size": 100,
            "overlap": 20,
            "boundary_type": "sentence"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["text_id"] == text_id
    assert data["total_chunks"] > 0
    assert len(data["chunks"]) > 0
    assert data["chunks"][0]["index"] == 0


@pytest.mark.asyncio
async def test_delete_text(client: AsyncClient):
    """Тест удаления текста"""
    # Создаем текст
    create_response = await client.post(
        "/texts/",
        json={
            "title": "Текст для удаления",
            "content": "Содержимое",
            "storage_type": "database"
        }
    )

    text_id = create_response.json()["id"]

    # Удаляем
    response = await client.delete(f"/texts/{text_id}")

    assert response.status_code == 204

    # Проверяем что удалён
    get_response = await client.get(f"/texts/{text_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_create_text_validation(client: AsyncClient):
    """Тест валидации при создании текста"""
    # Без content для database
    response = await client.post(
        "/texts/",
        json={
            "title": "Текст без content",
            "storage_type": "database"
        }
    )

    assert response.status_code == 400

    # Без file_path для file
    response = await client.post(
        "/texts/",
        json={
            "title": "Текст без file_path",
            "storage_type": "file"
        }
    )

    assert response.status_code == 400
