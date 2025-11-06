"""
Simplified conftest for unit tests (no API dependencies)
"""
import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
