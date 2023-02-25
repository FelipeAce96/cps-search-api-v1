import pytest

from app import say_hello, root


@pytest.mark.asyncio
async def test_root():
    result = await root()
    assert result == {'message': 'Hello World'}

