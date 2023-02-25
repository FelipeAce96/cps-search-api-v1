import pytest

from app import root


@pytest.mark.asyncio
async def test_root():
    result = await root()
    assert result == {'message': 'Hello World'}

