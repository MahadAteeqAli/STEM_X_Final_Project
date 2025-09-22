# tests/test_main.py

import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import Session, select

from main import app
from database import create_db_and_tables, engine
from models import User

@pytest.fixture(scope="module", autouse=True)
def initialize_db():
    # Recreate tables before tests
    create_db_and_tables()
    yield
    # (Optional) tear-down here

@pytest.mark.asyncio
async def test_signup_and_login():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Signup
        resp = await ac.post("/signup", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123"
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "user_id" in data

        # Login
        resp = await ac.post("/login",
            data={"username": "alice", "password": "password123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert resp.status_code == 200
        token = resp.json().get("access_token")
        assert token

@pytest.mark.asyncio
async def test_cbf_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Login to get token
        await ac.post("/signup", json={
            "username": "bob",
            "email": "bob@example.com",
            "password": "pass456"
        })
        login = await ac.post("/login",
            data={"username": "bob", "password": "pass456"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Call CBF
        resp = await ac.get("/recommend", params={"q": "Data Science", "k": 3}, headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "recommendations" in body
        assert len(body["recommendations"]) == 3

@pytest.mark.asyncio
async def test_cf_and_hybrid_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Login to get token
        login = await ac.post("/login",
            data={"username": "bob", "password": "pass456"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # CF-only
        resp_cf = await ac.get("/recommend/cf", params={"user_id": 1, "k": 3}, headers=headers)
        assert resp_cf.status_code == 200
        cf_body = resp_cf.json()
        assert len(cf_body["recommendations"]) == 3

        # Hybrid
        resp_h = await ac.get("/recommend/hybrid", params={"user_id": 1, "q": "Science", "k": 3}, headers=headers)
        assert resp_h.status_code == 200
        h_body = resp_h.json()
        assert "cbf" in h_body and "cf" in h_body and "hybrid" in h_body
        assert len(h_body["hybrid"]) == 3
