import json

import pytest

from bucket_manager import BucketManager
from web import night_watch_handoff


TOKEN = "t" * 48
BUCKET_ID = "eb528b630955"


class FakeMCP:
    def __init__(self):
        self.routes = {}

    def custom_route(self, path, methods):
        def decorator(handler):
            for method in methods:
                self.routes[(method, path)] = handler
            return handler
        return decorator


class Request:
    def __init__(self, body, token=TOKEN):
        self._body = json.dumps(body).encode("utf-8")
        self.headers = {"authorization": f"Bearer {token}"} if token else {}

    async def body(self):
        return self._body


class Manager:
    def __init__(self):
        self.buckets = {
            BUCKET_ID: {
                "id": BUCKET_ID,
                "metadata": {
                    "name": "night_watch_morning_handoff",
                    "dont_surface": True,
                },
                "content": "legacy bootstrap template",
            },
            "other": {
                "id": "other",
                "metadata": {"name": "ordinary_memory", "dont_surface": False},
                "content": "must stay unchanged",
            },
        }
        self.updated_ids = []

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)

    async def update(self, bucket_id, **updates):
        self.updated_ids.append(bucket_id)
        self.buckets[bucket_id]["content"] = updates["content"]
        return True


def payload(**overrides):
    base = {
        "schema_version": "0.1",
        "revision": 1,
        "session_status": "sleeping",
        "sleep_session_id": "sleep-20260831-001",
        "sleep_started_at": "2026-08-31T23:00:00-07:00",
        "expected_end_at": "2026-09-01T08:00:00-07:00",
        "updated_at": "2026-08-31T23:00:00-07:00",
        "wake_count": 0,
        "wake_summaries": [],
        "closure": None,
    }
    base.update(overrides)
    return base


def handler(monkeypatch, manager=None):
    monkeypatch.setenv("OMBRE_NIGHT_WATCH_HANDOFF_TOKEN", TOKEN)
    monkeypatch.setenv("OMBRE_NIGHT_WATCH_HANDOFF_BUCKET_ID", BUCKET_ID)
    manager = manager or Manager()
    monkeypatch.setattr(night_watch_handoff.sh, "bucket_mgr", manager, raising=False)
    mcp = FakeMCP()
    night_watch_handoff.register(mcp)
    return mcp.routes[("POST", "/api/night-watch/morning-handoff")], manager


@pytest.mark.asyncio
async def test_bridge_updates_only_configured_hidden_bucket(monkeypatch):
    call, manager = handler(monkeypatch)
    response = await call(Request(payload()))
    result = json.loads(response.body)
    assert response.status_code == 200
    assert result == {"status": "updated", "revision": 1}
    assert manager.updated_ids == [BUCKET_ID]
    assert manager.buckets["other"]["content"] == "must stay unchanged"
    saved = json.loads(manager.buckets[BUCKET_ID]["content"])
    assert saved["bucket_role"] == "night_watch_morning_handoff"
    assert saved["wake_summaries"] == []


@pytest.mark.asyncio
async def test_bridge_rejects_wrong_or_missing_dedicated_token(monkeypatch):
    call, manager = handler(monkeypatch)
    for token in ("wrong", ""):
        response = await call(Request(payload(), token=token))
        assert response.status_code == 401
    assert manager.updated_ids == []


@pytest.mark.asyncio
async def test_bridge_fails_closed_without_configuration(monkeypatch):
    monkeypatch.delenv("OMBRE_NIGHT_WATCH_HANDOFF_TOKEN", raising=False)
    monkeypatch.delenv("OMBRE_NIGHT_WATCH_HANDOFF_BUCKET_ID", raising=False)
    manager = Manager()
    monkeypatch.setattr(night_watch_handoff.sh, "bucket_mgr", manager, raising=False)
    mcp = FakeMCP()
    night_watch_handoff.register(mcp)
    response = await mcp.routes[("POST", "/api/night-watch/morning-handoff")](Request(payload()))
    assert response.status_code == 503
    assert manager.updated_ids == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"bucket_id": "other"},
        {"wake_count": 1},
        {"wake_summaries": [{"wake_id": "x"}]},
        {"sleep_started_at": "2026-08-31T23:00:00"},
        {"revision": 0},
        {"closure": {"all_wakes_closed": True}},
    ],
)
async def test_bridge_rejects_unknown_or_invalid_payloads(monkeypatch, changes):
    call, manager = handler(monkeypatch)
    response = await call(Request(payload(**changes)))
    assert response.status_code == 400
    assert manager.updated_ids == []


@pytest.mark.asyncio
async def test_bridge_rejects_misconfigured_target_bucket(monkeypatch):
    call, manager = handler(monkeypatch)
    manager.buckets[BUCKET_ID]["metadata"]["name"] = "ordinary_memory"
    response = await call(Request(payload()))
    assert response.status_code == 409
    assert manager.updated_ids == []


@pytest.mark.asyncio
async def test_bridge_requires_strict_hidden_bucket_flag(monkeypatch):
    call, manager = handler(monkeypatch)
    manager.buckets[BUCKET_ID]["metadata"]["dont_surface"] = "true"
    response = await call(Request(payload()))
    assert response.status_code == 409
    assert manager.updated_ids == []


@pytest.mark.asyncio
async def test_bridge_enforces_monotonic_revision_and_idempotency(monkeypatch):
    call, manager = handler(monkeypatch)
    first = await call(Request(payload()))
    assert first.status_code == 200
    same = await call(Request(payload()))
    assert json.loads(same.body)["status"] == "idempotent"
    stale = await call(Request(payload(updated_at="2026-09-01T00:00:00-07:00")))
    assert stale.status_code == 409
    gap = await call(Request(payload(revision=3)))
    assert gap.status_code == 409
    second = await call(Request(payload(revision=2, updated_at="2026-09-01T00:00:00-07:00")))
    assert second.status_code == 200
    assert manager.updated_ids == [BUCKET_ID, BUCKET_ID]


@pytest.mark.asyncio
async def test_real_bucket_manager_leaves_every_non_target_bucket_unchanged(
    monkeypatch, test_config
):
    manager = BucketManager(test_config, embedding_engine=None)
    target_id = await manager.create(
        "legacy bootstrap template",
        name="night_watch_morning_handoff",
        domain=["night_watch"],
        bucket_id_override=BUCKET_ID,
        defer_derived_index=True,
    )
    assert target_id == BUCKET_ID
    assert await manager.update(BUCKET_ID, dont_surface=True)
    await manager.create(
        "main memory must stay byte-for-byte stable",
        name="ordinary_memory",
        domain=["life"],
        bucket_id_override="ordinary-main-bucket",
        defer_derived_index=True,
    )
    before = {
        bucket["id"]: json.dumps(bucket, ensure_ascii=False, sort_keys=True)
        for bucket in await manager.list_all(include_archive=True)
        if bucket["id"] != BUCKET_ID
    }

    call, _ = handler(monkeypatch, manager)
    response = await call(Request(payload()))

    assert response.status_code == 200
    after = {
        bucket["id"]: json.dumps(bucket, ensure_ascii=False, sort_keys=True)
        for bucket in await manager.list_all(include_archive=True)
        if bucket["id"] != BUCKET_ID
    }
    assert after == before
