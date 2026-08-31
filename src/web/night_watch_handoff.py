"""Narrow, token-isolated writer for the Night Watch morning handoff bucket.

This route deliberately does not reuse Dashboard or MCP credentials.  Its token
can update exactly one configured bucket and the request schema is rendered by
the server, so callers cannot smuggle arbitrary ``trace`` fields or another
bucket id through the bridge.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import _shared as sh


_ROUTE = "/api/night-watch/morning-handoff"
_EXPECTED_BUCKET_NAME = "night_watch_morning_handoff"
_MAX_BODY_BYTES = 16_384
_MAX_RENDERED_CHARS = 12_000
_STATUSES = {"idle", "sleeping", "completed", "failed_safe"}
_OUTCOMES = {"think", "silent", "defer"}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "revision",
    "session_status",
    "sleep_session_id",
    "sleep_started_at",
    "expected_end_at",
    "updated_at",
    "wake_count",
    "wake_summaries",
    "closure",
}
_WAKE_KEYS = {
    "wake_id",
    "woke_at",
    "outcome",
    "chosen_topic",
    "selection_reason",
    "concise_reflection",
    "tools_used",
    "requires_attention",
}
_CLOSURE_KEYS = {
    "all_activities_unique",
    "all_acks_completed",
    "all_wakes_closed",
    "delivery_disabled",
    "external_actions_empty",
}


def _configured_token() -> str:
    return os.environ.get("OMBRE_NIGHT_WATCH_HANDOFF_TOKEN", "").strip()


def _configured_bucket_id() -> str:
    return os.environ.get("OMBRE_NIGHT_WATCH_HANDOFF_BUCKET_ID", "").strip()


def _authorized(request: Request, token: str) -> bool:
    header = str(request.headers.get("authorization") or "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if len(token) < 32 or not supplied:
        return False
    return hmac.compare_digest(
        hashlib.sha256(supplied.encode("utf-8")).digest(),
        hashlib.sha256(token.encode("utf-8")).digest(),
    )


def _text(value: Any, field: str, maximum: int, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{field} is required")
    if len(value) > maximum:
        raise ValueError(f"{field} is too long")
    return value


def _instant(value: Any, field: str, *, required: bool = False) -> str | None:
    text = _text(value, field, 64, required=required)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return text


def _validate_wake(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict) or set(item) - _WAKE_KEYS:
        raise ValueError("wake_summaries contains unknown fields")
    outcome = _text(item.get("outcome"), "outcome", 16, required=True)
    if outcome not in _OUTCOMES:
        raise ValueError("outcome is invalid")
    tools = item.get("tools_used", [])
    if not isinstance(tools, list) or len(tools) > 6:
        raise ValueError("tools_used must contain at most 6 entries")
    clean_tools = [_text(tool, "tools_used[]", 160, required=True) for tool in tools]
    attention = item.get("requires_attention", False)
    if not isinstance(attention, bool):
        raise ValueError("requires_attention must be boolean")
    return {
        "wake_id": _text(item.get("wake_id"), "wake_id", 160, required=True),
        "woke_at": _instant(item.get("woke_at"), "woke_at", required=True),
        "outcome": outcome,
        "chosen_topic": _text(item.get("chosen_topic"), "chosen_topic", 500),
        "selection_reason": _text(item.get("selection_reason"), "selection_reason", 1000),
        "concise_reflection": _text(item.get("concise_reflection"), "concise_reflection", 2000, required=True),
        "tools_used": clean_tools,
        "requires_attention": attention,
    }


def _validate_payload(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict) or set(body) - _TOP_LEVEL_KEYS:
        raise ValueError("request contains unknown fields")
    if body.get("schema_version") != "0.1":
        raise ValueError("schema_version must be 0.1")
    revision = body.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("revision must be a positive integer")
    status = body.get("session_status")
    if status not in _STATUSES:
        raise ValueError("session_status is invalid")
    session_id = _text(
        body.get("sleep_session_id"),
        "sleep_session_id",
        160,
        required=status != "idle",
    )
    summaries = body.get("wake_summaries", [])
    if not isinstance(summaries, list) or len(summaries) > 3:
        raise ValueError("wake_summaries must contain at most 3 entries")
    clean_summaries = [_validate_wake(item) for item in summaries]
    wake_count = body.get("wake_count")
    if wake_count != len(clean_summaries):
        raise ValueError("wake_count must match wake_summaries")
    closure = body.get("closure")
    if closure is not None:
        if not isinstance(closure, dict) or set(closure) - _CLOSURE_KEYS:
            raise ValueError("closure contains unknown fields")
        if set(closure) != _CLOSURE_KEYS or not all(isinstance(v, bool) for v in closure.values()):
            raise ValueError("closure must contain all boolean safety fields")
    clean = {
        "schema_version": "0.1",
        "bucket_role": _EXPECTED_BUCKET_NAME,
        "revision": revision,
        "session_status": status,
        "sleep_session_id": session_id,
        "sleep_started_at": _instant(body.get("sleep_started_at"), "sleep_started_at", required=status != "idle"),
        "expected_end_at": _instant(body.get("expected_end_at"), "expected_end_at", required=status != "idle"),
        "updated_at": _instant(body.get("updated_at"), "updated_at", required=True),
        "wake_count": wake_count,
        "wake_summaries": clean_summaries,
        "closure": closure,
    }
    if status in {"completed", "failed_safe"} and closure is None:
        raise ValueError("completed sessions require closure")
    rendered = json.dumps(clean, ensure_ascii=False, indent=2, sort_keys=True)
    if len(rendered) > _MAX_RENDERED_CHARS:
        raise ValueError("rendered handoff is too large")
    return clean


def _current_revision(bucket: dict[str, Any]) -> int:
    try:
        current = json.loads(str(bucket.get("content") or ""))
        revision = current.get("revision")
        return revision if isinstance(revision, int) and not isinstance(revision, bool) else 0
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def register(mcp) -> None:
    @mcp.custom_route(_ROUTE, methods=["POST"])
    async def night_watch_morning_handoff(request: Request) -> Response:
        token = _configured_token()
        bucket_id = _configured_bucket_id()
        if len(token) < 32 or not bucket_id:
            return JSONResponse({"error": "bridge_disabled"}, status_code=503)
        if not _authorized(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        raw = await request.body()
        if len(raw) > _MAX_BODY_BYTES:
            return JSONResponse({"error": "request_too_large"}, status_code=413)
        try:
            body = json.loads(raw.decode("utf-8"))
            clean = _validate_payload(body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return JSONResponse({"error": "invalid_handoff", "detail": str(exc)}, status_code=400)

        bucket = await sh.bucket_mgr.get(bucket_id)
        metadata = (bucket or {}).get("metadata") or {}
        if (
            not bucket
            or metadata.get("name") != _EXPECTED_BUCKET_NAME
            or metadata.get("dont_surface") is not True
        ):
            return JSONResponse({"error": "configured_bucket_mismatch"}, status_code=409)

        current_revision = _current_revision(bucket)
        rendered = json.dumps(clean, ensure_ascii=False, indent=2, sort_keys=True)
        if clean["revision"] == current_revision and str(bucket.get("content") or "") == rendered:
            return JSONResponse({"status": "idempotent", "revision": current_revision})
        if clean["revision"] <= current_revision:
            return JSONResponse(
                {"error": "stale_revision", "current_revision": current_revision},
                status_code=409,
            )
        if clean["revision"] != current_revision + 1:
            return JSONResponse(
                {"error": "revision_gap", "current_revision": current_revision},
                status_code=409,
            )
        updated = await sh.bucket_mgr.update(
            bucket_id,
            content=rendered,
            event_actor="night_watch_handoff_bridge",
        )
        if not updated:
            return JSONResponse({"error": "update_failed"}, status_code=500)
        return JSONResponse({"status": "updated", "revision": clean["revision"]})
