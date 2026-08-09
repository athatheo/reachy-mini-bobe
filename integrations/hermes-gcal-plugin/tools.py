"""Scoped Google Calendar tools for Hermes: list + create, nothing else.

Google's OAuth scopes cannot express "create but never modify" — the calendar
scope that allows inserts also allows updates and deletes. The restriction is
therefore enforced at the tool layer: this plugin registers exactly two tools
(list events, create event) and deliberately registers no update/delete/move
tools, so the model has no way to alter or remove existing events.

Install: symlink this directory to ``~/.hermes/plugins/gcal/``, set
``GCAL_SERVICE_ACCOUNT_FILE`` and ``CALENDAR_ID`` in ``~/.hermes/.env``,
enable the plugin, then restart the gateway. Add the ``gcal`` toolset to the
platforms that should see it (``platform_toolsets`` in config.yaml).
"""

# ruff: noqa: D103 — runs inside Hermes, matching its plugin conventions.

import os
import json
import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_SERVICE = None


def _calendar_service() -> Any:
    """Build (once) the Calendar API client from the service-account file."""
    global _SERVICE
    if _SERVICE is None:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        sa_path = os.path.expanduser(os.getenv("GCAL_SERVICE_ACCOUNT_FILE", ""))
        if not sa_path or not os.path.exists(sa_path):
            raise RuntimeError("GCAL_SERVICE_ACCOUNT_FILE is not set or the file does not exist")
        credentials = service_account.Credentials.from_service_account_file(sa_path, scopes=_SCOPES)
        _SERVICE = build("calendar", "v3", credentials=credentials, cache_discovery=False)
    return _SERVICE


def _calendar_id() -> str:
    calendar_id = (os.getenv("CALENDAR_ID") or "").strip()
    if not calendar_id:
        raise RuntimeError("CALENDAR_ID is not set")
    return calendar_id


def _handle_list(params: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    try:
        service = _calendar_service()
        request: Dict[str, Any] = {
            "calendarId": _calendar_id(),
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": min(int(params.get("max_results") or 20), 50),
        }
        if params.get("time_min"):
            request["timeMin"] = params["time_min"]
        if params.get("time_max"):
            request["timeMax"] = params["time_max"]
        result = service.events().list(**request).execute()
        events = [
            {
                "summary": item.get("summary", "(no title)"),
                "start": (item.get("start") or {}).get("dateTime") or (item.get("start") or {}).get("date"),
                "end": (item.get("end") or {}).get("dateTime") or (item.get("end") or {}).get("date"),
                "location": item.get("location"),
                "description": (item.get("description") or "")[:200] or None,
            }
            for item in result.get("items", [])
        ]
        return json.dumps({"success": True, "events": events, "count": len(events)})
    except Exception as exc:
        logger.exception("calendar_list_events failed")
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def _handle_create(params: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    try:
        summary = (params.get("summary") or "").strip()
        start_time = (params.get("start_time") or "").strip()
        end_time = (params.get("end_time") or "").strip()
        if not summary or not start_time or not end_time:
            return json.dumps({"success": False, "error": "summary, start_time, and end_time are required"})
        timezone = (params.get("timezone") or "Europe/Zurich").strip()
        body: Dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start_time, "timeZone": timezone},
            "end": {"dateTime": end_time, "timeZone": timezone},
        }
        if params.get("description"):
            body["description"] = str(params["description"])[:2000]
        if params.get("location"):
            body["location"] = str(params["location"])[:500]
        service = _calendar_service()
        created = service.events().insert(calendarId=_calendar_id(), body=body).execute()
        return json.dumps(
            {
                "success": True,
                "summary": created.get("summary"),
                "start": (created.get("start") or {}).get("dateTime"),
                "link": created.get("htmlLink"),
                "note": "Created. Reminder: you cannot edit or delete events — ask the user to do that themselves.",
            }
        )
    except Exception as exc:
        logger.exception("calendar_create_event failed")
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def register(ctx: Any) -> None:
    """Plugin entry point — register the two (and only two) calendar tools."""
    ctx.register_tool(
        name="calendar_list_events",
        toolset="gcal",
        schema={
            "name": "calendar_list_events",
            "description": (
                "List upcoming events from the user's Google Calendar. Read-only. "
                "Times are ISO 8601; defaults to upcoming events when no range is given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {"type": "string", "description": "ISO 8601 lower bound, e.g. 2026-08-10T00:00:00+02:00"},
                    "time_max": {"type": "string", "description": "ISO 8601 upper bound"},
                    "max_results": {"type": "integer", "description": "Max events to return (default 20, cap 50)"},
                },
            },
        },
        handler=_handle_list,
        description="List Google Calendar events (read-only).",
    )
    ctx.register_tool(
        name="calendar_create_event",
        toolset="gcal",
        schema={
            "name": "calendar_create_event",
            "description": (
                "Create a NEW event in the user's Google Calendar. There are no tools to "
                "edit, move, or delete events — if the user asks for that, explain they "
                "must do it themselves in Google Calendar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start_time": {"type": "string", "description": "ISO 8601 start, e.g. 2026-08-10T09:00:00"},
                    "end_time": {"type": "string", "description": "ISO 8601 end"},
                    "description": {"type": "string", "description": "Optional details"},
                    "location": {"type": "string", "description": "Optional location"},
                    "timezone": {"type": "string", "description": "IANA timezone (default Europe/Zurich)"},
                },
                "required": ["summary", "start_time", "end_time"],
            },
        },
        handler=_handle_create,
        description="Create a new Google Calendar event (create-only; no edit/delete).",
    )
    logger.info("gcal-scoped plugin registered: list + create only")
