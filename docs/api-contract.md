# API Contract

The authoritative machine-readable version is the OpenAPI schema at
`http://127.0.0.1:8000/openapi.json` while `aegisflow serve` is running, and the interactive
docs at `/docs`. This page is the summary.

## REST

| Method | Path | View | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | — | Liveness plus provider, policy and database state. Suitable as a container healthcheck. |
| `GET` | `/api/stats` | A | Counts by severity, behaviour and zone for the header tiles. |
| `GET` | `/api/clips` | A | Processed clips available for playback. |
| `GET` | `/api/clips/{clip_id}/video` | A | Stream the annotated MP4. |
| `GET` | `/api/events` | B, C | Filtered, paginated history. |
| `GET` | `/api/events/export` | C | CSV or JSON download of the filtered log. |
| `GET` | `/api/events/{event_id}` | A, B | One compliance record. |
| `GET` | `/api/policy` | — | Parsed rules plus the derived severity matrix. |
| `WS` | `/ws/alerts` | A, B | Live HIGH/CRITICAL alerts. |

`/api/events/export` is registered **before** `/api/events/{event_id}`; otherwise `export`
would be captured as an event id. There is a test for that.

### Filters on `/api/events` and `/api/events/export`

The three the assignment requires for View C, plus two for drill-down:

| Parameter | Type | Notes |
|---|---|---|
| `date_from`, `date_to` | ISO 8601 datetime | Inclusive. `422` if `from` is after `to`. |
| `severity` | repeatable | `LOW` \| `MEDIUM` \| `HIGH` \| `CRITICAL`. Repeat for OR. |
| `behavior_class` | repeatable | One of the four parsed classes. Repeat for OR. |
| `zone` | string | e.g. `Zone-1` |
| `clip_id` | string | all events from one clip |
| `limit`, `offset` | int | pagination; `limit` max 500 |
| `format` | `csv` \| `json` | export only |

Filters intersect (AND across parameters, OR within a repeated one). An unknown severity or
behaviour class is a `422`, not a silent empty result.

### Response shape

`GET /api/events` returns a page, so the client can render "1-50 of 312" without a second call:

```jsonc
{
  "total": 312,     // matching records, ignoring limit/offset
  "limit": 50,
  "offset": 0,
  "items": [ /* EventOut */ ]
}
```

`EventOut` carries the nine assignment-mandated fields plus `confidence`, `detection_method`
and `severity_rationale`. Surface the rationale in the UI — it is what shows a reviewer that
the tier came from the policy rather than from a guess.

## WebSocket

```jsonc
{
  "type": "violation",            // "violation" | "heartbeat" | "run_status"
  "sent_at": "2026-09-01T10:45:00Z",
  "payload": { /* a full compliance record, same shape as EventOut */ }
}
```

Only `HIGH` and `CRITICAL` events are published. `LOW` and `MEDIUM` are database-only by
policy (assignment Module 3), and the dashboard reads those from `GET /api/events`.

A `heartbeat` is sent every 25 s of idle so a client can show a live indicator without
polling, and so intermediaries do not close the connection. Each client gets its own bounded
queue: a stalled browser tab drops its own oldest messages and never blocks the pipeline.

## Errors

| Status | Meaning |
|---|---|
| `404` | Unknown event or clip id |
| `422` | Invalid filter value, or an inverted date range |
| `503` | `/api/policy` when `rules.json` is missing — run `aegisflow policy parse` |

`/api/health` never raises; it reports component state in its body instead.
