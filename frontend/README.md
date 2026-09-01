# Operations Dashboard (Module 5)

The dashboard the assignment asks for, built and wired to the live API. No build step, no
`node_modules`: three files served straight off disk by the FastAPI app.

```
index.html        markup + the four view containers
assets/app.css    design system - severity tokens, layout, light/dark themes
assets/app.js     API client, WebSocket alert feed, view rendering
favicon.svg       served at /favicon.ico by api/app.py
```

## Running it

The API serves the dashboard at `/` when `frontend/index.html` exists:

```bash
python -m aegisflow serve          # then open http://127.0.0.1:8000
```

`api/app.py::_mount_dashboard` mounts `assets/` at `/assets` and routes `/favicon.ico` to
`favicon.svg`. If `index.html` is absent, `/` redirects to `/docs` instead — the API does not
depend on the frontend being present.

Seed some events first, or every view is empty:

```bash
python scripts/seed_db.py --synthetic
```

## Why no framework

A Vite + React + TypeScript setup was the original plan. It was dropped because it buys
nothing here and costs a reviewer a `npm install` before they can see anything: there are
four read-mostly views over one REST API and one WebSocket, and no shared client state worth
a store. Plain ES modules keep the whole surface reviewable in one sitting and keep the
repository free of a lockfile and a `dist/` build artefact.

## The views

A severity tile row (`GET /api/stats`) sits above all four.

| View | Endpoints | What it shows |
|---|---|---|
| Live | `/api/clips`, `/api/events`, `WS /ws/alerts` | Processed clips with their annotated video, and alerts as they arrive |
| Timeline | `/api/events` | Events in time order, for spotting clusters |
| History | `/api/events` (filtered), `/api/events/export` | Filterable audit table with the full report fields, exportable |
| Policy | `/api/policy` | The parsed rule set — the evidence that rules came from the PDF |

Endpoint shapes are specified in [`../docs/api-contract.md`](../docs/api-contract.md).

## Conventions worth keeping

- **Severity colours are tokens, never literals.** `--sev-low` blue, `--sev-medium` green,
  `--sev-high` amber, `--sev-critical` red, defined once in `app.css`. The tiers themselves
  come from the policy parser; the frontend only maps a tier name to a token.
- **Every element `app.js` reaches for has an `id` in `index.html`.** CI asserts this — a
  renamed id fails the build rather than silently blanking a view.
- **The socket reconnects with backoff.** A dropped connection must degrade to polling, not
  to a dead page.
