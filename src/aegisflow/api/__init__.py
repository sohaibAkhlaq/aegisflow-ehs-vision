"""FastAPI application: REST routes, WebSocket alerts, dashboard hosting."""

from aegisflow.api.app import create_app, serve

__all__ = ["create_app", "serve"]
