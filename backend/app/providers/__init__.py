"""LLM provider layer: normalized chat types and provider implementations.

Independent of FastAPI and the database. The HTTP API translates its own schemas into
these types and back, so providers never see HTTP concerns.
"""
