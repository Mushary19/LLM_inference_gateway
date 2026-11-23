-- Identity marker for the dedicated test database (test-db service only).
--
-- Pytest refuses to migrate or truncate any database whose comment doesn't match this
-- value exactly (see backend/tests/database_guard.py), so tests can't run against some
-- other server that happens to have a "_test" database. Pytest only reads the marker.
--
-- Runs automatically when a fresh test-db volume is initialized. For a volume that was
-- initialized before this file existed, apply it once (see README, "Testing").
COMMENT ON DATABASE inferenceflow_test IS 'inferenceflow-test-db';
