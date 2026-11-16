#!/bin/sh
# Starts the test-redis server and provisions the identity marker pytest verifies
# (backend/tests/redis_guard.py). Redis keeps no data on disk here, so the marker is
# set again every time the container starts. Pytest only reads it.
set -eu

docker-entrypoint.sh redis-server --save "" --appendonly no &
server=$!
trap 'kill -TERM "$server"; wait "$server"' TERM INT

until redis-cli ping >/dev/null 2>&1; do sleep 0.1; done
redis-cli SET inferenceflow:test-redis:identity inferenceflow-test-redis >/dev/null

wait "$server"
