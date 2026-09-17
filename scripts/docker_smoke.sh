#!/usr/bin/env bash
# Build the image from this checkout, run exactly that image, and check it answers like the code does.
#
#   bash scripts/docker_smoke.sh
#
# Exits non-zero if Docker is unavailable, the image doesn't build or turn healthy, runs as root,
# carries pandas, disagrees with requirements.txt, or returns different numbers from tests/golden.json.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
IMAGE_BASE="python:3.12-slim-bookworm"
name="fnd-smoke-$$"
iid_file="$(mktemp)"
freeze_file="$(mktemp)"

fail() { echo "docker_smoke: FAIL: $*" >&2; exit 1; }
cleanup() {
  docker rm -f "$name" >/dev/null 2>&1 || true
  rm -f "$iid_file" "$freeze_file"
}
trap cleanup EXIT

command -v docker >/dev/null 2>&1 || fail "docker is not installed"
docker info >/dev/null 2>&1 || fail "the Docker daemon is not running"
if ! docker image inspect "$IMAGE_BASE" >/dev/null 2>&1; then
  echo "docker_smoke: $IMAGE_BASE is not local; the build will download it"
fi

echo "docker_smoke: building"
docker build --quiet --iidfile "$iid_file" . >/dev/null || fail "the image did not build"
iid="$(cat "$iid_file")"
[ -n "$iid" ] || fail "the build wrote no image id"

docker run -d --name "$name" -p 127.0.0.1::8080 "$iid" >/dev/null
port="$(docker port "$name" 8080/tcp | head -n 1 | sed 's/.*://')"
base="http://127.0.0.1:${port}"
echo "docker_smoke: running ${iid:7:12} on $base"

health="starting"
for _ in $(seq 1 60); do
  if [ "$(docker inspect --format '{{.State.Running}}' "$name")" != "true" ]; then
    docker logs "$name" >&2 || true
    fail "the container exited"
  fi
  health="$(docker inspect --format '{{.State.Health.Status}}' "$name")"
  [ "$health" = "healthy" ] && break
  sleep 1
done
[ "$health" = "healthy" ] || { docker logs "$name" >&2 || true; fail "not healthy after 60 s (status: $health)"; }

uid="$(docker exec "$name" id -u)"
[ "$uid" != "0" ] || fail "the container runs as root"

if docker exec "$name" python -c "import pandas" >/dev/null 2>&1; then
  fail "pandas is installed in the image"
fi
pandas_error="$(docker exec "$name" python -c "import pandas" 2>&1 || true)"
case "$pandas_error" in
  *ModuleNotFoundError*) ;;
  *) fail "import pandas did not fail with ModuleNotFoundError: $pandas_error" ;;
esac

docker exec "$name" pip check >/dev/null || fail "pip check found broken requirements"
docker exec "$name" pip freeze > "$freeze_file"

"$PY" - "$base" "$freeze_file" <<'EOF' || fail "the running image does not match the code"
import json
import re
import sys
import urllib.error
import urllib.request

base, freeze_path = sys.argv[1], sys.argv[2]


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(lines):
    out = {}
    for line in lines:
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", line.strip())
        if m:
            out[norm(m.group(1))] = m.group(2)
    return out


installed = pins(open(freeze_path))
wanted = pins(open("requirements.txt"))
assert installed == wanted, f"pip freeze {installed} != requirements.txt {wanted}"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


golden = json.load(open("tests/golden.json"))
for name in ("wire_article_dateline", "sensational_article_dateline", "mit_water"):
    record = golden["inputs"][name]
    for route, key in (("/api/predict", "predict"), ("/api/predict/simple", "simple")):
        status, raw = call("POST", route, {"text": record["text"]})
        body = json.loads(raw)
        assert status == record[key]["status"], (name, route, status)
        check = body.pop("input_check")
        assert body == record[key]["body"], (name, route, body)
        assert check["status"] in {"ok", "no_signal"}, check

status, raw = call("POST", "/api/predict", json.load(open("samples/wire_report.json")))
body = json.loads(raw)
assert status == 200 and "content_model" in body["model_breakdown"] and "input_check" in body, body

status, raw = call("GET", "/health")
health = json.loads(raw)
assert status == 200 and health["sklearn_version"] == "1.7.0", health
assert health["models"] == {"format_model": "available", "content_model": "available"}, health

status, _ = call("GET", "/console")
assert status == 404, f"/console answered {status}"

status, raw = call("GET", "/?sample=sensational_dateline")
assert status == 200 and b'data-state="disagreement"' in raw, status

print("docker_smoke: responses match tests/golden.json and requirements.txt")
EOF

echo "docker_smoke: PASS (healthy, uid $uid, no pandas, pins match, golden numbers match)"
