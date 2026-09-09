import json
import os
import threading
import time
import urllib.error
import urllib.request

FDO_OWNER_URL = os.environ.get("FDO_OWNER_URL", "http://localhost:8043")
FDO_GUID_MAP_PATH = os.environ.get(
    "FDO_GUID_MAP_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "guid_machine_map.json"),
)
FDO_POLL_INTERVAL_SECONDS = float(os.environ.get("FDO_POLL_INTERVAL_SECONDS", "5"))
FDO_STALE_AFTER_SECONDS = float(os.environ.get("FDO_STALE_AFTER_SECONDS", "30"))

_lock = threading.Lock()
_device_cache = {}  # guid (hex str) -> {"to2_completed": bool}
_last_poll_success_at = None  # epoch seconds, None until the first successful poll


def load_guid_machine_map():
    try:
        with open(FDO_GUID_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _fetch_owner_devices():
    url = f"{FDO_OWNER_URL}/api/v1/owner/devices"
    with urllib.request.urlopen(url, timeout=3) as resp:
        return json.loads(resp.read())


def _poll_once():
    global _last_poll_success_at
    devices = _fetch_owner_devices()
    cache = {d["guid"]: {"to2_completed": bool(d.get("to2_completed"))} for d in devices if d.get("guid")}
    with _lock:
        _device_cache.clear()
        _device_cache.update(cache)
        _last_poll_success_at = time.time()


def poll_owner_devices():
    while True:
        try:
            _poll_once()
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            print(f"[fdo_client] Owner server poll failed: {exc}")
        time.sleep(FDO_POLL_INTERVAL_SECONDS)


def start_poller():
    thread = threading.Thread(target=poll_owner_devices, daemon=True)
    thread.start()
    return thread


def get_fdo_status(machine_id):
    guid_map = load_guid_machine_map()
    guid = next((g for g, info in guid_map.items() if info.get("machineId") == machine_id), None)

    with _lock:
        onboarded = bool(guid and _device_cache.get(guid, {}).get("to2_completed"))
        stale = _last_poll_success_at is None or (time.time() - _last_poll_success_at) > FDO_STALE_AFTER_SECONDS

    return {"onboarded": onboarded, "guid": guid, "stale": stale}
