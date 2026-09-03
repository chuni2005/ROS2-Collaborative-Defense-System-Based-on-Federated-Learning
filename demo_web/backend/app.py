import os
import threading
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, request

import fdo_client

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

MACHINE_COUNT = 5

SCORE_THRESHOLD = 50          # 信任分數低於此值視為異常
ABNORMAL_SUSTAIN_SECONDS = 4  # 異常持續幾秒才觸發截斷（3~5 秒）
BLOCK_DURATION_SECONDS = 7    # 觸發後截斷幾秒（5~10 秒）
BLOCK_HISTORY_LIMIT = 20

lock = threading.Lock()
machines = {
    i: {
        "id": i,
        "name": f"機台{i}",
        "score": None,          # 最新一次收到的信任分數
        "last_update": None,    # 最近一次成功處理資料的時間 (epoch seconds)
        "abnormal_since": None, # 分數持續低於門檻的起始時間
        "blocked_until": None,  # 目前截斷狀態會維持到什麼時候
        "block_history": [],    # 觸發截斷的紀錄
    }
    for i in range(1, MACHINE_COUNT + 1)
}


def now_iso(ts=None):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else datetime.now(timezone.utc).isoformat()


def compute_diagnosis(machine_id, row):
    # TODO: 換成真正的 AI 信任分數判斷（由負責 API 串接的人接手）。
    # row 是機台送來的一列資料；這裡先回傳一個固定的高分做預設值。
    return 100.0


def is_blocked(m, now):
    return m["blocked_until"] is not None and now < m["blocked_until"]


def compute_light(m, now):
    if is_blocked(m, now):
        return "red"
    if m["score"] is None:
        return "gray"
    if m["score"] < SCORE_THRESHOLD:
        return "yellow"
    return "green"


def status_text(m, now):
    if is_blocked(m, now):
        return "已截斷"
    if m["score"] is None:
        return "尚無資料"
    if m["score"] < SCORE_THRESHOLD:
        return "異常"
    return "正常"


app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/machines")
def list_machines():
    return jsonify([{"id": i, "name": f"機台{i}"} for i in range(1, MACHINE_COUNT + 1)])


@app.get("/api/machines/status")
def machine_status():
    now = time.time()
    with lock:
        rows = [
            {"id": m["id"], "name": m["name"], "light": compute_light(m, now), "score": m["score"]}
            for m in machines.values()
        ]
    for row in rows:
        fdo_status = fdo_client.get_fdo_status(row["id"])
        row["fdoOnboarded"] = fdo_status["onboarded"]
        row["fdoStale"] = fdo_status["stale"]
    return jsonify(rows)


@app.post("/api/machines/select")
def select_machine():
    machine_id = (request.get_json(silent=True) or {}).get("machineId")
    with lock:
        m = machines.get(machine_id)
    if not m:
        return jsonify({"error": "not found"}), 404
    return jsonify({"id": m["id"], "name": m["name"]})


@app.post("/api/ingest")
def ingest():
    # Gate A: 機台身分要靠真的 FDO 上線紀錄，不是靠來源 IP。
    guid = request.headers.get("X-Device-Guid")
    guid_entry = fdo_client.load_guid_machine_map().get(guid) if guid else None
    if guid_entry is None:
        return jsonify({"error": "unknown device guid", "guid": guid}), 403

    machine_id = guid_entry["machineId"]
    fdo_status = fdo_client.get_fdo_status(machine_id)
    if not fdo_status["onboarded"]:
        return jsonify({"error": "device not onboarded", "machineId": machine_id}), 403

    # Gate B: 信任分數持續異常會被截斷，邏輯不變。
    now = time.time()
    row = request.get_json(silent=True) or {}

    with lock:
        m = machines[machine_id]

        if is_blocked(m, now):
            return jsonify({"machineId": machine_id, "status": "dropped"}), 200

        score = row.get("score")
        if score is None:
            score = compute_diagnosis(machine_id, row)
        score = float(score)

        m["score"] = score
        m["last_update"] = now

        if score < SCORE_THRESHOLD:
            if m["abnormal_since"] is None:
                m["abnormal_since"] = now
            elif now - m["abnormal_since"] >= ABNORMAL_SUSTAIN_SECONDS:
                m["blocked_until"] = now + BLOCK_DURATION_SECONDS
                m["block_history"].insert(0, {"time": now_iso(now), "score": score})
                del m["block_history"][BLOCK_HISTORY_LIMIT:]
                m["abnormal_since"] = None
        else:
            m["abnormal_since"] = None

    return jsonify({"machineId": machine_id, "status": "ok"}), 200


@app.get("/api/machines/<int:machine_id>/diagnosis")
def diagnosis(machine_id):
    now = time.time()
    with lock:
        m = machines.get(machine_id)
        if not m:
            return jsonify({"error": "not found"}), 404

        blocked = is_blocked(m, now)
        details = []
        if m["last_update"] is None:
            details.append("尚未收到資料")
        else:
            details.append(f"最後更新: {datetime.fromtimestamp(m['last_update']).strftime('%H:%M:%S')}")
        details.append(f"信任分數門檻: {SCORE_THRESHOLD}")
        if blocked:
            details.append(f"已截斷，剩餘 {m['blocked_until'] - now:.1f} 秒")
        elif m["abnormal_since"] is not None:
            details.append(
                f"異常持續 {now - m['abnormal_since']:.1f} 秒（達 {ABNORMAL_SUSTAIN_SECONDS} 秒將截斷訊息）"
            )

        return jsonify(
            {
                "machineId": m["id"],
                "status": status_text(m, now),
                "details": details,
                "score": m["score"],
                "blocked": blocked,
                "blockedSecondsRemaining": max(0.0, m["blocked_until"] - now) if blocked else 0.0,
            }
        )


@app.get("/api/machines/<int:machine_id>/publisher-status")
def publisher_status(machine_id):
    now = time.time()
    with lock:
        m = machines.get(machine_id)
        if not m:
            return jsonify({"error": "not found"}), 404

        blocked = is_blocked(m, now)
        return jsonify(
            {
                "machineId": m["id"],
                "blocked": blocked,
                "blockedSecondsRemaining": max(0.0, m["blocked_until"] - now) if blocked else 0.0,
                "history": list(m["block_history"]),
            }
        )


if __name__ == "__main__":
    fdo_client.start_poller()
    app.run(port=5181, debug=False)
