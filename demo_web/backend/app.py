import os
import random
import threading
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, request

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

MACHINE_COUNT = 5
LIGHTS = ["green", "green", "green", "yellow", "red", "gray"]
DIAGNOSIS_TEXT = {"green": "正常", "yellow": "警告", "red": "異常", "gray": "離線"}
MESSAGE_SAMPLES = [
    "心跳封包已送出",
    "CPU 使用率過高",
    "偵測到異常封包流量",
    "已完成本輪聯邦學習訓練",
    "與伺服器連線逾時，重試中",
    "感測器讀值超出正常範圍",
    "節點重新啟動完成",
]

lock = threading.Lock()
machines = {
    i: {
        "id": i,
        "name": f"機台{i}",
        "light": "green",
        "score": 100.0,
        "duration": 0,
        "messages": [],
    }
    for i in range(1, MACHINE_COUNT + 1)
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def tick():
    while True:
        with lock:
            for m in machines.values():
                if random.random() < 0.15:
                    m["light"] = random.choice(LIGHTS)

                m["score"] = min(100.0, max(0.0, m["score"] + random.randint(-5, 5)))
                m["duration"] = 0 if m["light"] == "gray" else m["duration"] + 1

                if random.random() < 0.3:
                    level = {"red": "error", "yellow": "warning"}.get(m["light"], "info")
                    m["messages"].append(
                        {
                            "timestamp": now_iso(),
                            "level": level,
                            "content": random.choice(MESSAGE_SAMPLES),
                        }
                    )
                    del m["messages"][:-100]
        time.sleep(1)


app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/machines")
def list_machines():
    with lock:
        return jsonify([{"id": m["id"], "name": m["name"]} for m in machines.values()])


@app.get("/api/machines/status")
def machine_status():
    with lock:
        return jsonify(
            [{"id": m["id"], "name": m["name"], "light": m["light"]} for m in machines.values()]
        )


@app.post("/api/machines/select")
def select_machine():
    machine_id = (request.get_json(silent=True) or {}).get("machineId")
    with lock:
        m = machines.get(machine_id)
    if not m:
        return jsonify({"error": "not found"}), 404
    return jsonify({"id": m["id"], "name": m["name"]})


@app.get("/api/machines/<int:machine_id>/diagnosis")
def diagnosis(machine_id):
    with lock:
        m = machines.get(machine_id)
        if not m:
            return jsonify({"error": "not found"}), 404
        status = DIAGNOSIS_TEXT[m["light"]]
        return jsonify(
            {
                "machineId": m["id"],
                "status": status,
                "details": [
                    f"最近心跳: {datetime.now().strftime('%H:%M:%S')}",
                    f"信心分數: {m['score']:.1f}",
                    f"狀態持續: {m['duration']} 秒",
                ],
                "score": m["score"],
                "durationSeconds": m["duration"],
            }
        )


@app.get("/api/machines/<int:machine_id>/messages")
def messages(machine_id):
    with lock:
        m = machines.get(machine_id)
        if not m:
            return jsonify({"error": "not found"}), 404
        return jsonify(list(reversed(m["messages"][-30:])))


if __name__ == "__main__":
    threading.Thread(target=tick, daemon=True).start()
    app.run(port=5181, debug=False)
