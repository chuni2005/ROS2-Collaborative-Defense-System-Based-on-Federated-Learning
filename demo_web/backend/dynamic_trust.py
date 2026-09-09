import json
import math
import os
import random

# 這是 Dynamic_Trust_Evaluation/fuzzy_inference.py 的精簡版：只保留「環境風險
# 模糊推論」那一段，不依賴 pandas/xgboost。目前還沒有訓練好的 XGBoost 模型
# （negotiated with user：先不管 XGBoost，之後模型好了再換掉 _simulate_env_context
# 這個函式，改成吃機台端真的送來的 35 個特徵）。

CONFIG_PATH = os.environ.get(
    "FUZZY_CONFIG_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "Dynamic_Trust_Evaluation",
        "fuzzy_threshold_config.json",
    ),
)

BASE_THRESHOLD = 0.60  # 對應信任分數 0~100 尺度的 60 分
MAX_ADJUSTMENT = 0.10  # 環境風險最多把門檻往上/下拉 10 分


def _load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_CONFIG = _load_config()


def _triangle_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if b < x < c:
        return (c - x) / (c - b) if c != b else 1.0
    return 0.0


def _fuzzify(feature_name, value):
    conf = _CONFIG[feature_name]
    l_verts = conf["L_vertices"]
    m_verts = conf["M_vertices"]
    h_verts = conf["H_vertices"]
    mu_l = 1.0 if value <= l_verts[1] else _triangle_membership(value, *l_verts)
    mu_m = _triangle_membership(value, *m_verts)
    mu_h = 1.0 if value >= h_verts[1] else _triangle_membership(value, *h_verts)
    return mu_l, mu_m, mu_h


def simulate_env_context():
    # 還沒有真的機台把 35 個特徵送進來，先用設定檔裡每個特徵本來就記錄的
    # mu（平均值）/ sigma（標準差）做常態分布抽樣，模擬出合理的特徵數值。
    context = {}
    for feature_name, conf in _CONFIG.items():
        mu = conf.get("mu", 0.0)
        sigma = conf.get("sigma", 1.0)
        context[feature_name] = max(0.0, random.gauss(mu, sigma))
    return context


def env_risk_score(env_context):
    mu_l_list, mu_m_list, mu_h_list = [], [], []
    for feature_name, value in env_context.items():
        if feature_name not in _CONFIG or value is None or math.isnan(value):
            continue
        mu_l, mu_m, mu_h = _fuzzify(feature_name, value)
        mu_l_list.append(mu_l)
        mu_m_list.append(mu_m)
        mu_h_list.append(mu_h)

    if not mu_l_list:
        return 0.0

    rule_safe = sum(mu_l_list) / len(mu_l_list)
    rule_warn = max(mu_m_list)
    rule_danger = max(mu_h_list)

    denominator = rule_safe + rule_warn + rule_danger
    if denominator == 0:
        return 0.0
    return ((rule_safe * -1.0) + (rule_warn * 0.0) + (rule_danger * 1.0)) / denominator


def compute_dynamic_threshold():
    """回傳 (threshold_0_100, risk) — threshold 是信任分數(0~100 尺度)的動態門檻。"""
    if not _CONFIG:
        return 50.0, 0.0
    context = simulate_env_context()
    risk = env_risk_score(context)
    threshold = (BASE_THRESHOLD + risk * MAX_ADJUSTMENT) * 100
    return threshold, risk
