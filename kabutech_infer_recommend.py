from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime

# ---------------------------
# 1) CONFIG
# ---------------------------
MODEL_PATH = r"runs/classify/train/weights/best.pt" # Update if different path!!!!

# Map class -> severity
SEVERITY = {
    "bacterial_blotch_disease": "moderate",
    "dry_bubble_disease": "moderate",
    "green_molds_disease": "high",
    "healthy_fruiting_bag": "none",
    "healthy_mushroom": "none",
}

# Target ranges (adjust later, based on your references)
TARGETS = {
    "temp_c": (20.0, 28.0),
    "humidity_rh": (85.0, 95.0),
    "light_lux": (50.0, 300.0),
    "substrate_moisture_pct": (55.0, 65.0),
}

CONF_THRESHOLD = 0.70  # confidence gate

# ---------------------------
# 2) RULE ENGINE
# ---------------------------
def recommend(pred_class: str, conf: float, env: dict):
    severity = SEVERITY.get(pred_class, "unknown")
    alerts = []
    actions = []

    # Confidence gate
    if conf < CONF_THRESHOLD:
        alerts.append(f"Low confidence ({conf:.2f}).")
        actions += [
            "Retake photo with better focus and lighting; keep subject centered.",
            "If symptoms persist, isolate and monitor the bag/mushroom."
        ]
        return severity, alerts, actions

    # Environment checks (only if provided)
    def check_range(key, unit=""):
        lo, hi = TARGETS[key]
        v = env.get(key, None)
        if v is None:
            return
        if v < lo or v > hi:
            alerts.append(f"{key} out of range: {v}{unit} (target {lo}-{hi}{unit})")

    if "temp_c" in env: check_range("temp_c", "°C")
    if "humidity_rh" in env: check_range("humidity_rh", "%")
    if "light_lux" in env: check_range("light_lux", " lux")
    if "substrate_moisture_pct" in env: check_range("substrate_moisture_pct", "%")

    # Disease-specific recommendations
    if pred_class == "green_molds_disease":
        actions += [
            "HIGH severity: isolate/remove contaminated bag immediately to prevent spread.",
            "Sterilize tools and nearby surfaces.",
            "Reduce overly-wet conditions; increase fresh-air exchange."
        ]
    elif pred_class == "dry_bubble_disease":
        actions += [
            "MODERATE severity: isolate affected bag.",
            "Avoid direct misting on fruiting bodies; improve airflow.",
            "Sanitize handling area and tools."
        ]
    elif pred_class == "bacterial_blotch_disease":
        actions += [
            "MODERATE severity: reduce surface moisture (avoid water droplets on caps).",
            "Increase ventilation; avoid overcrowding.",
            "Sanitize area to reduce bacterial spread."
        ]
    else:
        actions += [
            "No disease detected: continue monitoring.",
            "Maintain chamber conditions within target ranges."
        ]

    # Optional substrate quality heuristic
    q = env.get("substrate_quality_score", None)
    if q is not None and q < 0.6:
        alerts.append("Substrate quality flagged as poor (score < 0.6).")
        actions.append("Consider replacing/refreshing substrate and reviewing sterilization/pasteurization.")

    return severity, alerts, actions

# ---------------------------
# 3) INPUT (manual for now)
# ---------------------------
def get_env_manual(): #REPLACE ME WITH read_sensors() WHEN YOU HAVE THE SENSOR SETUP READY
    # Leave blank to skip a field.
    def get_float(prompt):
        s = input(prompt).strip()
        return None if s == "" else float(s)

    env = {}
    v = get_float("Temperature (°C) [Enter to skip]: ")
    if v is not None: env["temp_c"] = v

    v = get_float("Humidity (%RH) [Enter to skip]: ")
    if v is not None: env["humidity_rh"] = v

    v = get_float("Light intensity (lux) [Enter to skip]: ")
    if v is not None: env["light_lux"] = v

    v = get_float("Substrate moisture (%) [Enter to skip]: ")
    if v is not None: env["substrate_moisture_pct"] = v

    v = get_float("Substrate quality score (0..1) [Enter to skip]: ")
    if v is not None: env["substrate_quality_score"] = v

    return env

# ---------------------------
# 4) MAIN
# ---------------------------
def main():
    model = YOLO(MODEL_PATH)

    img_path = input("Path to image: ").strip().strip('"')
    p = Path(img_path)
    if not p.exists():
        print("Image path not found.")
        return

    env = get_env_manual()

    r = model.predict(source=str(p), verbose=False)[0]
    pred_id = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    pred_class = model.names[pred_id]

    severity, alerts, actions = recommend(pred_class, conf, env)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": str(p),
        "predicted_class": pred_class,
        "confidence": round(conf, 4),
        "severity": severity,
        "environment": env,
        "alerts": alerts,
        "actions": actions,
    }

    print("\n=== KabuTech Result ===")
    print(json.dumps(result, indent=2))

    # Log to file
    log_path = Path("kabutech_log.jsonl")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"\nLogged to: {log_path.resolve()}")

if __name__ == "__main__":
    main()
