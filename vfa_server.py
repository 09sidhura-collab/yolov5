"""Local browser bridge for the Vehicle Footage Analyser.

Place this file and vfa.py in the root of the local YOLOv5 repository,
then run it with the same Anaconda environment used by VFA.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
VFA_SCRIPT = ROOT / "vfa.py"
DASHBOARD = ROOT / "vehicle_search_dashboard_integrated.html"
UPLOAD_DIR = ROOT / "vfa_local_runs" / "uploads"
REPORT_DIR = ROOT / "vfa_local_runs" / "reports"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv", ".webm"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB; adjust for local needs.

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.get("/")
def home():
    if not DASHBOARD.exists():
        return error(f"Dashboard not found: {DASHBOARD.name}. Put it next to vfa_server.py.", 500)
    return send_from_directory(ROOT, DASHBOARD.name)


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "python": sys.executable,
        "vfa_script_found": VFA_SCRIPT.exists(),
        "yolov5_root": str(ROOT),
    })


@app.post("/api/analyze")
def analyze():
    if not VFA_SCRIPT.is_file():
        return error("vfa.py was not found beside vfa_server.py. Place both files in the YOLOv5 repository root.", 500)
    if not (ROOT / "hubconf.py").is_file():
        return error("YOLOv5 hubconf.py was not found in this folder. Start the service from the root of your cloned ultralytics/yolov5 repository.", 500)
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename:
        return error("Choose a video file first.")
    original_name = secure_filename(uploaded.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return error("Unsupported video extension. Try MP4, AVI, MOV, MKV, MPG, WMV or WEBM.")

    run_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = UPLOAD_DIR / f"{run_id}_{original_name or ('footage' + suffix)}"
    report_path = REPORT_DIR / f"vehicle_results_{run_id}.xlsx"
    uploaded.save(video_path)

    # stdout/stdin intentionally inherit the Anaconda terminal. vfa.py uses
    # input() for CCTV start time and OpenCV GUI windows for gate calibration.
    command = [sys.executable, str(VFA_SCRIPT), "--video", str(video_path), "--output-xlsx", str(report_path)]
    print("\n[VFA] Starting local analysis:", " ".join(f'"{x}"' if " " in x else x for x in command), flush=True)
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False)
    except Exception as exc:
        return error(f"Could not launch vfa.py: {exc}", 500)

    if completed.returncode != 0:
        return error(f"vfa.py exited with code {completed.returncode}. Check the Anaconda terminal for the Python/OpenCV/YOLOv5 error. The video was saved locally as {video_path.name}.", 500)
    if not report_path.is_file() or report_path.stat().st_size == 0:
        return error("vfa.py finished but did not create the expected Excel report. Check the Anaconda terminal.", 500)

    return jsonify({
        "message": "Vehicle analysis finished successfully.",
        "report_name": report_path.name,
        "report_url": f"/api/report/{report_path.name}",
    })


@app.get("/api/report/<name>")
def get_report(name: str):
    # Only allow report filenames from the generated report directory.
    if not re.fullmatch(r"vehicle_results_[a-f0-9]{12}\.xlsx", name):
        return error("Invalid report name.", 400)
    path = REPORT_DIR / name
    if not path.is_file():
        return error("Report not found. It may have been removed.", 404)
    return send_file(path, as_attachment=True, download_name=name, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.errorhandler(413)
def too_large(_exc):
    return error("The video is larger than the configured upload limit (8 GiB). Increase MAX_UPLOAD_BYTES in vfa_server.py if needed.", 413)


if __name__ == "__main__":
    print("Vehicle Footage Analyser local bridge", flush=True)
    print(f"Python environment: {sys.executable}", flush=True)
    print(f"YOLOv5 root: {ROOT}", flush=True)
    print("Open http://127.0.0.1:5000 in your browser.", flush=True)
    print("Keep this terminal open while running video analysis.", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
