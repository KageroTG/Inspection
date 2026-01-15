import json
from datetime import datetime
import time

# ================================================================
# CONFIGURATION
# ================================================================
METADATA_PATH = (
    r"D:\NIK WORK\ROBOMY\PROJECTS\MultiThreading\results\metadata_20251209_125621.json"
)
TIME_WINDOW = 20  # seconds per batch
CRACK_CLASS = "cracks"
CONF_THRESHOLD = 0.5
OUTPUT_HTML = "batched_output3test.html"


# ================================================================
# LOAD METADATA
# ================================================================
def load_metadata(path):
    """Load full metadata JSON."""
    with open(path, "r") as f:
        return json.load(f)


# ================================================================
# TIMESTAMP CONVERSION
# ================================================================
def convert_timestamp_to_unix(timestamp_str):
    """Convert HH:MM:SS timestamp to Unix timestamp."""
    # Assuming the timestamp is in the format HH:MM:SS
    # We'll convert it to a datetime object and then to Unix timestamp
    # Using today's date since no date is provided
    today = datetime.now().date()
    time_obj = datetime.strptime(timestamp_str, "%H:%M:%S").time()
    dt_obj = datetime.combine(today, time_obj)
    return dt_obj.timestamp()


# ================================================================
# BATCHING LOGIC (UNIX TIMESTAMPS) - FOR ALL CLASSES
# ================================================================
def make_batches(detections):
    """Batch all detections using UNIX timestamps."""
    detections.sort(key=lambda x: x["unix_timestamp"])  # sort by time

    batches = []
    current_batch = []
    start_time = None

    for det in detections:
        ts = det["unix_timestamp"]

        if start_time is None:
            start_time = ts
            current_batch = [det]
            continue

        if ts - start_time <= TIME_WINDOW:
            current_batch.append(det)
        else:
            batches.append((start_time, current_batch))
            start_time = ts
            current_batch = [det]

    if current_batch:
        batches.append((start_time, current_batch))

    return batches


# ================================================================
# PER-SECOND, PER-CLASS DEDUPLICATION
# ================================================================
def dedup_per_second_per_class(detections):
    """
    For each integer second and class, keep only the detection with
    the highest confidence.
    """
    best_by_key = {}  # (second, class_lower) -> detection

    for det in detections:
        ts = det["unix_timestamp"]
        cls = det["class"].lower()
        sec = int(ts)  # integer Unix second
        key = (sec, cls)

        existing = best_by_key.get(key)
        if existing is None or det["confidence"] > existing["confidence"]:
            best_by_key[key] = det

    return list(best_by_key.values())


# ================================================================
# HTML REPORT
# ================================================================
def generate_html(batches):
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Batched Detection Results</title>
<style>
body {
  font-family: Arial, sans-serif;
  margin: 20px;
  background: #f4f6f8;
}
.batch-container {
  margin-bottom: 20px;
  border: 1px solid #ccc;
  border-radius: 5px;
  overflow: hidden;
}
.batch-header {
  background: #0078d4;
  color: white;
  padding: 10px;
  font-weight: bold;
}
.class-section {
  padding: 10px;
  margin: 5px;
  border-radius: 5px;
}
.potholes-section {
  background: #ffebee;
  border: 1px solid #f44336;
}
.raveling-section {
  background: #e8f5e9;
  border: 1px solid #4caf50;
}
.cracks-section {
  background: #fff4ce;
  border: 1px solid #d4a017;
}
summary {
  padding: 10px;
  background: #e3e7ed;
  cursor: pointer;
  font-weight: bold;
  border-radius: 5px;
}
details {
  background: white;
  border: 1px solid #ccc;
  margin-top: 8px;
  padding: 10px;
  border-radius: 5px;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin-top: 10px;
}
th, td {
  padding: 6px;
  border: 1px solid #ddd;
}
th {
  background: #0078d4;
  color: white;
}
</style>
</head>
<body>
<h2>Batched Detection Results (>= 0.5 confidence)</h2>
<p>Only detections with confidence above 0.5 are shown.</p>
"""

    for idx, (start_time, detections) in enumerate(batches, 1):
        end_time = start_time + TIME_WINDOW

        # Convert Unix timestamps back to HH:MM:SS for display
        start_time_str = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        end_time_str = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

        # Get the start and end seconds for display
        start_sec = int(start_time) % 60
        end_sec = int(end_time) % 60

        html += f"""
<div class="batch-container">
  <div class="batch-header">Batch {idx} — Time Window: {start_time_str} → {end_time_str}</div>
"""

        # Separate detections by class
        pothole_dets = [d for d in detections if d["class"].lower() == "potholes"]
        raveling_dets = [d for d in detections if d["class"].lower() == "raveling"]
        crack_dets = [
            d for d in detections if d["class"].lower() == CRACK_CLASS.lower()
        ]

        # Display potholes directly (not collapsible)
        if pothole_dets:
            html += f"""
  <div class="class-section potholes-section">
    <h3>Potholes ({start_sec}-{end_sec})</h3>
    <table>
      <tr><th>#</th><th>Frame</th><th>Timestamp</th><th>Class</th><th>Confidence</th><th>BBox</th><th>Track ID</th></tr>
"""
            for i, d in enumerate(pothole_dets, 1):
                track_id = d.get("track_id", "N/A")
                html += f"""
      <tr>
        <td>{i}</td>
        <td>{d['frame']}</td>
        <td>{d['timestamp']}</td>
        <td>{d['class']}</td>
        <td>{d['confidence']:.3f}</td>
        <td>{d['bbox']}</td>
        <td>{track_id}</td>
      </tr>
"""
            html += "    </table></div>"

        # Display raveling directly (not collapsible)
        if raveling_dets:
            html += f"""
  <div class="class-section raveling-section">
    <h3>Raveling ({start_sec}-{end_sec})</h3>
    <table>
      <tr><th>#</th><th>Frame</th><th>Timestamp</th><th>Class</th><th>Confidence</th><th>BBox</th><th>Track ID</th></tr>
"""
            for i, d in enumerate(raveling_dets, 1):
                track_id = d.get("track_id", "N/A")
                html += f"""
      <tr>
        <td>{i}</td>
        <td>{d['frame']}</td>
        <td>{d['timestamp']}</td>
        <td>{d['class']}</td>
        <td>{d['confidence']:.3f}</td>
        <td>{d['bbox']}</td>
        <td>{track_id}</td>
      </tr>
"""
            html += "    </table></div>"

        # Display cracks in collapsible format
        if crack_dets:
            html += f"""
  <div class="class-section cracks-section">
    <details>
      <summary>Cracks ({start_sec}-{end_sec}) — {len(crack_dets)} detections</summary>
      <table>
        <tr><th>#</th><th>Frame</th><th>Timestamp</th><th>Class</th><th>Confidence</th><th>BBox</th><th>Track ID</th></tr>
"""
            for i, d in enumerate(crack_dets, 1):
                track_id = d.get("track_id", "N/A")
                html += f"""
        <tr>
          <td>{i}</td>
          <td>{d['frame']}</td>
          <td>{d['timestamp']}</td>
          <td>{d['class']}</td>
          <td>{d['confidence']:.3f}</td>
          <td>{d['bbox']}</td>
          <td>{track_id}</td>
        </tr>
"""
            html += "      </table></details></div>"

        html += "</div>"

    html += "</body></html>"
    return html


# ================================================================
# SAVE HTML
# ================================================================
def save_html(content):
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[✔] HTML saved to: {OUTPUT_HTML}")


# ================================================================
# MAIN
# ================================================================
def main():
    print(f"[+] Loading metadata: {METADATA_PATH}")

    data = load_metadata(METADATA_PATH)
    detections = data["detections"]

    # ----------------------------
    # Convert timestamps to Unix format
    # ----------------------------
    for det in detections:
        det["unix_timestamp"] = convert_timestamp_to_unix(det["timestamp"])

    # ----------------------------
    # Filter by confidence
    # ----------------------------
    detections = [d for d in detections if d["confidence"] >= CONF_THRESHOLD]
    print(f"[+] Detections after filtering (>= {CONF_THRESHOLD}): {len(detections)}")

    # ----------------------------
    # Per-second, per-class dedup
    # ----------------------------
    before_dedup = len(detections)
    detections = dedup_per_second_per_class(detections)
    after_dedup = len(detections)
    print(
        f"[+] Detections after per-second, per-class dedup: {after_dedup} "
        f"(removed {before_dedup - after_dedup})"
    )

    # Count detections by class
    crack_count = sum(
        1 for d in detections if d["class"].lower() == CRACK_CLASS.lower()
    )
    pothole_count = sum(1 for d in detections if d["class"].lower() == "potholes")
    raveling_count = sum(1 for d in detections if d["class"].lower() == "raveling")

    print(f"[+] Cracks detected: {crack_count}")
    print(f"[+] Potholes detected: {pothole_count}")
    print(f"[+] Raveling detected: {raveling_count}")

    # ----------------------------
    # Create batches for all detections
    # ----------------------------
    print(f"[+] Creating batches...")
    batches = make_batches(detections)
    print(f"[+] Total batches created: {len(batches)}")

    html = generate_html(batches)
    save_html(html)


# ================================================================
if __name__ == "__main__":
    main()
