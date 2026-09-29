"""BeamGuardAI: upload an image, a zip of images, or a video and get a challan decision.

    pip install -r requirements.txt
    streamlit run app.py

Decision for every upload:
    GENERATE CHALLAN  strong headlight glare (single image), or strong glare that
                      holds over several frames in a row (video)
    REVIEW            glare is in the uncertain band, or too short-lived in a video
    NO CHALLAN        no lit headlights, or glare below the review level
"""
import io
import os
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

TARGET_WIDTH = 1280            # every frame is scaled to this width so the pixel rules behave the same
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
MAX_ZIP_IMAGES = 300
MAX_UNZIPPED_BYTES = 300 * 1024 * 1024
MAX_VIDEO_SAMPLES = 400
DECISIONS = ["GENERATE CHALLAN", "REVIEW", "NO CHALLAN"]


# ---------- light analysis ----------
def _halo(gray, x, y, cw, ch):
    """Mean brightness (0-1) of the thin ring just around a light. Lamps glow into the dark
    around them (high value); signs and reflectors have sharp edges and a dark surround."""
    H, W = gray.shape
    pad = max(int(0.5 * max(cw, ch)), 3)
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + cw + pad, W), min(y + ch + pad, H)
    outer = gray[y0:y1, x0:x1].astype(np.float32)
    mask = np.ones(outer.shape, bool)
    mask[y - y0:y - y0 + ch, x - x0:x - x0 + cw] = False
    return float(outer[mask].mean() / 255.0) if mask.any() else 0.0


def detect_lights(frame, core_thr=240, min_area=8, top_ignore=0.35):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray >= core_thr).astype(np.uint8)
    mask[: int(h * top_ignore)] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    lights = []
    for i in range(1, n):
        x, y, cw, ch, area = [int(v) for v in stats[i]]
        if area < min_area:
            continue
        if max(cw, ch) / max(min(cw, ch), 1) > 3.5 or area / (cw * ch) < 0.45:
            continue                                    # lane paint and other line-like shapes
        px = frame[y:y + ch, x:x + cw].reshape(-1, 3).astype(np.float32)
        b, g, r = px.mean(axis=0)
        halo = _halo(gray, x, y, cw, ch)
        size = min(1.0, area / (0.0006 * h * w))
        bloom = min(1.0, halo / 0.6)
        lights.append({"box": (x, y, x + cw, y + ch), "area": area, "cx": x + cw / 2, "cy": y + ch / 2,
                       "halo": round(halo, 3), "red": bool(r > 1.4 * max(g, b) + 5),
                       "glare": round(0.55 * size + 0.45 * bloom, 3)})
    return lights


def pair_lights(lights):
    used, pairs, cands = set(), [], []
    for i in range(len(lights)):
        for j in range(i + 1, len(lights)):
            a, b = lights[i], lights[j]
            wmax = max(a["box"][2] - a["box"][0], b["box"][2] - b["box"][0])
            hmax = max(a["box"][3] - a["box"][1], b["box"][3] - b["box"][1])
            dx, dy = abs(a["cx"] - b["cx"]), abs(a["cy"] - b["cy"])
            ratio = max(a["area"], b["area"]) / max(min(a["area"], b["area"]), 1)
            if dy <= 0.8 * hmax and ratio <= 3 and 1.5 * wmax <= dx <= 14 * wmax and a["red"] == b["red"]:
                cands.append((dx, i, j))
    for dx, i, j in sorted(cands):
        if i in used or j in used:
            continue
        used.update((i, j))
        pairs.append((i, j))
    return pairs


def predict_lights(frame, big_frac=0.002, bloom_min=0.33, min_head_area=40):
    """Classifies each light. A headlight must show a visible bloom: signs and reflectors
    have sharp edges and almost no glow, real lamps glow into the dark around them."""
    h, w = frame.shape[:2]
    lights = detect_lights(frame)
    paired = {k for p in pair_lights(lights) for k in p}
    for k, L in enumerate(lights):
        glows = L["halo"] >= bloom_min and L["area"] >= min_head_area
        big = L["area"] >= big_frac * h * w
        single_strong = L["halo"] >= 0.45 and L["area"] >= 150
        bw, bh = L["box"][2] - L["box"][0], L["box"][3] - L["box"][1]
        stripe = (max(bw, bh) / max(min(bw, bh), 1) >= 1.6 and L["area"] / (bw * bh) < 0.62) \
            or (bh > 1.3 * bw and L["area"] < big_frac * h * w)      # lane paint: slanted or tall and narrow
        if stripe:
            L["kind"] = "other light"
        elif L["red"] and k in paired:
            L["kind"] = "tail light"
        elif glows and not L["red"] and (k in paired or big or single_strong):
            L["kind"] = "headlight"
        else:
            L["kind"] = "other light"       # signs, reflectors, street lamps, small distant lights
    heads = [L for L in lights if L["kind"] == "headlight"]
    return lights, max((L["glare"] for L in heads), default=0.0)


def prepare(frame):
    h, w = frame.shape[:2]
    if w == TARGET_WIDTH:
        return frame
    interp = cv2.INTER_AREA if w > TARGET_WIDTH else cv2.INTER_LINEAR
    return cv2.resize(frame, (TARGET_WIDTH, max(int(h * TARGET_WIDTH / w), 1)), interpolation=interp)


def decode_image(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


COLORS = {"headlight": (0, 255, 0), "tail light": (0, 0, 255), "other light": (255, 160, 0)}


def draw(frame, lights):
    canvas = frame.copy()
    for L in lights:
        x1, y1, x2, y2 = L["box"]
        cv2.rectangle(canvas, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), COLORS[L["kind"]], 3)
    return canvas


# ---------- cached analysis ----------
@st.cache_data(show_spinner=False, max_entries=500)
def analyze_image(data):
    frame = decode_image(data)
    if frame is None:
        return None
    lights, glare = predict_lights(prepare(frame))
    return {"glare": float(glare), "headlights": sum(L["kind"] == "headlight" for L in lights)}


@st.cache_data(show_spinner=False, max_entries=50)
def annotated_image(data):
    frame = prepare(decode_image(data))
    lights, _ = predict_lights(frame)
    return cv2.cvtColor(draw(frame, lights), cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False, max_entries=10)
def analyze_video(data, suffix, every_n, max_samples=MAX_VIDEO_SAMPLES):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    samples, best_glare, best_frame, idx = [], -1.0, None, 0
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        while len(samples) < max_samples:
            if not cap.grab():
                break
            if idx % every_n == 0:
                ok, frame = cap.retrieve()
                if ok:
                    frame = prepare(frame)
                    lights, glare = predict_lights(frame)
                    heads = sum(L["kind"] == "headlight" for L in lights)
                    samples.append((idx, float(glare), int(heads)))
                    if glare > best_glare:
                        best_glare = glare
                        best_frame = cv2.cvtColor(draw(frame, lights), cv2.COLOR_BGR2RGB)
            idx += 1
        cap.release()
    finally:
        os.remove(path)
    return {"samples": samples, "fps": fps, "best_frame": best_frame}


# ---------- decisions ----------
def decide_image(glare, heads, yes_thr, review_thr):
    if heads == 0:
        return "NO CHALLAN", "No lit headlights found in the image."
    if glare >= yes_thr:
        return "GENERATE CHALLAN", f"Headlight glare {glare:.2f} is at or above the challan level ({yes_thr:.2f})."
    if glare >= review_thr:
        return "REVIEW", f"Headlight glare {glare:.2f} is between the review level ({review_thr:.2f}) and the challan level ({yes_thr:.2f})."
    return "NO CHALLAN", f"Headlight glare {glare:.2f} is below the review level ({review_thr:.2f})."


def longest_run(samples, threshold):
    run = best = 0
    for _, glare, heads in samples:
        run = run + 1 if heads > 0 and glare >= threshold else 0
        best = max(best, run)
    return best


def decide_video(samples, yes_thr, review_thr, k):
    if not samples:
        return "REVIEW", "No frames could be read from this video.", 0.0, 0
    peak = max(g for _, g, h in samples if h > 0) if any(h > 0 for _, _, h in samples) else 0.0
    heads = max(h for _, _, h in samples)
    if longest_run(samples, yes_thr) >= k:
        return "GENERATE CHALLAN", f"Glare stayed at or above {yes_thr:.2f} for {k} or more sampled frames in a row (peak {peak:.2f}).", peak, heads
    if longest_run(samples, review_thr) >= k or peak >= yes_thr:
        return "REVIEW", f"Glare peaked at {peak:.2f}, but it did not stay above {yes_thr:.2f} for {k} frames in a row.", peak, heads
    if heads == 0:
        return "NO CHALLAN", "No lit headlights found in the video.", peak, heads
    return "NO CHALLAN", f"Glare never held above the review level ({review_thr:.2f}) for {k} frames in a row (peak {peak:.2f}).", peak, heads


def show_decision(decision, reason):
    if decision == "GENERATE CHALLAN":
        st.error(f"### GENERATE CHALLAN\n{reason}")
    elif decision == "REVIEW":
        st.warning(f"### REVIEW\n{reason}")
    else:
        st.success(f"### NO CHALLAN\n{reason}")


# ---------- reading uploads ----------
def expand_uploads(files):
    """Turns the uploaded files into (name, kind, bytes) items. Zips are opened in memory."""
    items, problems = [], []
    for f in files:
        name, lower = f.name, f.name.lower()
        data = f.getvalue()
        if lower.endswith(IMAGE_EXTS):
            items.append((name, "image", data))
        elif lower.endswith(VIDEO_EXTS):
            items.append((name, "video", data))
        elif lower.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    members = [m for m in archive.infolist() if not m.is_dir()]
                    if sum(m.file_size for m in members) > MAX_UNZIPPED_BYTES:
                        problems.append(f"{name}: too large once unpacked.")
                        continue
                    count = 0
                    for m in members:
                        base = os.path.basename(m.filename)
                        if base.startswith(".") or not base.lower().endswith(IMAGE_EXTS):
                            continue
                        if count >= MAX_ZIP_IMAGES:
                            problems.append(f"{name}: only the first {MAX_ZIP_IMAGES} images were used.")
                            break
                        items.append((f"{name}/{base}", "image", archive.read(m)))
                        count += 1
                    if count == 0:
                        problems.append(f"{name}: no images found inside the zip.")
            except zipfile.BadZipFile:
                problems.append(f"{name}: not a valid zip file.")
        else:
            problems.append(f"{name}: unsupported file type.")
    return items, problems


# ---------- app ----------
st.set_page_config(page_title="BeamGuardAI - Challan check", layout="wide")

st.title("BeamGuardAI: high-beam challan check")
st.caption(
    "Upload an image, a zip of images, or a video. The app looks for headlights and measures their "
    "glare, then tells you whether to generate a challan. This is a prototype: the glare levels are "
    "not calibrated on labeled data, so an authorized person should confirm before a challan is issued."
)

st.sidebar.header("Decision rules")
yes_thr = st.sidebar.slider("Generate challan at glare of", 0.50, 1.00, 0.90, 0.01)
review_thr = min(st.sidebar.slider("Send for review from glare of", 0.20, 1.00, 0.60, 0.01), yes_thr)
st.sidebar.header("Video")
every_n = st.sidebar.slider("Analyze every Nth frame", 1, 30, 5)
k_frames = st.sidebar.slider("Frames in a row needed", 1, 10, 3)
with st.sidebar.expander("How the decision is made"):
    st.write(
        "The app finds very bright spots, keeps the ones that glow into the dark around them the way "
        "real lamps do (road signs and reflectors do not), and pairs left and right lamps. The glare "
        "score (0 to 1) grows with the size and glow of the brightest headlight. A single image can "
        "only show one moment, so a video is a stronger basis: it needs glare to hold for several "
        "sampled frames in a row before it says to generate a challan."
    )

uploads = st.file_uploader(
    "Upload images, a zip of images, or a video",
    type=[e.strip(".") for e in IMAGE_EXTS + VIDEO_EXTS] + ["zip"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Upload a file above to get a decision.")
    st.stop()

items, problems = expand_uploads(uploads)
for message in problems:
    st.warning(message)
if not items:
    st.stop()

rows, details = [], {}
with st.spinner("Analyzing..."):
    for name, kind, data in items:
        if kind == "image":
            result = analyze_image(data)
            if result is None:
                st.error(f"{name}: could not read this image.")
                continue
            decision, reason = decide_image(result["glare"], result["headlights"], yes_thr, review_thr)
            rows.append({"File": name, "Type": "image", "Headlights": result["headlights"],
                         "Glare score": round(result["glare"], 3), "Decision": decision})
            details[name] = {"kind": "image", "data": data, "decision": decision, "reason": reason}
        else:
            suffix = os.path.splitext(name)[1] or ".mp4"
            result = analyze_video(data, suffix, every_n)
            decision, reason, peak, heads = decide_video(result["samples"], yes_thr, review_thr, k_frames)
            rows.append({"File": name, "Type": "video", "Headlights": heads,
                         "Glare score": round(peak, 3), "Decision": decision})
            details[name] = {"kind": "video", "result": result, "decision": decision, "reason": reason}

if not rows:
    st.stop()

table = pd.DataFrame(rows)
table["_order"] = table["Decision"].map({d: i for i, d in enumerate(DECISIONS)})
table = table.sort_values(["_order", "Glare score"], ascending=[True, False]).drop(columns="_order").reset_index(drop=True)

if len(table) > 1:
    counts = table["Decision"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files checked", len(table))
    c2.metric("Generate challan", int(counts.get("GENERATE CHALLAN", 0)))
    c3.metric("Review", int(counts.get("REVIEW", 0)))
    c4.metric("No challan", int(counts.get("NO CHALLAN", 0)))
    st.dataframe(table, hide_index=True)
    st.download_button("Download results as CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name="challan_decisions.csv", mime="text/csv")

selected = st.selectbox("Show details for", table["File"].tolist()) if len(table) > 1 else table["File"].iloc[0]
info = details[selected]

show_decision(info["decision"], info["reason"])

if info["kind"] == "image":
    st.image(annotated_image(info["data"]),
             caption="Green: headlights. Orange: other lights (signs, reflectors, street lamps). Red: tail lights.")
    st.caption("This decision is based on one image. A video gives a more reliable result.")
else:
    result = info["result"]
    if result["samples"]:
        frames = [s[0] for s in result["samples"]]
        chart = pd.DataFrame({"Glare score": [s[1] if s[2] > 0 else 0.0 for s in result["samples"]]},
                             index=pd.Index(frames, name="Frame"))
        st.line_chart(chart)
        if result["best_frame"] is not None:
            st.image(result["best_frame"], caption="Frame with the strongest headlight glare.")
    st.caption(f"Analyzed {len(result['samples'])} frames (every {every_n}th).")