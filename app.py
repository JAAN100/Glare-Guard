"""BeamGuardAI review dashboard.

Run it on your own machine or deploy it (for example on Streamlit Community Cloud):

    pip install -r requirements.txt
    streamlit run app.py

Data sources (choose one in the sidebar):
    Upload files  - upload results.zip (violations.db + evidence images), a .db or a .csv,
                    plus evidence images from your computer. Works anywhere, including
                    Streamlit Community Cloud. Changes live only in your browser session:
                    use the download buttons to keep them.
    Local files   - read output/violations.db and output/evidence/ next to this script.
                    Approve/Reject and new records are saved to those files.
    Demo data     - made-up rows to try the layout.
"""
import io
import os
import re
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

STATUSES = ["pending", "approved", "rejected"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DB_EXTS = (".db", ".sqlite", ".sqlite3")
MAX_UNZIPPED_BYTES = 300 * 1024 * 1024  # guard against oversized or malicious zips

COLUMNS = [
    "id", "vehicle_track_id", "plate_number", "timestamp", "camera_id",
    "mean_brightness", "bright_ratio", "violation_score", "image_path",
    "review_status",
]
TABLE_COLUMNS = [
    "id", "timestamp", "camera_id", "vehicle_track_id",
    "plate_number", "violation_score", "review_status",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_track_id INTEGER,
    plate_number TEXT,
    timestamp TEXT,
    camera_id TEXT,
    mean_brightness REAL,
    bright_ratio REAL,
    violation_score REAL,
    image_path TEXT,
    review_status TEXT DEFAULT 'pending'
)
"""


# ---------- helpers ----------
def basename(path):
    return re.split(r"[\\/]", str(path))[-1]


def clean(value, fallback=""):
    """Show blanks instead of 'nan' / 'None'."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return fallback
    return value


def to_python(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def normalize(df):
    """Make any uploaded table look like the violations table."""
    df = df.copy()
    defaults = {
        "vehicle_track_id": None, "plate_number": None, "timestamp": "",
        "camera_id": "", "mean_brightness": 0.0, "bright_ratio": 0.0,
        "violation_score": 0.0, "image_path": "", "review_status": "pending",
    }
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    for column in ("mean_brightness", "bright_ratio", "violation_score"):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).drop_duplicates("id")
    df["id"] = df["id"].astype(int)
    df["timestamp"] = df["timestamp"].fillna("").astype(str)
    df["image_path"] = df["image_path"].fillna("").astype(str)
    df["review_status"] = df["review_status"].where(
        df["review_status"].isin(STATUSES), "pending"
    )
    return df[COLUMNS].sort_values("timestamp", ascending=False).reset_index(drop=True)


def empty_table():
    return normalize(pd.DataFrame(columns=COLUMNS))


def demo_violations():
    """Made-up rows so you can try the dashboard before any real detections exist."""
    now = datetime.now()
    rows = []
    for i, (plate, score, status) in enumerate(
        [("DEMO-001", 0.91, "pending"), ("DEMO-002", 0.84, "approved"),
         ("DEMO-003", 0.78, "rejected"), ("DEMO-004", 0.88, "pending")],
        start=1,
    ):
        rows.append({
            "id": i, "vehicle_track_id": 10 + i, "plate_number": plate,
            "timestamp": (now - timedelta(minutes=7 * i)).isoformat(timespec="seconds"),
            "camera_id": "CAM01", "mean_brightness": 200.0 + i,
            "bright_ratio": 0.2 + 0.05 * i, "violation_score": score,
            "image_path": "", "review_status": status,
        })
    return normalize(pd.DataFrame(rows))


# ---------- reading data ----------
def read_db_bytes(data):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        with closing(sqlite3.connect(path)) as conn:
            return pd.read_sql_query("SELECT * FROM violations", conn)
    finally:
        os.remove(path)


def load_upload(uploaded):
    """Returns (dataframe, {image name: bytes}, error message or None)."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    try:
        if name.endswith(".zip"):
            images, df = {}, None
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [m for m in archive.infolist() if not m.is_dir()]
                if sum(m.file_size for m in members) > MAX_UNZIPPED_BYTES:
                    return None, {}, "The zip is too large once unpacked."
                for member in members:
                    base = basename(member.filename)
                    if base.startswith("."):  # skips macOS '._' files
                        continue
                    lower = base.lower()
                    if lower.endswith(DB_EXTS) and df is None:
                        df = read_db_bytes(archive.read(member))
                    elif lower.endswith(IMAGE_EXTS):
                        images[base] = archive.read(member)
            if df is None:
                return None, images, "No database (.db) found inside the zip."
            return normalize(df), images, None
        if name.endswith(DB_EXTS):
            return normalize(read_db_bytes(data)), {}, None
        if name.endswith(".csv"):
            return normalize(pd.read_csv(io.BytesIO(data))), {}, None
        return None, {}, "Unsupported file type. Use .zip, .db or .csv."
    except Exception as exc:
        return None, {}, f"Could not read {uploaded.name}: {exc}"


def load_violations(db_path):
    if not os.path.exists(db_path):
        return None, f"Database file not found: {db_path}"
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            df = pd.read_sql_query("SELECT * FROM violations", conn)
    except Exception as exc:
        return None, f"Could not read the violations table: {exc}"
    return normalize(df), None


# ---------- writing data ----------
def set_status(db_path, violation_id, status):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE violations SET review_status = ? WHERE id = ?",
            (status, int(violation_id)),
        )
        conn.commit()


def insert_record(db_path, record):
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fields = [c for c in COLUMNS if c != "id"]
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(SCHEMA)
        cursor = conn.execute(
            f"INSERT INTO violations ({', '.join(fields)}) "
            f"VALUES ({', '.join('?' * len(fields))})",
            [record[f] for f in fields],
        )
        conn.commit()
        return cursor.lastrowid


def build_db_bytes(df):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(SCHEMA)
            rows = [
                tuple(to_python(v) for v in row)
                for row in df[COLUMNS].itertuples(index=False, name=None)
            ]
            conn.executemany(
                f"INSERT INTO violations ({', '.join(COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(COLUMNS))})",
                rows,
            )
            conn.commit()
        with open(path, "rb") as handle:
            return handle.read()
    finally:
        os.remove(path)


def build_zip_bytes(df, images):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("violations.db", build_db_bytes(df))
        for name, data in images.items():
            archive.writestr(f"evidence/{name}", data)
    return buffer.getvalue()


def find_image(image_path, images, evidence_dir):
    """Uploaded images first, then the path in the database, then the local evidence folder."""
    if not image_path:
        return None
    name = basename(image_path)
    if name in images:
        return images[name]
    if os.path.exists(image_path):
        return image_path
    candidate = os.path.join(evidence_dir, name)
    return candidate if os.path.exists(candidate) else None


# ---------- app ----------
st.set_page_config(page_title="BeamGuardAI Dashboard", layout="wide")
state = st.session_state

st.sidebar.header("Data")
mode = st.sidebar.radio("Data source", ["Upload files", "Local files", "Demo data"])
demo = mode == "Demo data"
local = mode == "Local files"

if state.get("mode") != mode:  # switching source starts fresh
    state["mode"] = mode
    state["df"] = None
    state["images"] = {}
    state["sig"] = None

db_path = "output/violations.db"
evidence_dir = "output/evidence"
error = None

if mode == "Upload files":
    results_file = st.sidebar.file_uploader(
        "Results: results.zip, .db or .csv", type=["zip", "db", "sqlite", "sqlite3", "csv"]
    )
    image_files = st.sidebar.file_uploader(
        "Evidence images (optional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if results_file is not None:
        signature = (results_file.name, results_file.size)
        if state["sig"] != signature:
            loaded_df, loaded_images, error = load_upload(results_file)
            if error is None:
                state["df"], state["images"], state["sig"] = loaded_df, loaded_images, signature
    elif state["sig"] is not None:  # the results file was removed
        state["df"], state["images"], state["sig"] = None, {}, None
    if state["df"] is None:
        state["df"], state["images"] = empty_table(), {}
    for image in image_files or []:
        state["images"][basename(image.name)] = image.getvalue()
    st.sidebar.caption(
        f"{len(state['images'])} evidence image(s) loaded. Changes last only for this "
        "browser session: download the results below to keep them."
    )
elif local:
    db_path = st.sidebar.text_input("Database file", db_path)
    evidence_dir = st.sidebar.text_input("Evidence folder", evidence_dir)
    state["df"], error = load_violations(db_path)
else:
    if state["sig"] != "demo":
        state["df"], state["images"], state["sig"] = demo_violations(), {}, "demo"

df = state["df"]
images = state.get("images", {})

st.title("BeamGuardAI Dashboard")
st.caption(
    "Potential high-beam violations detected by the system, for review by an "
    "authorized person. Nothing here is an automatic fine."
)

if state.get("flash"):
    st.success(state.pop("flash"))
if demo:
    st.warning("DEMO DATA: these rows are made up and are not real detections.")
if error:
    st.error(error)
if df is None:
    st.info(
        "Run the pipeline on a video in the Kaggle notebook, commit it, download "
        "`output/violations.db` and `output/evidence/` from the Output tab, and "
        "place them next to this script. Or switch the data source in the sidebar."
    )
    st.stop()

# export (upload mode): keeps this session's changes
if mode == "Upload files" and not df.empty:
    st.sidebar.download_button(
        "Download results.zip (database + images)",
        data=build_zip_bytes(df, images),
        file_name="results.zip",
        mime="application/zip",
    )
    st.sidebar.download_button(
        "Download table as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="violations.csv",
        mime="text/csv",
    )


# ---------- add a record ----------
def add_record():
    with st.expander("Add a record manually (with an image from your computer)"):
        with st.form("add_record", clear_on_submit=True):
            plate = st.text_input("Plate number (optional)")
            camera = st.text_input("Camera ID", "CAM01")
            score = st.slider("Violation score", 0.0, 1.0, 0.5, 0.01)
            when = st.text_input("Timestamp", datetime.now().isoformat(timespec="seconds"))
            image = st.file_uploader("Evidence image", type=["jpg", "jpeg", "png"])
            submitted = st.form_submit_button("Add record")
        if not submitted:
            return
        image_name = ""
        if image is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", basename(image.name))
            image_name = f"manual_{stamp}_{safe}"
        record = {
            "vehicle_track_id": None, "plate_number": plate.strip() or None,
            "timestamp": when.strip(), "camera_id": camera.strip(),
            "mean_brightness": 0.0, "bright_ratio": 0.0,
            "violation_score": float(score), "image_path": image_name,
            "review_status": "pending",
        }
        if local:
            insert_record(db_path, record)
            if image is not None:
                os.makedirs(evidence_dir, exist_ok=True)
                with open(os.path.join(evidence_dir, image_name), "wb") as handle:
                    handle.write(image.getvalue())
        else:
            new_id = int(state["df"]["id"].max()) + 1 if not state["df"].empty else 1
            row = pd.DataFrame([{**record, "id": new_id}])
            combined = row if state["df"].empty else pd.concat([state["df"], row], ignore_index=True)
            state["df"] = normalize(combined)
            if image is not None:
                state["images"][image_name] = image.getvalue()
        state["flash"] = "Record added."
        st.rerun()


if not demo:
    add_record()

if df.empty:
    st.info(
        "Nothing here yet. Upload a results file in the sidebar, or add a record "
        "above. (The pipeline only logs a candidate after a headlight scores high "
        "for several frames in a row on a video.)"
    )
    st.stop()

# ---------- summary ----------
counts = df["review_status"].value_counts()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total candidates", len(df))
c2.metric("Pending", int(counts.get("pending", 0)))
c3.metric("Approved", int(counts.get("approved", 0)))
c4.metric("Rejected", int(counts.get("rejected", 0)))
c5.metric("Average score", f"{df['violation_score'].mean():.0%}")

# ---------- filters ----------
selected_statuses = st.sidebar.multiselect("Review status", STATUSES, default=STATUSES)
min_score = st.sidebar.slider("Minimum score", 0.0, 1.0, 0.0, 0.05)

filtered = df[
    df["review_status"].isin(selected_statuses) & (df["violation_score"] >= min_score)
]

st.subheader("Violation candidates")
st.dataframe(filtered[TABLE_COLUMNS], hide_index=True)

if filtered.empty:
    st.info("No candidates match the current filters.")
    st.stop()

# ---------- detail and review ----------
st.subheader("Review a candidate")


def label(violation_id):
    row = filtered[filtered["id"] == violation_id].iloc[0]
    plate = clean(row["plate_number"], "plate unknown")
    return f"#{violation_id} | {plate} | score {row['violation_score']:.0%} | {row['review_status']}"


selected_id = st.selectbox("Candidate", filtered["id"].tolist(), format_func=label)
row = filtered[filtered["id"] == selected_id].iloc[0]

left, right = st.columns([3, 2])

with left:
    image = find_image(row["image_path"], images, evidence_dir)
    if image is not None:
        st.image(image, caption=basename(row["image_path"]))
    else:
        st.info("No evidence image found for this candidate. Upload it in the sidebar.")

with right:
    st.write(f"**Time:** {row['timestamp']}")
    st.write(f"**Camera:** {clean(row['camera_id'], '-')}")
    st.write(f"**Plate:** {clean(row['plate_number'], 'not read')}")
    st.write(f"**Vehicle track ID:** {clean(row['vehicle_track_id'], '-')}")
    st.write(f"**Violation score:** {row['violation_score']:.0%}")
    st.write(f"**Mean brightness:** {row['mean_brightness']:.1f}")
    st.write(f"**Bright pixel ratio:** {row['bright_ratio']:.1%}")
    st.write(f"**Status:** {row['review_status']}")

    b1, b2, b3 = st.columns(3)
    for col, text, new_status in [
        (b1, "Approve", "approved"),
        (b2, "Reject", "rejected"),
        (b3, "Reset", "pending"),
    ]:
        if col.button(text, disabled=demo, key=f"{new_status}_{selected_id}"):
            if local:
                set_status(db_path, selected_id, new_status)
            else:
                state["df"].loc[state["df"]["id"] == selected_id, "review_status"] = new_status
            st.rerun()
    if demo:
        st.caption("Review buttons are disabled in demo mode.")