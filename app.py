"""BeamGuardAI review dashboard.

Run it on your own machine (not inside Kaggle):

    pip install -r requirements.txt
    streamlit run app.py

Put the files you downloaded from the Kaggle notebook next to this script:

    dashboard/
        app.py
        output/
            violations.db
            evidence/violation_*.jpg
"""
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="BeamGuardAI Dashboard", layout="wide")

STATUSES = ["pending", "approved", "rejected"]
TABLE_COLUMNS = [
    "id", "timestamp", "camera_id", "vehicle_track_id",
    "plate_number", "violation_score", "review_status",
]


def load_violations(db_path):
    if not os.path.exists(db_path):
        return None, f"Database file not found: {db_path}"
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM violations ORDER BY timestamp DESC", conn
            )
    except Exception as exc:
        return None, f"Could not read the violations table: {exc}"
    return df, None


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
            "id": i,
            "vehicle_track_id": 10 + i,
            "plate_number": plate,
            "timestamp": (now - timedelta(minutes=7 * i)).isoformat(timespec="seconds"),
            "camera_id": "CAM01",
            "mean_brightness": 200.0 + i,
            "bright_ratio": 0.2 + 0.05 * i,
            "violation_score": score,
            "image_path": "",
            "review_status": status,
        })
    return pd.DataFrame(rows)


def set_status(db_path, violation_id, status):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE violations SET review_status = ? WHERE id = ?",
            (status, int(violation_id)),
        )
        conn.commit()


def find_image(image_path, evidence_dir):
    """The database stores Kaggle paths, so fall back to the local evidence folder."""
    if not image_path:
        return None
    if os.path.exists(image_path):
        return image_path
    candidate = os.path.join(evidence_dir, os.path.basename(image_path))
    return candidate if os.path.exists(candidate) else None


# ---------- sidebar ----------
st.sidebar.header("Data")
db_path = st.sidebar.text_input("Database file", "output/violations.db")
evidence_dir = st.sidebar.text_input("Evidence folder", "output/evidence")
demo = st.sidebar.checkbox("Show demo data (no database needed)")

st.title("BeamGuardAI Dashboard")
st.caption(
    "Potential high-beam violations detected by the system, for review by an "
    "authorized person. Nothing here is an automatic fine."
)

if demo:
    df, error = demo_violations(), None
    st.warning("DEMO DATA: these rows are made up and are not real detections.")
else:
    df, error = load_violations(db_path)

if error:
    st.error(error)
    st.info(
        "Run the pipeline on a video in the Kaggle notebook, commit it, download "
        "`output/violations.db` and `output/evidence/` from the Output tab, and "
        "place them next to this script. Or tick the demo option in the sidebar."
    )
    st.stop()

if df.empty:
    st.info(
        "The database is empty. Nothing has been logged yet: `run_pipeline()` "
        "only logs a candidate after a headlight scores high for several frames "
        "in a row on a video."
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
    plate = row["plate_number"] or "plate unknown"
    return f"#{violation_id} | {plate} | score {row['violation_score']:.0%} | {row['review_status']}"


selected_id = st.selectbox("Candidate", filtered["id"].tolist(), format_func=label)
row = filtered[filtered["id"] == selected_id].iloc[0]

left, right = st.columns([3, 2])

with left:
    image = find_image(row["image_path"], evidence_dir)
    if image:
        st.image(image, caption=os.path.basename(image))
    else:
        st.info("No evidence image found for this candidate.")

with right:
    st.write(f"**Time:** {row['timestamp']}")
    st.write(f"**Camera:** {row['camera_id']}")
    st.write(f"**Plate:** {row['plate_number'] or 'not read'}")
    st.write(f"**Vehicle track ID:** {row['vehicle_track_id']}")
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
            set_status(db_path, selected_id, new_status)
            st.rerun()
    if demo:
        st.caption("Review buttons are disabled in demo mode.")
