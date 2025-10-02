# app_streamlit.py — CMU-MOSI viewer (Video • Audio • Expressions • Evaluation)
# Ready for Streamlit Cloud: uses relative paths (no C:\...).
# Drop your aligned file next to this script as: MOSI_aligned_combined.xlsx

import os
import glob
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional deps (only needed for waveform + MAE)
try:
    import librosa
except Exception:
    librosa = None

# ------------------- Defaults -------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = str(BASE_DIR / "MOSI_aligned_combined.xlsx")
VIDEO_DIR = BASE_DIR / "Video"
AUDIO_DIR = BASE_DIR / "Audio"
VIDEO_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

DEFAULT_SUS_URL = ""   # paste your Google Form link here (or fill from sidebar)

PAGE_TITLE = "CMU-MOSI Aligned Viewer (Video • Audio • Expressions • Evaluation)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# ------------------- Helpers -------------------
def coerce_word(x):
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    s = str(x)
    if s.startswith("[b'") and s.endswith("']"):
        return s[3:-2]
    if s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    return s

def get_cols_by_prefix(df, prefix):
    pref = prefix.lower()
    cols = [c for c in df.columns if c.lower().startswith(pref)]
    def tail_num(c):
        d = "".join(ch for ch in c if c and ch.isdigit())
        return int(d) if d else 10**9
    return sorted(cols, key=tail_num)

def detect_facet_columns(df):
    lower_map = {c.lower(): c for c in df.columns}
    known = ["joy","anger","sadness","surprise","fear","disgust","contempt",
             "neutral","valence","engagement","smile","brow_raise","brow_furrow"]
    named_map = {}
    for k in known:
        for cand in (f"facet_{k}", f"facet-emotion_{k}", f"emotion_{k}", k):
            if cand in lower_map:
                named_map[k] = lower_map[cand]
                break
    generic_cols = [c for c in df.columns if c.lower().startswith("facet_")]
    return named_map, generic_cols

def nearest_idx(val, arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0
    return int(np.argmin(np.abs(arr - float(val))))

def download_youtube_video(video_id):
    """Download video+audio from YouTube for given MOSI ID"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        subprocess.run([
            "yt-dlp", "-f", "mp4",
            "-o", f"{VIDEO_DIR}/{video_id}.%(ext)s",
            url
        ], check=True)

        subprocess.run([
            "yt-dlp", "-f", "bestaudio", "--extract-audio",
            "--audio-format", "wav", "--audio-quality", "0",
            "-o", f"{AUDIO_DIR}/{video_id}.%(ext)s",
            url
        ], check=True)

        return True
    except Exception as e:
        st.warning(f"Could not download video {video_id}: {e}")
        return False

def get_local_paths(video_id):
    v = VIDEO_DIR / f"{video_id}.mp4"
    a = AUDIO_DIR / f"{video_id}.wav"
    return (v if v.exists() else None, a if a.exists() else None)

# ------------------- Cached loaders -------------------
@st.cache_data(show_spinner=False)
def load_aligned(path_or_buf, sheet=None):
    p = str(path_or_buf)
    if p.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(path_or_buf, sheet_name=sheet if sheet is not None else 0)
    else:
        df = pd.read_csv(path_or_buf)

    need = ["video_id","word_start","word_end"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV/XLSX missing required columns: {miss}")

    if "word_mid" not in df.columns:
        df["word_mid"] = (pd.to_numeric(df["word_start"])+pd.to_numeric(df["word_end"])) / 2.0
    if "word" not in df.columns:
        df["word"] = ""

    df["word"] = df["word"].apply(coerce_word)
    for c in ["word_start","word_end","word_mid"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values(["video_id","word_mid"]).reset_index(drop=True)

def make_subtitles(df_video: pd.DataFrame, gap: float = 0.6) -> pd.DataFrame:
    if df_video.empty:
        return pd.DataFrame(columns=["start","end","mid","text"])
    w = df_video[["word_start","word_end","word"]].copy().sort_values("word_start")
    w["word"] = w["word"].astype(str).str.strip()
    w = w[w["word"] != ""]
    starts = w["word_start"].to_numpy(); ends = w["word_end"].to_numpy()
    seg_id = [0]
    for i in range(1, len(w)):
        seg_id.append(seg_id[-1] + 1 if float(starts[i]-ends[i-1]) > gap else seg_id[-1])
    w["seg_id"] = seg_id
    subs = (w.groupby("seg_id", as_index=False)
              .agg(start=("word_start","min"),
                   end=("word_end","max"),
                   text=("word", lambda s: " ".join(s))))
    subs["mid"] = (subs["start"] + subs["end"]) / 2.0
    return subs[["start","end","mid","text"]]

@st.cache_data(show_spinner=False)
def get_subtitles_for_video(df_all: pd.DataFrame, video_id: str, gap: float = 0.6) -> pd.DataFrame:
    return make_subtitles(df_all[df_all["video_id"].astype(str)==str(video_id)].copy(), gap=gap)

# ------------------- Waveform -------------------
@st.cache_data(show_spinner=False)
def load_waveform_for_vid(vid: str, target_sr: int = 16000):
    if librosa is None:
        return None, None, "librosa_not_installed"
    _, a = get_local_paths(vid)
    if a and a.exists():
        try:
            y, sr = librosa.load(a, sr=target_sr, mono=True)
            t = np.arange(len(y), dtype=np.float32) / float(sr)
            return t, y.astype(np.float32), None
        except Exception as e:
            return None, None, f"decode_failed: {e}"
    v, _ = get_local_paths(vid)
    if v and v.exists():
        try:
            y, sr = librosa.load(v, sr=target_sr, mono=True)
            t = np.arange(len(y), dtype=np.float32) / float(sr)
            return t, y.astype(np.float32), None
        except Exception as e:
            return None, None, f"ffmpeg_decode_failed: {e}"
    return None, None, "no_audio_source"

def downsample_for_plot(t, y, max_points=4000):
    if t is None or y is None or len(y) == 0:
        return None, None
    n = len(y)
    if n <= max_points:
        return t, y
    idx = np.linspace(0, n-1, num=max_points, dtype=int)
    return t[idx], y[idx]

# ------------------- Sidebar: load aligned -------------------
st.sidebar.header("Data source")
mode = st.sidebar.radio("Load aligned data from:", ["Default path", "Upload file"], index=0)

start_load = time.perf_counter()
if mode == "Upload file":
    up = st.sidebar.file_uploader("Upload aligned CSV/XLSX", type=["csv","xlsx","xls"])
    if up is None:
        st.warning("Please upload your aligned file to continue.")
        st.stop()
    df_all = load_aligned(up)
    loaded_path_display = "(uploaded)"
else:
    if not DEFAULT_DATA_PATH or not os.path.exists(DEFAULT_DATA_PATH):
        st.warning("No default aligned file found in repo. Please upload one from the sidebar.")
        st.stop()
    df_all = load_aligned(DEFAULT_DATA_PATH)
    loaded_path_display = DEFAULT_DATA_PATH

load_time = time.perf_counter() - start_load
st.sidebar.metric("Data load time", f"{load_time:.2f}s")
st.sidebar.caption(f"Using: {loaded_path_display}")
st.sidebar.success(f"Rows loaded: {len(df_all):,}")

# ------------------- Silence filter -------------------
hide_silence = st.sidebar.checkbox("Hide silence tokens (sp/sil)", True)
if hide_silence and "word" in df_all.columns:
    SILENCE = {"sp","sil","[sp]","[sil]","<sp>","<sil>",""}
    df_all = df_all[~df_all["word"].astype(str).str.lower().isin(SILENCE)].reset_index(drop=True)
    st.sidebar.info(f"Rows after silence filter: {len(df_all):,}")

# ------------------- Video selection -------------------
video_ids = df_all["video_id"].astype(str).unique().tolist()
if not video_ids:
    st.error("No video_ids found.")
    st.stop()

vid = st.selectbox("Choose a video_id", video_ids, index=0)
df = df_all[df_all["video_id"].astype(str) == str(vid)].copy().reset_index(drop=True)
if df.empty:
    st.error("No rows for selected video.")
    st.stop()

# ------------------- Layout -------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Playback")
    if "cur_t" not in st.session_state:
        st.session_state.cur_t = float(df["word_mid"].iloc[0]) if len(df) else 0.0

    st.session_state.cur_t = st.slider(
        "Time (s)", min_value=float(df["word_start"].min()), 
        max_value=float(df["word_end"].max()),
        value=st.session_state.cur_t, step=0.001, format="%.3f",
        key="time_slider",
    )
    cur_t = float(st.session_state.cur_t)

    idx = nearest_idx(cur_t, df["word_mid"].values)
    row = df.iloc[idx]
    st.metric("Current word", row["word"])

    subs = get_subtitles_for_video(df_all, vid, gap=0.6)
    if not subs.empty:
        j = nearest_idx(cur_t, subs["mid"].values)
        st.markdown("**Subtitle (now)**")
        st.write(subs.iloc[j]["text"])

    # Video handling
    st.markdown("**Video**")
    v_local, a_local = get_local_paths(vid)
    if not v_local or not a_local:
        st.info(f"Local media not found for {vid}, attempting download...")
        ok = download_youtube_video(vid)
        v_local, a_local = get_local_paths(vid) if ok else (None, None)

    if v_local and v_local.exists():
        st.video(str(v_local))
    else:
        st.error("No playable video found (may be expired).")

with right:
    st.subheader("Audio waveform")
    t_wav, y_wav, wav_err = load_waveform_for_vid(vid, target_sr=16000)
    if t_wav is None or y_wav is None:
        st.info(f"Waveform unavailable: {wav_err}")
    else:
        tt, yy = downsample_for_plot(t_wav, y_wav, max_points=4000)
        wav_fig = go.Figure()
        wav_fig.add_trace(go.Scatter(x=tt, y=yy, mode="lines", name="waveform"))
        ws, we = float(row["word_start"]), float(row["word_end"])
        wav_fig.add_vline(x=cur_t, line_width=2, line_dash="dash")
        st.plotly_chart(wav_fig, use_container_width=True)
