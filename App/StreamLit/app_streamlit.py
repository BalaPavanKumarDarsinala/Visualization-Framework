# app_streamlit.py — CMU-MOSI viewer (Video • Audio • Expressions • Evaluation)
# Ready for Streamlit Cloud: uses relative paths (no C:\...).
# Drop your aligned file next to this script as: MOSI_aligned_combined.xlsx

import os
import glob
import time
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

# ------------------- Defaults (relative paths for Cloud) -------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = str(BASE_DIR / "MOSI_aligned_combined.xlsx")
DEFAULT_VIDEO_DIR = ""   # leave empty on Cloud (use YouTube fallback)
DEFAULT_AUDIO_DIR = ""   # leave empty on Cloud
DEFAULT_SUS_URL   = ""   # paste your Google Form link here (or fill from sidebar)

PAGE_TITLE = "CMU-MOSI Aligned Viewer (Video • Audio • Expressions • Evaluation)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# ------------------- Small helpers -------------------
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

def find_local_video(video_dir, vid):
    if not video_dir:
        return None
    for ext in (".mp4",".avi",".mov",".mkv",".webm",".m4v"):
        p = Path(video_dir) / f"{vid}{ext}"
        if p.is_file(): return str(p)
    gl = glob.glob(str(Path(video_dir) / f"{vid}.*"))
    return gl[0] if gl else None

def find_local_audio(audio_dir, vid):
    if not audio_dir:
        return None
    for ext in (".wav",".mp3",".m4a",".ogg",".flac"):
        p = Path(audio_dir) / f"{vid}{ext}"
        if p.is_file(): return str(p)
    gl = glob.glob(str(Path(audio_dir) / f"{vid}.*"))
    return gl[0] if gl else None

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

# Emotion mapping
BASIC_EMOS = ["joy","anger","sadness","surprise","fear","disgust","contempt","neutral"]
def pretty_emo(name: str) -> str: return name.replace("_"," ").title()

def pick_emotion_text(row: pd.Series, facet_map: dict, facet_generic: list[str]):
    have = {e: facet_map[e] for e in BASIC_EMOS if e in facet_map}
    if have:
        cols = [have[e] for e in have]
        try:
            vals = row[cols].astype(float).values
            if not np.all(np.isnan(vals)):
                idx = int(np.nanargmax(vals))
                emo = list(have.keys())[idx]
                score = float(vals[idx]) if not np.isnan(vals[idx]) else 0.0
                display = {"joy":"Happy","anger":"Angry","sadness":"Sad",
                           "surprise":"Surprised","fear":"Afraid","disgust":"Disgusted",
                           "contempt":"Contempt","neutral":"Neutral"}.get(emo, pretty_emo(emo))
                return display, score, cols
        except Exception:
            pass
    if "valence" in facet_map:
        try:
            v = float(row[facet_map["valence"]])
            if v > 0.2: return "Happy", v, [facet_map["valence"]]
            if v < -0.2: return "Sad",   v, [facet_map["valence"]]
            return "Neutral", v, [facet_map["valence"]]
        except Exception:
            pass
    if "smile" in facet_map:
        try:
            s = float(row[facet_map["smile"]])
            if s > 0.5: return "Happy", s, [facet_map["smile"]]
        except Exception:
            pass
    au12 = [c for c in facet_generic if "au12" in c.lower()]
    if au12:
        try:
            v = float(row[au12[0]])
            if v > 0.5: return "Happy", v, [au12[0]]
        except Exception:
            pass
    return "Neutral", 0.0, []

# Waveform + downsampling (used by audio panel & MAE)
@st.cache_data(show_spinner=False)
def load_waveform_for_vid(vid: str, audio_dir: str, video_dir: str, target_sr: int = 16000):
    if librosa is None:
        return None, None, "librosa_not_installed"
    a_path = find_local_audio(audio_dir, vid) if audio_dir else None
    if a_path and os.path.isfile(a_path):
        try:
            y, sr = librosa.load(a_path, sr=target_sr, mono=True)
            t = np.arange(len(y), dtype=np.float32) / float(sr)
            return t, y.astype(np.float32), None
        except Exception:
            pass
    v_path = find_local_video(video_dir, vid) if video_dir else None
    if v_path and os.path.isfile(v_path):
        try:
            y, sr = librosa.load(v_path, sr=target_sr, mono=True)
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

# ------------------- Evaluation: MAE (word mid vs audio RMS peaks) -------------------
@st.cache_data(show_spinner=False)
def compute_sync_mae(vid: str, df_video: pd.DataFrame, audio_dir: str, video_dir: str,
                     sr: int = 16000, frame_length: int = 1024, hop_length: int = 512,
                     peak_quantile: float = 0.70, window_s: float = 0.35):
    if librosa is None:
        return {"ok": False, "reason": "librosa_not_installed"}
    t, y, err = load_waveform_for_vid(vid, audio_dir, video_dir, target_sr=sr)
    if y is None:
        return {"ok": False, "reason": err}
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    thr = np.nanquantile(rms, peak_quantile)
    middle = (rms[1:-1] >= rms[:-2]) & (rms[1:-1] > rms[2:]) & (rms[1:-1] >= thr)
    peak_idx = np.where(middle)[0] + 1
    peak_times = times[peak_idx]
    if peak_times.size == 0:
        return {"ok": False, "reason": "no_peaks_found"}
    mids = pd.to_numeric(df_video["word_mid"], errors="coerce").astype(float).values
    errors, chosen = [], []
    half = float(window_s)
    for m in mids:
        mask = (peak_times >= (m - half)) & (peak_times <= (m + half))
        cand = peak_times[mask]
        p = float(cand[np.argmin(np.abs(cand - m))]) if cand.size else float(times[nearest_idx(m, times)])
        errors.append(abs(m - p)); chosen.append(p)
    errors = np.array(errors, dtype=float)
    return {
        "ok": True,
        "n_words": int(len(mids)),
        "n_peaks": int(len(peak_times)),
        "mae": float(np.nanmean(errors)),
        "median": float(np.nanmedian(errors)),
        "p90": float(np.nanpercentile(errors, 90)),
        "window_s": float(window_s),
        "peak_quantile": float(peak_quantile),
        "peak_times": peak_times.astype(float),
        "picked_times": np.array(chosen, dtype=float),
        "rms_times": times.astype(float),
        "rms_vals": rms.astype(float),
    }

# ------------------- Sidebar: data source + load time -------------------
st.sidebar.header("Data source")
mode = st.sidebar.radio("Load aligned data from:", ["Default path", "Upload file"], index=0)

start_load = time.perf_counter()
if mode == "Upload file":
    up = st.sidebar.file_uploader("Upload aligned CSV/XLSX", type=["csv","xlsx","xls"])
    if up is None:
        st.stop()
    df_all = load_aligned(up)
    loaded_path_display = "(uploaded)"
else:
    st.sidebar.text_input("Default data path", DEFAULT_DATA_PATH, key="data_path_echo", disabled=True)
    df_all = load_aligned(DEFAULT_DATA_PATH)
    loaded_path_display = DEFAULT_DATA_PATH
load_time = time.perf_counter() - start_load
st.sidebar.metric("Data load time", f"{load_time:.2f}s")
st.sidebar.caption(f"Using: {loaded_path_display}")
st.sidebar.success(f"Rows loaded: {len(df_all):,}")

# Silence filter
hide_silence = st.sidebar.checkbox("Hide silence tokens (sp/sil)", True)
if hide_silence and "word" in df_all.columns:
    SILENCE = {"sp","sil","[sp]","[sil]","<sp>","<sil>",""}
    df_all = df_all[~df_all["word"].astype(str).str.lower().isin(SILENCE)].reset_index(drop=True)
    st.sidebar.info(f"Rows after silence filter: {len(df_all):,}")

# Media + SUS
st.sidebar.header("Media sources")
video_dir = st.sidebar.text_input("Local video folder (optional)", DEFAULT_VIDEO_DIR)
audio_dir = st.sidebar.text_input("Local audio folder (optional)", DEFAULT_AUDIO_DIR)
use_youtube = st.sidebar.checkbox("Fallback to YouTube embed if local video not found", True)

st.sidebar.header("SUS (Usability)")
sus_url = DEFAULT_SUS_URL or st.sidebar.text_input("SUS Google Form URL", "")
if sus_url:
    st.sidebar.markdown(f"[Open SUS survey]({sus_url})")

# ------------------- Video selection -------------------
video_ids = df_all["video_id"].astype(str).unique().tolist()
if not video_ids:
    st.error("No video_ids found. Check your aligned file or turn off 'Hide silence'.")
    st.stop()

vid = st.selectbox("Choose a video_id", video_ids, index=0)
df = df_all[df_all["video_id"].astype(str) == str(vid)].copy().reset_index(drop=True)
if df.empty:
    st.error("No rows for selected video.")
    st.stop()

facet_map, facet_generic = detect_facet_columns(df)

# Time bounds (reach true last word)
t_min = float(np.nanmin(pd.to_numeric(df["word_start"], errors="coerce")))
t_max_words = float(np.nanmax(pd.to_numeric(df["word_end"],   errors="coerce")))
t_max_slider = float(np.nextafter(t_max_words, np.inf))
cur_default = float(df["word_mid"].iloc[0]) if len(df) else t_min

# ------------------- Layout -------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Playback")
    if "cur_t" not in st.session_state:
        st.session_state.cur_t = cur_default

    st.session_state.cur_t = st.slider(
        "Time (s)",
        min_value=t_min,
        max_value=t_max_slider,
        value=min(st.session_state.cur_t, t_max_words),
        step=0.001,
        format="%.3f",
        key="time_slider",
    )
    cur_t = float(st.session_state.cur_t)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⏮ Start"):
            st.session_state.cur_t = t_min; st.rerun()
    with c2:
        if st.button("⏭ Last word"):
            st.session_state.cur_t = t_max_words; st.rerun()
    with c3:
        if st.button("↦ Next word"):
            mids = df["word_mid"].values
            later = mids[mids > cur_t + 1e-6]
            if len(later): st.session_state.cur_t = float(later.min()); st.rerun()

    idx = nearest_idx(cur_t, df["word_mid"].values)
    row = df.iloc[idx]
    st.metric("Current word", row["word"])

    subs = get_subtitles_for_video(df_all, vid, gap=0.6)
    if not subs.empty:
        j = nearest_idx(cur_t, subs["mid"].values)
        st.markdown("**Subtitle (now)**")
        st.write(subs.iloc[j]["text"])
    else:
        st.caption("No subtitle segments (no words after filtering).")

    # Emotion text
    emo_text, emo_score, emo_cols = pick_emotion_text(row, facet_map, facet_generic)
    color_map = {"Happy":"green","Angry":"red","Sad":"royalblue","Surprised":"orange",
                 "Afraid":"purple","Disgusted":"olive","Contempt":"gray","Neutral":"gray"}
    st.markdown(
        f"<div style='font-size:26px; font-weight:700; color:{color_map.get(emo_text,'black')};'>"
        f"Emotion (now): {emo_text}</div>",
        unsafe_allow_html=True
    )
    st.metric("Emotion score", f"{emo_score:.2f}")

    # Video (local or YouTube)
    st.markdown("**Video**")
    local_path = find_local_video(video_dir, vid) if video_dir else None
    if local_path:
        st.video(local_path)
    elif use_youtube:
        st.video(f"https://www.youtube.com/watch?v={vid}")
        st.caption("Using YouTube embed (set a local folder for offline playback).")
    else:
        st.info("No video found. Provide a local folder or enable YouTube embed in the sidebar.")

    st.download_button(
        "Download this video's rows (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{vid}_aligned.csv",
        mime="text/csv"
    )

with right:
    st.subheader("Audio waveform")
    t_wav, y_wav, wav_err = load_waveform_for_vid(vid, audio_dir, video_dir, target_sr=16000)
    if t_wav is None or y_wav is None:
        if wav_err == "librosa_not_installed":
            st.warning("Waveform requires `librosa` and `soundfile`. Install them to enable this panel.")
        elif wav_err == "no_audio_source":
            st.info("No audio file found and no local video to decode. "
                    "Place <video_id>.wav in an Audio folder or a <video_id>.mp4 in a Video folder.")
        else:
            st.warning(f"Could not decode audio: {wav_err}")
    else:
        tt, yy = downsample_for_plot(t_wav, y_wav, max_points=4000)
        wav_fig = go.Figure()
        wav_fig.add_trace(go.Scatter(x=tt, y=yy, mode="lines", name="waveform"))
        ws, we = float(row["word_start"]), float(row["word_end"])
        wav_fig.add_vline(x=cur_t, line_width=2, line_dash="dash")
        ymin = float(np.nanmin(yy)) if len(yy) else -1.0
        ymax = float(np.nanmax(yy)) if len(yy) else 1.0
        pad = 0.05 * (ymax - ymin + 1e-6)
        wav_fig.add_shape(type="rect", x0=ws, x1=we, y0=ymin - pad, y1=ymax + pad,
                          line=dict(width=0), fillcolor="rgba(200,200,200,0.3)")

        # If MAE has been computed, overlay peak marks
        mae_res = st.session_state.get("mae_res")
        if mae_res and mae_res.get("ok") and "peak_times" in mae_res:
            wav_fig.add_trace(go.Scatter(
                x=mae_res["peak_times"], y=[0.0]*len(mae_res["peak_times"]),
                mode="markers", name="audio peaks", marker=dict(size=6, symbol="x")
            ))

        wav_fig.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10),
                              xaxis_title="time (s)", yaxis_title="amplitude", hovermode="x")
        st.plotly_chart(wav_fig, use_container_width=True)

    st.subheader("Timeline & Features")
    cov_cols_v = get_cols_by_prefix(df, "covarep_f")
    vis_cols_v = get_cols_by_prefix(df, "visual_f")
    default_cov = cov_cols_v[:3] if cov_cols_v else []
    default_vis = vis_cols_v[:3] if vis_cols_v else []
    sel_cov = st.multiselect("COVAREP features to plot", cov_cols_v, default=default_cov)
    sel_vis = st.multiselect("OpenFace (visual) features to plot", vis_cols_v, default=default_vis)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["word_mid"], y=[0]*len(df), mode="markers",
        marker=dict(size=6), name="words",
        hovertext=df["word"], hoverinfo="text+x"
    ))
    for c in sel_cov:
        vals = pd.to_numeric(df[c], errors="coerce").values
        if not np.all(np.isnan(vals)):
            mu, sd = np.nanmean(vals), np.nanstd(vals)
            v = (vals - mu) / (sd + 1e-8) if sd > 0 else vals
            fig.add_trace(go.Scatter(x=df["word_mid"], y=v, mode="lines",
                                     name=c, hovertemplate=f"{c}: %{{y:.2f}} @ %{{x:.2f}}s<extra></extra>"))
    for c in sel_vis:
        vals = pd.to_numeric(df[c], errors="coerce").values
        if not np.all(np.isnan(vals)):
            mu, sd = np.nanmean(vals), np.nanstd(vals)
            v = (vals - mu) / (sd + 1e-8) if sd > 0 else vals
            fig.add_trace(go.Scatter(x=df["word_mid"], y=v, mode="lines",
                                     name=c, hovertemplate=f"{c}: %{{y:.2f}} @ %{{x:.2f}}s<extra></extra>"))
    fig.add_vline(x=cur_t, line_width=2, line_dash="dash",
                  annotation_text=f"{cur_t:.2f}s", annotation_position="top right")
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      xaxis_title="time (s)", yaxis_title="(z-score, per feature)", hovermode="x")
    st.plotly_chart(fig, use_container_width=True)

# ------------------- FACET panel -------------------
st.subheader("Facial Expressions (FACET)")
facet_map_all, facet_generic_all = detect_facet_columns(df)
if facet_map_all:
    emo_names = [e for e in BASIC_EMOS if e in facet_map_all]
    cols = [facet_map_all[k] for k in emo_names]
    try:
        i_cur = nearest_idx(float(st.session_state.cur_t), df["word_mid"].values)
        cur_vals = df.iloc[i_cur:i_cur+1][cols].astype(float).iloc[0].to_dict()
    except Exception:
        cur_vals = {k:0.0 for k in cols}
    bar = go.Figure([go.Bar(x=[e.title() for e in emo_names],
                            y=[cur_vals.get(facet_map_all[e], 0.0) for e in emo_names])])
    bar.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10),
                      yaxis_title="Score / prob.", xaxis_title="Emotion")
    st.plotly_chart(bar, use_container_width=True)

    topk = st.slider("Show top-k emotions over time", 3, min(6, len(cols)), 5)
    means = df[cols].astype(float).mean().sort_values(ascending=False)
    top_cols = list(means.index[:topk])
    emo_fig = go.Figure()
    for c in top_cols:
        emo_fig.add_trace(go.Scatter(x=df["word_mid"], y=df[c].astype(float), mode="lines",
                                     name=c.replace("facet_","").title()))
    emo_fig.add_vline(x=float(st.session_state.cur_t), line_width=2, line_dash="dash")
    emo_fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                          xaxis_title="time (s)", yaxis_title="score", hovermode="x")
    st.plotly_chart(emo_fig, use_container_width=True)
elif any(c.lower().startswith("facet_") for c in df.columns):
    st.caption("Generic FACET columns detected. Named emotion columns will give nicer labels.")
else:
    st.info("No FACET columns found. Re-run alignment with the FACET .csd to produce 'facet_*' columns.")

# ------------------- Transcript window -------------------
st.subheader("Transcript window")
window = st.slider("Rows around cursor", 3, 30, 10)
cur_idx = nearest_idx(float(st.session_state.cur_t), df["word_mid"].values)
lo = max(0, cur_idx - window); hi = min(len(df), cur_idx + window + 1)
cov_cols_show = get_cols_by_prefix(df, "covarep_f")
vis_cols_show = get_cols_by_prefix(df, "visual_f")
cols_to_show = ["video_id","word_start","word_end","word_mid","word"]
if "label" in df.columns: cols_to_show.append("label")
cols_to_show += cov_cols_show[:2] + vis_cols_show[:2]
st.dataframe(df.iloc[lo:hi][cols_to_show], use_container_width=True)

# ------------------- Evaluation (sidebar): MAE -------------------
st.sidebar.header("Evaluation")
st.sidebar.caption("Sync accuracy = MAE between word midpoints and nearest audio RMS peaks.")
if librosa is None:
    st.sidebar.warning("Install `librosa` + `soundfile` to enable MAE.")
else:
    peak_q = st.sidebar.slider("Peak threshold (RMS quantile)", 0.50, 0.95, 0.70, 0.01)
    win_s  = st.sidebar.slider("Search window (± seconds)", 0.10, 0.80, 0.35, 0.05)
    if st.sidebar.button("Compute MAE (current video)"):
        with st.spinner("Computing MAE…"):
            res = compute_sync_mae(
                vid, df, audio_dir=audio_dir, video_dir=video_dir,
                sr=16000, frame_length=1024, hop_length=512,
                peak_quantile=peak_q, window_s=win_s
            )
        st.session_state["mae_res"] = res

    res = st.session_state.get("mae_res")
    if res and res.get("ok"):
        st.sidebar.success(f"MAE: {res['mae']:.03f}s")
        st.sidebar.write(f"Median: {res['median']:.03f}s • P90: {res['p90']:.03f}s")
        st.sidebar.caption(f"Words: {res['n_words']} • Peaks: {res['n_peaks']} • Window ±{res['window_s']:.2f}s • q={res['peak_quantile']:.2f}")
    elif res and not res.get("ok"):
        st.sidebar.warning(f"MAE unavailable: {res.get('reason','unknown')}")
