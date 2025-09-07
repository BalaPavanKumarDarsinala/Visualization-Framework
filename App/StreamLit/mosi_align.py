# -*- coding: utf-8 -*-
"""
CMU-MOSI aligner (Words + COVAREP + FACET + Labels)
- Robust reading of MOSI .csd via MMSDK (handles object/vlen in HDF5)
- Drops silence tokens (sp/sil/etc.)
- Light cleaning on COVAREP & FACET (smoothing + outlier clipping)
- Aligns by word midpoint -> nearest audio/visual frame
- Exports combined CSV + Excel; writes a skip report for any video with issues
"""

from mmsdk import mmdataset
import pandas as pd
import numpy as np
import os, time, re
from math import ceil

# =================== USER PATHS (.csd) ===================
COVAREP_PATH = r"C:\Users\balap\Desktop\Visualization_Main_Project\multimodal-viz\Data\CMU-MOSI\CMU_MOSI_COVAREP.csd"
WORDS_PATH   = r"C:\Users\balap\Desktop\Visualization_Main_Project\multimodal-viz\Data\CMU-MOSI\CMU_MOSI_TimestampedWords.csd"
LABELS_PATH  = r"C:\Users\balap\Desktop\Visualization_Main_Project\multimodal-viz\Data\CMU-MOSI\CMU_MOSI_Opinion_Labels.csd"
# FACET 4.2 (visual expressions)
VISUAL_PATH  = r"C:\Users\balap\Desktop\Visualization_Main_Project\multimodal-viz\Data\CMU-MOSI\CMU_MOSI_Visual_Facet_42.csd"

# =================== OUTPUTS ===================
OUT_DIR        = r"C:\Users\balap\Desktop\Visualization_Main_Project\multimodal-viz\StreamLit"
os.makedirs(OUT_DIR, exist_ok=True)
COMBINED_CSV   = os.path.join(OUT_DIR, "MOSI_aligned_combined.csv")
COMBINED_XLSX  = os.path.join(OUT_DIR, "MOSI_aligned_combined.xlsx")
SKIP_REPORT    = os.path.join(OUT_DIR, "MOSI_skip_report.csv")

# =================== KNOBS =====================
KEEP_AUDIO_DIMS   = 5      # first N COVAREP dims to export
KEEP_VISUAL_DIMS  = 16     # first N FACET dims to export
INCLUDE_VISUALS   = True   # export FACET
SAMPLE_LIMIT      = None   # e.g., 10 for quick test; None = all videos
BATCH_SIZE        = 9999   # all in one go
PROGRESS_EVERY    = 25

# ======== Cleaning / Preprocessing toggles ========
DROP_SILENCE      = True
SILENCE_TOKENS    = {"sp","sil","[sp]","[sil]","<sp>","<sil>",""}  # lowercased

# COVAREP hygiene (audio)
Z_SCORE_COVAREP   = False      # keep False; the app z-scores for plotting
FILLNAN_AUDIO     = True       # keep small NaN gaps from breaking plots
FILLNAN_LIMIT     = 2
SMOOTH_COVAREP    = 5          # light smoothing (3–5 is good)
CLIP_COVAREP      = True       # clip outliers for prettier axes
CLIP_PCT          = (1, 99)    # clip to 1st–99th percentile band

# FACET hygiene (visual)
SMOOTH_FACET      = 5          # light smoothing (3–5)
CLIP_FACET        = True       # clip outliers
FACET_CLIP_PCT    = (1, 99)
# ===================================================

# Global cache for discovered FACET names
_FACET_NAMES = None

# =================== Helpers =====================
def _excel_engine():
    try:
        import xlsxwriter; return "xlsxwriter"
    except Exception:
        try:
            import openpyxl; return "openpyxl"
        except Exception:
            return None

def safe_name(s: str) -> str:
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_").strip()
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s.lower().strip("_")

def decode_word(x):
    if isinstance(x, bytes):
        try: return x.decode("utf-8", errors="ignore")
        except Exception: return str(x)
    s = str(x)
    if s.startswith("[b'") and s.endswith("']"): return s[3:-2]
    if s.startswith("b'") and s.endswith("'"):   return s[2:-1]
    return s

def extract_token(x):
    if isinstance(x, (list, tuple, np.ndarray)) and len(x)>0:
        return extract_token(x[0])
    return decode_word(x)

def nearest_index(mid, intervals):
    mids = (intervals[:,0] + intervals[:,1]) / 2.0
    return int(np.argmin(np.abs(mids - mid)))

def nanfill_forwardback(x, limit=2):
    """Forward/back fill short NaN runs (pandas ffill/bfill; no deprecation)."""
    s = pd.Series(x, dtype="float32")
    try:
        s = s.ffill(limit=limit).bfill(limit=limit)
    except TypeError:
        s = s.fillna(method="ffill", limit=limit).fillna(method="bfill", limit=limit)
    return s.to_numpy(dtype="float32")

def smooth_1d(x, w):
    if not w or w <= 1 or len(x) < 2:
        return x
    k = int(min(w, len(x)))
    return np.convolve(x, np.ones(k, dtype="float32")/k, mode="same")

def clip_1d(x, p_lo, p_hi):
    if np.all(np.isnan(x)):
        return x
    lo = float(np.nanpercentile(x, p_lo))
    hi = float(np.nanpercentile(x, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return x
    return np.clip(x, lo, hi)

def preprocess_covarep(feat):
    arr = np.array(feat, dtype=np.float32, copy=True)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    T, D = arr.shape
    for d in range(D):
        col = arr[:, d]
        if FILLNAN_AUDIO:
            col = nanfill_forwardback(col, FILLNAN_LIMIT)
        if CLIP_COVAREP:
            col = clip_1d(col, *CLIP_PCT)
        if SMOOTH_COVAREP and SMOOTH_COVAREP > 1:
            col = smooth_1d(col, SMOOTH_COVAREP)
        arr[:, d] = col
    if Z_SCORE_COVAREP:
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        arr = (arr - mu) / (sd + 1e-8)
    return arr

def preprocess_facet(feat):
    arr = np.array(feat, dtype=np.float32, copy=True)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    T, D = arr.shape
    for d in range(D):
        col = arr[:, d]
        if CLIP_FACET:
            col = clip_1d(col, *FACET_CLIP_PCT)
        if SMOOTH_FACET and SMOOTH_FACET > 1:
            col = smooth_1d(col, SMOOTH_FACET)
        arr[:, d] = col
    return arr

# =================== Safe HDF5 readers =====================
def h5_read_tokens(dset):
    try:
        arr = dset.asstr()[...]
        if arr.ndim == 2 and arr.shape[1] >= 1:
            return [str(arr[i,0]) for i in range(arr.shape[0])]
        elif arr.ndim == 1:
            return [str(v) for v in arr.tolist()]
    except Exception:
        pass
    toks=[]; shape = getattr(dset,"shape",())
    N = shape[0] if shape else 0; ndim = len(shape) if shape else 0
    for i in range(N):
        try:
            val = dset[i,0] if (ndim==2 and shape[1]>=1) else dset[i]
            toks.append(extract_token(val))
        except Exception:
            toks.append("")
    return toks

def h5_read_intervals(dset):
    try:
        raw = dset[...]
        arr = np.asarray(raw, dtype=object)
        if arr.ndim==2 and arr.shape[1]==2:
            try: return arr.astype(np.float32)
            except Exception:
                out = np.empty(arr.shape, dtype=np.float32)
                for i in range(arr.shape[0]):
                    out[i,0] = float(arr[i,0]); out[i,1] = float(arr[i,1])
                return out
    except Exception:
        pass
    shape = getattr(dset,"shape",()); N = shape[0] if shape else 0
    out = np.empty((N,2), dtype=np.float32)
    for i in range(N):
        r = np.array(dset[i], dtype=object).ravel()
        out[i,0] = float(r[0]); out[i,1] = float(r[1])
    return out

def h5_read_numeric(dset):
    try:
        arr = dset[...]
        a = np.asarray(arr)
        if a.ndim==2 and a.dtype != object:
            return a.astype(np.float32, copy=False)
        if a.ndim==2 and a.dtype == object:
            try: return a.astype(np.float32)
            except Exception: pass
    except Exception:
        pass
    T,D = dset.shape
    out = np.empty((T,D), dtype=np.float32)
    for t in range(T):
        row = np.array(dset[t], dtype=object).ravel()
        for d in range(D): out[t,d] = float(row[d])
    return out

# =================== MMSDK helpers =====================
def list_cs_names(ds):
    names=[]
    try: names = list(getattr(ds,"computational_sequences",{}).keys())
    except Exception: pass
    if not names:
        try: names = list(getattr(ds,"data",{}).keys())
        except Exception: pass
    return names

def resolve_alias(ds, preferred):
    names = list_cs_names(ds)
    for p in preferred:
        if p in names: return p
    if len(names)==1: return names[0]
    raise RuntimeError(f"Could not resolve alias. Preferred={preferred}, available={names}")

def cs_get(ds, alias):
    try: return ds[alias]
    except Exception: pass
    csmap = getattr(ds,"computational_sequences",None)
    if isinstance(csmap, dict) and alias in csmap: return csmap[alias]
    data = getattr(ds,"data",None)
    if isinstance(data, dict) and alias in data: return data[alias]
    raise KeyError(f"alias_not_found:{alias}")

def cs_get_item(cs, vid):
    try: return cs[vid]
    except Exception: pass
    data = getattr(cs,"data",None)
    if isinstance(data, dict) and vid in data: return data[vid]
    return None

# =================== Per-modality unpack =====================
def get_words_for_vid(ds_words, alias_words, vid):
    import h5py
    cs = cs_get(ds_words, alias_words)
    wobj = cs_get_item(cs, vid)
    if wobj is None: raise KeyError("missing_vid_in_words")
    if isinstance(wobj, h5py.Group):
        win = h5_read_intervals(wobj["intervals"])
        toks = h5_read_tokens(wobj["features"])
        return win, toks
    win_raw = wobj.get("intervals")
    if win_raw is None: raise KeyError("words_intervals_missing")
    win = np.asarray(win_raw, dtype=object)
    if not (win.ndim==2 and win.shape[1]==2):
        out = np.empty((len(win),2), dtype=np.float32)
        for i,r in enumerate(win):
            rr = np.array(r, dtype=object).ravel()
            out[i,0] = float(rr[0]); out[i,1] = float(rr[1])
        win = out
    else:
        win = win.astype(np.float32)
    f = wobj.get("features")
    arr = np.asarray(f, dtype=object) if f is not None else np.empty((0,1), dtype=object)
    if arr.ndim==2 and arr.shape[1]>=1:
        toks = [extract_token(arr[i,0]) for i in range(arr.shape[0])]
    else:
        toks = [extract_token(v) for v in arr]
    return win, toks

def get_numeric_for_vid(ds_mod, alias, vid):
    import h5py
    cs = cs_get(ds_mod, alias)
    obj = cs_get_item(cs, vid)
    if obj is None: raise KeyError(f"missing_vid_in_{alias}")
    if isinstance(obj, h5py.Group):
        iv = h5_read_intervals(obj["intervals"])
        fv = h5_read_numeric(obj["features"])
        return iv, fv
    iv = obj.get("intervals"); fv = obj.get("features")
    if iv is None or fv is None: raise KeyError(f"{alias}_missing_arrays")
    iv = np.asarray(iv, dtype=object)
    if not (iv.ndim==2 and iv.shape[1]==2):
        iv = h5_read_intervals(obj["intervals"])
    else:
        try: iv = iv.astype(np.float32)
        except Exception: iv = h5_read_intervals(obj["intervals"])
    fv = np.asarray(fv)
    if fv.ndim!=2:
        fv = h5_read_numeric(obj["features"])
    elif fv.dtype==object:
        out = np.empty(fv.shape, dtype=np.float32)
        for r in range(fv.shape[0]):
            row = np.array(fv[r], dtype=object).ravel()
            for c in range(fv.shape[1]): out[r,c] = float(row[c])
        fv = out
    else:
        fv = fv.astype(np.float32, copy=False)
    return iv, fv

def get_label_for_vid(ds_labels, alias_labels, vid):
    import h5py
    if alias_labels is None: return None
    cs = cs_get(ds_labels, alias_labels)
    obj = cs_get_item(cs, vid)
    if obj is None: return None
    if isinstance(obj, h5py.Group):
        for k in ("features","value","values","data"):
            if k in obj:
                try:
                    v = np.array(obj[k]).flatten()
                    return float(v[0])
                except Exception:
                    continue
        return None
    v = obj.get("features") or obj.get("value")
    if v is None: return None
    try: return float(np.array(v).flatten()[0])
    except Exception: return None

# =================== FACET name discovery =====================
def _canonical_emotion(name: str) -> str | None:
    n = safe_name(name)
    emo = {
        "joy":"joy","happy":"joy","happiness":"joy","smile":"smile",
        "anger":"anger","angry":"anger",
        "sad":"sadness","sadness":"sadness",
        "surprise":"surprise","surprised":"surprise",
        "fear":"fear","scared":"fear",
        "disgust":"disgust","contempt":"contempt",
        "neutral":"neutral",
        "valence":"valence","engagement":"engagement",
        "brow_raise":"brow_raise","browfurrow":"brow_furrow","brow_furrow":"brow_furrow",
    }
    if re.fullmatch(r"au\d+(_r|_c)?", n):
        return n
    return emo.get(n)

def discover_facet_names(ds_visual, v_alias) -> list[str] | None:
    import h5py
    try:
        cs = cs_get(ds_visual, v_alias)
        try: vids = list(cs.keys())
        except Exception: vids = list(getattr(cs,"data",{}).keys())
        if not vids: return None
        obj = cs_get_item(cs, vids[0])
        if not isinstance(obj, h5py.Group): return None
        if "features" not in obj: return None
        fs = obj["features"]
        D = fs.shape[1] if len(fs.shape)==2 else None
        for k in ["column_names","columns","feature_names","features_names","dim_names",
                  "names","Channels","channels","Description","desc"]:
            if k in fs.attrs:
                raw = fs.attrs[k]
                if isinstance(raw, (list, tuple, np.ndarray)):
                    cand = [decode_word(x) for x in raw]
                else:
                    cand = [decode_word(raw)]
                cand = [safe_name(x) for x in cand if str(x).strip()!=""]
                if D is not None and len(cand)==D:
                    return cand
                if D is not None and len(cand)==1 and ("," in cand[0] or ";" in cand[0]):
                    parts = re.split(r"[;,]\s*", cand[0])
                    parts = [safe_name(p) for p in parts if p]
                    if len(parts)==D: return parts
        return None
    except Exception:
        return None

# =================== Align one video =====================
def align_video(vid, ds_audio, a_alias, ds_words, w_alias, ds_labels, l_alias, ds_visual, v_alias):
    global _FACET_NAMES
    w_int, tokens = get_words_for_vid(ds_words, w_alias, vid)
    a_int, a_feat = get_numeric_for_vid(ds_audio, a_alias, vid)
    v_int, v_feat = (None, None)
    if INCLUDE_VISUALS and v_alias is not None:
        try:
            v_int, v_feat = get_numeric_for_vid(ds_visual, v_alias, vid)
        except KeyError:
            v_int, v_feat = (None, None)

    # light preprocessing
    if a_feat is not None and a_feat.ndim == 2: a_feat = preprocess_covarep(a_feat)
    if v_feat is not None and v_feat.ndim == 2: v_feat = preprocess_facet(v_feat)

    # discover facet names once per run
    if INCLUDE_VISUALS and _FACET_NAMES is None and v_feat is not None and v_feat.ndim==2:
        names = discover_facet_names(ds_visual, v_alias)
        if names and len(names) == v_feat.shape[1]:
            mapped = [safe_name(_canonical_emotion(nm) or nm) for nm in names]
            _FACET_NAMES = mapped
            print("Detected FACET channels:", _FACET_NAMES[:12], "..." if len(_FACET_NAMES)>12 else "")

    label_scalar = get_label_for_vid(ds_labels, l_alias, vid)

    if w_int is None or len(w_int)==0: return pd.DataFrame(), "empty_words_intervals"
    if tokens is None or len(tokens)==0: return pd.DataFrame(), "empty_words_tokens"

    n = min(len(w_int), len(tokens))
    w_int   = w_int[:n]
    tokens  = tokens[:n]

    rows=[]
    ka = min(KEEP_AUDIO_DIMS,  a_feat.shape[1]) if (a_feat is not None and a_feat.ndim==2) else 0
    kv = min(KEEP_VISUAL_DIMS, v_feat.shape[1]) if (v_feat is not None and v_feat.ndim==2) else 0

    for i in range(n):
        tok_raw  = tokens[i]
        tok_norm = tok_raw.strip().lower()
        if DROP_SILENCE and tok_norm in SILENCE_TOKENS:
            continue

        mid = float((w_int[i,0] + w_int[i,1]) / 2.0)
        row = {
            "video_id":   vid,
            "word_start": float(w_int[i,0]),
            "word_end":   float(w_int[i,1]),
            "word_mid":   mid,
            "word":       tok_raw,
        }
        # audio
        if ka>0:
            ai = nearest_index(mid, a_int)
            for d in range(ka):
                row[f"covarep_f{d}"] = float(a_feat[ai, d])
        else:
            for d in range(KEEP_AUDIO_DIMS): row[f"covarep_f{d}"] = np.nan

        # FACET visual (named if available)
        if kv>0:
            vi = nearest_index(mid, v_int)
            for d in range(kv):
                if _FACET_NAMES and d < len(_FACET_NAMES):
                    row[f"facet_{_FACET_NAMES[d]}"] = float(v_feat[vi, d])
                else:
                    row[f"facet_f{d}"] = float(v_feat[vi, d])

        if label_scalar is not None:
            row["label"] = label_scalar

        rows.append(row)

    return pd.DataFrame(rows), None

# =================== Main =====================
def main():
    print("Loading MOSI datasets …")
    ds_words  = mmdataset({"words":   WORDS_PATH})
    ds_audio  = mmdataset({"covarep": COVAREP_PATH})
    ds_labels = mmdataset({"labels":  LABELS_PATH})
    ds_visual = mmdataset({"visual":  VISUAL_PATH}) if INCLUDE_VISUALS else mmdataset({})

    # resolve aliases (support common variants)
    w_alias = resolve_alias(ds_words,  ["words","Words"])
    a_alias = resolve_alias(ds_audio,  ["covarep","COVAREP"])
    l_alias = resolve_alias(ds_labels, ["labels","Labels","opinion","Opinion"]) if list_cs_names(ds_labels) else None
    v_alias = resolve_alias(
        ds_visual,
        ["FACET_4.2","FACET 4.2","FACET_4.1","FACET 4.1","Visual_Facet_42","Visual_Facet_41","visual"]
    ) if (INCLUDE_VISUALS and list_cs_names(ds_visual)) else None

    # ids
    try:
        w_cs = cs_get(ds_words, w_alias)
        try: words_ids = set(w_cs.keys())
        except Exception: words_ids = set(getattr(w_cs,"data",{}).keys())
    except Exception as e:
        raise RuntimeError(f"Cannot access words cs '{w_alias}': {e}")

    try:
        a_cs = cs_get(ds_audio, a_alias)
        try: audio_ids = set(a_cs.keys())
        except Exception: audio_ids = set(getattr(a_cs,"data",{}).keys())
    except Exception as e:
        raise RuntimeError(f"Cannot access covarep cs '{a_alias}': {e}")

    common = sorted(list(words_ids & audio_ids))
    if SAMPLE_LIMIT is not None:
        common = common[:int(SAMPLE_LIMIT)]
    print(f"  words vids: {len(words_ids)} | audio vids: {len(audio_ids)} | common: {len(common)}")
    if not common:
        raise SystemExit("No common video ids. Check that all four paths point to MOSI (not MOSEI).")

    # reset outputs
    if os.path.exists(COMBINED_CSV): os.remove(COMBINED_CSV)
    header_written=False; skip_records=[]; t0=time.perf_counter()
    num_batches = ceil(len(common)/BATCH_SIZE)

    for b in range(num_batches):
        s,e = b*BATCH_SIZE, min((b+1)*BATCH_SIZE, len(common))
        batch_ids = common[s:e]
        print(f"\n=== Batch {b+1}/{num_batches}: videos {s+1}-{e} ===")
        for i, vid in enumerate(batch_ids, start=1):
            t_vid = time.perf_counter()
            try:
                df, reason = align_video(vid, ds_audio, a_alias, ds_words, w_alias, ds_labels, l_alias, ds_visual, v_alias)
                if df.empty:
                    skip_records.append({"video_id": vid, "reason": reason or "no_rows"})
                    print(f"[WARN] Skipped {vid}: {reason or 'no_rows'}")
                else:
                    if reason: skip_records.append({"video_id": vid, "reason": reason})
                    if not header_written:
                        df.to_csv(COMBINED_CSV, index=False, mode="w", encoding="utf-8"); header_written=True
                    else:
                        df.to_csv(COMBINED_CSV, index=False, mode="a", header=False, encoding="utf-8")
            except KeyError as ke:
                r = str(ke).strip("'")
                skip_records.append({"video_id": vid, "reason": r})
                print(f"[WARN] Skipped {vid}: {r}")
            except Exception as e:
                skip_records.append({"video_id": vid, "reason": f"exception: {e}"})
                print(f"[WARN] Skipped {vid}: exception {e}")

            if (i % PROGRESS_EVERY)==0:
                print(f"  processed {i}/{len(batch_ids)} | last ~{time.perf_counter()-t_vid:0.2f}s")

    if skip_records:
        pd.DataFrame(skip_records).to_csv(SKP:=SKIP_REPORT, index=False, encoding="utf-8")
        print(f"\nSkip report written to: {os.path.abspath(SKP)} (rows={len(skip_records)})")

    if os.path.exists(COMBINED_CSV):
        try:
            eng = _excel_engine()
            if eng is None: raise RuntimeError("Install xlsxwriter or openpyxl")
            print("\nBuilding Excel from combined CSV …")
            df_all = pd.read_csv(COMBINED_CSV)
            with pd.ExcelWriter(COMBINED_XLSX, engine=eng) as writer:
                if len(df_all) <= 1_048_576:
                    df_all.to_excel(writer, sheet_name="data", index=False)
                else:
                    start=0; sheet_no=1
                    while start < len(df_all):
                        end = min(start+1_048_576, len(df_all))
                        df_all.iloc[start:end].to_excel(writer, sheet_name=f"data_{sheet_no}", index=False)
                        sheet_no+=1; start=end
            print(f"Excel saved to: {os.path.abspath(COMBINED_XLSX)} (rows={len(df_all)})")
        except Exception as e:
            print(f"[WARN] Could not write Excel: {e}\nCSV is here: {os.path.abspath(COMBINED_CSV)}")
    else:
        print("\nNo combined CSV was written.")

    mins = (time.perf_counter()-t0)/60.0
    print(f"\nAll done ✅  Total time: {mins:0.1f} min")
    print("Combined CSV:", os.path.abspath(COMBINED_CSV) if os.path.exists(COMBINED_CSV) else "(not created)")
    print("Excel XLSX  :", os.path.abspath(COMBINED_XLSX) if os.path.exists(COMBINED_XLSX) else "(not created)")
    print("Output folder:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
