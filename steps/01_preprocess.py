"""Preprocess PCM history -> per-PCM windows for the CNN."""

import argparse
import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


PCM_ID_LIST = [
    445, 361, 328,  33, 135, 225, 343, 220, 347, 241, 245, 320, 238,  67,
    449, 420, 422, 421, 334, 293,  44, 246,  53,  52, 247,  64,  61,  47,
     71, 331, 211, 321, 329, 221,   3, 121,  63, 327, 210,  62, 342,  48,
    358, 162, 184,  36,  11,  70,   7,   1, 182, 315, 318,  10,  39, 153,
    297,  29, 183, 154,  12, 442, 304, 319, 370, 120, 119, 208, 306, 209,
    176, 242, 314, 387, 142, 226, 410, 353, 305,  17, 214, 326, 215, 173,
    415, 324, 325, 283, 309, 117, 330,  79, 284, 101,   4,  50,  23, 207,
    161, 164, 163,  41,  56, 379, 167,  66, 166,  76,  98, 294,  40, 295,
    204, 281, 137, 280, 322, 279, 317,  46, 116, 446,  65, 425, 431, 102,
    346, 256, 257, 203, 355, 254, 165,  43, 175, 168,  18,  20, 345, 235,
    229, 267, 231, 303,  69, 253, 408,  73,  54, 239, 124,  74,  85, 301,
    187, 450,  21, 181, 217,  38, 216,  57,  77,  34, 228,  72, 333,  15,
     75, 139, 323, 386,  78,  14, 213,  83, 336, 109, 426, 144,  91, 223,
    338, 428, 316, 302, 310, 237,  16, 234, 230, 232, 233, 380, 236, 438,
     35, 212, 354, 155, 112,  42, 441, 414, 339,   6, 424, 151, 444, 388,
     94,  22, 143, 138, 195,   9,  55,   5, 185, 248, 118, 403,  37, 292,
    270,  90, 344, 125, 261, 434, 222, 128, 180, 206,   8,  45, 437, 436,
    369, 335,  99, 407, 100, 406,  26, 448, 110, 126, 243, 300, 443, 196,
     19, 383, 340, 427, 430, 159,  84,  96, 127,  81, 453, 399, 122,  80,
    240, 171,
]

CHANNEL_TO_IDS = {
    "CH1": {"level": 3, "margin": 13, "swr": 27},
    "CH2": {"level": 4, "margin": 14, "swr": 28},
    "CH3": {"level": 5, "margin": 15, "swr": None},
    "CH4": {"level": 6, "margin": 16, "swr": None},
    "CH5": {"level": 7, "margin": 17, "swr": None},
}
TYPE_NAMES = {"level": "Level", "margin": "Margin", "swr": "SWR"}


def load_raw(start, end):
    import pyodbc
    server = os.environ.get("PCM_SQL_SERVER", "localhost")
    db = os.environ.get("PCM_SQL_DB", "PCM_AC")
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};DATABASE={db};Trusted_Connection=yes;"
    )
    pcm_ids_str = ",".join(str(x) for x in PCM_ID_LIST)
    start_str = start.strftime("%Y%m%dT000000")
    end_str = (end + timedelta(days=1)).strftime("%Y%m%dT000000")
    q = f"""
        SELECT h.pk_index, h.pcm_id, h.id, h.value, h.timestamp,
               p.name, p.detail, p.thrsh_alert_lower, p.thrsh_alert_upper
        FROM [PCM_AC].[dbo].[type30_history] h
        INNER JOIN [PCM_AC].[dbo].[type30_points] p
            ON h.id = p.id AND h.pcm_id = p.pcm_id
        WHERE p.enabled = 1
          AND h.pcm_id IN ({pcm_ids_str})
          AND h.timestamp >= ? AND h.timestamp < ?
    """
    print(f"querying SQL ({start.date()} -> {end.date()}, {len(PCM_ID_LIST)} PCMs)")
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql_query(q, conn, params=[start_str, end_str])
    conn.close()
    print(f"loaded {len(df):,} rows")
    return df


def make_wide_df(raw):
    raw = raw.copy()
    raw["datetime"] = pd.to_datetime(
        raw["timestamp"], format="%Y%m%dT%H%M%S", errors="coerce"
    )
    raw = raw.dropna(subset=["datetime"]).sort_values(["pcm_id", "datetime"])

    id2ch, id2type = {}, {}
    for ch, ids in CHANNEL_TO_IDS.items():
        for t, mid in ids.items():
            if mid is not None:
                id2ch[mid] = ch
                id2type[mid] = TYPE_NAMES[t]

    raw["channel"] = raw["id"].map(id2ch)
    raw["measurement_type"] = raw["id"].map(id2type)
    raw = raw[raw["measurement_type"].notna()].copy()
    raw["unit"] = raw["measurement_type"].map({"SWR": "points"}).fillna("dB")
    raw["measurement_id"] = (
        raw["channel"] + "_" + raw["measurement_type"] + "_" + raw["unit"]
    )

    thresh = (
        raw[["pcm_id", "measurement_id", "thrsh_alert_lower", "thrsh_alert_upper"]]
        .drop_duplicates(subset=["pcm_id", "measurement_id"])
        .set_index(["pcm_id", "measurement_id"])
    )

    wide = raw.pivot_table(
        index=["pcm_id", "datetime"], columns="measurement_id",
        values="value", aggfunc="mean",
    ).sort_index()

    # drop columns that are mostly zeros (dead sensors)
    zero_frac = (wide == 0).mean()
    dead = zero_frac[zero_frac > 0.95].index.tolist()
    if dead:
        print(f"dropping dead channels: {dead}")
        wide = wide.drop(columns=dead)
    feat_cols = list(wide.columns)
    print(f"features ({len(feat_cols)}): {feat_cols}")

    # is_failure_event = any feature out of its alert thresholds
    fail = pd.Series(0, index=wide.index, dtype=np.int8)
    for pcm_id in wide.index.get_level_values("pcm_id").unique():
        m = wide.index.get_level_values("pcm_id") == pcm_id
        sub = wide.loc[m]
        f = np.zeros(len(sub), dtype=bool)
        for col in feat_cols:
            key = (int(pcm_id), col)
            if key not in thresh.index:
                continue
            lo = float(thresh.loc[key, "thrsh_alert_lower"])
            hi = float(thresh.loc[key, "thrsh_alert_upper"])
            v = sub[col].values.astype(float)
            f |= np.isfinite(v) & ((v < lo) | (v > hi))
        fail.loc[m] = f.astype(np.int8)
    print(f"raw failure rows: {int(fail.sum()):,}")

    # fill small native gaps before resampling
    parts = []
    for _, g in wide.groupby(level="pcm_id"):
        g = g.interpolate(method="linear", limit=10).ffill().bfill()
        parts.append(g)
    wide = pd.concat(parts).sort_index()

    # 1H resample, mean for features, max for the failure flag
    parts = []
    for pcm_id, g in wide.groupby(level="pcm_id"):
        g = g.droplevel("pcm_id")
        g_feat = g.resample("1h").mean()
        g_fail = fail.loc[pcm_id].resample("1h").max().fillna(0).astype(int)
        idx = pd.MultiIndex.from_tuples(
            [(pcm_id, dt) for dt in g_feat.index], names=["pcm_id", "datetime"]
        )
        g_feat.index = idx
        g_feat["is_failure_event"] = g_fail.values
        parts.append(g_feat)
    out = pd.concat(parts).sort_index()
    print(f"hourly wide_df: {out.shape}, "
          f"{int(out['is_failure_event'].sum()):,} failure hours")
    return out


def make_windows(df_scaled, y_lbl, w, overlap, lead_h):
    stride = max(1, int(w * (1.0 - overlap)))
    Xs, ys, pcms = [], [], []
    for pcm_id, g in df_scaled.groupby(level="pcm_id"):
        n = len(g)
        if n < w:
            print(f"  pcm {pcm_id}: only {n} rows, skip")
            continue
        if pcm_id in y_lbl.index.get_level_values("pcm_id"):
            fs = y_lbl.loc[[pcm_id]].droplevel("pcm_id")
            fail_dts = fs[fs == 1].index
        else:
            fail_dts = pd.DatetimeIndex([])
        nw = (n - w) // stride + 1
        X = np.empty((nw, w, g.shape[1]), dtype=np.float32)
        y = np.zeros(nw, dtype=np.int8)
        dts = g.index.get_level_values("datetime")
        for j in range(nw):
            s = j * stride
            e = s + w
            X[j] = g.iloc[s:e].values.astype(np.float32)
            we = pd.Timestamp(dts[e - 1])
            for fd in fail_dts:
                if we < fd <= we + timedelta(hours=lead_h):
                    y[j] = 1
                    break
        Xs.append(X)
        ys.append(y)
        pcms.extend([int(pcm_id)] * nw)
    if not Xs:
        raise RuntimeError(f"no windows produced for w={w}")
    return (
        np.concatenate(Xs),
        np.concatenate(ys),
        np.array(pcms, dtype=np.int64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window_size", type=int, default=336)
    ap.add_argument("--lead_time", type=int, default=24)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--from_intermediate", default=None)
    ap.add_argument("--save_intermediate", default=None)
    ap.add_argument("--start_date", default="2018-01-01")
    ap.add_argument("--end_date", default=None)
    args = ap.parse_args()

    w = args.window_size
    lead = args.lead_time
    out_dir = args.out_dir or f"preprocessed_pcm_data_{w}h_{lead}h"
    os.makedirs(out_dir, exist_ok=True)
    print(f"w={w}h, lead={lead}h, out={out_dir}")

    # 1. load 1H wide_df from intermediate parquet, or rebuild from SQL
    if args.from_intermediate and os.path.exists(args.from_intermediate):
        print(f"loading intermediate {args.from_intermediate}")
        wide = pd.read_parquet(args.from_intermediate)
        if not isinstance(wide.index, pd.MultiIndex):
            wide = wide.set_index(["pcm_id", "datetime"])
        wide.index.names = ["pcm_id", "datetime"]
        # parquet roundtrip can demote the datetime level dtype, force it back
        dt = pd.to_datetime(wide.index.get_level_values("datetime"))
        wide.index = wide.index.set_levels(dt.unique(), level="datetime")
    else:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = (datetime.strptime(args.end_date, "%Y-%m-%d")
               if args.end_date else datetime.now())
        raw = load_raw(start, end)
        wide = make_wide_df(raw)
        if args.save_intermediate:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_intermediate)),
                        exist_ok=True)
            wide.reset_index().to_parquet(args.save_intermediate, index=False)
            print(f"saved intermediate {args.save_intermediate}")
    print(f"wide_df shape: {wide.shape}")

    fail = wide["is_failure_event"].copy()
    feats = wide.drop(columns=["is_failure_event"])

    # 2. chronological 70/15/15 per PCM
    train_idx, val_idx, test_idx, skipped = [], [], [], []
    for pcm_id, g in feats.groupby(level="pcm_id"):
        g = g.sort_index(level="datetime")
        n = len(g)
        n_test = int(n * 0.15)
        n_val = int(n * 0.15)
        n_train = n - n_val - n_test
        if n_train < w or n_val < w or n_test < w:
            skipped.append(int(pcm_id))
            continue
        idx = g.index
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
    if skipped:
        print(f"skipped {len(skipped)} pcms (too short for w={w})")

    train_idx = pd.MultiIndex.from_tuples(train_idx, names=feats.index.names)
    val_idx = pd.MultiIndex.from_tuples(val_idx, names=feats.index.names)
    test_idx = pd.MultiIndex.from_tuples(test_idx, names=feats.index.names)
    train_df = feats.loc[train_idx]
    val_df = feats.loc[val_idx]
    test_df = feats.loc[test_idx]

    # 3. fill small gaps inside each split
    def fill(df):
        parts = []
        for _, g in df.groupby(level="pcm_id"):
            g = g.interpolate(method="linear", limit=3).ffill().bfill()
            parts.append(g)
        return pd.concat(parts).sort_index()

    train_df = fill(train_df)
    val_df = fill(val_df)
    test_df = fill(test_df)

    # 4. fit a RobustScaler per pcm on train, apply to val/test
    scalers = {}
    parts = []
    for pcm_id, g in train_df.groupby(level="pcm_id"):
        sc = RobustScaler()
        parts.append(pd.DataFrame(
            sc.fit_transform(g), index=g.index, columns=g.columns
        ))
        scalers[int(pcm_id)] = sc
    train_scaled = pd.concat(parts).sort_index()

    def transform(df):
        parts = []
        for pcm_id, g in df.groupby(level="pcm_id"):
            sc = scalers[int(pcm_id)]
            parts.append(pd.DataFrame(
                sc.transform(g), index=g.index, columns=g.columns
            ))
        return pd.concat(parts).sort_index()

    val_scaled = transform(val_df)
    test_scaled = transform(test_df)

    y_train_lbl = fail.reindex(train_scaled.index).fillna(0).astype(int)
    y_val_lbl = fail.reindex(val_scaled.index).fillna(0).astype(int)
    y_test_lbl = fail.reindex(test_scaled.index).fillna(0).astype(int)

    # 5. windowing: 75% overlap on train, none on val/test
    print("windowing train")
    X_tr, y_tr, p_tr = make_windows(train_scaled, y_train_lbl, w, 0.75, lead)
    print("windowing val")
    X_va, y_va, p_va = make_windows(val_scaled, y_val_lbl, w, 0.0, lead)
    print("windowing test")
    X_te, y_te, p_te = make_windows(test_scaled, y_test_lbl, w, 0.0, lead)

    # 6. save
    np.save(os.path.join(out_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(out_dir, "y_train.npy"), y_tr)
    np.save(os.path.join(out_dir, "pcm_train.npy"), p_tr)
    np.save(os.path.join(out_dir, "X_val.npy"), X_va)
    np.save(os.path.join(out_dir, "y_val.npy"), y_va)
    np.save(os.path.join(out_dir, "pcm_val.npy"), p_va)
    np.save(os.path.join(out_dir, "X_test.npy"), X_te)
    np.save(os.path.join(out_dir, "y_test.npy"), y_te)
    np.save(os.path.join(out_dir, "pcm_test.npy"), p_te)
    joblib.dump(scalers, os.path.join(out_dir, "pcm_scalers.pkl"))

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "window_length": w,
            "lead_time_hours": lead,
            "features": list(feats.columns),
            "train": int(len(X_tr)),
            "val": int(len(X_va)),
            "test": int(len(X_te)),
        }, f, indent=2)

    print(f"done: train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")


if __name__ == "__main__":
    main()
