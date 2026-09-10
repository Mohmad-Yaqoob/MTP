"""
tribe_ablation.py — 7-condition modality ablation over a folder of clips,
with Glasser (HCP-MMP1) parcellation of the outputs.

For each clip it builds the events dataframe ONCE using TRIBE's own
preprocessing pipeline, then runs model.predict() on seven filtered views
of that dataframe:

    VAT  video + audio + text
    VA   video + audio
    VT   video + text
    AT   audio + text
    V    video only
    A    audio only
    T    text only

Modalities are dropped by removing their rows from the events dataframe.
The model then substitutes zeros for any modality missing from the batch.

Usage

Dry run on 2 clips:
    python tribe_ablation.py --limit 2
Full sweep, parcels only:
    python tribe_ablation.py --no-vertices --dtype float16
Outputs
    <clip>/events.csv
    <clip>/parcels_VAT.npy
    <clip>/parcels_VA.npy
    <clip>/parcels_VT.npy
    <clip>/parcels_AT.npy
    <clip>/parcels_V.npy
    <clip>/parcels_A.npy
    <clip>/parcels_T.npy

    <clip>/preds_<condition>.npy
        Full 20,484 vertex predictions unless --no-vertices.

    parcel_labels.json
        360 Glasser parcel names.

    _manifest.csv
        One row per clip x condition.

    _failures.csv
        Clips that failed.

Re-running skips clips whose parcel outputs already exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CLIPS_DIR = Path(r"D:\IIT Madras\MTP\modality_attribution\videos")

VIDEO_EXTS = (".mp4",)

N_CORTICAL_VERTICES = 20484  # fsaverage5, both hemispheres

CONDITIONS: dict[str, set[str]] = {
    "VAT": {"video", "audio", "text"},
    "VA":  {"video", "audio"},
    "VT":  {"video", "text"},
    "AT":  {"audio", "text"},
    "V":   {"video"},
    "A":   {"audio"},
    "T":   {"text"},
}

# Glasser parcellation

_PARCELS: tuple[list[str], list[np.ndarray]] | None = None

def get_parcels(mesh: str = "fsaverage5") -> tuple[list[str], list[np.ndarray]]:
    global _PARCELS
    if _PARCELS is not None:
        return _PARCELS
    from tribev2.utils import get_hcp_labels

    labels: list[str] = []
    indices: list[np.ndarray] = []

    for hemi, suffix in (("left", "-lh"), ("right", "-rh")):
        hemi_labels = get_hcp_labels(mesh=mesh, combine=False, hemi=hemi)
        for name, verts in hemi_labels.items():
            label = f"{name}{suffix}"
            if label.startswith("?") or len(verts) == 0:
                continue

            labels.append(label)
            indices.append(np.asarray(verts, dtype=np.int64))

    _PARCELS = (labels, indices)

    return _PARCELS

def parcellate(preds: np.ndarray, mesh: str = "fsaverage5") -> np.ndarray:
    labels, indices = get_parcels(mesh)
    if preds.shape[1] < N_CORTICAL_VERTICES:
        raise ValueError(
            f"expected at least {N_CORTICAL_VERTICES} cortical vertices, "
            f"got {preds.shape[1]}"
        )
    cortex = preds[:, :N_CORTICAL_VERTICES]

    out = np.empty((cortex.shape[0], len(labels)), dtype=np.float32)
    for i, idx in enumerate(indices):
        out[:, i] = cortex[:, idx].mean(axis=1)

    return out

# Event types

def classify_types(events: pd.DataFrame) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}

    for t in events["type"].dropna().unique():
        if t == "Video":
            mapping[t] = "video"
        elif t == "Audio":
            mapping[t] = "audio"
        elif t == "Word":
            mapping[t] = "text"
        else:
            mapping[t] = None

    return mapping

def filter_events(df: pd.DataFrame, mapping: dict[str, str | None],keep: set[str],) -> pd.DataFrame:
    wanted = df["type"].map(mapping).isin(keep)
    return df[wanted].copy()

# Per-clip processing

def clip_outputs_complete(out_dir: Path, conditions: list[str],) -> bool:
    return all((out_dir / f"parcels_{condition}.npy").is_file() for condition in conditions)

def build_events(model, video_path: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    events_csv = out_dir / "events.csv"
    if events_csv.is_file():
        df = pd.read_csv(events_csv)
        print(f"events: reused {events_csv} ({len(df)} rows)")
        return df

    audio_path = video_path.parent.parent / "audio" / f"{video_path.stem}.wav"
    transcript_path = video_path.parent.parent / "transcripts" / f"{video_path.stem}.json"

    if not audio_path.is_file():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    if not transcript_path.is_file():
        raise FileNotFoundError(f"transcript not found: {transcript_path}")

    tsv_path = audio_path.with_suffix(".tsv")
    if not tsv_path.is_file():
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        rows = []
        for sequence_id, segment in enumerate(data["segments"]):
            sentence = segment.get("text", "").replace('"', "")
            for word in segment.get("words", []):
                if "start" not in word or "end" not in word:
                    continue
                rows.append({
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": sequence_id,
                    "sentence": sentence,
                })

        pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False,)
        print(f"  transcript: created {tsv_path.name}")

    video_events = model.get_events_dataframe(video_path=str(video_path))

    video_events = video_events[video_events["type"] == "Video"].copy()

    audio_events = model.get_events_dataframe(audio_path=str(audio_path))

    audio_events = audio_events[audio_events["type"].isin(["Audio", "Word"])].copy()
    
    clip_end = (video_events["start"] + video_events["duration"]).max()
    before = len(audio_events)
    audio_events = audio_events[audio_events["start"] < clip_end].copy()
    print(f"  trimmed {before - len(audio_events)} events past {clip_end:.2f}s")

    df = pd.concat([video_events, audio_events], ignore_index=True)
    df = df.sort_values(["start", "type"]).reset_index(drop=True)
    df.to_csv(events_csv, index=False)
    print(f"  events: built {len(df)} rows")
    return df

def resolve_mapping(df: pd.DataFrame, type_mapping: dict[str, str | None] | None,) -> dict[str, str | None]:
    if type_mapping is None:
        type_mapping = classify_types(df)
        return type_mapping

    for t in df["type"].dropna().unique():
        if t not in type_mapping:
            type_mapping.update(classify_types(df[df["type"] == t]))
            print(f"  note: new event type "f"'{t}' -> {type_mapping[t]}")
    return type_mapping

def process_clip( model, video_path: Path, out_dir: Path, conditions: list[str], dtype: str, type_mapping: dict[str, str | None] | None, verbose_first: bool, save_vertices: bool,) -> tuple[list[dict], dict[str, str | None]]:
    df = build_events(model,video_path,out_dir,)
    type_mapping = resolve_mapping(df,type_mapping,)

    rows: list[dict] = []
    shapes: dict[str, tuple] = {}
    spans: dict[str, tuple] = {}
    parcels: dict[str, np.ndarray] = {}

    for condition in conditions:
        sub = filter_events(df, type_mapping, CONDITIONS[condition],)
        if len(sub) == 0:
            print(f"  {condition:<4} "
                f"skipped — no rows survive the filter" )
            rows.append(
                {
                    "clip": video_path.stem,
                    "condition": condition,
                    "n_events": 0,
                    "n_segments": 0,
                    "n_vertices": 0,
                    "n_parcels": 0,
                    "status": "empty",
                }
            )
            continue
        t0 = time.time()
        preds, segments = model.predict(events=sub, verbose=verbose_first)
        preds = np.asarray(preds)
        par = parcellate(preds).astype(dtype)
        np.save(out_dir / f"parcels_{condition}.npy", par)
        # ts = np.array([float(getattr(s, "start", i)) for i, s in enumerate(segments)], dtype=np.float32)
        # np.save(out_dir / f"times_{condition}.npy", ts)
        ts = np.array([[float(getattr(s, "start", i)),float(getattr(s, "stop", getattr(s, "end", i + 1)))] for i, s in enumerate(segments)], dtype=np.float32)
        np.save(out_dir / f"times_{condition}.npy", ts)

        parcels[condition] = par
        if save_vertices:
            np.save(out_dir / f"preds_{condition}.npy", preds.astype(dtype))
        shapes[condition] = preds.shape
        # spans[condition] = (float(ts[0]), float(ts[-1]), len(ts))
        spans[condition] = (float(ts[0, 0]), float(ts[-1, 1]), len(ts))

        print(
            f"  {condition:<4} "
            f"events={len(sub):<6} "
            f"preds={preds.shape} "
            f"parcels={par.shape} "
            f"({time.time() - t0:.1f}s)"
        )

        rows.append({
            "clip": video_path.stem,
            "condition": condition,
            "n_events": len(sub),
            "n_segments": preds.shape[0],
            "n_vertices": preds.shape[1],
            "n_parcels": par.shape[1],
            "t_start": spans[condition][0],
            "t_end": spans[condition][1],
            "status": "ok",
        })
    if len(set(spans.values())) > 1:
        print(f"  !! conditions span different time windows: {spans}")
        print("     arrays are not positionally comparable — align on times_*.npy before subtracting")

    if len(set(shapes.values())) > 1:
        print(
            f"  !! segment counts differ across conditions: "
            f"{shapes}"
        )
        print(
            "     conditions are not row-comparable — "
            "check that remove_empty_segments is False"
        )

    if "VAT" in parcels and "V" in parcels:
        a = parcels["VAT"]
        b = parcels["V"]
        if (a.shape == b.shape and np.allclose(a, b)):
            print(
                "  !! VAT and V predictions are IDENTICAL — "
                "the filtering is not actually dropping modalities. "
                "Stop and check the type mapping above."
            )

        elif a.shape == b.shape:
            d = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
            print(
                f"  sanity: mean |VAT - V| "
                f"over parcels = {d:.5f} "
                f"(non-zero, good)"
            )
    return rows, type_mapping

# Main

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the 7-condition modality ablation "
            "over a folder of clips."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    ap.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR,)
    ap.add_argument("--out",type=Path,default=(Path(__file__).resolve().parent/ "outputs"/ "ablation"),)
    ap.add_argument("--cache",type=Path,default=(Path(__file__).resolve().parent/ "cache"), )
    ap.add_argument("--conditions",type=str,default=",".join(CONDITIONS), help=( "comma-separated subset of "+ ",".join(CONDITIONS)),)
    ap.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help=(
            "float16 halves disk use "
            "at negligible cost here"
        ),
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "process only the first N clips "
            "(use 1 for a dry run)"
        ),
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "redo clips whose outputs "
            "already exist"
        ),
    )
    ap.add_argument(
        "--no-vertices",
        action="store_true",
        help=(
            "save only the 360-parcel arrays, "
            "not the full 20484-vertex arrays"
        ),
    )
    args = ap.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in CONDITIONS]

    if bad:
        sys.exit(
            f"unknown conditions: {bad}; "
            f"valid: {list(CONDITIONS)}"
        )
    if not args.clips_dir.is_dir():
        sys.exit(
            f"clips dir does not exist: "
            f"{args.clips_dir}"
        )
    clips = sorted(f for f in args.clips_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS)

    if not clips:
        sys.exit(
            f"no videos ({'/'.join(VIDEO_EXTS)}) "
            f"in {args.clips_dir}"
        )
    if args.limit:
        clips = clips[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"[ablation] clips      : "
        f"{len(clips)} from {args.clips_dir}"
    )
    print(
        f"[ablation] conditions : "
        f"{conditions}"
    )
    print(
        f"[ablation] out        : "
        f"{args.out}"
    )
    labels, _ = get_parcels()

    (args.out / "parcel_labels.json").write_text(
        json.dumps(labels, indent=1)
    )

    print(
        f"[ablation] parcels: {len(labels)} "
        f"(labels -> {args.out / 'parcel_labels.json'})"
    )
    from tribev2.demo_utils import TribeModel
    print(
        "[ablation] loading "
        "facebook/tribev2 "
        "(first run downloads model files) ..."
    )

    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(args.cache),
    )
    model.remove_empty_segments = False
    print(
        "[ablation] "
        "remove_empty_segments = False"
    )

    manifest: list[dict] = []

    failures: list[dict] = []

    type_mapping: dict[str, str | None] | None = None

    for i, video_path in enumerate(clips, 1):
        out_dir = args.out / video_path.stem
        print(
            f"\n[{i}/{len(clips)}] "
            f"{video_path.name}"
        )
        if not args.overwrite and clip_outputs_complete(out_dir, conditions):
            print( "  already complete, skipping" )
            continue
        try:
            rows, type_mapping = process_clip(
                model=model,
                video_path=video_path,
                out_dir=out_dir,
                conditions=conditions,
                dtype=args.dtype,
                type_mapping=type_mapping,
                verbose_first=(i == 1),
                save_vertices=not args.no_vertices,
            )
            manifest.extend(rows)
        except Exception as exc:
            print( f"  FAILED: {exc}")
            traceback.print_exc()

            failures.append({
                    "clip": video_path.stem,
                    "error": repr(exc),
                }
            )

        if manifest:
            pd.DataFrame(manifest).to_csv(args.out / "_manifest.csv", index=False)

        if failures:
            pd.DataFrame(failures).to_csv(args.out / "_failures.csv", index=False)

    print(
        f"\n[ablation] done. "
        f"{len(manifest)} clip-conditions written, "
        f"{len(failures)} clips failed."
    )
    if failures:
        print(
            f"[ablation] see "
            f"{args.out / '_failures.csv'}"
        )

if __name__ == "__main__":
    main()