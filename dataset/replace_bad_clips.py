import ast
import subprocess
from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"D:\IIT Madras\MTP\modality_attribution")

CSV_FILE = BASE_DIR / "clips.csv"
VIDEO_DIR = BASE_DIR / "videos"

CMD_DIR = BASE_DIR / "cache" / "cmd_metadata"

WINDOW_SECONDS = 100
MIN_DURATION = 120
SEED = 42

GENRES = ["Comedy", "Horror"]

BAD_IDS = {
    "Comedy": [
        "6La5YCYlMZY",
        "8O8_FMhW9dY",
        "9zbF578--dE",
        "pWRCxdh4PTM",
    ],
    "Horror": [
        "Irq818Ek0Ro",
        "LJ8UoUA2_uE",
        "RObb2QfBnUs",
        "ZP73cUcxidQ",
    ],
}


def parse_genres(s):
    if not isinstance(s, str):
        return []

    try:
        value = ast.literal_eval(s)

        if isinstance(value, list):
            return [g.strip() for g in value]

    except (SyntaxError, ValueError):
        pass

    return []


def assign_single_genre(genre_list):
    matches = [g for g in genre_list if g in GENRES]

    if len(matches) == 1:
        return matches[0]

    return None


def load_candidates():
    clips = pd.read_csv(CMD_DIR / "clips.csv")
    movies = pd.read_csv(CMD_DIR / "movie_info.csv")
    durations = pd.read_csv(CMD_DIR / "durations.csv")

    movies = movies.assign(
        genre_list=movies["genre"].apply(parse_genres)
    )

    df = clips.merge(
        movies[["imdbid", "genre_list"]],
        on="imdbid",
        how="inner",
    )

    df = df.merge(
        durations,
        on="videoid",
        how="inner",
    )

    df = df[df["duration"] >= MIN_DURATION]

    df = df.assign(
        genre=df["genre_list"].apply(assign_single_genre)
    )

    df = df[df["genre"].notna()]

    return df


def get_actual_duration(video_file):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_file),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def download_candidate(clip_id, url, start_seconds):
    VIDEO_DIR.mkdir(exist_ok=True)

    output_file = VIDEO_DIR / f"{clip_id}.mp4"

    end_seconds = start_seconds + WINDOW_SECONDS

    print()
    print(f"[download] {clip_id}")
    print(f"           {start_seconds}s -> {end_seconds}s")

    command = [
        "yt-dlp",
        "--download-sections",
        f"*{start_seconds}-{end_seconds}",
        "--force-keyframes-at-cuts",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format",
        "mp4",
        "--retries",
        "3",
        "--no-playlist",
        "-o",
        str(output_file),
        url,
    ]

    result = subprocess.run(command)

    if result.returncode != 0 or not output_file.exists():
        print("[download] FAILED")
        return None

    duration = get_actual_duration(output_file)

    print(f"[download] actual duration: {duration:.3f}s")

    return duration


def main():

    df = pd.read_csv(CSV_FILE)

    print(f"Current dataset: {len(df)} clips")

    existing_ids = set(df["clip_id"].astype(str))

    candidates = load_candidates()

    candidates = candidates.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    replacements = []

    for genre in GENRES:

        print()
        print("=" * 60)
        print(f"Finding replacements for {genre}")
        print("=" * 60)

        needed = BAD_IDS[genre]

        genre_candidates = candidates[
            candidates["genre"] == genre
        ]

        found = 0

        for _, candidate in genre_candidates.iterrows():

            if found >= len(needed):
                break

            clip_id = str(candidate["videoid"])

            # Never use an ID already in our dataset.
            if clip_id in existing_ids:
                continue

            url = f"https://www.youtube.com/watch?v={clip_id}"

            cmd_duration = float(candidate["duration"])

            start_seconds = int(
                max(
                    0,
                    (cmd_duration - WINDOW_SECONDS) // 2
                )
            )

            actual_duration = download_candidate(
                clip_id,
                url,
                start_seconds,
            )

            if actual_duration is None:
                continue

            # Accept only approximately 100 seconds.
            if not (99 <= actual_duration <= 101):

                print(
                    f"[reject] {clip_id}: "
                    f"{actual_duration:.3f}s"
                )

                output_file = VIDEO_DIR / f"{clip_id}.mp4"

                if output_file.exists():
                    output_file.unlink()

                continue

            print(f"[ACCEPT] {clip_id}")

            old_id = needed[found]

            replacements.append({
                "old_id": old_id,
                "new_id": clip_id,
                "genre": genre,
                "url": url,
                "start_seconds": start_seconds,
                "duration": actual_duration,
            })

            existing_ids.add(clip_id)

            found += 1

        if found < len(needed):

            print()
            print(
                f"ERROR: only found {found}/"
                f"{len(needed)} replacements for {genre}"
            )

            return

    print()
    print("=" * 60)
    print("REPLACEMENTS FOUND")
    print("=" * 60)

    for r in replacements:
        print(
            f"{r['genre']}: "
            f"{r['old_id']} -> {r['new_id']} "
            f"({r['duration']:.3f}s)"
        )

    # Backup current CSV before modifying it.
    backup_file = BASE_DIR / "clips_before_replacement.csv"
    df.to_csv(backup_file, index=False)

    print()
    print(f"Backup saved to: {backup_file}")

    # Replace rows.
    for r in replacements:

        mask = df["clip_id"].astype(str) == r["old_id"]

        df.loc[mask, "clip_id"] = r["new_id"]
        df.loc[mask, "url"] = r["url"]
        df.loc[mask, "start_seconds"] = r["start_seconds"]

    df.to_csv(CSV_FILE, index=False)

    print()
    print(f"Updated: {CSV_FILE}")

    print()
    print("Final counts:")
    print(df["genre"].value_counts().to_string())


if __name__ == "__main__":
    main()