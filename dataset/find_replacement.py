import ast
import subprocess
from pathlib import Path

import pandas as pd

from config import CACHE

CMD_DIR = CACHE / "cmd_metadata"

CLIPS_CSV = Path(r"D:\IIT Madras\MTP\modality_attribution\clips.csv")

GENRE = "Horror"
WINDOW_SECONDS = 100
MIN_DURATION = 120


def parse_genres(s) -> list[str]:
    if not isinstance(s, str):
        return []

    try:
        v = ast.literal_eval(s)
        return [g.strip() for g in v] if isinstance(v, list) else []
    except (SyntaxError, ValueError):
        return []


def assign_single_genre(genre_list: list[str]) -> str | None:
    intersect = [g for g in genre_list if g in ["Comedy", "Horror"]]
    return intersect[0] if len(intersect) == 1 else None


def check_available(url: str) -> bool:
    command = [
        "yt-dlp",
        "--simulate",
        "--no-playlist",
        url,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return result.returncode == 0


def main():
    current = pd.read_csv(CLIPS_CSV)

    existing_ids = set(
        current["clip_id"].astype(str)
    )

    print(f"Current clips: {len(current)}")
    print(
        current["genre"].value_counts().to_string()
    )

    print(
        f"\nLooking for a new {GENRE} clip "
        f"not already in clips.csv..."
    )

    clips = pd.read_csv(CMD_DIR / "clips.csv")
    movies = pd.read_csv(CMD_DIR / "movie_info.csv")
    durations = pd.read_csv(CMD_DIR / "durations.csv")

    movies = movies.assign(genre_list=movies["genre"].apply(parse_genres))

    df = clips.merge(movies[["imdbid", "genre_list"]], on="imdbid", how="inner")

    df = df.merge(durations,on="videoid",how="inner")

    df = df[ df["duration"] >= MIN_DURATION]

    df = df.assign(genre=df["genre_list"].apply(assign_single_genre))

    df = df[ df["genre"] == GENRE ]

    df = df[~df["videoid"].astype(str).isin(existing_ids)]

    # Randomize candidates
    df = df.sample(frac=1,random_state=42 )

    print(
        f"New {GENRE} candidates to check: "
        f"{len(df)}"
    )


    for i, (_, row) in enumerate(df.iterrows(), 1):

        clip_id = str(row["videoid"])
        url = ( f"https://www.youtube.com/watch?v={clip_id}")

        print(
            f"[{i}/{len(df)}] checking "
            f"{clip_id} ... ",
            end="",
            flush=True,
        )

        if not check_available(url):
            print("UNAVAILABLE")
            continue

        print("AVAILABLE")

        duration = float(row["duration"])

        start_seconds = max(
            0,
            int(
                (duration - WINDOW_SECONDS) // 2
            ),
        )

        replacement = {
            "clip_id": clip_id,
            "url": url,
            "start_seconds": start_seconds,
            "genre": GENRE,
        }

        print("\n===================================")
        print("FOUND REPLACEMENT")
        print("===================================")

        for key, value in replacement.items():
            print(f"{key}: {value}")

    

        failed_id = "01qhgR0WsnA"

        current = current[current["clip_id"].astype(str) != failed_id]

        current = pd.concat(
            [
                current,
                pd.DataFrame([replacement]),
            ],
            ignore_index=True,
        )

        current = current.sort_values(
            ["genre", "clip_id"]
        ).reset_index(drop=True)

        current.to_csv(
            CLIPS_CSV,
            index=False,
        )

        print(
            f"\nReplaced {failed_id} "
            f"with {clip_id}"
        )

        print("\nFinal counts:")
        print(
            current["genre"].value_counts().to_string()
        )

        print(
            f"\nUpdated: {CLIPS_CSV}"
        )

        return

    print(
        "\nCould not find an available "
        "replacement."
    )


if __name__ == "__main__":
    main()