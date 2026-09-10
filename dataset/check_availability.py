import ast
import subprocess
from pathlib import Path

import pandas as pd

from config import CACHE

CMD_DIR = CACHE / "cmd_metadata"

OUTPUT_CSV = Path(r"D:\IIT Madras\MTP\modality_attribution\clips.csv")

GENRES = ["Comedy", "Horror"]
N_PER_GENRE = 100
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
    intersect = [g for g in genre_list if g in GENRES]
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

    clips = pd.read_csv(CMD_DIR / "clips.csv")
    movies = pd.read_csv(CMD_DIR / "movie_info.csv")
    durations = pd.read_csv(CMD_DIR / "durations.csv")

    print(
        f"[check] CMD raw: "
        f"{len(clips)} clips, "
        f"{len(movies)} movies, "
        f"{len(durations)} durations"
    )

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

    # Keep scenes at least 120 seconds
    df = df[df["duration"] >= MIN_DURATION]

    # Keep only unambiguous Comedy/Horror movies
    df = df.assign(
        genre=df["genre_list"].apply(assign_single_genre)
    )

    df = df[df["genre"].notna()]

    print("\n[check] Candidate pool:")
    print(df["genre"].value_counts().to_string())

    available = []

    for genre in GENRES:

        genre_df = df[df["genre"] == genre].copy()

        # Randomize candidates so we don't always test the same ordering
        genre_df = genre_df.sample(
            frac=1,
            random_state=42,
        )

        count = 0

        print(f"\n[check] Checking {genre}...")

        for _, row in genre_df.iterrows():

            clip_id = str(row["videoid"])
            url = f"https://www.youtube.com/watch?v={clip_id}"

            print(
                f"[{genre}] checking "
                f"{count + 1}/{len(genre_df)}: "
                f"{clip_id}",
                end=" ... ",
                flush=True,
            )

            if check_available(url):

                print("AVAILABLE")

                duration = float(row["duration"])

                start_seconds = max(
                    0,
                    int((duration - WINDOW_SECONDS) // 2),
                )

                available.append(
                    {
                        "clip_id": clip_id,
                        "url": url,
                        "start_seconds": start_seconds,
                        "genre": genre,
                    }
                )

                count += 1

                if count == N_PER_GENRE:
                    print(
                        f"[check] Got {N_PER_GENRE} "
                        f"available {genre} clips."
                    )
                    break

            else:
                print("UNAVAILABLE")

        if count < N_PER_GENRE:

            raise SystemExit(
                f"\n[ERROR] Only found {count} "
                f"available {genre} clips. "
                f"Need {N_PER_GENRE}."
            )

    result = pd.DataFrame(available)

    result = result.sort_values(
        ["genre", "clip_id"]
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n===================================")
    print("Availability check complete")
    print("===================================")

    print(
        result["genre"].value_counts().to_string()
    )

    print(
        f"\nWrote: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()