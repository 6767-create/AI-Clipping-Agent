
import csv
import re
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel

INPUT = Path("input")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

HYPE_WORDS = {
    "wow": 1.4, "no way": 2.0, "insane": 1.8,
    "crazy": 1.3, "lets go": 1.8, "clutch": 1.5,
    "incredible": 1.5, "amazing": 1.1
}

def run(cmd):
    subprocess.run(cmd, check=True)

def duration_seconds(path):
    result = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ], text=True)
    return float(result.strip())

def transcribe(path):
    model = WhisperModel(
        "base", device="cpu", compute_type="int8"
    )
    segments, _ = model.transcribe(
        str(path), vad_filter=True
    )
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments if s.text.strip()
    ]

def find_highlights(segments, duration):
    candidates = []

    for start in range(0, int(duration), 10):
        end = min(start + 35, duration)
        text = " ".join(
            s["text"] for s in segments
            if s["end"] >= start and s["start"] <= end
        )
        low = text.lower()
        score = sum(
            weight for word, weight in HYPE_WORDS.items()
            if word in low
        )

        if text.strip():
            candidates.append((score, start, end, text))

    candidates.sort(reverse=True)
    chosen = []

    for item in candidates:
        if all(abs(item[1] - x[1]) >= 23 for x in chosen):
            chosen.append(item)
        if len(chosen) == 8:
            break

    return sorted(chosen, key=lambda x: x[1])

def make_clip(source, start, end, name):
    output = OUTPUT / f"{name}.mp4"

    run([
        "ffmpeg", "-y", "-ss", str(start),
        "-i", str(source), "-t", str(end-start),
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", str(output)
    ])
    return output.name

def process_all():
    videos = [
        p for p in INPUT.iterdir()
        if p.suffix.lower() in
        [".mp4", ".mov", ".mkv", ".webm"]
    ]

    if not videos:
        print("No authorized videos found in input/")
        return

    queue = []

    for video in videos:
        print("Analyzing:", video.name)
        segments = transcribe(video)
        duration = duration_seconds(video)

        moments = find_highlights(segments, duration)

        for i, (score, start, end, text) in enumerate(moments, 1):
            name = f"{video.stem}_clip_{i:02d}"
            filename = make_clip(video, start, end, name)

            queue.append({
                "filename": filename,
                "source": video.name,
                "start": start,
                "end": end,
                "score": score,
                "title": text[:85],
                "approved": "no"
            })

    with open(OUTPUT / "review_queue.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=queue[0].keys())
        writer.writeheader()
        writer.writerows(queue)

    print("Finished! Review your clips before publishing.")

if __name__ == "__main__":
    process_all()
