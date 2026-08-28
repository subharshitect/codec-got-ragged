#!/usr/bin/env python3
"""Extract codec/bitstream frame data into clean CSV/JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import statistics
import sys
from pathlib import Path
from fractions import Fraction

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.progress import tqdm


FRAME_FIELDS = [
    "frame_id",
    "source_index",
    "display_order_index",
    "decode_order_index",
    "key_frame",
    "pict_type",
    "pts_time",
    "dts_time",
    "display_time_seconds",
    "decode_time_seconds",
    "frame_image",
    "best_effort_timestamp_time",
    "pkt_duration_time",
    "pkt_pos",
    "pkt_size",
    "coded_picture_number",
    "display_picture_number",
    "mv_count",
    "mv_mean_magnitude",
    "mv_max_magnitude",
    "mv_source_neg_count",
    "mv_source_pos_count",
    "mv_source_unknown_count",
    "mv_zero_count",
    "mv_zero_ratio",
]

PACKET_FIELDS = [
    "packet_index",
    "pts_time",
    "dts_time",
    "duration_time",
    "size",
    "pos",
    "flags",
]

MOTION_VECTOR_FIELDS = [
    "frame_id",
    "source_index",
    "source",
    "w",
    "h",
    "src_x",
    "src_y",
    "dst_x",
    "dst_y",
    "motion_x",
    "motion_y",
    "motion_scale",
    "motion_x_scaled",
    "motion_y_scaled",
    "motion_magnitude",
]


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout or "{}")


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "Command failed without output"
        raise SystemExit(message)


def read_stream(video: Path) -> dict:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(video),
        ]
    )
    return (data.get("streams") or [{}])[0]


def number(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_text(value: object) -> str:
    parsed = number(value)
    if parsed is None:
        return ""
    return str(int(parsed))


def text(value: object) -> str:
    if value in (None, "N/A"):
        return ""
    return str(value)


def parse_rate(value: object) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return number(value)


def fps_label(fps: float) -> str:
    return str(fps).rstrip("0").rstrip(".").replace(".", "p")


def make_analysis_video(video: Path, fps: float, encoded_dir: Path) -> Path:
    if fps <= 0:
        return video
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required for FPS>0 re-encoding but was not found on PATH")

    encoded_dir.mkdir(parents=True, exist_ok=True)
    output = encoded_dir / f"{video.stem}_fps_{fps_label(fps)}.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-r",
        str(fps),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    run_command(command)
    return output


def extract_frame_images(video: Path, frames_dir: Path) -> int:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to decode frame images but was not found on PATH")

    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(list(frames_dir.glob("frame_*.png")), desc="clear frame images", unit="file"):
        path.unlink()

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-start_number",
        "0",
        str(frames_dir / "frame_%06d.png"),
    ]
    run_command(command)
    return len(list(frames_dir.glob("frame_*.png")))


def sort_index(rows: list[dict[str, str]], keys: list[str]) -> dict[str, int]:
    def sort_key(row: dict[str, str]) -> tuple:
        values = []
        for key in keys:
            parsed = number(row.get(key))
            values.append((parsed is None, parsed if parsed is not None else math.inf))
        values.append((False, number(row["source_index"]) or 0))
        return tuple(values)

    return {row["frame_id"]: index for index, row in enumerate(sorted(rows, key=sort_key))}


def motion_vectors_from_frame(frame: dict, frame_id: str, source_index: int) -> list[dict[str, str]]:
    rows = []
    for side_data in frame.get("side_data_list", []) or []:
        vectors = side_data.get("motion_vectors") or []
        for vector in vectors:
            motion_x = number(vector.get("motion_x"))
            motion_y = number(vector.get("motion_y"))
            scale = number(vector.get("motion_scale")) or 1.0
            scaled_x = motion_x / scale if motion_x is not None else None
            scaled_y = motion_y / scale if motion_y is not None else None
            magnitude = None
            if scaled_x is not None and scaled_y is not None:
                magnitude = math.hypot(scaled_x, scaled_y)

            rows.append(
                {
                    "frame_id": frame_id,
                    "source_index": str(source_index),
                    "source": text(vector.get("source")),
                    "w": int_text(vector.get("w")),
                    "h": int_text(vector.get("h")),
                    "src_x": int_text(vector.get("src_x")),
                    "src_y": int_text(vector.get("src_y")),
                    "dst_x": int_text(vector.get("dst_x")),
                    "dst_y": int_text(vector.get("dst_y")),
                    "motion_x": int_text(vector.get("motion_x")),
                    "motion_y": int_text(vector.get("motion_y")),
                    "motion_scale": text(vector.get("motion_scale")),
                    "motion_x_scaled": "" if scaled_x is None else f"{scaled_x:.6f}",
                    "motion_y_scaled": "" if scaled_y is None else f"{scaled_y:.6f}",
                    "motion_magnitude": "" if magnitude is None else f"{magnitude:.6f}",
                }
            )
    return rows


def motion_stats(rows: list[dict[str, str]]) -> dict[str, str]:
    magnitudes = [number(row["motion_magnitude"]) for row in rows]
    magnitudes = [value for value in magnitudes if value is not None]

    neg_count = 0
    pos_count = 0
    unknown_count = 0
    for row in rows:
        source = number(row["source"])
        if source is None:
            unknown_count += 1
        elif source < 0:
            neg_count += 1
        elif source > 0:
            pos_count += 1
        else:
            unknown_count += 1

    zero_count = sum(1 for value in magnitudes if value == 0)
    count = len(rows)

    return {
        "mv_count": str(count),
        "mv_mean_magnitude": f"{sum(magnitudes) / len(magnitudes):.6f}" if magnitudes else "",
        "mv_max_magnitude": f"{max(magnitudes):.6f}" if magnitudes else "",
        "mv_source_neg_count": str(neg_count),
        "mv_source_pos_count": str(pos_count),
        "mv_source_unknown_count": str(unknown_count),
        "mv_zero_count": str(zero_count),
        "mv_zero_ratio": f"{zero_count / count:.6f}" if count else "",
    }


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow({field: row.get(field, "") for field in fields})


def packet_stats_by_frame_type(frames: list[dict[str, str]]) -> dict[str, dict[str, float | int | None]]:
    output = {}
    for pict_type in ["I", "P", "B"]:
        sizes = [number(row.get("pkt_size")) for row in frames if row.get("pict_type") == pict_type]
        sizes = [size for size in sizes if size is not None]
        output[pict_type] = {
            "count": len(sizes),
            "total_bytes": int(sum(sizes)) if sizes else 0,
            "min_bytes": int(min(sizes)) if sizes else None,
            "max_bytes": int(max(sizes)) if sizes else None,
            "mean_bytes": statistics.mean(sizes) if sizes else None,
            "median_bytes": statistics.median(sizes) if sizes else None,
        }
    return output


def mean_gap(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    gaps = [right - left for left, right in zip(values, values[1:])]
    return statistics.mean(gaps)


def build_summary(
    frames: list[dict[str, str]],
    stream: dict,
    input_fps: float | None,
    analysis_video_fps: float | None,
) -> dict:
    pict_counts = {pict_type: 0 for pict_type in ["I", "P", "B"]}
    for row in frames:
        pict_type = row.get("pict_type")
        if pict_type in pict_counts:
            pict_counts[pict_type] += 1

    i_frames = [row for row in frames if row.get("pict_type") == "I"]
    i_display_indices = [number(row.get("display_order_index")) for row in i_frames]
    i_display_times = [number(row.get("display_time_seconds")) for row in i_frames]
    i_display_indices = [value for value in i_display_indices if value is not None]
    i_display_times = [value for value in i_display_times if value is not None]

    return {
        "total_frames": len(frames),
        "frame_type_counts": pict_counts,
        "total_keyframes": sum(1 for row in frames if row.get("key_frame") == "1"),
        "i_frame_count": len(i_frames),
        "average_distance_between_i_frames": {
            "frames": mean_gap(i_display_indices),
            "seconds": mean_gap(i_display_times),
        },
        "packet_size_by_frame_type": packet_stats_by_frame_type(frames),
        "video": {
            "duration": stream.get("duration"),
            "input_fps": input_fps,
            "analysis_video_fps": analysis_video_fps,
            "codec_name": stream.get("codec_name"),
            "profile": stream.get("profile"),
            "level": stream.get("level"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--fps", type=float, default=0.0, help="Analysis FPS. Use 0 to analyze the original encode.")
    parser.add_argument("--out", default="outputs/extracted", help="Extraction output folder")
    parser.add_argument("--encoded-out", default="outputs/encoded", help="Folder for FPS>0 analysis encodes")
    args = parser.parse_args()

    original_video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out)
    encoded_dir = Path(args.encoded_out)
    frames_dir = out_dir / "frame_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fps < 0:
        raise SystemExit("--fps must be 0 or a positive number")
    if not original_video.exists():
        raise SystemExit(f"Video not found: {original_video}")
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required but was not found on PATH")

    input_stream = read_stream(original_video)
    input_fps = parse_rate(input_stream.get("avg_frame_rate")) or parse_rate(input_stream.get("r_frame_rate"))
    video = make_analysis_video(original_video, args.fps, encoded_dir)
    image_count = extract_frame_images(video, frames_dir)
    stream = read_stream(video)
    frame_data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-export_side_data",
            "+mvs",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-of",
            "json",
            str(video),
        ]
    )
    packet_data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-of",
            "json",
            str(video),
        ]
    )

    analysis_video_fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))

    packets = []
    packet_by_pos = {}
    packet_items = packet_data.get("packets", []) or []
    for index, packet in enumerate(tqdm(packet_items, desc="extract packets", unit="packet")):
        row = {
            "packet_index": str(index),
            "pts_time": text(packet.get("pts_time")),
            "dts_time": text(packet.get("dts_time")),
            "duration_time": text(packet.get("duration_time")),
            "size": int_text(packet.get("size")),
            "pos": int_text(packet.get("pos")),
            "flags": text(packet.get("flags")),
        }
        packets.append(row)
        if row["pos"]:
            packet_by_pos[row["pos"]] = row

    frames = []
    motion_vectors = []
    frames_json = []

    frame_items = frame_data.get("frames", []) or []
    for source_index, frame in enumerate(tqdm(frame_items, desc="extract frames", unit="frame")):
        frame_id = f"f{source_index:06d}"
        frame_vectors = motion_vectors_from_frame(frame, frame_id, source_index)
        stats = motion_stats(frame_vectors)
        packet = packet_by_pos.get(int_text(frame.get("pkt_pos")), {})

        row = {
            "frame_id": frame_id,
            "source_index": str(source_index),
            "display_order_index": "",
            "decode_order_index": "",
            "key_frame": int_text(frame.get("key_frame")),
            "pict_type": text(frame.get("pict_type")),
            "pts_time": text(frame.get("pts_time") or frame.get("best_effort_timestamp_time")),
            "dts_time": text(packet.get("dts_time") or frame.get("pkt_dts_time")),
            "display_time_seconds": "",
            "decode_time_seconds": "",
            "frame_image": "",
            "best_effort_timestamp_time": text(frame.get("best_effort_timestamp_time")),
            "pkt_duration_time": text(frame.get("pkt_duration_time")),
            "pkt_pos": int_text(frame.get("pkt_pos")),
            "pkt_size": int_text(frame.get("pkt_size") or packet.get("size")),
            "coded_picture_number": int_text(frame.get("coded_picture_number")),
            "display_picture_number": int_text(frame.get("display_picture_number")),
            **stats,
        }
        frames.append(row)
        motion_vectors.extend(frame_vectors)
        frames_json.append({**row, "raw_side_data_types": [text(item.get("side_data_type")) for item in frame.get("side_data_list", []) or []]})

    display_indices = sort_index(frames, ["pts_time", "best_effort_timestamp_time"])
    decode_indices = sort_index(frames, ["dts_time", "pkt_pos", "coded_picture_number"])
    for row in tqdm(frames, desc="assign frame order", unit="frame"):
        row["display_order_index"] = str(display_indices[row["frame_id"]])
        row["decode_order_index"] = str(decode_indices[row["frame_id"]])
        row["display_time_seconds"] = row["pts_time"] or row["best_effort_timestamp_time"]
        row["decode_time_seconds"] = row["dts_time"]
        row["frame_image"] = f"frame_images/frame_{int(row['display_order_index']):06d}.png"
    frame_by_id = {frame["frame_id"]: frame for frame in frames}
    for row in tqdm(frames_json, desc="sync frame json", unit="frame"):
        row["display_order_index"] = str(display_indices[row["frame_id"]])
        row["decode_order_index"] = str(decode_indices[row["frame_id"]])
        matching_frame = frame_by_id[row["frame_id"]]
        row["display_time_seconds"] = matching_frame["display_time_seconds"]
        row["decode_time_seconds"] = matching_frame["decode_time_seconds"]
        row["frame_image"] = matching_frame["frame_image"]

    if image_count != len(frames):
        raise SystemExit(f"Decoded {image_count} frame images but ffprobe reported {len(frames)} frames")

    metadata = {
        "input_video": str(original_video),
        "analysis_video": str(video),
        "extractor": "ffprobe",
        "analysis_fps": {
            "mode": "reencoded" if args.fps > 0 else "original",
            "requested_fps": args.fps,
            "input_fps": input_fps,
            "analysis_video_fps": analysis_video_fps,
            "note": "FPS=0 extracts from the original encoded video. FPS>0 re-encodes first, then extracts from that encoded analysis video.",
        },
        "codec": {
            "codec_name": stream.get("codec_name"),
            "profile": stream.get("profile"),
            "level": stream.get("level"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "duration": stream.get("duration"),
            "bit_rate": stream.get("bit_rate"),
            "nb_frames": stream.get("nb_frames"),
        },
        "counts": {
            "frames": len(frames),
            "packets": len(packets),
            "motion_vectors": len(motion_vectors),
            "frame_images": image_count,
        },
        "notes": [
            "display_order_index is sorted by PTS/best-effort timestamp.",
            "decode_order_index is sorted by matched packet DTS, then packet position when DTS is missing.",
            "When FPS>0, extracted rows come from the re-encoded analysis video, not from post-extraction sampling.",
            "frame_image points to a decoded PNG image in display order.",
            "motion vectors are written only when ffprobe exposes motion-vector side data.",
            "reference frame lists, QP, and block details are not guessed when absent.",
        ],
    }
    summary = build_summary(frames, stream, input_fps, analysis_video_fps)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "frames.json").write_text(json.dumps(frames_json, indent=2), encoding="utf-8")
    write_csv(out_dir / "frames.csv", frames, FRAME_FIELDS)
    write_csv(out_dir / "packets.csv", packets, PACKET_FIELDS)
    write_csv(out_dir / "motion_vectors.csv", motion_vectors, MOTION_VECTOR_FIELDS)

    print(f"Wrote extraction outputs to {out_dir}")


if __name__ == "__main__":
    main()
