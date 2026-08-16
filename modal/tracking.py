"""Face tracking over video frames, faithful port of ``shared/utils/videoFaceTracking.ts``.

Produces a per-frame face resolver with interpolation and appearance /
disappearance extension, matching the local server and browser pipelines.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

FACE_MATCH_MAX_SCORE = 1.5
HIGH_CONFIDENCE_FACE = 0.9
FACE_MATCH_DISTANCE_RATIO = 1.5
SMALL_FACE_MAX_SIZE = 36
FAST_MOTION_RATIO = 0.2

DETECTION_PRIORITY = 2
INTERPOLATED_PRIORITY = 1
EXTENSION_PRIORITY = 0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def center_distance(a: dict, b: dict) -> float:
    ax = a["x"] + (a["width"] / 2)
    ay = a["y"] + (a["height"] / 2)
    bx = b["x"] + (b["width"] / 2)
    by = b["y"] + (b["height"] / 2)
    return math.hypot(ax - bx, ay - by)


def overlap_ratio(a: dict, b: dict) -> float:
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["width"], b["x"] + b["width"])
    bottom = min(a["y"] + a["height"], b["y"] + b["height"])
    intersection = max(0, right - left) * max(0, bottom - top)
    smallest_area = max(1, min(a["width"] * a["height"], b["width"] * b["height"]))
    return intersection / smallest_area


def size_difference(a: dict, b: dict) -> float:
    return (
        abs(a["width"] - b["width"]) / max(1, max(a["width"], b["width"]))
        + abs(a["height"] - b["height"]) / max(1, max(a["height"], b["height"]))
    )


def match_score(a: dict, b: dict) -> float:
    average_size = max(1, (a["width"] + a["height"] + b["width"] + b["height"]) / 4)
    return center_distance(a, b) / average_size + size_difference(a, b)


def is_high_confidence_match(a: dict, b: dict) -> bool:
    smaller_max_size = min(max(a["width"], a["height"]), max(b["width"], b["height"]))
    close_enough = center_distance(a, b) <= smaller_max_size * FACE_MATCH_DISTANCE_RATIO
    return (
        max(a["confidence"], b["confidence"]) >= HIGH_CONFIDENCE_FACE
        and (close_enough or overlap_ratio(a, b) >= 0.1)
    )


def can_match(a: dict, b: dict, score: float) -> bool:
    return score <= FACE_MATCH_MAX_SCORE or is_high_confidence_match(a, b)


def match_faces(previous_faces: list[dict], next_faces: list[dict]) -> list[tuple[int, int]]:
    remaining_previous = set(range(len(previous_faces)))
    remaining_next = set(range(len(next_faces)))
    pairs: list[tuple[int, int]] = []

    while remaining_previous and remaining_next:
        best_previous = -1
        best_next = -1
        best_score = math.inf

        for previous_index in remaining_previous:
            for next_index in remaining_next:
                score = match_score(previous_faces[previous_index], next_faces[next_index])
                if score < best_score:
                    best_score = score
                    best_previous = previous_index
                    best_next = next_index

        if (
            best_previous < 0
            or best_next < 0
            or not can_match(previous_faces[best_previous], next_faces[best_next], best_score)
        ):
            break

        pairs.append((best_previous, best_next))
        remaining_previous.remove(best_previous)
        remaining_next.remove(best_next)

    return pairs


class Track:
    def __init__(self) -> None:
        self.detections: list[dict] = []


def build_tracks(samples: list[dict]) -> list[Track]:
    track_id_by_face_key: dict[str, int] = {}
    tracks: list[Track] = []
    next_track_id = 0

    def ensure_track(keys: list[str]) -> int:
        nonlocal next_track_id
        for key in keys:
            existing = track_id_by_face_key.get(key)
            if existing is not None:
                return existing
        track_id = next_track_id
        next_track_id += 1
        tracks.append(Track())
        for key in keys:
            track_id_by_face_key[key] = track_id
        return track_id

    for sample_index in range(len(samples) - 1):
        previous_faces = samples[sample_index]["faces"]
        next_faces = samples[sample_index + 1]["faces"]
        for from_index, to_index in match_faces(previous_faces, next_faces):
            ensure_track([f"{sample_index}:{from_index}", f"{sample_index + 1}:{to_index}"])

    for sample_index, sample in enumerate(samples):
        for face_index, face in enumerate(sample["faces"]):
            track_id = track_id_by_face_key.get(f"{sample_index}:{face_index}")
            if track_id is None:
                track_id = ensure_track([f"{sample_index}:{face_index}"])
            tracks[track_id].detections.append({
                "frame_index": sample["frameIndex"],
                "face": face,
            })

    return [track for track in tracks if track.detections]


def remove_isolated_tracks(tracks: list[Track], samples: list[dict], window_frames: int) -> list[Track]:
    kept: list[Track] = []
    for track in tracks:
        if len(track.detections) > 1:
            kept.append(track)
            continue
        detection = track.detections[0]
        isolated = True
        for sample in samples:
            frame_distance = abs(sample["frameIndex"] - detection["frame_index"])
            if frame_distance == 0 or frame_distance > window_frames:
                continue
            for face in sample["faces"]:
                if can_match(detection["face"], face, match_score(detection["face"], face)):
                    isolated = False
                    break
            if not isolated:
                break
        if not isolated:
            kept.append(track)
    return kept


def tracks_compatible(previous_face: dict, next_face: dict) -> bool:
    max_size = max(
        previous_face["width"], previous_face["height"],
        next_face["width"], next_face["height"],
    )
    return center_distance(previous_face, next_face) <= max_size * FACE_MATCH_DISTANCE_RATIO or overlap_ratio(previous_face, next_face) >= 0.1


def merge_tracks(tracks: list[Track], gap_frames: int) -> list[Track]:
    if gap_frames <= 0:
        return tracks
    ordered = sorted(tracks, key=lambda track: track.detections[0]["frame_index"])
    merged: list[Track] = []
    for track in ordered:
        current_start_frame = track.detections[0]["frame_index"]
        current_first_face = track.detections[0]["face"]
        best_previous: Optional[Track] = None
        best_distance = math.inf
        for index in range(len(merged) - 1, -1, -1):
            previous_track = merged[index]
            previous_end_frame = previous_track.detections[-1]["frame_index"]
            gap = current_start_frame - previous_end_frame
            if gap <= 0 or gap > gap_frames:
                continue
            previous_last_face = previous_track.detections[-1]["face"]
            if not tracks_compatible(previous_last_face, current_first_face):
                continue
            distance = center_distance(previous_last_face, current_first_face)
            if distance < best_distance:
                best_distance = distance
                best_previous = previous_track
        if best_previous is not None:
            best_previous.detections.extend(track.detections)
            best_previous.detections.sort(key=lambda det: det["frame_index"])
            continue
        merged.append(track)
    return merged


def track_is_unstable(track: Track) -> bool:
    detections = track.detections
    if not detections:
        return False
    size_sum = 0.0
    motion_sum = 0.0
    motion_count = 0
    has_fast_segment = False
    for index, detection in enumerate(detections):
        face = detection["face"]
        size_sum += max(face["width"], face["height"])
        if index > 0:
            previous_face = detections[index - 1]["face"]
            gap = max(1, detections[index]["frame_index"] - detections[index - 1]["frame_index"])
            motion_per_frame = center_distance(previous_face, face) / gap
            motion_sum += motion_per_frame
            motion_count += 1
            segment_size = max(
                max(previous_face["width"], previous_face["height"]),
                max(face["width"], face["height"]),
            )
            if motion_per_frame / max(1, segment_size) > FAST_MOTION_RATIO:
                has_fast_segment = True
    average_size = size_sum / len(detections)
    if average_size < SMALL_FACE_MAX_SIZE:
        return True
    average_motion = motion_sum / motion_count if motion_count > 0 else 0
    return average_motion / max(1, average_size) > FAST_MOTION_RATIO or has_fast_segment


def smooth_track(track: Track, window_size: int, smooth_position: bool = True) -> None:
    half = math.floor(window_size / 2)
    for index, detection in enumerate(track.detections):
        start = max(0, index - half)
        end = min(len(track.detections) - 1, index + half)
        weight_sum = 0.0
        x = 0.0
        y = 0.0
        width = 0.0
        height = 0.0
        for other_index in range(start, end + 1):
            other = track.detections[other_index]["face"]
            weight = 1 / (abs(other_index - index) + 1)
            x += other["x"] * weight
            y += other["y"] * weight
            width += other["width"] * weight
            height += other["height"] * weight
            weight_sum += weight
        face = detection["face"]
        if smooth_position:
            face["x"] = round(x / weight_sum)
            face["y"] = round(y / weight_sum)
        face["width"] = round(width / weight_sum)
        face["height"] = round(height / weight_sum)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    return sorted_values[math.floor(len(sorted_values) / 2)]


def face_center(face: dict) -> tuple[float, float]:
    return face["x"] + (face["width"] / 2), face["y"] + (face["height"] / 2)


def estimate_global_motion(tracks: list[Track]) -> dict[int, dict]:
    samples_by_frame: dict[int, list[dict]] = {}
    for track in tracks:
        for index in range(1, len(track.detections)):
            current = track.detections[index]
            previous = track.detections[index - 1]
            if current["frame_index"] != previous["frame_index"] + 1:
                continue
            previous_center = face_center(previous["face"])
            current_center = face_center(current["face"])
            samples_by_frame.setdefault(previous["frame_index"], []).append({
                "dx": current_center[0] - previous_center[0],
                "dy": current_center[1] - previous_center[1],
            })
    motion: dict[int, dict] = {}
    for frame_index, samples in samples_by_frame.items():
        motion[frame_index] = {
            "dx": median([sample["dx"] for sample in samples]),
            "dy": median([sample["dy"] for sample in samples]),
        }
    return motion


def camera_motion_at(global_motion: dict[int, dict], frame_index: int) -> dict:
    max_distance = 60
    for distance in range(max_distance + 1):
        after = global_motion.get(frame_index + distance)
        if after is not None:
            return after
        before = global_motion.get(frame_index - distance)
        if before is not None:
            return before
    return {"dx": 0, "dy": 0}


def sum_camera_motion(global_motion: dict[int, dict], start_frame: int, end_frame: int) -> dict:
    dx = 0.0
    dy = 0.0
    for frame_index in range(start_frame, end_frame + 1):
        motion = camera_motion_at(global_motion, frame_index)
        dx += motion["dx"]
        dy += motion["dy"]
    return {"dx": dx, "dy": dy}


def project_before_track(track: Track, frame_index: int, global_motion: dict[int, dict]) -> dict:
    first_detection = track.detections[0]
    frames_before = first_detection["frame_index"] - frame_index
    first_face = first_detection["face"]
    if frames_before <= 0:
        return {**first_face}
    residual_vx = 0.0
    residual_vy = 0.0
    max_shift = max(4, min(48, first_face["width"] * 0.6))
    if len(track.detections) >= 2:
        second_detection = track.detections[1]
        span = max(1, second_detection["frame_index"] - first_detection["frame_index"])
        face_vx = (second_detection["face"]["x"] - first_face["x"]) / span
        face_vy = (second_detection["face"]["y"] - first_face["y"]) / span
        camera_motion = sum_camera_motion(global_motion, first_detection["frame_index"], second_detection["frame_index"] - 1)
        camera_vx = camera_motion["dx"] / span
        camera_vy = camera_motion["dy"] / span
        residual_vx = clamp(face_vx - camera_vx, -max_shift, max_shift)
        residual_vy = clamp(face_vy - camera_vy, -max_shift, max_shift)
    camera_shift = sum_camera_motion(global_motion, frame_index, first_detection["frame_index"] - 1)
    velocity_width = 0.0
    velocity_height = 0.0
    if len(track.detections) >= 2:
        second = track.detections[1]["face"]
        span = max(1, track.detections[1]["frame_index"] - first_detection["frame_index"])
        velocity_width = (second["width"] - first_face["width"]) / span
        velocity_height = (second["height"] - first_face["height"]) / span
    residual_shift_x = clamp(residual_vx * frames_before, -max_shift, max_shift)
    residual_shift_y = clamp(residual_vy * frames_before, -max_shift, max_shift)
    total_shift_x = clamp(camera_shift["dx"] + residual_shift_x, -max_shift, max_shift)
    total_shift_y = clamp(camera_shift["dy"] + residual_shift_y, -max_shift, max_shift)
    return {
        **first_face,
        "x": round(first_face["x"] - total_shift_x),
        "y": round(first_face["y"] - total_shift_y),
        "width": round(clamp(first_face["width"] - (velocity_width * frames_before), first_face["width"] * 0.75, first_face["width"] * 1.25)),
        "height": round(clamp(first_face["height"] - (velocity_height * frames_before), first_face["height"] * 0.75, first_face["height"] * 1.25)),
    }


def project_after_track(track: Track, frame_index: int, global_motion: dict[int, dict]) -> dict:
    last_detection = track.detections[-1]
    frames_after = frame_index - last_detection["frame_index"]
    last_face = last_detection["face"]
    if frames_after <= 0:
        return {**last_face}
    residual_vx = 0.0
    residual_vy = 0.0
    max_shift = max(4, min(48, last_face["width"] * 0.6))
    if len(track.detections) >= 2:
        previous_detection = track.detections[-2]
        span = max(1, last_detection["frame_index"] - previous_detection["frame_index"])
        face_vx = (last_face["x"] - previous_detection["face"]["x"]) / span
        face_vy = (last_face["y"] - previous_detection["face"]["y"]) / span
        camera_motion = sum_camera_motion(global_motion, previous_detection["frame_index"], last_detection["frame_index"] - 1)
        camera_vx = camera_motion["dx"] / span
        camera_vy = camera_motion["dy"] / span
        residual_vx = clamp(face_vx - camera_vx, -max_shift, max_shift)
        residual_vy = clamp(face_vy - camera_vy, -max_shift, max_shift)
    camera_shift = sum_camera_motion(global_motion, last_detection["frame_index"] + 1, frame_index)
    velocity_width = 0.0
    velocity_height = 0.0
    if len(track.detections) >= 2:
        previous = track.detections[-2]["face"]
        span = max(1, last_detection["frame_index"] - track.detections[-2]["frame_index"])
        velocity_width = (last_face["width"] - previous["width"]) / span
        velocity_height = (last_face["height"] - previous["height"]) / span
    residual_shift_x = clamp(residual_vx * frames_after, -max_shift, max_shift)
    residual_shift_y = clamp(residual_vy * frames_after, -max_shift, max_shift)
    total_shift_x = clamp(camera_shift["dx"] + residual_shift_x, -max_shift, max_shift)
    total_shift_y = clamp(camera_shift["dy"] + residual_shift_y, -max_shift, max_shift)
    return {
        **last_face,
        "x": round(last_face["x"] + total_shift_x),
        "y": round(last_face["y"] + total_shift_y),
        "width": round(clamp(last_face["width"] + (velocity_width * frames_after), last_face["width"] * 0.75, last_face["width"] * 1.25)),
        "height": round(clamp(last_face["height"] + (velocity_height * frames_after), last_face["height"] * 0.75, last_face["height"] * 1.25)),
    }


def interpolate_face(face_a: dict, face_b: dict, t: float) -> dict:
    return {
        **face_a,
        "x": round(face_a["x"] + ((face_b["x"] - face_a["x"]) * t)),
        "y": round(face_a["y"] + ((face_b["y"] - face_a["y"]) * t)),
        "width": round(face_a["width"] + ((face_b["width"] - face_a["width"]) * t)),
        "height": round(face_a["height"] + ((face_b["height"] - face_a["height"]) * t)),
    }


def clamp_to_frame(face: dict, frame_width: int, frame_height: int) -> dict:
    return {
        **face,
        "x": clamp(face["x"], 0, max(0, frame_width - face["width"])),
        "y": clamp(face["y"], 0, max(0, frame_height - face["height"])),
    }


def fill_track_gaps(track: Track, gap_frames: int, resolved_frames: dict[int, list[dict]]) -> None:
    for index in range(len(track.detections) - 1):
        current = track.detections[index]
        next_detection = track.detections[index + 1]
        gap = next_detection["frame_index"] - current["frame_index"]
        if gap <= 1 or gap > gap_frames:
            continue
        for frame_index in range(current["frame_index"] + 1, next_detection["frame_index"]):
            t = (frame_index - current["frame_index"]) / gap
            resolved_frames.setdefault(frame_index, []).append(interpolate_face(current["face"], next_detection["face"], t))


def fill_track_extension(
    track: Track,
    appearance_frames: int,
    disappearance_frames: int,
    frame_count: int,
    resolved_frames: dict[int, list[dict]],
    global_motion: dict[int, dict],
    frame_width: int,
    frame_height: int,
) -> None:
    first_detection = track.detections[0]
    last_detection = track.detections[-1]
    if len(track.detections) < 2:
        return
    if appearance_frames > 0:
        first_frame = max(0, first_detection["frame_index"] - appearance_frames)
        for frame_index in range(first_frame, first_detection["frame_index"]):
            resolved_frames.setdefault(frame_index, []).append(clamp_to_frame(project_before_track(track, frame_index, global_motion), frame_width, frame_height))
    if disappearance_frames > 0:
        last_frame = min(frame_count - 1, last_detection["frame_index"] + disappearance_frames)
        for frame_index in range(last_detection["frame_index"] + 1, last_frame + 1):
            resolved_frames.setdefault(frame_index, []).append(clamp_to_frame(project_after_track(track, frame_index, global_motion), frame_width, frame_height))


def deduplicate_frame_faces(faces: list[dict]) -> list[dict]:
    def priority(source: str) -> int:
        if source == "detection":
            return DETECTION_PRIORITY
        if source == "interpolated":
            return INTERPOLATED_PRIORITY
        return EXTENSION_PRIORITY

    sorted_faces = sorted(
        faces,
        key=lambda face: (-priority(face.get("source", "detection")), -face["confidence"]),
    )
    kept: list[dict] = []
    for face in sorted_faces:
        is_duplicate = False
        for other in kept:
            min_size = min(face["width"], face["height"], other["width"], other["height"])
            close_centers = center_distance(face, other) <= min_size * 0.3
            sizes_similar = max(face["width"], face["height"], other["width"], other["height"]) <= min_size * 1.5
            both_detections = face.get("source") == "detection" and other.get("source") == "detection"
            if both_detections:
                is_duplicate = sizes_similar and (overlap_ratio(face, other) >= 0.5 or close_centers)
            else:
                is_duplicate = overlap_ratio(face, other) >= 0.2 or close_centers
            if is_duplicate:
                break
        if not is_duplicate:
            kept.append(face)
    kept.sort(key=lambda face: face["width"] * face["height"], reverse=True)
    return kept


def create_face_resolver(
    samples: list[dict],
    gap_frames: int = 4,
    appearance_frames: int = 0,
    disappearance_frames: int = 0,
) -> Callable[[int], list[dict]]:
    if not samples:
        return lambda frame_index: []

    normalized = sorted(samples, key=lambda sample: sample["frameIndex"])
    frame_count = normalized[-1]["frameIndex"] + 1
    frame_width = 0
    frame_height = 0
    for sample in normalized:
        for face in sample["faces"]:
            frame_width = max(frame_width, face["x"] + face["width"])
            frame_height = max(frame_height, face["y"] + face["height"])
    gap_frames = max(0, gap_frames)
    appearance_frames = max(0, appearance_frames)
    disappearance_frames = max(0, disappearance_frames)
    tracks = merge_tracks(
        remove_isolated_tracks(build_tracks(normalized), normalized, 2),
        gap_frames,
    )
    global_motion = estimate_global_motion(tracks)
    resolved_frames: dict[int, list[dict]] = {}

    for track in tracks:
        unstable = track_is_unstable(track)
        smooth_track(track, 5, not unstable)
        for detection in track.detections:
            resolved_frames.setdefault(detection["frame_index"], []).append({**detection["face"], "source": "detection"})
        fill_track_gaps(track, gap_frames, resolved_frames)
        fill_track_extension(
            track,
            appearance_frames,
            disappearance_frames,
            frame_count,
            resolved_frames,
            global_motion,
            frame_width,
            frame_height,
        )

    for frame_index, faces in resolved_frames.items():
        resolved_frames[frame_index] = deduplicate_frame_faces(faces)

    def resolve(frame_index: int) -> list[dict]:
        if frame_index < 0 or frame_index >= frame_count:
            return []
        return resolved_frames.get(frame_index, [])

    return resolve
