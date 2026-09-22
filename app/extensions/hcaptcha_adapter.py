# -*- coding: utf-8 -*-
import asyncio
import itertools
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import httpx
import matplotlib
import numpy as np
from hcaptcha_challenger.agent.challenger import AgentV, RoboticArm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from hcaptcha_challenger.models import (
    CaptchaResponse,
    ChallengeTypeEnum,
    PointCoordinate,
    RequestType,
    SpatialPath,
)
from loguru import logger
from playwright.async_api import Locator, TimeoutError as PlaywrightTimeoutError

from extensions.numbered_line_solver import solve_numbered_line_drag


_EMPTY_CHECKCAPTCHA_GRACE_SECONDS = 5.0

_GRID_TICK_INSTRUCTION = (
    "Read every coordinate from the axis tick labels of the coordinate-grid image directly "
    "(those ticks are labelled with page coordinates). Do not report raw pixel positions of "
    "the rendered image."
)

# The upstream `create_coordinate_grid` helper renders the challenge screenshot into a
# matplotlib figure (figsize 10x10 => 1000x1000 px) whose axis ticks are labelled with
# *page* coordinates, expecting the model to read those ticks. GLM returns the pixel
# coordinates of that rendered figure instead, so the plot area layout is reproduced once
# here in order to convert model output back into page coordinates.
GridGeometry = tuple[float, float, float, float]

_GRID_GEOMETRY_CACHE: dict[tuple[int, ...], GridGeometry | None] = {}


def _grid_plot_geometry(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_lines: int,
    y_lines: int,
) -> GridGeometry | None:
    """Reproduce the upstream figure layout and return the plot area in image pixels.

    Returns ``(left, top, width, height)`` measured from the top-left corner of the grid
    image, or ``None`` when the layout cannot be reproduced confidently.
    """
    key = (round(x_min), round(x_max), round(y_min), round(y_max), x_lines, y_lines)
    if key in _GRID_GEOMETRY_CACHE:
        return _GRID_GEOMETRY_CACHE[key]

    geometry: GridGeometry | None = None
    try:
        figure, axis = plt.subplots(figsize=(10, 10))
        axis.imshow(np.zeros((2, 2, 3), dtype=np.uint8), extent=(x_min, x_max, y_max, y_min))
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_max, y_min)
        axis.spines["left"].set_position(("data", x_min))
        axis.spines["bottom"].set_position(("data", y_max))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        x_ticks = np.linspace(x_min, x_max, x_lines)
        y_ticks = np.linspace(y_min, y_max, y_lines)
        axis.set_xticks(x_ticks)
        axis.set_yticks(y_ticks)
        axis.tick_params(axis="both", which="major", labelsize=10)
        axis.set_xticklabels([str(round(tick)) for tick in x_ticks])
        axis.set_yticklabels([str(round(tick)) for tick in y_ticks])
        axis.set_xlabel("X Coordinate")
        axis.set_ylabel("Y Coordinate")
        axis.set_title("Image with Coordinate Grid")
        plt.tight_layout()
        figure.canvas.draw()
        canvas_width, canvas_height = figure.canvas.get_width_height()
        box = axis.get_window_extent()
        plt.close(figure)

        left = float(box.x0)
        top = float(canvas_height - box.y1)
        plot_width = float(box.width)
        plot_height = float(box.height)
        if (
            0.0 <= left < canvas_width
            and 0.0 <= top < canvas_height
            and 0.0 < plot_width <= canvas_width
            and 0.0 < plot_height <= canvas_height
            and plot_width >= canvas_width * 0.5
            and plot_height >= canvas_height * 0.5
        ):
            geometry = (left, top, plot_width, plot_height)
    except Exception as err:
        logger.warning("Could not reproduce hCaptcha grid geometry: {!r}", err)

    _GRID_GEOMETRY_CACHE[key] = geometry
    return geometry


def _map_model_points_to_page(
    points: list[Any],
    *,
    challenge_bbox: dict[str, float],
    geometry: GridGeometry | None,
) -> list[Any]:
    """Convert model output from grid-image pixels into page coordinates.

    The remap is only adopted when it puts strictly more points inside the challenge area
    than the raw values did, so genuinely page-space answers are left untouched.
    """
    if geometry is None or not points:
        return points

    bx = float(challenge_bbox["x"])
    by = float(challenge_bbox["y"])
    bw = float(challenge_bbox["width"])
    bh = float(challenge_bbox["height"])

    def _inside(px: float, py: float) -> bool:
        return bx <= px <= bx + bw and by <= py <= by + bh

    left, top, plot_width, plot_height = geometry
    remapped = [
        (
            bx + (float(point.x) - left) / plot_width * bw,
            by + (float(point.y) - top) / plot_height * bh,
        )
        for point in points
    ]

    raw_hits = sum(1 for point in points if _inside(float(point.x), float(point.y)))
    mapped_hits = sum(1 for px, py in remapped if _inside(px, py))
    if mapped_hits <= raw_hits:
        return points

    logger.info(
        "Remapped hCaptcha grid pixels to page coordinates | "
        "geometry=({:.1f}, {:.1f}, {:.1f}, {:.1f}) | raw_hits={} mapped_hits={}",
        left,
        top,
        plot_width,
        plot_height,
        raw_hits,
        mapped_hits,
    )
    for point, (px, py) in zip(points, remapped):
        point.x = px
        point.y = py
    return points


def _longest_contiguous_run(values: np.ndarray) -> list[int]:
    runs: list[list[int]] = []
    for value in values.tolist():
        value = int(value)
        if not runs or value != runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return max(runs, key=len, default=[])


def _detect_task_canvas_bounds(challenge_screenshot: Path) -> tuple[int, int, int, int] | None:
    image = cv2.imread(str(challenge_screenshot))
    if image is None:
        return None

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    task_pixels = (hsv[:, :, 1] > 10) | (hsv[:, :, 2] < 235)

    # The prompt header occupies the top of the challenge. The task canvas is the longest
    # colored run below it, regardless of whether hCaptcha renders the 320px or 330px layout.
    row_counts = task_pixels.sum(axis=1)
    row_indexes = np.flatnonzero(
        (row_counts > width * 0.35) & (np.arange(height) >= int(height * 0.23))
    )
    task_rows = _longest_contiguous_run(row_indexes)
    if len(task_rows) < height * 0.45:
        return None

    task_mask = task_pixels[task_rows[0] : task_rows[-1] + 1]
    column_indexes = np.flatnonzero(task_mask.sum(axis=0) > len(task_rows) * 0.2)
    if not len(column_indexes):
        return None

    return (int(column_indexes.min()), task_rows[0], int(column_indexes.max()), task_rows[-1])


def _detect_task_canvas_origin(challenge_screenshot: Path) -> tuple[int, int] | None:
    bounds = _detect_task_canvas_bounds(challenge_screenshot)
    return bounds[:2] if bounds is not None else None


def _detect_clickable_grid_bounds(challenge_screenshot: Path) -> tuple[int, int, int, int] | None:
    image = cv2.imread(str(challenge_screenshot))
    task_bounds = _detect_task_canvas_bounds(challenge_screenshot)
    if image is None or task_bounds is None:
        return None

    task_x_min, task_y_min, task_x_max, task_y_max = task_bounds
    task_height = task_y_max - task_y_min + 1
    task_width = task_x_max - task_x_min + 1
    if task_width < task_height * 1.25:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    badge_centers: list[tuple[float, float]] = []
    image_height, image_width = image.shape[:2]
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (
            task_y_min <= y <= task_y_max
            and image_width * 0.07 <= width <= image_width * 0.16
            and image_height * 0.04 <= height <= image_height * 0.11
            and 1.25 <= width / max(height, 1) <= 2.2
            and area >= width * height * 0.45
        ):
            continue
        badge_centers.append((x + width / 2, y + height / 2))

    best_group: list[tuple[float, float]] = []
    x_tolerance = image_width * 0.04
    for center in badge_centers:
        group = [item for item in badge_centers if abs(item[0] - center[0]) <= x_tolerance]
        if len(group) > len(best_group):
            best_group = group

    if len(best_group) < 3:
        return None
    badge_y_values = [center[1] for center in best_group]
    if max(badge_y_values) - min(badge_y_values) < task_height * 0.45:
        return None

    reference_x = sum(center[0] for center in best_group) / len(best_group)
    task_midpoint = (task_x_min + task_x_max) / 2
    if reference_x < task_midpoint:
        return task_x_max - task_height + 1, task_y_min, task_x_max, task_y_max
    return task_x_min, task_y_min, task_x_min + task_height - 1, task_y_max


def _map_image_bounds_to_page(
    image_bounds: tuple[int, int, int, int],
    *,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float],
) -> tuple[float, float, float, float] | None:
    image = cv2.imread(str(challenge_screenshot))
    if image is None:
        return None

    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = image_bounds
    scale_x = float(challenge_bbox["width"]) / image_width
    scale_y = float(challenge_bbox["height"]) / image_height
    return (
        float(challenge_bbox["x"]) + x_min * scale_x,
        float(challenge_bbox["y"]) + y_min * scale_y,
        float(challenge_bbox["x"]) + x_max * scale_x,
        float(challenge_bbox["y"]) + y_max * scale_y,
    )


def _point_answer_validation_error(
    points: list[Any],
    *,
    challenge_bbox: dict[str, float] | None,
    clickable_bounds: tuple[float, float, float, float] | None,
) -> str | None:
    if not points:
        return "model returned no click points"
    if challenge_bbox is None:
        return None

    # Allow generous tolerance around challenge bounds for validation
    challenge_bounds = (
        float(challenge_bbox["x"]) - 30.0,
        float(challenge_bbox["y"]) - 30.0,
        float(challenge_bbox["x"]) + float(challenge_bbox["width"]) + 30.0,
        float(challenge_bbox["y"]) + float(challenge_bbox["height"]) + 30.0,
    )
    for point in points:
        coordinates = float(point.x), float(point.y)
        if not _point_inside_bounds(coordinates, challenge_bounds):
            return f"point {coordinates} is outside challenge bounds {challenge_bounds}"
    return None


def _point_inside_bounds(
    point: tuple[float, float], bounds: tuple[float, float, float, float]
) -> bool:
    x, y = point
    x_min, y_min, x_max, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


def _build_point_prompt(
    user_prompt: str,
    *,
    challenge_bbox: dict[str, float] | None,
    clickable_bounds: tuple[float, float, float, float] | None,
) -> str:
    if clickable_bounds is not None:
        x_min, y_min, x_max, y_max = clickable_bounds
        constraint = (
            "The clickable tile grid is within page coordinates "
            f"x={x_min:.0f}..{x_max:.0f}, y={y_min:.0f}..{y_max:.0f}. "
            "Every returned point must be inside this grid. Repeated count badges and example "
            "animals outside the grid are references only, regardless of which side they occupy."
        )
    elif challenge_bbox is not None:
        x_min = float(challenge_bbox["x"])
        y_min = float(challenge_bbox["y"])
        x_max = x_min + float(challenge_bbox["width"])
        y_max = y_min + float(challenge_bbox["height"])
        constraint = (
            "Every returned point must be inside the visible challenge bounds "
            f"x={x_min:.0f}..{x_max:.0f}, y={y_min:.0f}..{y_max:.0f}."
        )
    else:
        return user_prompt
    return f"{user_prompt}\n\n{constraint}\n\n{_GRID_TICK_INSTRUCTION}"


def _is_count_selection_question(question: str) -> bool:
    normalized = question.lower()
    return "animal" in normalized and "count" in normalized


def _entity_centers(captcha_payload: Any, crumb_id: int) -> list[tuple[int, int]]:
    tasklist = getattr(captcha_payload, "tasklist", None) or []
    if crumb_id < 0 or crumb_id >= len(tasklist):
        return []

    centers: list[tuple[int, int]] = []
    for entity in getattr(tasklist[crumb_id], "entities", None) or []:
        coords = getattr(entity, "coords", None) or []
        if len(coords) < 2:
            return []
        centers.append((int(coords[0]), int(coords[1])))
    return centers


def _map_canvas_points_to_page(
    points: list[tuple[float, float]],
    *,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[tuple[int, int]]:
    if not points or not challenge_bbox:
        return []

    canvas_origin = _detect_task_canvas_origin(challenge_screenshot)
    image = cv2.imread(str(challenge_screenshot))
    if canvas_origin is None or image is None:
        return []

    image_height, image_width = image.shape[:2]
    scale_x = float(challenge_bbox["width"]) / image_width
    scale_y = float(challenge_bbox["height"]) / image_height
    origin_x, origin_y = canvas_origin
    return [
        (
            int(round(float(challenge_bbox["x"]) + (origin_x + x) * scale_x)),
            int(round(float(challenge_bbox["y"]) + (origin_y + y) * scale_y)),
        )
        for x, y in points
    ]


def _payload_source_points(
    *,
    captcha_payload: Any,
    crumb_id: int,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[tuple[int, int]]:
    centers = _entity_centers(captcha_payload, crumb_id)
    return _map_canvas_points_to_page(
        centers, challenge_screenshot=challenge_screenshot, challenge_bbox=challenge_bbox
    )


def _correct_drag_source_points(
    paths: list[Any],
    *,
    captcha_payload: Any,
    crumb_id: int,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[Any]:
    centers = _entity_centers(captcha_payload, crumb_id)
    if not paths or len(centers) != len(paths) or not challenge_bbox:
        return paths

    resolved_sources = _payload_source_points(
        captcha_payload=captcha_payload,
        crumb_id=crumb_id,
        challenge_screenshot=challenge_screenshot,
        challenge_bbox=challenge_bbox,
    )
    if len(resolved_sources) != len(paths):
        logger.warning("Could not locate hCaptcha drag canvas; keeping model source coordinates")
        return paths

    path_order = sorted(range(len(paths)), key=lambda index: paths[index].start_point.y)
    source_order = sorted(resolved_sources, key=lambda point: point[1])
    for path_index, source in zip(path_order, source_order):
        path = paths[path_index]
        previous = (path.start_point.x, path.start_point.y)
        path.start_point.x, path.start_point.y = source
        logger.info("Corrected hCaptcha drag source from model={} to payload={}", previous, source)

    return paths


def _extract_outline_targets(
    challenge_screenshot: Path,
) -> list[tuple[np.ndarray, tuple[float, float]]]:
    image = cv2.imread(str(challenge_screenshot))
    canvas_origin = _detect_task_canvas_origin(challenge_screenshot)
    if image is None or canvas_origin is None:
        return []

    origin_x, origin_y = canvas_origin
    task_canvas = image[origin_y:, origin_x:]
    hsv = cv2.cvtColor(task_canvas, cv2.COLOR_BGR2HSV)
    outline_mask = ((hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 140)).astype(np.uint8) * 255
    outline_mask[:, int(task_canvas.shape[1] * 0.68) :] = 0
    outline_mask = cv2.morphologyEx(outline_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(outline_mask)
    targets: list[tuple[np.ndarray, tuple[float, float]]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if not (400 <= area <= 3000 and width >= 35 and height >= 35):
            continue
        if x >= task_canvas.shape[1] * 0.68 or y >= task_canvas.shape[0] * 0.88:
            continue

        component = (labels == index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        targets.append((contour, center))

    return targets


def _decode_entity_contour(content: bytes) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 4:
        return None

    mask = (image[:, :, 3] > 20).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _match_outline_contours(
    sources: list[np.ndarray], targets: list[tuple[np.ndarray, tuple[float, float]]]
) -> tuple[list[int], list[float]] | None:
    if not sources or len(targets) < len(sources):
        return None

    scores = [
        [cv2.matchShapes(source, target[0], cv2.CONTOURS_MATCH_I1, 0) for target in targets]
        for source in sources
    ]
    assignments = itertools.permutations(range(len(targets)), len(sources))
    best = min(
        assignments, key=lambda item: sum(scores[i][target] for i, target in enumerate(item))
    )
    assigned_scores = [scores[index][target] for index, target in enumerate(best)]
    if max(assigned_scores) > 0.16:
        return None
    return list(best), assigned_scores


def _marker_segment_color(
    image: np.ndarray, center: tuple[int, int]
) -> tuple[float, float, float] | None:
    x, y = center
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    distance_squared = (xx - x) ** 2 + (yy - y) ** 2
    scale = image.shape[1] / 500.0
    inner_radius = max(5, int(round(8 * scale)))
    outer_radius = max(15, int(round(25 * scale)))
    annulus = (distance_squared >= inner_radius**2) & (distance_squared <= outer_radius**2)
    pixels = image[annulus]
    hsv_pixels = hsv[annulus]
    if not len(pixels):
        return None

    bright_threshold = np.percentile(hsv_pixels[:, 2], 65)
    line_pixels = pixels[(hsv_pixels[:, 2] > bright_threshold) & (hsv_pixels[:, 1] > 20)]
    if len(line_pixels) < 20:
        return None
    blue, green, red = np.median(line_pixels, axis=0)
    return float(blue), float(green), float(red)


def _select_line_gap_markers(
    markers: list[tuple[tuple[float, float], tuple[float, float, float]]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not 4 <= len(markers) <= 7:
        return None

    # Segment 3 is cyan and segment 5 is yellow across the observed line-puzzle variants.
    yellow_scores = [red + green - 2 * blue for _, (blue, green, red) in markers]
    yellow_order = sorted(range(len(markers)), key=yellow_scores.__getitem__, reverse=True)
    if yellow_scores[yellow_order[0]] - yellow_scores[yellow_order[1]] < 35:
        return None
    marker_five_index = yellow_order[0]

    remaining = [index for index in range(len(markers)) if index != marker_five_index]
    cyan_scores = {
        index: markers[index][1][0] + markers[index][1][1] - 2 * markers[index][1][2]
        for index in remaining
    }
    cyan_order = sorted(remaining, key=cyan_scores.__getitem__, reverse=True)
    if cyan_scores[cyan_order[0]] - cyan_scores[cyan_order[1]] < 10:
        return None

    return markers[cyan_order[0]][0], markers[marker_five_index][0]


def _extract_line_target(challenge_screenshot: Path) -> tuple[float, float] | None:
    image = cv2.imread(str(challenge_screenshot))
    canvas_origin = _detect_task_canvas_origin(challenge_screenshot)
    if image is None or canvas_origin is None:
        return None

    origin_x, origin_y = canvas_origin
    task_width = image.shape[1] - origin_x
    fixed_line_width = int(task_width * 0.78)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fixed_line = gray[origin_y:, origin_x : origin_x + fixed_line_width]
    scale = image.shape[1] / 500.0
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(fixed_line, (5, 5), 1),
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(18, int(round(25 * scale))),
        param1=80,
        param2=18,
        minRadius=max(5, int(round(7 * scale))),
        maxRadius=max(10, int(round(14 * scale))),
    )
    if circles is None:
        return None

    markers = []
    for local_x, local_y, _ in np.round(circles[0]).astype(int):
        center = (int(local_x + origin_x), int(local_y + origin_y))
        color = _marker_segment_color(image, center)
        if color is not None:
            markers.append((center, color))

    selected = _select_line_gap_markers(markers)
    if selected is None:
        return None
    marker_three, marker_five = selected
    return (
        (marker_three[0] + marker_five[0]) / 2 - origin_x,
        (marker_three[1] + marker_five[1]) / 2 - origin_y,
    )


def _is_line_completion_question(question: str) -> bool:
    normalized = question.lower()
    return "segment" in normalized and "line" in normalized


def _resolve_line_path(
    *,
    captcha_payload: Any,
    crumb_id: int,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[SpatialPath] | None:
    question = ""
    with suppress(Exception):
        question = captcha_payload.get_requester_question()
    if not _is_line_completion_question(question):
        return None

    source_points = _payload_source_points(
        captcha_payload=captcha_payload,
        crumb_id=crumb_id,
        challenge_screenshot=challenge_screenshot,
        challenge_bbox=challenge_bbox,
    )
    if len(source_points) != 1:
        logger.warning("Could not resolve numbered hCaptcha line locally; falling back to LLM")
        return None

    numbered_solution = None
    if challenge_bbox:
        try:
            numbered_solution = solve_numbered_line_drag(challenge_screenshot, challenge_bbox)
        except Exception as err:
            logger.warning("Numbered-circle template matching failed: {!r}", err)

    if numbered_solution is not None:
        target_points = [numbered_solution.end]
        strategy = (
            f"numbered-circles:1-{numbered_solution.digit_count}/"
            f"source={numbered_solution.source_label}/score={numbered_solution.score:.3f}"
        )
    else:
        target = _extract_line_target(challenge_screenshot)
        target_points = _map_canvas_points_to_page(
            [target] if target is not None else [],
            challenge_screenshot=challenge_screenshot,
            challenge_bbox=challenge_bbox,
        )
        strategy = "color-markers"

    if len(target_points) != 1:
        logger.warning("Could not resolve numbered hCaptcha line locally; falling back to LLM")
        return None

    path = SpatialPath(
        start_point=PointCoordinate(x=source_points[0][0], y=source_points[0][1]),
        end_point=PointCoordinate(x=target_points[0][0], y=target_points[0][1]),
    )
    logger.info(
        "Resolved numbered hCaptcha line deterministically | strategy={} from={} to={}",
        strategy,
        (path.start_point.x, path.start_point.y),
        (path.end_point.x, path.end_point.y),
    )
    return [path]


async def _resolve_outline_paths(
    *,
    captcha_payload: Any,
    crumb_id: int,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[SpatialPath] | None:
    tasklist = getattr(captcha_payload, "tasklist", None) or []
    if crumb_id < 0 or crumb_id >= len(tasklist):
        return None
    entities = getattr(tasklist[crumb_id], "entities", None) or []
    if len(entities) < 2:
        return None

    question = ""
    with suppress(Exception):
        question = captcha_payload.get_requester_question().lower()
    if "outline" not in question:
        return None

    source_points = _payload_source_points(
        captcha_payload=captcha_payload,
        crumb_id=crumb_id,
        challenge_screenshot=challenge_screenshot,
        challenge_bbox=challenge_bbox,
    )
    targets = _extract_outline_targets(challenge_screenshot)
    if len(source_points) != len(entities) or len(targets) < len(entities):
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            responses = await asyncio.gather(
                *(client.get(str(entity.entity_uri)) for entity in entities)
            )
        for response in responses:
            response.raise_for_status()
        source_contours = [_decode_entity_contour(response.content) for response in responses]
    except Exception as err:
        logger.warning("Could not load hCaptcha drag entities for outline matching: {!r}", err)
        return None

    if any(contour is None for contour in source_contours):
        return None
    matched = _match_outline_contours(source_contours, targets)
    if matched is None:
        logger.warning("hCaptcha outline topology match was not confident; falling back to LLM")
        return None

    assignment, scores = matched
    target_points = _map_canvas_points_to_page(
        [targets[target][1] for target in assignment],
        challenge_screenshot=challenge_screenshot,
        challenge_bbox=challenge_bbox,
    )
    if len(target_points) != len(source_points):
        return None

    paths = [
        SpatialPath(
            start_point=PointCoordinate(x=source[0], y=source[1]),
            end_point=PointCoordinate(x=target[0], y=target[1]),
        )
        for source, target in zip(source_points, target_points)
    ]
    logger.info(
        "Resolved hCaptcha outline paths by topology | scores={} paths={}",
        [round(score, 4) for score in scores],
        [
            {
                "from": (path.start_point.x, path.start_point.y),
                "to": (path.end_point.x, path.end_point.y),
            }
            for path in paths
        ],
    )
    return paths


def _build_drag_prompt(user_prompt: str, *, source_points: list[tuple[int, int]]) -> str:
    details = (
        f"Authoritative draggable centers from the challenge payload: {source_points}. "
        "Use these exact start_point values and reason only about each end_point."
    )
    if _is_line_completion_question(user_prompt):
        details += (
            " For numbered line completion, locate fixed segments 3 and 5 and the two exposed "
            "ends they present. Translate the entire segment 4 between them. Its end_point is "
            "the center of segment 4 after placement and should lie in the geometric corridor "
            "between segments 3 and 5, usually near the midpoint of their numbered circles. "
            "Reject any candidate at marker 3, marker 5, or a distant unrelated empty area."
        )
    return f"{user_prompt}\n\n{details}"


def _cancel_pending_empty_response(agent: Any) -> None:
    pending = getattr(agent, "_epic_empty_response_task", None)
    if pending is not None and not pending.done():
        pending.cancel()
    agent._epic_empty_response_task = None


async def _queue_empty_checkcaptcha_response(
    agent: Any, response: Any, *, grace_seconds: float = _EMPTY_CHECKCAPTCHA_GRACE_SECONDS
) -> bool:
    if "/checkcaptcha/" not in str(getattr(response, "url", "")):
        return False
    body = await response.body()
    if body and body.strip():
        _cancel_pending_empty_response(agent)
        return False

    _cancel_pending_empty_response(agent)

    async def enqueue_failure_after_grace_period():
        try:
            await asyncio.sleep(grace_seconds)
            if agent._captcha_response_queue.empty():
                agent._captcha_response_queue.put_nowait(
                    CaptchaResponse.model_validate(
                        {"pass": False, "error": "empty_checkcaptcha_response"}
                    )
                )
        except asyncio.CancelledError:
            return

    agent._epic_empty_response_task = asyncio.create_task(enqueue_failure_after_grace_period())
    logger.warning(
        "hCaptcha check returned an empty response; waiting {:.1f}s for the paired result | "
        "status={}",
        grace_seconds,
        getattr(response, "status", "unknown"),
    )
    return True


def _apply_empty_checkcaptcha_patch() -> None:
    if getattr(AgentV._task_handler, "_epic_empty_response_patch", False):
        return

    original_task_handler = AgentV._task_handler

    async def patched_task_handler(self: AgentV, response: Any):
        with suppress(Exception):
            if await _queue_empty_checkcaptcha_response(self, response):
                return
        return await original_task_handler(self, response)

    patched_task_handler._epic_empty_response_patch = True
    AgentV._task_handler = patched_task_handler


def _patch_robotic_arm_safety() -> None:
    if getattr(RoboticArm, "_epic_safety_patch", False):
        return

    orig_click_by_mouse = RoboticArm.click_by_mouse

    async def safe_click_by_mouse(self: RoboticArm, locator: Locator) -> bool:
        try:
            bbox = await locator.bounding_box()
            if bbox is None:
                logger.warning("Locator has no bounding box; click aborted | locator={}", locator)
                return False
            center_x = bbox["x"] + bbox["width"] / 2
            center_y = bbox["y"] + bbox["height"] / 2
            await self.page.mouse.move(center_x, center_y)
            await self.page.mouse.click(center_x, center_y, delay=150)
            return True
        except Exception as err:
            logger.warning("Failed to click locator by mouse: {!r}", err)
            return False

    async def safe_refresh_challenge(self: RoboticArm) -> bool:
        selectors = [
            "//div[@class='refresh button']",
            "//div[contains(@class, 'refresh') and contains(@class, 'button')]",
            "//div[contains(@class, 'refresh')]",
            "//button[contains(@class, 'refresh')]",
            "//div[contains(@class, 'reload')]",
            "button[aria-label*='refresh' i]",
            "div[aria-label*='refresh' i]",
            "button[title*='refresh' i]",
            "#refresh",
            ".refresh",
        ]
        try:
            refresh_frame = await self.get_challenge_frame_locator()
            frames_to_try = [refresh_frame] if refresh_frame else []
            for frame in self.page.frames:
                if "hcaptcha.com" in frame.url and "frame=challenge" in frame.url and frame not in frames_to_try:
                    frames_to_try.append(frame)

            for frame in frames_to_try:
                for sel in selectors:
                    try:
                        loc = frame.locator(sel).first
                        if await loc.count() > 0 and await loc.is_visible():
                            bbox = await loc.bounding_box()
                            if bbox:
                                cx = bbox["x"] + bbox["width"] / 2
                                cy = bbox["y"] + bbox["height"] / 2
                                await self.page.mouse.move(cx, cy)
                                await self.page.mouse.click(cx, cy, delay=150)
                                logger.info("Clicked hCaptcha refresh button via selector: {}", sel)
                                return True
                            else:
                                await loc.click(force=True, timeout=2000)
                                logger.info("Clicked hCaptcha refresh button with force=True: {}", sel)
                                return True
                    except Exception:
                        continue
            logger.warning("No visible refresh button found across candidate selectors and frames")
            return False
        except Exception as err:
            logger.warning("Failed to click refresh button: {!r}", err)
            return False

    orig_check_challenge_type = RoboticArm.check_challenge_type

    async def safe_check_challenge_type(self: RoboticArm) -> RequestType | ChallengeTypeEnum | None:
        try:
            with suppress(Exception):
                await self.page.wait_for_selector(self.challenge_selector, timeout=1000)

            frame_challenge = await self.get_challenge_frame_locator()
            if frame_challenge is None:
                logger.warning("Cannot find valid challenge frame in check_challenge_type")
                return None

            return await orig_check_challenge_type(self)
        except Exception as err:
            logger.warning("Error checking challenge type: {!r}", err)
            return None

    RoboticArm.click_by_mouse = safe_click_by_mouse
    RoboticArm.refresh_challenge = safe_refresh_challenge
    RoboticArm.check_challenge_type = safe_check_challenge_type

    orig_solve_captcha = AgentV._solve_captcha

    async def safe_solve_captcha(self: AgentV):
        attempts = getattr(self, "_epic_solve_attempts", 0) + 1
        self._epic_solve_attempts = attempts
        if attempts > 4:
            logger.warning("Reached maximum captcha solve attempts ({}); aborting loop to avoid timeout", attempts)
            self._epic_solve_attempts = 0
            return None

        challenge_type = await self._review_challenge_type()
        if challenge_type is None:
            logger.warning("Challenge type is None; refreshing challenge")
            await self.page.wait_for_timeout(2000)
            await self.robotic_arm.refresh_challenge()
            challenge_type = await self._review_challenge_type()
            if challenge_type is None:
                logger.warning("Challenge type still None after refresh; returning")
                self._epic_solve_attempts = 0
                return None

        type_str = getattr(challenge_type, "value", str(challenge_type))
        logger.debug(
            f"Start Challenge - type={type_str} count={self.robotic_arm.signal_crumb_count}"
        )

        try:
            with suppress(Exception):
                if self.config.ignore_request_questions and self._captcha_payload:
                    for q in self.config.ignore_request_questions:
                        if q in self._captcha_payload.get_requester_question():
                            await self.page.wait_for_timeout(2000)
                            await self.robotic_arm.refresh_challenge()
                            return await self._solve_captcha()

            match challenge_type:
                case RequestType.IMAGE_LABEL_BINARY:
                    if RequestType.IMAGE_LABEL_BINARY not in self.config.ignore_request_types:
                        res = await self.robotic_arm.challenge_image_label_binary()
                        self._epic_solve_attempts = 0
                        return res
                case ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT:
                    if (
                        RequestType.IMAGE_LABEL_AREA_SELECT not in self.config.ignore_request_types
                        and ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT not in self.config.ignore_request_types
                    ):
                        res = await self.robotic_arm.challenge_image_label_select(challenge_type)
                        self._epic_solve_attempts = 0
                        return res
                case ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT:
                    if (
                        RequestType.IMAGE_LABEL_AREA_SELECT not in self.config.ignore_request_types
                        and ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT not in self.config.ignore_request_types
                    ):
                        res = await self.robotic_arm.challenge_image_label_select(challenge_type)
                        self._epic_solve_attempts = 0
                        return res
                case ChallengeTypeEnum.IMAGE_DRAG_SINGLE:
                    if (
                        RequestType.IMAGE_DRAG_DROP not in self.config.ignore_request_types
                        and ChallengeTypeEnum.IMAGE_DRAG_SINGLE not in self.config.ignore_request_types
                    ):
                        res = await self.robotic_arm.challenge_image_drag_drop(challenge_type)
                        self._epic_solve_attempts = 0
                        return res
                case ChallengeTypeEnum.IMAGE_DRAG_MULTI:
                    if (
                        RequestType.IMAGE_DRAG_DROP not in self.config.ignore_request_types
                        and ChallengeTypeEnum.IMAGE_DRAG_MULTI not in self.config.ignore_request_types
                    ):
                        res = await self.robotic_arm.challenge_image_drag_drop(challenge_type)
                        self._epic_solve_attempts = 0
                        return res
                case _:
                    logger.warning(f"Unknown types of challenges: {challenge_type}")

            await self.page.wait_for_timeout(2000)
            await self.robotic_arm.refresh_challenge()
            res = await self._solve_captcha()
            self._epic_solve_attempts = 0
            return res
        except Exception as err:
            logger.warning(f"ChallengeException - type={type_str} err={err!r}")
            await self.page.wait_for_timeout(3000)
            await self.robotic_arm.refresh_challenge()
            res = await self._solve_captcha()
            self._epic_solve_attempts = 0
            return res

    AgentV._solve_captcha = safe_solve_captcha
    RoboticArm._epic_safety_patch = True


def apply_hcaptcha_drag_patch() -> None:
    _apply_empty_checkcaptcha_patch()
    _patch_robotic_arm_safety()

    if not getattr(RoboticArm.challenge_image_label_select, "_epic_point_bounds_patch", False):

        async def patched_challenge_image_label_select(self: RoboticArm, job_type: Any):
            frame_challenge = await self.get_challenge_frame_locator()
            if frame_challenge is None:
                logger.warning("Challenge frame not found before crumb loop; refreshing")
                await self.refresh_challenge()
                return

            crumb_count = await self.check_crumb_count()
            cache_key = self.config.create_cache_key(self.captcha_payload)

            for cid in range(crumb_count):
                await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)
                frame_challenge = await self.get_challenge_frame_locator()
                if frame_challenge is None:
                    logger.warning("Challenge frame lost at crumb {}/{}", cid + 1, crumb_count)
                    await self.refresh_challenge()
                    return

                raw, projection = await self._capture_spatial_mapping(
                    frame_challenge, cache_key, cid
                )
                challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
                challenge_bbox = await challenge_view.bounding_box()
                base_prompt = self._match_user_prompt(job_type)
                image_grid_bounds = (
                    _detect_clickable_grid_bounds(raw)
                    if _is_count_selection_question(base_prompt)
                    else None
                )
                clickable_bounds = None
                if image_grid_bounds is not None and challenge_bbox is not None:
                    clickable_bounds = _map_image_bounds_to_page(
                        image_grid_bounds, challenge_screenshot=raw, challenge_bbox=challenge_bbox
                    )

                user_prompt = _build_point_prompt(
                    base_prompt, challenge_bbox=challenge_bbox, clickable_bounds=clickable_bounds
                )
                response = await self._spatial_point_reasoner(
                    challenge_screenshot=raw,
                    grid_divisions=projection,
                    auxiliary_information=user_prompt,
                )
                logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')

                # The model answers in grid-image pixels, so convert them back into page
                # coordinates first, then only guard against residual out-of-bounds values.
                # Points are never dropped: dropping one makes the crumb short of the
                # required selection count and guarantees a failed challenge.
                if challenge_bbox is not None and getattr(response, "points", None):
                    bx = float(challenge_bbox["x"])
                    by = float(challenge_bbox["y"])
                    bw = float(challenge_bbox["width"])
                    bh = float(challenge_bbox["height"])
                    x_min, y_min = bx, by
                    x_max, y_max = bx + bw, by + bh

                    response.points = _map_model_points_to_page(
                        response.points,
                        challenge_bbox=challenge_bbox,
                        geometry=_grid_plot_geometry(
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            x_lines=self.config.coordinate_grid.x_line_space_num,
                            y_lines=self.config.coordinate_grid.y_line_space_num,
                        ),
                    )

                    valid_points = []
                    for pt in response.points:
                        px = float(pt.x)
                        py = float(pt.y)
                        clamped_x = max(x_min + 4.0, min(x_max - 4.0, px))
                        clamped_y = max(y_min + 4.0, min(y_max - 4.0, py))
                        if clamped_x != px or clamped_y != py:
                            logger.info(
                                "Clamped point ({:.1f}, {:.1f}) -> ({:.1f}, {:.1f}) to challenge edge",
                                px,
                                py,
                                clamped_x,
                                clamped_y,
                            )
                        pt.x = clamped_x
                        pt.y = clamped_y
                        valid_points.append(pt)
                    response.points = valid_points

                # Only reject if no points could be extracted or clamped
                if not getattr(response, "points", None):
                    logger.warning("No valid point answers found; clicking challenge center")
                    if challenge_bbox is not None:
                        center_x = float(challenge_bbox["x"]) + float(challenge_bbox["width"]) / 2
                        center_y = float(challenge_bbox["y"]) + float(challenge_bbox["height"]) / 2
                        await self.page.mouse.click(center_x, center_y, delay=180)
                    await self.refresh_challenge()
                    return

                self._spatial_point_reasoner.cache_response(
                    path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
                )
                for point in response.points:
                    await self.page.mouse.click(point.x, point.y, delay=180)
                    await self.page.wait_for_timeout(500)

                with suppress(PlaywrightTimeoutError):
                    submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                    await self.click_by_mouse(submit_btn)

        patched_challenge_image_label_select._epic_point_bounds_patch = True
        RoboticArm.challenge_image_label_select = patched_challenge_image_label_select

    if getattr(RoboticArm.challenge_image_drag_drop, "_epic_drag_source_patch", False):
        return

    async def patched_challenge_image_drag_drop(self: RoboticArm, job_type: Any):
        frame_challenge = await self.get_challenge_frame_locator()
        if frame_challenge is None:
            logger.warning("Challenge frame not found before drag loop; refreshing")
            await self.refresh_challenge()
            return

        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)
            frame_challenge = await self.get_challenge_frame_locator()
            if frame_challenge is None:
                logger.warning("Challenge frame lost at crumb {}/{}", cid + 1, crumb_count)
                await self.refresh_challenge()
                return

            raw, projection = await self._capture_spatial_mapping(frame_challenge, cache_key, cid)
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            challenge_bbox = await challenge_view.bounding_box()
            user_prompt = self._match_user_prompt(job_type)
            paths = _resolve_line_path(
                captcha_payload=self.captcha_payload,
                crumb_id=cid,
                challenge_screenshot=raw,
                challenge_bbox=challenge_bbox,
            )
            if paths is None:
                paths = await _resolve_outline_paths(
                    captcha_payload=self.captcha_payload,
                    crumb_id=cid,
                    challenge_screenshot=raw,
                    challenge_bbox=challenge_bbox,
                )
            if paths is None:
                source_points = _payload_source_points(
                    captcha_payload=self.captcha_payload,
                    crumb_id=cid,
                    challenge_screenshot=raw,
                    challenge_bbox=challenge_bbox,
                )
                response = await self._spatial_path_reasoner(
                    challenge_screenshot=raw,
                    grid_divisions=projection,
                    auxiliary_information=_build_drag_prompt(
                        user_prompt, source_points=source_points
                    ),
                )
                logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
                self._spatial_path_reasoner.cache_response(
                    path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
                )
                paths = _correct_drag_source_points(
                    response.paths,
                    captcha_payload=self.captcha_payload,
                    crumb_id=cid,
                    challenge_screenshot=raw,
                    challenge_bbox=challenge_bbox,
                )

            for path in paths:
                await self._perform_drag_drop(path)

            with suppress(PlaywrightTimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                await self.click_by_mouse(submit_btn)

    patched_challenge_image_drag_drop._epic_drag_source_patch = True
    RoboticArm.challenge_image_drag_drop = patched_challenge_image_drag_drop
    logger.info("hCaptcha local solvers, point guards, and response patches loaded")
