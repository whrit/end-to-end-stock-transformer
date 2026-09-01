"""F084-F100 and reusable batched clean-path geometry metrics."""

from typing import Dict, Sequence

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_session_returns,
    path_efficiency_from_returns,
    safe_divide,
    safe_nanmean,
    safe_nanquantile,
)


GEOMETRY_KEYS = (
    "path_efficiency", "arc_chord_ratio", "signed_chord_area",
    "abs_chord_area", "max_chord_deviation", "rms_chord_deviation",
    "turning_rate", "mean_turn_angle", "p90_turn_angle",
    "mean_abs_curvature", "p90_abs_curvature", "curvature_energy",
)


def trapezoid(values: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Integrate with the NumPy 2.x name or the NumPy 1.x fallback."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = getattr(np, "trapz")
    return implementation(values, x, axis=axis)


def geometry_summaries(
    returns: np.ndarray, min_obs: int, chunk_size: int, eps: float
) -> Dict[str, np.ndarray]:
    stocks = returns.shape[1]
    result = {
        key: np.full(stocks, np.nan, dtype="float64") for key in GEOMETRY_KEYS
    }
    result["path_efficiency"] = path_efficiency_from_returns(returns)
    for chunk_start in range(0, stocks, chunk_size):
        chunk_end = min(stocks, chunk_start + chunk_size)
        chunk = returns[:, chunk_start:chunk_end]
        counts = np.isfinite(chunk).sum(axis=0)
        for count in np.unique(counts[counts >= min_obs]):
            local_columns = np.flatnonzero(counts == count)
            selected = chunk[:, local_columns]
            mask = np.isfinite(selected)
            packed = selected.T[mask.T].reshape(len(local_columns), int(count)).T
            y = np.cumsum(packed, axis=0)
            y_mean = y.mean(axis=0)
            y_std = y.std(axis=0, ddof=0)
            usable = np.isfinite(y_std) & (y_std > eps)
            x = np.linspace(0.0, 1.0, int(count), dtype="float64")[:, None]
            standardized = safe_divide(y - y_mean[None, :], y_std[None, :])
            dx = np.diff(x[:, 0])[:, None]
            dy = np.diff(standardized, axis=0)
            segment_length = np.sqrt(dx * dx + dy * dy)
            arc_length = segment_length.sum(axis=0)
            chord_length = np.sqrt(
                1.0 + (standardized[-1] - standardized[0]) ** 2
            )
            chord = standardized[0][None, :] + (
                standardized[-1] - standardized[0]
            )[None, :] * x
            deviation = standardized - chord
            with np.errstate(invalid="ignore"):
                signed_area = trapezoid(deviation, x[:, 0], axis=0)
                abs_area = trapezoid(np.abs(deviation), x[:, 0], axis=0)
            max_deviation = np.max(np.abs(deviation), axis=0)
            rms_deviation = np.sqrt(np.mean(deviation ** 2, axis=0))

            turning = ((packed[1:-1] * packed[2:]) < 0).sum(axis=0)
            turning_rate = turning / float(max(int(count) - 2, 1))
            first = np.stack((np.broadcast_to(dx[:-1], dy[:-1].shape), dy[:-1]), axis=2)
            second = np.stack((np.broadcast_to(dx[1:], dy[1:].shape), dy[1:]), axis=2)
            dot = (first * second).sum(axis=2)
            norm_product = np.sqrt((first ** 2).sum(axis=2) * (second ** 2).sum(axis=2))
            cosine = np.clip(safe_divide(dot, norm_product), -1.0, 1.0)
            angles = np.arccos(cosine)

            ab_x = dx[:-1]
            ab_y = dy[:-1]
            bc_x = dx[1:]
            bc_y = dy[1:]
            ac_x = ab_x + bc_x
            ac_y = ab_y + bc_y
            cross = np.abs(ab_x * ac_y - ab_y * ac_x)
            length_ab = np.sqrt(ab_x ** 2 + ab_y ** 2)
            length_bc = np.sqrt(bc_x ** 2 + bc_y ** 2)
            length_ac = np.sqrt(ac_x ** 2 + ac_y ** 2)
            curvature = safe_divide(
                2.0 * cross, length_ab * length_bc * length_ac
            )
            metrics = {
                "arc_chord_ratio": safe_divide(arc_length, chord_length),
                "signed_chord_area": signed_area,
                "abs_chord_area": abs_area,
                "max_chord_deviation": max_deviation,
                "rms_chord_deviation": rms_deviation,
                "turning_rate": turning_rate,
                "mean_turn_angle": safe_nanmean(angles, axis=0),
                "p90_turn_angle": safe_nanquantile(angles, 0.90, axis=0),
                "mean_abs_curvature": safe_nanmean(np.abs(curvature), axis=0),
                "p90_abs_curvature": safe_nanquantile(np.abs(curvature), 0.90, axis=0),
                "curvature_energy": safe_nanmean(curvature ** 2, axis=0),
            }
            targets = chunk_start + local_columns
            for key, values in metrics.items():
                cleaned = np.asarray(values, dtype="float64")
                cleaned[~usable] = np.nan
                result[key][targets] = cleaned
    return result


def turning_rate(values: np.ndarray, min_obs: int) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    output = np.full(values.shape[1], np.nan, dtype="float64")
    for column_group_count in np.unique(count[count >= min_obs]):
        columns = np.flatnonzero(count == column_group_count)
        selected = values[:, columns]
        packed = selected.T[np.isfinite(selected).T].reshape(
            len(columns), int(column_group_count)
        ).T
        changes = ((packed[1:-1] * packed[2:]) < 0).sum(axis=0)
        output[columns] = changes / float(max(int(column_group_count) - 2, 1))
    return output


def _standardized_path_length(
    values: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    columns: Sequence[int],
    eps: float,
) -> np.ndarray:
    output = np.full(len(columns), np.nan, dtype="float64")
    selected = values[:, columns]
    counts = np.isfinite(selected).sum(axis=0)
    for count in np.unique(counts[counts >= 2]):
        local = np.flatnonzero(counts == count)
        group = selected[:, local]
        packed = group.T[np.isfinite(group).T].reshape(len(local), int(count)).T
        y = np.cumsum(packed, axis=0)
        std = source_std[np.asarray(columns)[local]]
        mean = source_mean[np.asarray(columns)[local]]
        usable = np.isfinite(std) & (std > eps)
        standardized = safe_divide(y - mean[None, :], std[None, :])
        dx = 1.0 / float(int(count) - 1)
        length = np.sqrt(dx * dx + np.diff(standardized, axis=0) ** 2).sum(axis=0)
        length[~usable] = np.nan
        output[local] = length
    return output


def roughness_ratios(
    returns: np.ndarray,
    coarser: Sequence[np.ndarray],
    min_obs: int,
    chunk_size: int,
    eps: float,
):
    stocks = returns.shape[1]
    outputs = [np.full(stocks, np.nan, dtype="float64") for _ in coarser]
    for chunk_start in range(0, stocks, chunk_size):
        chunk_end = min(stocks, chunk_start + chunk_size)
        columns = np.arange(chunk_start, chunk_end)
        selected = returns[:, columns]
        counts = np.isfinite(selected).sum(axis=0)
        for count in np.unique(counts[counts >= min_obs]):
            local = np.flatnonzero(counts == count)
            global_columns = columns[local]
            group = selected[:, local]
            packed = group.T[np.isfinite(group).T].reshape(len(local), int(count)).T
            y = np.cumsum(packed, axis=0)
            mean = np.full(stocks, np.nan, dtype="float64")
            std = np.full(stocks, np.nan, dtype="float64")
            mean[global_columns] = y.mean(axis=0)
            std[global_columns] = y.std(axis=0, ddof=0)
            x_step = 1.0 / float(int(count) - 1)
            standardized = safe_divide(
                y - mean[global_columns][None, :], std[global_columns][None, :]
            )
            length1 = np.sqrt(
                x_step * x_step + np.diff(standardized, axis=0) ** 2
            ).sum(axis=0)
            length1[std[global_columns] <= eps] = np.nan
            for output, coarse in zip(outputs, coarser):
                coarse_length = _standardized_path_length(
                    coarse, mean, std, global_columns, eps
                )
                output[global_columns] = safe_divide(coarse_length, length1)
    return outputs


def compute(context: DayContext):
    base = cached_geometry(context, "1m", context.returns)
    ret5 = cached_session_returns(context, 5)
    ret15 = cached_session_returns(context, 15)
    ret30 = cached_session_returns(context, 30)
    rough5, rough15 = roughness_ratios(
        context.returns,
        (ret5, ret15),
        context.config.min_geometry_obs,
        context.config.stock_chunk_size,
        context.config.eps,
    )
    return {
        "path_efficiency": base["path_efficiency"],
        "arc_chord_ratio": base["arc_chord_ratio"],
        "signed_chord_area": base["signed_chord_area"],
        "abs_chord_area": base["abs_chord_area"],
        "max_chord_deviation": base["max_chord_deviation"],
        "rms_chord_deviation": base["rms_chord_deviation"],
        "turning_rate_1m": base["turning_rate"],
        "turning_rate_5m": turning_rate(ret5, context.config.min_geometry_obs),
        "turning_rate_15m": turning_rate(ret15, context.config.min_geometry_obs),
        "turning_rate_30m": turning_rate(ret30, context.config.min_geometry_obs),
        "mean_turn_angle": base["mean_turn_angle"],
        "p90_turn_angle": base["p90_turn_angle"],
        "mean_abs_curvature": base["mean_abs_curvature"],
        "p90_abs_curvature": base["p90_abs_curvature"],
        "curvature_energy": base["curvature_energy"],
        "roughness_5m_1m": rough5,
        "roughness_15m_1m": rough15,
    }


def cached_geometry(
    context: DayContext, cache_name: str, returns: np.ndarray
) -> Dict[str, np.ndarray]:
    key = "geometry:{}".format(cache_name)
    if key not in context.cache:
        context.cache[key] = geometry_summaries(
            returns,
            context.config.min_geometry_obs,
            context.config.stock_chunk_size,
            context.config.eps,
        )
    return context.cache[key]
