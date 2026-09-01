"""F229-F240: cached extended spectra and multiscale curvature."""

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_nonnegative_field,
    cached_session_returns,
)
from market_guided_stock_alpha.features.intraday.geometry import cached_geometry
from market_guided_stock_alpha.features.intraday.spectral import cached_fft


def compute(context: DayContext):
    volume = cached_nonnegative_field(context, "volume")
    log_volume = np.where(np.isfinite(volume), np.log1p(volume), np.nan)
    ret_fft = cached_fft(context, "ret", context.returns)
    absret_fft = cached_fft(context, "absret", context.abs_returns)
    volume_fft = cached_fft(context, "volume", log_volume)
    result = {
        "ret_fft_low_high_logratio": ret_fft["low_high_logratio"],
        "absret_fft_low_high_logratio": absret_fft["low_high_logratio"],
        "volume_fft_low_high_logratio": volume_fft["low_high_logratio"],
        "ret_spectral_slope": ret_fft["spectral_slope"],
        "absret_spectral_slope": absret_fft["spectral_slope"],
        "volume_spectral_slope": volume_fft["spectral_slope"],
    }
    for size in (5, 15, 30):
        geometry = cached_geometry(
            context,
            "{}m".format(size),
            cached_session_returns(context, size),
        )
        result["mean_abs_curvature_{}m".format(size)] = geometry[
            "mean_abs_curvature"
        ]
        result["curvature_energy_{}m".format(size)] = geometry[
            "curvature_energy"
        ]
    return result
