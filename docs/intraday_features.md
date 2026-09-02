# Intraday Price-Volume Features (248D)

## Overview

Intraday Price-Volume Features is a fixed 248-dimensional daily representation constructed from one stock's minute-level `open`, `high`, `low`, `close`, `volume`, and `amount` observations. Each eligible stock-day produces one 248D vector; minute bars are not passed directly to the forecasting model.

The representation was designed, organized, and implemented in this project. It uses standard statistical, financial, signal-processing, and geometric tools—including skewness, realized variation, concentration measures, Fourier transforms, entropy, path length, and curvature—but does not claim those mathematical tools themselves as original. The project contribution is the feature selection and combination, the session-aware multiscale construction, the frequency and path-shape representations, and the complete frozen F001–F248 system.

The intraday representation is combined with daily stock features in Stage 2. Alpha158 follows the public [Microsoft Qlib](https://github.com/microsoft/qlib) definition and is not redefined here.

The canonical implementation is the source of truth:

- registry and order: `src/market_guided_stock_alpha/features/intraday/registry.py`;
- shared contracts: `src/market_guided_stock_alpha/features/intraday/common.py`;
- computation orchestration: `src/market_guided_stock_alpha/features/intraday/compute.py`.

## Data and Session Contract

### Input grid

The canonical session contains exactly 240 ordered trading bars:

- morning session: positions 1–120;
- afternoon session: positions 121–240.

For stock $i$, date $d$, and normal trading position $m$, the input fields are

$$
O_m,\ H_m,\ L_m,\ C_m,\ V_m,\ A_m,
$$

where $A_m$ is the minute `amount`. Volume and amount are per-bar increments, not cumulative session totals. The six fields must be aligned on the same 240-position grid. The implementation does not interpolate, forward-fill, backward-fill, or fill across dates.

### Structural-inactive minutes

Structural inactivity is detected from the same-date cross-section rather than from a hard-coded clock time. Let $\mathcal U_{d,m}$ be stocks for which both volume and amount are observed at position $m$. Define

$$
z_{d,m}=
\frac{
\lvert{i\in\mathcal U_{d,m}:V_{i,d,m}=0,\ A_{i,d,m}=0}\rvert
}{
\lvert\mathcal U_{d,m}\rvert
}.
$$

Position $m$ is structural inactive when $z_{d,m}\ge\tau_z$, with default $\tau_z=0.80$. Such rows are excluded from price, return, event, volume, spectral, and geometry calculations for every stock. Fixed time-of-day and 60-minute block boundaries remain attached to their original 240-position grid; they are not shifted to replace excluded rows.

### Coverage and eligibility

Coverage is computed from the unmodified close grid:

$$
\mathrm{coverage}_{i,d}=
\frac{
\lvert{m:C_{i,d,m}\text{ is finite}}\rvert
}{240}.
$$

The default eligibility threshold is 0.90. Factors for an ineligible stock-day are returned as missing, and the daily table writer emits eligible rows only. Structural-inactive rows do not reduce this close-based coverage measure.

### Minute-return convention

After structural-inactive rows and invalid prices are excluded, valid positions keep their original order. If $m_1,\ldots,m_M$ are the remaining valid close positions, then

$$
r_{m_1}=\log\frac{C_{m_1}}{O_{m_1}},
\qquad
r_{m_k}=\log\frac{C_{m_k}}{C_{m_{k-1}}},\quad k\ge2.
$$

Thus a missing individual bar is skipped rather than imputed, and the next valid close is compared with the last valid close. All prices used in a logarithm must be positive.

### Session-aware aggregation

The 5m, 15m, and 30m series are non-overlapping block aggregates. Morning and afternoon are reshaped separately, so no block crosses the midday break:

| Scale | Morning blocks | Afternoon blocks | Total blocks |
|---|---:|---:|---:|
| 5m | 24 | 24 | 48 |
| 15m | 8 | 8 | 16 |
| 30m | 4 | 4 | 8 |

Block return, block volume, and block amount are sums of their finite 1m observations. A block with no finite observation is missing. The four fixed position blocks used elsewhere are

$$
B_1=1{:}60,\quad B_2=61{:}120,\quad
B_3=121{:}180,\quad B_4=181{:}240.
$$

### Invalid observations and output

Default minimum observations are 10 paired points for 1m correlations, 4 blocks for multiscale correlations, 20 points for FFT summaries, and 5 points for standardized geometry. Non-positive denominators, insufficient samples, zero variance, zero usable spectral power, degenerate geometry, and other undefined cases return `NaN`. Positive and negative infinity are converted to `NaN`. The frozen numerical output is `float32` and preserves registry order. This feature layer emits raw stock-day values; it does not perform cross-sectional normalization, winsorization, label-based selection, or any transformation fitted across dates.

## Common Notation

| Symbol | Meaning |
|---|---|
| $r_m$ | cleaned minute log return under the convention above |
| $a_m=\lvert r_m\rvert$ | absolute minute return |
| $O_m,H_m,L_m,C_m$ | minute open, high, low, and close |
| $V_m,A_m$ | non-negative minute volume and amount |
| $RV=\sum_m r_m^2$ | 1m realized variance |
| $P_m=A_m/V_m$ | minute VWAP, defined only when both inputs are positive |
| $P_{day}=\sum_m A_m/\sum_m V_m$ | daily VWAP over jointly valid observations |
| $B_j$ | one of the four fixed 60-position blocks |
| session | morning positions 1–120 or afternoon positions 121–240 |
| block | one non-overlapping 5m, 15m, or 30m session-local aggregate |
| $\varepsilon$ | numerical stabilizer, default $10^{-12}$, used only where stated |

For non-negative weights $w_j$, let $q_j=w_j/\sum_k w_k$. The common concentration summaries are

$$
\mathrm{HHI}(w)=\sum_jq_j^2,
\qquad
\mathrm{Entropy}(w)=
-\frac{\sum_{j:q_j>0}q_j\log q_j}{\log K},
$$

where $K$ is the number of finite input positions, including finite zero-weight positions, and must exceed one. Unless specified otherwise, standard deviations use the corrected sample denominator; geometry standardization uses population standard deviation, matching the implementation.

## Feature Family Overview

Categories and counts below are generated from the frozen registry, not inferred from prose.

| Category | Factor IDs | Count | Purpose |
|---|---|---:|---|
| `distribution` | F001–F012 | 12 | return shape, signs, runs, and tail concentration |
| `volatility` | F013–F028; F197–F212 | 32 | realized variation, jumps, dependence, concentration, ranges, and trend |
| `liquidity` | F029–F040 | 12 | volume/amount concentration, illiquidity, and VWAP dispersion |
| `price_volume` | F041–F052; F213–F228 | 28 | contemporaneous and lead-lag coupling, imbalance, and weighted returns |
| `time_of_day` | F053–F068 | 16 | open, morning, afternoon, and close-window structure |
| `spectral` | F069–F083; F229–F234 | 21 | frequency allocation, entropy, dominant period, ratios, and slopes |
| `geometry` | F084–F100; F235–F240 | 23 | cumulative-return path shape, turning, curvature, and roughness |
| `position_blocks` | F101–F132; F241–F248 | 40 | four fixed 60-position local summaries |
| `multiscale` | F133–F142; F177–F196 | 30 | 5m/15m/30m dependence, shape, and concentration |
| `cross_session` | F143–F148; F161–F176 | 22 | first/last and morning/afternoon contrasts and session shapes |
| `event_timing` | F149–F160 | 12 | extreme values and their normalized intraday positions |
| **Total** | **F001–F248** | **248** | frozen stock-day representation |

## Detailed Definitions

### Distribution and tails — F001–F012

`ret_skew` and `ret_kurt` are corrected sample skewness and Fisher corrected excess kurtosis. Sign and run features are

$$
\frac{1}{M}\sum_m\mathbf1(r_m>0),\qquad
\frac{1}{M}\sum_m\mathbf1(|r_m|\le10^{-12}),
$$

the fraction of adjacent non-zero return pairs whose signs differ, and the longest strictly positive/negative run divided by $M$. `absret_q90`, `absret_q95`, and `absret_q99` are the corresponding quantiles of $|r|$. Tail asymmetry is

$$
\frac{Q_{.95}(r)-|Q_{.05}(r)|}
{Q_{.95}(r)+|Q_{.05}(r)|},
$$

and `extreme_rv_share` is the share of $RV$ contributed by the largest $\lceil0.05M\rceil$ absolute returns.

### Volatility, jumps, and dependence — F013–F028

The base volatility summaries are

$$
BV=\frac{\pi}{2}\sum_{m=2}^{M}|r_m||r_{m-1}|,
\quad
\mathrm{jump\_share}=\frac{\max(RV-BV,0)}{RV},
$$

$$
\mathrm{downside\_rv}=\sum_{r_m<0}r_m^2,
\quad
\mathrm{RQ}=\frac{M}{3}\sum_mr_m^4.
$$

Minute-bar Parkinson and Rogers–Satchell quantities are summed across valid bars:

$$
\sum_m\frac{[\log(H_m/L_m)]^2}{4\log2},
$$

$$
\sum_m\left[
\log(H_m/O_m)\log(H_m/C_m)+
\log(L_m/O_m)\log(L_m/C_m)
\right].
$$

Maximum drawdown and drawup use the minute-close path. F023–F026 are lag-1 and lag-5 correlations of $r$ and $|r|$. F027–F028 compare squared 5m/15m block returns with 1m $RV$.

### Liquidity and concentration — F029–F040

The family contains volume HHI, normalized volume entropy, volume Gini, amount HHI, maximum-minute and top-5%-minute volume shares, zero-volume ratio, and volume coefficient of variation. The Amihud-like observations are $|r_m|/A_m$ for $A_m>0$, summarized by their mean and 90th percentile.

For positive minute volume and amount, define $P_m=A_m/V_m$. `vwap_path_absdev` and `minute_vwap_dispersion` are respectively

$$
\mathrm{Mean}\left|\log(P_m/P_{day})\right|,
\qquad
\mathrm{Std}\left[\log(P_m/P_{day})\right].
$$

### Price-volume coupling — F041–F052 and F213–F228

Let $v_m=\log(1+V_m)$, $u_m=\log(1+A_m)$, and $\Delta v_m=v_m-v_{m-1}$. F041–F048 contain contemporaneous correlations between returns/absolute returns and $v$ or $\Delta v$, plus one-step lead-lag correlations in both directions. Signed imbalances are

$$
\frac{\sum_m\mathrm{sign}(r_m)V_m}{\sum_mV_m},
\qquad
\frac{\sum_m\mathrm{sign}(r_m)A_m}{\sum_mA_m}.
$$

F051–F052 compare mean returns in the top and bottom volume quintiles.

F213–F218 repeat the contemporaneous and bidirectional lead-lag construction with log amount $u_m$. F219–F221 locate close, open, and daily VWAP relative to the day's price range:

$$
\log(C_{last}/P_{day}),\quad
\log(O_{first}/P_{day}),\quad
\frac{P_{day}-L_{day}}{H_{day}-L_{day}}.
$$

F222–F224 are volume-weighted return, amount-weighted return, and the correlation between $|r_m|$ and $|\log(P_m/P_{day})|$. F225–F226 weight signed absolute returns by volume or amount. F227–F228 are the fractions of volume at positive- and negative-return positions.

### Time of day — F053–F068

The fixed windows are `open30` = 1–30, `midmorning` = 31–120, `midafternoon` = 121–210, and `close30` = 211–240. For each window $S$, the implementation records

$$
R(S)=\sum_{m\in S}r_m,
\quad
\mathrm{RVShare}(S)=\frac{\sum_{m\in S}r_m^2}{RV},
$$

$$
\mathrm{VShare}(S)=\frac{\sum_{m\in S}V_m}{\sum_mV_m},
\quad
\mathrm{Range}(S)=\log\frac{\max_{m\in S}H_m}{\min_{m\in S}L_m}.
$$

Structural-inactive rows remain absent, but window boundaries do not move.

### Frequency-domain representation — F069–F083 and F229–F234

The spectral summaries are evaluated independently on three cleaned, order-preserving sequences:

1. minute return $r_m$;
2. absolute minute return $|r_m|$;
3. log volume $\log(1+V_m)$.

Missing and structural-inactive positions are not filled. For each stock, the finite observations are packed in their original order. Let the resulting sequence have length $M$, center it as $\tilde x_m=x_m-\bar x$, and compute the real FFT:

$$
X_k=\sum_{m=0}^{M-1}\tilde x_m e^{-2\pi i km/M}.
$$

The DC term $k=0$ is removed. For the remaining bins,

$$
P_k=|X_k|^2,
\qquad
p_k=\frac{P_k}{\sum_{j>0}P_j},
\qquad
T_k=\frac{M}{k},
\qquad
f_k=\frac{k}{M}.
$$

The five base summaries for each sequence are

$$
\mathrm{low\_share}=\sum_{k:T_k\ge60}p_k,
\qquad
\mathrm{high\_share}=\sum_{k:T_k<15}p_k,
$$

$$
\mathrm{spectral\_entropy}=
-\frac{\sum_kp_k\log p_k}{\log K},
\qquad
\mathrm{centroid}=\sum_kp_kf_k,
$$

$$
\mathrm{dominant\_period}=T_{\arg\max_k P_k},
$$

where $K$ is the total number of non-DC real-FFT bins. Zero-power bins contribute zero to the entropy numerator. Constant or near-constant sequences with total non-DC power at or below $\varepsilon$ return missing spectral summaries.

The three enhanced low/high ratios are

$$
\log\frac{P_{low}+\varepsilon}{P_{high}+\varepsilon},
$$

computed only when both low- and high-frequency bin sets exist and total usable power is positive. Spectral slope is the OLS coefficient $b$ in

$$
\log(P_k+\varepsilon)=a+b\log f_k,
$$

using all positive non-DC frequencies and requiring at least three bins.

### Geometry and path shape — F084–F100 and F235–F240

For the finite cleaned returns, define the cumulative path

$$
y_m=\sum_{j\le m}r_j,
\qquad
x_m=\frac{m-1}{M-1},
$$

and its standardized ordinate

$$
\tilde y_m=\frac{y_m-\bar y}{s_y},
\qquad s_y=\mathrm{Std}_{\mathrm{ddof}=0}(y).
$$

The 2D path is $\gamma_m=(x_m,\tilde y_m)$. When $s_y\le\varepsilon$, standardized geometry is undefined.

`path_efficiency` uses the unstandardized cumulative path:

$$
\frac{|y_M-y_1|}{\sum_{m=2}^{M}|y_m-y_{m-1}|}.
$$

For standardized geometry, arc length and chord length are

$$
L=\sum_{m=2}^{M}\|\gamma_m-\gamma_{m-1}\|_2,
\qquad
D=\|\gamma_M-\gamma_1\|_2,
$$

and `arc_chord_ratio` is $L/D$. Interpolate the endpoint chord by

$$
\ell(x)=\tilde y_1+(\tilde y_M-\tilde y_1)
\frac{x-x_1}{x_M-x_1},
\qquad
d_m=\tilde y_m-\ell(x_m).
$$

The signed and absolute chord areas are trapezoidal integrals of $d(x)$ and $|d(x)|$. Maximum and RMS chord deviation are $\max_m|d_m|$ and $\sqrt{M^{-1}\sum_md_m^2}$.

At an internal path point, a strict turn occurs when the adjacent non-zero path increments have opposite signs. Turning rate divides the number of such changes by $M-2$. Directional change angle is

$$
\theta_m=\arccos
\frac{u_m\cdot v_m}{\|u_m\|\|v_m\|},
$$

where $u_m=\gamma_m-\gamma_{m-1}$ and $v_m=\gamma_{m+1}-\gamma_m$. F094–F095 are its mean and 90th percentile.

For three consecutive standardized path points $A,B,C$, let $a,b,c$ be their side lengths and $\Delta$ the triangle area. Discrete curvature is

$$
\kappa=\frac{4\Delta}{abc}.
$$

F096–F098 are mean absolute curvature, its 90th percentile, and curvature energy $\mathrm{Mean}(\kappa^2)$. Degenerate triples return missing rather than infinity.

For `roughness_5m_1m` and `roughness_15m_1m`, the coarse cumulative paths reuse the 1m path's mean and standard deviation, remap coarse time to $[0,1]$, and divide coarse arc length by 1m arc length. F235–F240 instead recompute mean absolute curvature and curvature energy on separately standardized 5m, 15m, and 30m paths.

### Position blocks — F101–F132 and F241–F248

Each fixed 60-position block records return, share of full-day RV, share of full-day volume, share of full-day amount, log high/low range, path efficiency, absolute chord area, and curvature energy. F241–F244 add within-block downside-RV share; F245–F248 add within-block signed-volume imbalance.

The four blocks follow raw session positions. Missing or structural-inactive observations do not cause boundaries to shift.

### Multiscale summaries — F133–F142 and F177–F196

F133–F142 contain the 30m/1m RV ratio; lag-1 autocorrelation of block returns and absolute block returns at 5m, 15m, and 30m; and path efficiency at those three scales.

F177–F196 recompute corrected return skewness, corrected excess kurtosis, strict sign-change rate, RV HHI, RV entropy, and volume HHI at 5m/15m/30m. Volume entropy is retained at 5m and 15m only. The weight for RV concentration is squared block return; the weight for volume concentration is block volume.

The multiscale system therefore covers 1m, 5m, 15m, and 30m views without joining a morning block to an afternoon block. Related scale-aware features also appear in F027–F028, F090–F100, F210–F212, and F235–F240.

### Cross-session contrasts — F143–F148

The first-versus-last 60-minute contrasts are return difference, RV log-ratio, and volume-share difference. The morning-versus-afternoon contrasts use the corresponding two blocks on each side. RV ratios use

$$
\log\frac{RV_L+\varepsilon}{RV_R+\varepsilon},
$$

but return missing when both sides are zero.

### Event timing — F149–F160

F149–F151 are the maximum minute return, minimum minute return (the negative sign is retained), and maximum absolute minute return. F152–F160 record the first normalized positions of those events, maximum volume, maximum amount, intraday high, intraday low, maximum-drawdown end, and maximum-drawup end.

Structural-inactive rows are removed before event search. If $M_a$ globally active rows remain, a selected row index $j\in\{0,\ldots,M_a-1\}$ is reported as $j/(M_a-1)$. Individual-stock missing observations remain missing within this active grid and are not replaced. Ties use the first occurrence.

### Morning and afternoon shapes — F161–F176

Morning and afternoon separately recompute corrected return skewness, corrected excess kurtosis, downside-RV share, path efficiency, absolute chord area, curvature energy, signed-volume imbalance, and volume HHI. The registry groups this family under `cross_session` because it supplies the two session-local components used to compare intraday shape.

### Full-day concentration, range, and trend — F197–F212

F197–F200 summarize 1m squared-return concentration: HHI, normalized entropy, maximum-minute RV share, and the share contributed by the largest $\lceil0.01M\rceil$ absolute returns. F201–F203 are median absolute return, return median absolute deviation, and return interquartile range.

Let $g_m=\log(H_m/L_m)$ on valid positive bars. F204–F206 are its mean, corrected sample standard deviation, and HHI of $g_m^2$. For trend features, regress cumulative return on normalized valid position:

$$
y_m=\alpha+\beta x_m+e_m.
$$

F207–F209 are $\beta$, $R^2$, and $\sqrt{M^{-1}\sum_me_m^2}$. F210–F212 are maximum absolute block returns at 5m, 15m, and 30m.

## Design Inspiration

The frequency-domain direction was inspired by *Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks*, particularly its use of frequency decomposition to study price-volume structure. The finalized factors here are this project's own FFT-based spectral summaries, not copies of that work's factor definitions.

The path-shape direction was inspired by *Large-Scale Asset Selection via Metric Dependence with Enriched High Frequency Information*, which treats high-frequency intraday information as a curve or functional object. The finalized arc/chord, chord-deviation, turning-angle, discrete-curvature, curvature-energy, and roughness definitions are those implemented in this project.

## Complete F001–F248 Mapping

The implementation column is relative to `src/market_guided_stock_alpha/features/`. Each ID, name, and category below is generated from the canonical registry. Formula families are defined once in the referenced section rather than repeated for every factor.

| ID | Name | Category | Definition section | Implementation |
|---|---|---|---|---|
| F001 | `ret_skew` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F002 | `ret_kurt` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F003 | `positive_ret_ratio` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F004 | `zero_ret_ratio` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F005 | `sign_change_ratio` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F006 | `longest_up_run_ratio` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F007 | `longest_down_run_ratio` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F008 | `absret_q90` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F009 | `absret_q95` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F010 | `absret_q99` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F011 | `tail_asymmetry` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F012 | `extreme_rv_share` | `distribution` | Distribution and tails | `intraday/distribution.py` |
| F013 | `realized_variance` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F014 | `bipower_variation` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F015 | `jump_share` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F016 | `downside_rv` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F017 | `downside_rv_share` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F018 | `realized_quarticity` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F019 | `intramin_parkinson_var` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F020 | `intramin_rs_var` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F021 | `intraday_max_drawdown` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F022 | `intraday_max_drawup` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F023 | `ret_acf1` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F024 | `ret_acf5` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F025 | `absret_acf1` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F026 | `absret_acf5` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F027 | `rv_ratio_5m_1m` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F028 | `rv_ratio_15m_1m` | `volatility` | Volatility, jumps, and dependence | `intraday/volatility.py` |
| F029 | `volume_hhi` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F030 | `volume_entropy` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F031 | `volume_gini` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F032 | `amount_hhi` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F033 | `max_minute_volume_share` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F034 | `top5pct_volume_share` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F035 | `zero_volume_ratio` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F036 | `volume_cv` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F037 | `illiq_mean` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F038 | `illiq_p90` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F039 | `vwap_path_absdev` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F040 | `minute_vwap_dispersion` | `liquidity` | Liquidity and concentration | `intraday/liquidity.py` |
| F041 | `corr_ret_volume` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F042 | `corr_absret_volume` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F043 | `corr_ret_dvolume` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F044 | `corr_absret_dvolume` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F045 | `volume_lead_ret_corr` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F046 | `volume_lead_absret_corr` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F047 | `ret_lead_volume_corr` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F048 | `absret_lead_volume_corr` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F049 | `signed_volume_imbalance` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F050 | `signed_amount_imbalance` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F051 | `high_volume_ret` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F052 | `high_low_volume_ret_diff` | `price_volume` | Price-volume coupling | `intraday/price_volume_base.py` |
| F053 | `open30_ret` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F054 | `midmorning_ret` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F055 | `midafternoon_ret` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F056 | `close30_ret` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F057 | `open30_rv_share` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F058 | `midmorning_rv_share` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F059 | `midafternoon_rv_share` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F060 | `close30_rv_share` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F061 | `open30_volume_share` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F062 | `midmorning_volume_share` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F063 | `midafternoon_volume_share` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F064 | `close30_volume_share` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F065 | `open30_range` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F066 | `midmorning_range` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F067 | `midafternoon_range` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F068 | `close30_range` | `time_of_day` | Time of day | `intraday/time_of_day.py` |
| F069 | `ret_fft_low_share` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F070 | `ret_fft_high_share` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F071 | `ret_fft_entropy` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F072 | `ret_fft_centroid` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F073 | `ret_fft_dominant_period` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F074 | `absret_fft_low_share` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F075 | `absret_fft_high_share` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F076 | `absret_fft_entropy` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F077 | `absret_fft_centroid` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F078 | `absret_fft_dominant_period` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F079 | `volume_fft_low_share` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F080 | `volume_fft_high_share` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F081 | `volume_fft_entropy` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F082 | `volume_fft_centroid` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F083 | `volume_fft_dominant_period` | `spectral` | Frequency-domain representation | `intraday/spectral.py` |
| F084 | `path_efficiency` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F085 | `arc_chord_ratio` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F086 | `signed_chord_area` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F087 | `abs_chord_area` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F088 | `max_chord_deviation` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F089 | `rms_chord_deviation` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F090 | `turning_rate_1m` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F091 | `turning_rate_5m` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F092 | `turning_rate_15m` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F093 | `turning_rate_30m` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F094 | `mean_turn_angle` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F095 | `p90_turn_angle` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F096 | `mean_abs_curvature` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F097 | `p90_abs_curvature` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F098 | `curvature_energy` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F099 | `roughness_5m_1m` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F100 | `roughness_15m_1m` | `geometry` | Geometry and path shape | `intraday/geometry.py` |
| F101 | `block1_ret` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F102 | `block1_rv_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F103 | `block1_volume_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F104 | `block1_amount_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F105 | `block1_range` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F106 | `block1_path_efficiency` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F107 | `block1_abs_chord_area` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F108 | `block1_curvature_energy` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F109 | `block2_ret` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F110 | `block2_rv_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F111 | `block2_volume_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F112 | `block2_amount_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F113 | `block2_range` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F114 | `block2_path_efficiency` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F115 | `block2_abs_chord_area` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F116 | `block2_curvature_energy` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F117 | `block3_ret` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F118 | `block3_rv_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F119 | `block3_volume_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F120 | `block3_amount_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F121 | `block3_range` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F122 | `block3_path_efficiency` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F123 | `block3_abs_chord_area` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F124 | `block3_curvature_energy` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F125 | `block4_ret` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F126 | `block4_rv_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F127 | `block4_volume_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F128 | `block4_amount_share` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F129 | `block4_range` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F130 | `block4_path_efficiency` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F131 | `block4_abs_chord_area` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F132 | `block4_curvature_energy` | `position_blocks` | Position blocks | `intraday/time_of_day.py` |
| F133 | `rv_ratio_30m_1m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F134 | `ret_acf1_5m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F135 | `ret_acf1_15m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F136 | `ret_acf1_30m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F137 | `absret_acf1_5m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F138 | `absret_acf1_15m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F139 | `absret_acf1_30m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F140 | `path_efficiency_5m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F141 | `path_efficiency_15m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F142 | `path_efficiency_30m` | `multiscale` | Multiscale summaries | `intraday/time_of_day.py` |
| F143 | `first60_last60_ret_diff` | `cross_session` | Cross-session contrasts | `intraday/time_of_day.py` |
| F144 | `first60_last60_rv_logratio` | `cross_session` | Cross-session contrasts | `intraday/time_of_day.py` |
| F145 | `first60_last60_volume_share_diff` | `cross_session` | Cross-session contrasts | `intraday/time_of_day.py` |
| F146 | `morning_afternoon_ret_diff` | `cross_session` | Cross-session contrasts | `intraday/time_of_day.py` |
| F147 | `morning_afternoon_rv_logratio` | `cross_session` | Cross-session contrasts | `intraday/time_of_day.py` |
| F148 | `morning_afternoon_volume_share_diff` | `cross_session` | Cross-session contrasts | `intraday/time_of_day.py` |
| F149 | `max_positive_minute_return` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F150 | `max_negative_minute_return` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F151 | `max_abs_minute_return` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F152 | `max_positive_return_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F153 | `max_negative_return_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F154 | `max_abs_return_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F155 | `max_volume_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F156 | `max_amount_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F157 | `intraday_high_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F158 | `intraday_low_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F159 | `max_drawdown_end_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F160 | `max_drawup_end_position` | `event_timing` | Event timing | `intraday/event_timing.py` |
| F161 | `morning_ret_skew` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F162 | `afternoon_ret_skew` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F163 | `morning_ret_kurt` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F164 | `afternoon_ret_kurt` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F165 | `morning_downside_rv_share` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F166 | `afternoon_downside_rv_share` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F167 | `morning_path_efficiency` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F168 | `afternoon_path_efficiency` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F169 | `morning_abs_chord_area` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F170 | `afternoon_abs_chord_area` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F171 | `morning_curvature_energy` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F172 | `afternoon_curvature_energy` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F173 | `morning_signed_volume_imbalance` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F174 | `afternoon_signed_volume_imbalance` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F175 | `morning_volume_hhi` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F176 | `afternoon_volume_hhi` | `cross_session` | Morning and afternoon shapes | `intraday/session_shape.py` |
| F177 | `ret_skew_5m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F178 | `ret_skew_15m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F179 | `ret_skew_30m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F180 | `ret_kurt_5m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F181 | `ret_kurt_15m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F182 | `ret_kurt_30m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F183 | `sign_change_ratio_5m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F184 | `sign_change_ratio_15m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F185 | `sign_change_ratio_30m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F186 | `rv_hhi_5m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F187 | `rv_hhi_15m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F188 | `rv_hhi_30m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F189 | `rv_entropy_5m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F190 | `rv_entropy_15m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F191 | `rv_entropy_30m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F192 | `volume_hhi_5m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F193 | `volume_hhi_15m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F194 | `volume_hhi_30m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F195 | `volume_entropy_5m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F196 | `volume_entropy_15m` | `multiscale` | Multiscale summaries | `intraday/multiscale.py` |
| F197 | `rv_hhi_1m` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F198 | `rv_entropy_1m` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F199 | `max_minute_rv_share` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F200 | `top1pct_rv_share` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F201 | `median_absret` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F202 | `ret_mad` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F203 | `ret_iqr` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F204 | `mean_minute_log_range` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F205 | `log_range_std` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F206 | `range_hhi` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F207 | `intraday_trend_slope` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F208 | `intraday_trend_r2` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F209 | `detrended_path_std` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F210 | `max_5m_abs_return` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F211 | `max_15m_abs_return` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F212 | `max_30m_abs_return` | `volatility` | Full-day concentration, range, and trend | `intraday/multiscale.py` |
| F213 | `corr_ret_amount` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F214 | `corr_absret_amount` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F215 | `amount_lead_ret_corr` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F216 | `amount_lead_absret_corr` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F217 | `ret_lead_amount_corr` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F218 | `absret_lead_amount_corr` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F219 | `close_vs_day_vwap` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F220 | `open_vs_day_vwap` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F221 | `day_vwap_location` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F222 | `volume_weighted_return` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F223 | `amount_weighted_return` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F224 | `corr_absret_vwap_deviation` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F225 | `signed_volume_impact` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F226 | `signed_amount_impact` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F227 | `positive_return_volume_share` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F228 | `negative_return_volume_share` | `price_volume` | Price-volume coupling | `intraday/price_volume_extended.py` |
| F229 | `ret_fft_low_high_logratio` | `spectral` | Frequency-domain representation | `intraday/spectral_geometry.py` |
| F230 | `absret_fft_low_high_logratio` | `spectral` | Frequency-domain representation | `intraday/spectral_geometry.py` |
| F231 | `volume_fft_low_high_logratio` | `spectral` | Frequency-domain representation | `intraday/spectral_geometry.py` |
| F232 | `ret_spectral_slope` | `spectral` | Frequency-domain representation | `intraday/spectral_geometry.py` |
| F233 | `absret_spectral_slope` | `spectral` | Frequency-domain representation | `intraday/spectral_geometry.py` |
| F234 | `volume_spectral_slope` | `spectral` | Frequency-domain representation | `intraday/spectral_geometry.py` |
| F235 | `mean_abs_curvature_5m` | `geometry` | Geometry and path shape | `intraday/spectral_geometry.py` |
| F236 | `mean_abs_curvature_15m` | `geometry` | Geometry and path shape | `intraday/spectral_geometry.py` |
| F237 | `mean_abs_curvature_30m` | `geometry` | Geometry and path shape | `intraday/spectral_geometry.py` |
| F238 | `curvature_energy_5m` | `geometry` | Geometry and path shape | `intraday/spectral_geometry.py` |
| F239 | `curvature_energy_15m` | `geometry` | Geometry and path shape | `intraday/spectral_geometry.py` |
| F240 | `curvature_energy_30m` | `geometry` | Geometry and path shape | `intraday/spectral_geometry.py` |
| F241 | `block1_downside_rv_share` | `position_blocks` | Position blocks | `intraday/multiscale.py` |
| F242 | `block2_downside_rv_share` | `position_blocks` | Position blocks | `intraday/multiscale.py` |
| F243 | `block3_downside_rv_share` | `position_blocks` | Position blocks | `intraday/multiscale.py` |
| F244 | `block4_downside_rv_share` | `position_blocks` | Position blocks | `intraday/multiscale.py` |
| F245 | `block1_signed_volume_imbalance` | `position_blocks` | Position blocks | `intraday/multiscale.py` |
| F246 | `block2_signed_volume_imbalance` | `position_blocks` | Position blocks | `intraday/multiscale.py` |
| F247 | `block3_signed_volume_imbalance` | `position_blocks` | Position blocks | `intraday/multiscale.py` |
| F248 | `block4_signed_volume_imbalance` | `position_blocks` | Position blocks | `intraday/multiscale.py` |
