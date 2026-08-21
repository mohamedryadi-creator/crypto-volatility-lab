# Theory: arbitrage-free crypto volatility and relative-value trading

This chapter states the economic question, derives the models used by the code, and fixes the
backtest protocol before looking at results. The objective is not to manufacture a high Sharpe
ratio. It is to test whether cross-sectional deviations from a smooth, arbitrage-free volatility
surface contain short-horizon information after paying the spread, fees, funding and delta hedge.

> **Research hypothesis.** At a fixed expiry, liquid options that are cheap relative to an
> arbitrage-free SSVI surface subsequently outperform vega-matched rich options over one- and
> four-hour horizons, net of executable costs.

BTC is the development market. Every modeling and selection choice is locked before applying the
same protocol to ETH, which serves as external validation.

## 1. Instruments, numeraire and notation

Deribit BTC and ETH inverse options are European, cash-settled and quoted in the base coin. A BTC
contract has multiplier one BTC. The exchange converts a USD or volatility order into coin units,
and uses the expiry forward in its volatility calculation. These conventions are documented in
the [Deribit inverse-option specification](https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options).

Let

- \(t\) be the observation time and \(T\) the year fraction to expiry;
- \(S_t\) be the spot/index price in USD per coin;
- \(F_t(T)\) be the forward price for expiry \(T\);
- \(D_t(T)\) be the USD discount factor;
- \(K\) be the strike;
- \(\sigma\) be Black implied volatility;
- \(k=\log(K/F_t(T))\) be log-forward moneyness;
- \(w(k,T)=\sigma^2(k,T)T\) be total implied variance.

The terminal coin payoff of an inverse call is

\[
\Pi_T^{C,\mathrm{coin}}=\frac{(S_T-K)^+}{S_T},
\]

and the put payoff is \((K-S_T)^+/S_T\). At time \(t\), a quoted coin premium
\(p_t^{\mathrm{coin}}\) is converted to a USD mark for calibration:

\[
p_t^{\mathrm{USD}}=S_t p_t^{\mathrm{coin}}.
\]

This conversion is a change of reporting unit, not an assertion that the option is linear. The
backtest keeps cash flows in their native coin units and converts the final marked P&L to USD with
the contemporaneous index. Mixing a coin premium with a USD Greek is a dimensional error.

## 2. Black-76 prices and implied volatility

For \(F>0\), \(K>0\), \(T>0\), \(\sigma>0\), define

\[
d_1=\frac{\log(F/K)+\tfrac12\sigma^2T}{\sigma\sqrt{T}},
\qquad
d_2=d_1-\sigma\sqrt{T}.
\]

The discounted Black-76 prices in USD are

\[
C=D\left(FN(d_1)-KN(d_2)\right),
\qquad
P=D\left(KN(-d_2)-FN(-d_1)\right).
\]

They satisfy put-call parity:

\[
C-P=D(F-K).
\]

The current normalized Tardis contract supplies `underlying_price`, which the runner uses as its
forward proxy with \(D=1\) over these short crypto maturities. A production extension should
estimate \(DF\) and \(D\) robustly from matched call-put pairs, downweight wide spreads and reject
pairs whose timestamps or contract conventions do not match. This limitation is kept explicit:
the public code does not claim that spot and forward are universally interchangeable.

### 2.1 No-arbitrage bounds and the IV solver

Before inversion, each quote must satisfy its discounted intrinsic lower bound and upper bound:

\[
D(F-K)^+\le C\le DF,
\qquad
D(K-F)^+\le P\le DK.
\]

Implied volatility is the root of

\[
f(\sigma)=P_{\text{Black}}(\sigma)-P_{\text{observed}}.
\]

The implementation uses a bracketed Brent solver. It never silently clips a price into the
admissible interval: invalid inputs, expired contracts and roots outside the configured bracket are
reported as data-quality failures. Close to intrinsic value, price-space errors are preferred to IV
errors because vega is nearly zero and IV becomes ill-conditioned.

### 2.2 Greeks

For an option expressed in USD, the forward vega is

\[
\mathcal V=D F\phi(d_1)\sqrt{T},
\]

and the forward delta is \(D N(d_1)\) for a call and \(-D N(-d_1)\) for a put. Gamma is

\[
\Gamma_F=\frac{D\phi(d_1)}{F\sigma\sqrt T}.
\]

The strategy neutralizes point-in-time provider vega cross-sectionally, then hedges residual
directional exposure with the corresponding inverse perpetual. The calibration independently
recomputes Black vega in USD for weighting. Coin-denominated option P&L and inverse-perpetual P&L
are calculated separately; for the hedge the engine uses Deribit's inverse net-transaction delta
proxy \(\Delta_{\mathrm{NTD}}=\Delta_{\mathrm{Black}}-p^{\mathrm{coin}}\), measured in BTC or
ETH, rather than applying a USD delta directly to a coin premium. A production run should report
the reconciliation between provider and model Greeks.

## 3. From a smile to a surface

A collection of independent splines can interpolate observed quotes but may imply negative
state-price density or calendar arbitrage. SVI provides a parsimonious parameterization in total
variance, while SSVI couples maturities and admits tractable sufficient conditions for static
arbitrage. The core reference is Gatheral and Jacquier,
[“Arbitrage-free SVI volatility surfaces”](https://arxiv.org/abs/1204.0646).

### 3.1 Raw SVI by maturity

For one maturity,

\[
w(k)=a+b\left[\rho(k-m)+\sqrt{(k-m)^2+\sigma_{\text{SVI}}^2}\right].
\]

The five parameters have interpretable effects:

- \(a\) shifts total variance;
- \(b\) controls wing slope;
- \(\rho\) controls skew asymmetry;
- \(m\) shifts the smile horizontally;
- \(\sigma_{\text{SVI}}\) controls ATM curvature.

The optimizer works in unconstrained coordinates and transforms them so that
\(b\ge0\), \(\sigma_{\text{SVI}}>0\) and \(|\rho|<1\). It also checks the minimum-variance
condition

\[
a+b\sigma_{\text{SVI}}\sqrt{1-\rho^2}\ge0.
\]

These restrictions are necessary for a sensible slice but are not a substitute for an explicit
butterfly-arbitrage test.

### 3.2 SSVI across maturities

Let \(\theta(T)=w(0,T)\) be ATM total variance and \(\varphi(\theta)>0\) a shape function. SSVI is

\[
w(k,\theta)=\frac{\theta}{2}\left[
1+\rho\varphi(\theta)k+
\sqrt{\left(\varphi(\theta)k+\rho\right)^2+1-\rho^2}
\right].
\]

The implementation uses a monotone interpolation of observed \(\theta(T)\) and a positive
power-law family for \(\varphi\). Parameters are accepted only if they satisfy the implemented
sufficient SSVI restrictions and pass numerical checks on a grid extending beyond observed
moneyness. Specifically, each fit is audited at 33 evenly spaced maturities across its calibrated
range and 401 log-moneyness nodes on \([-L,L]\), where
\(L=\max(1.25,\max_i|k_i|+0.25)\). The numerical audit is blocking: a failed surface supplies no
model prices to the signal or backtest.

## 4. Static no-arbitrage diagnostics

### 4.1 Calendar spreads

For each fixed log-forward moneyness, total variance must not decrease with maturity:

\[
\partial_T w(k,T)\ge0.
\]

This is evaluated on a common \(k\)-grid, not by comparing different observed strikes directly.
The code reports the minimum total-variance increment and the number of nodes below the configured
tolerance.

### 4.2 Butterfly arbitrage

For a twice differentiable total-variance smile, define

\[
g(k)=
\left(1-\frac{k w'(k)}{2w(k)}\right)^2
-\frac{[w'(k)]^2}{4}\left(\frac1{w(k)}+\frac14\right)
+\frac{w''(k)}2.
\]

Non-negative risk-neutral density requires \(g(k)\ge0\), together with appropriate wing behavior.
The audit also reprices normalized-forward calls with \(F=D=1\) and \(K=e^k\), then directly checks
non-negative prices, decreasing prices, vertical-spread slope bounds and convexity in strike.
Checking both analytical and price-space conditions catches implementation and interpolation
errors. Every count must be zero within the configured tolerance. The JSON report aggregates the
minimum margins and violation counts over all accepted surfaces, and separately counts surfaces
rejected by this audit.

### 4.3 Wing behavior

The asymptotic total-variance slopes are monitored against Lee-type moment bounds. Data are never
extrapolated mechanically to zero or infinite strike for a variance integral; tail assumptions are
made explicit and stressed.

## 5. Calibration inside a bid-ask market

The midpoint is not an executable truth. Bid, midpoint and ask prices are first inverted to total
variances \([w_i^B,w_i^A]\) and \(w_i^{\mathrm{mid}}\). For a raw-SVI slice prediction
\(\widehat w_i(\beta)\), define distance to that interval

\[
d_i(\beta)=
\begin{cases}
w_i^B-\widehat w_i, & \widehat w_i<w_i^B,\\
0, & w_i^B\le \widehat w_i\le w_i^A,\\
\widehat w_i-w_i^A, & \widehat w_i>w_i^A.
\end{cases}
\]

The primary calibration objective is

\[
\min_\beta
\sum_i \omega_i\,
\rho_H\!\left(\frac{d_i(\beta)}{h_i^w+\varepsilon}\right)
+\lambda_{\mathrm{mid}}
\sum_i\omega_i\left(\widehat w_i-w_i^{\mathrm{mid}}\right)^2
+\mathcal P_{\mathrm{arb}}(\beta),
\]

where \(h_i^w=(w_i^A-w_i^B)/2\), \(\rho_H\) is a robust loss, and \(\omega_i\) combines vega and
the inverse variance spread. The small midpoint term identifies a raw-SVI solution when many
predictions lie inside their spreads. Those slice fits estimate the monotone ATM variance curve;
the global SSVI fit then minimizes robust, vega-weighted total-variance errors with explicit
sufficient-condition penalties and hard post-fit rejection.

Filters are point-in-time and configured before evaluation:

- positive and non-crossed quotes;
- bounded relative spread;
- 7–120 days to expiry;
- a common 15-minute backward-looking snapshot;
- valid forward, strike and contract convention;
- minimum quotes per expiry and bounded log-moneyness;
- optional size/open-interest thresholds when those fields are available.

Raw-SVI slice fits retain total-variance RMSE and the fraction of predictions inside the variance
spread. The global fit retains weighted and unweighted total-variance RMSE. The synthetic JSON
reports mean global RMSE, while both runners report the numerical static-arbitrage grid, worst
margins and violation totals actually applied to accepted surfaces.

## 6. Risk-neutral density and model-free variance

For a smooth call surface, Breeden-Litzenberger gives the discounted state-price density:

\[
q_T(K)=\frac{1}{D(0,T)}\frac{\partial^2 C(K,T)}{\partial K^2}.
\]

Numerical extraction uses the fitted surface, never second differences of noisy raw quotes. The
returned diagnostics let the researcher inspect:

1. any negative mass removed before normalization;
2. finite-grid raw mass, \(\int q_T(K)\,dK\approx1\);
3. the first moment against the forward, \(\int Kq_T(K)\,dK\approx F_T\);
4. stability to strike-grid refinement and wing truncation in an explicit sensitivity run.

The API reports finite-grid mass, removed negative mass and distribution moments. The checked-in
synthetic demo exposes these diagnostics, but it does not automate a grid-refinement or wing stress;
those remain required before using tail estimates in an empirical report. The density supports tail
probabilities, quantiles and risk-neutral skewness. These are pricing objects under \(\mathbb Q\),
not forecasts under the physical measure.

The model-free annualized variance for maturity \(T\) is approximated by

\[
\operatorname{IVAR}_{t,T}\approx
\frac{2}{T D(0,T)}\left[
\int_0^{F_T}\frac{P(K,T)}{K^2}\,dK+
\int_{F_T}^{\infty}\frac{C(K,T)}{K^2}\,dK
\right].
\]

The numerical API integrates over the strike grid supplied by the caller. A serious empirical
report should run observed-strike, fitted-wing and stressed-wing grids separately rather than hide
the truncation choice. If realized variance is later compared with IVAR,
\(\operatorname{IVAR}-\operatorname{RV}\) is labeled an *ex-post proxy* for the variance risk
premium, not an identified physical expectation.

## 7. Relative-value signal

For a fitted model value \(M_i\), define a signed executable discrepancy with a no-trade region:

\[
e_i=
\begin{cases}
A_i-M_i, & M_i>A_i \quad(\text{cheap; buy}),\\
B_i-M_i, & M_i<B_i \quad(\text{rich; sell}),\\
0, & \text{otherwise}.
\end{cases}
\]

The score normalizes edge by both spread and vega:

\[
z_i=\frac{e_i}{\max(h_i,\varepsilon_p)}
\times\frac{\mathcal V_{\mathrm{ref}}}{\max(\mathcal V_i,\varepsilon_\mathcal V)}.
\]

Thus cheap candidates have a negative score and rich candidates a positive score. Quotes with a
model value inside the spread have zero signal. A configurable score cap prevents a single
ill-conditioned wing quote from dominating. A leave-one-out slice refit is a useful model risk
extension, but is not automated by the current runner and must not be presented as part of its
reported performance.

## 8. Portfolio construction

Signals are paired within the same expiry. A long cheap option and a short rich option receive
weights \(x_L>0\), \(x_S<0\) such that

\[
x_L\mathcal V_L+x_S\mathcal V_S=0.
\]

The remaining option delta is hedged with the same-asset inverse perpetual. A real two-sided quote
for that instrument is required at both entry and exit, and its arrival time may be at most one
configured resampling interval old. The engine never manufactures a perpetual market from an
option underlying or index value. Candidates are ranked by expected convergence edge net of the
entry and exit spread. A portfolio is rejected if it violates any of the following predeclared
limits:

- gross vega and gross premium budget;
- maximum absolute position per option leg;
- scenario loss over an underlying and volatility shock grid;
- stale or missing hedge quote;
- expiry mismatch or contract-convention mismatch.

The research engine allows fractional option quantities and fractional inverse-perpetual contract
counts, and caps option legs symmetrically so that vega neutrality is preserved. Any deployment
layer must round options and the perpetual USD amount to exchange contract increments, then
recompute neutrality before sending orders; order routing is deliberately outside this project. A
failed research constraint means no trade and is not repaired using future information.

## 9. Execution and P&L accounting

The signal is formed at snapshot \(t\), but the strategy cannot fill at that same timestamp. Entry
occurs at the next 15-minute snapshot:

- long legs cross the ask;
- short legs cross the bid;
- the perpetual hedge crosses the appropriate side;
- exit after 60 or 240 minutes crosses the opposite side.

If a required option quote is absent, or a matching perpetual quote is absent, invalid or older
than the configured freshness limit, the trade is skipped rather than filled at a stale mark or an
index proxy.
Option and perpetual fees are configurable. At publication, the checked-in scenario uses 3 bp per
option execution (subject to the 12.5%-of-premium cap) and 3.5 bp per taker perpetual execution,
matching the standard tier in the
[current Deribit fee schedule](https://support.deribit.com/hc/en-us/articles/25944746248989-Fees).
The rates are configurable so the study can be rerun at 0×, 1× and 2× costs. Current fees are not
misrepresented as historical fees.

### 9.1 Inverse-perpetual units and cash flows

The hedge is not modeled as a linear future. Let \(\Delta^{\mathrm{opt}}\) be the option portfolio's
net transaction delta in base coin, and let \(P_0\) be the executable entry price of the matching
inverse perpetual in USD per coin. The signed Deribit order amount is

\[
N^{\mathrm{USD}}=-\Delta^{\mathrm{opt}}P_0.
\]

Thus its base-coin position at entry is \(N^{\mathrm{USD}}/P_0=-\Delta^{\mathrm{opt}}\). Deribit
amounts are denominated in USD: BTC-PERPETUAL has a 10 USD contract size and ETH-PERPETUAL a 1 USD
contract size, so the signed contract count is \(N^{\mathrm{USD}}/c\). The research engine keeps
this count continuous and reports it; exchange rounding is a separate deployment concern. These
contract specifications are documented in Deribit's
[inverse-perpetual specification](https://support.deribit.com/hc/en-us/articles/31424954847133-Inverse-Perpetual).

For an exit fill \(P_1\), the exact coin-settled trading P&L of the inverse perpetual is

\[
\Pi_{\mathrm{perp}}^{\mathrm{coin}}
=N^{\mathrm{USD}}\left(\frac{1}{P_0}-\frac{1}{P_1}\right).
\]

The report converts this coin cash flow to USD using the exit perpetual midpoint as the available
index proxy. A long amount \(N^{\mathrm{USD}}>0\) therefore profits when \(P_1>P_0\); a short amount
has the opposite sign. For taker fee rate \(f\), each execution costs
\(f|N^{\mathrm{USD}}|/P\) in settlement coin, whose contemporaneous USD equivalent is
\(f|N^{\mathrm{USD}}|\). The modeled round-trip perpetual fee is consequently
\(2f|N^{\mathrm{USD}}|\).

For a realized funding fraction \(r_j\), positive funding is paid by longs. Its contemporaneous USD
equivalent is \(-N^{\mathrm{USD}}r_j\), since the coin position is \(N^{\mathrm{USD}}/X_j\).
Without a historical funding series, the configured constant sensitivity applies the same identity
over the holding interval; the default is zero.

For trade \(j\), net USD P&L is decomposed as

\[
\Pi_j^{\mathrm{net}}=
\Pi_j^{\mathrm{options}}+
\Pi_j^{\mathrm{hedge}}+
\Pi_j^{\mathrm{funding}}-
\operatorname{fees}_j.
\]

Here \(\Pi_j^{\mathrm{funding}}\) is signed: it is negative when the hedge pays funding and positive
when it receives funding.

The report also computes a Greek approximation

\[
\Delta V\approx
\Delta\,\Delta S+
\frac12\Gamma(\Delta S)^2+
\mathcal V\,\Delta\sigma+
\Theta\,\Delta t+
\text{residual},
\]

only as attribution. Actual bid/ask cash flows determine performance.

## 10. Statistical protocol

Free Tardis samples contain the first calendar day of each month. Intraday rows from the same day
are strongly dependent and do not create thousands of independent observations. The sampling unit
for inference is the **day**.

The chronological protocol is:

- BTC development: March 2020–December 2023;
- BTC validation: January–December 2024;
- BTC final test: January 2025–August 2026;
- ETH external validation: the final test period with every BTC choice frozen.

If a date is unavailable or is listed in a provider incident report, it is recorded and skipped.
No random row split is permitted.

Within each holding-period cohort, every completed trade is assigned to the UTC calendar date of its
exit. Net P&L and turnover are summed within that date *before* computing performance. Thus the
reported observation count, mean P&L, sample volatility, hit rate, profit factor and end-of-day
drawdown are daily quantities rather than trade-level quantities. The zero-risk-free-rate Sharpe is

\[
\widehat{SR}_{\mathrm{ann}}
=\sqrt{365}\,\frac{\overline{P\&L}_{\mathrm{day}}}
{s(P\&L_{\mathrm{day}})},
\]

using the 365-day crypto calendar and sample standard deviation. It is unavailable with fewer than
two observed UTC days; the probabilistic Sharpe additionally requires at least three. This
annualization describes the daily-P&L convention, not evidence that the sparse first-of-month sample
represents every calendar day.

Reported statistics also include cumulative P&L, daily mean edge, daily hit rate, daily turnover,
maximum end-of-day drawdown and exposure distributions. Confidence intervals resample daily P&L
totals in the main runner. The statistical library also exposes stationary and fixed-block
bootstraps, multiple-testing corrections, and deflated/probabilistic Sharpe diagnostics for
registered sensitivity analyses rather than selecting the largest Sharpe.

The result is considered economically positive only if it:

1. survives executable bid/ask prices and the 1× fee scenario;
2. has a day-clustered confidence interval that excludes a materially negative mean;
3. is not driven by one date, expiry or strike bucket;
4. remains directionally consistent on BTC final test and ETH lockbox;
5. passes the predeclared data and no-arbitrage diagnostics.

## 11. Data design and licensing

[Tardis.dev downloadable CSV datasets](https://docs.tardis.dev/downloadable-csv-files)
provide tick-level option-chain and quote files for the first day of each month without an API key.
Deribit coverage begins in 2019 and the grouped options sample begins in March 2020. The raw stream
is compressed. In the CLI workflow, download and preparation are separate:
`download --keep-raw` stages an exact file, `prepare` creates backward-looking snapshots, and the
user can remove the private raw staging copy after checking the transformation manifest. Isolated
monthly dates are resampled independently: quote state is cleared at every UTC-date boundary, so
the pipeline neither fabricates observations between samples nor accumulates expired symbols. The
programmatic download context deletes its raw payload by default.

The public repository contains code, configurations, theory and synthetic fixtures. It does not
contain raw or row-level exchange data. Provider terms can change; every user must read and accept
the current [Tardis terms](https://docs.tardis.dev/legal/terms-of-service) before downloading.
Generated real-data reports remain local by default. The project does not assume that a 15-minute
last-observation snapshot qualifies for a redistribution exception. This is a conservative
reproducibility design, not legal advice.

## 12. Limitations and falsification

- **First-of-month selection.** The free sample is not a random sample of trading days and may
  overrepresent weekends or month-boundary flows. Inference is conditional on this sampling rule.
- **Sparse independent dates.** Rich tick data do not remove the small number of independent days.
  Confidence intervals must remain wide when the evidence is weak.
- **Corpus completeness.** The runner audits expected monthly option and perpetual partitions and
  disables confirmatory claim eligibility when any are absent. It also requires an exact manifest
  match, strict parsing, the active configuration, non-empty row counts and valid SHA-256 checksums;
  unmanifested partitions or later file changes are rejected. Provider incidents still require a
  documented human review.
- **Crossing is not guaranteed fill.** Bid/ask crossing is more conservative than midpoint fills,
  but ignores queue depletion, latency and partial fills. Size caps reduce, not eliminate, this risk.
- **Model risk.** SSVI smooths noisy wings and can create apparent residuals. Leave-one-out and
  alternative calibration objectives are important follow-up sensitivity checks; they are not
  automated by the current runner.
- **Fee history.** The repository has no dated archive of fee tiers. It stresses configurable
  current rates rather than pretending they held historically.
- **Inverse-contract risk.** Coin denomination makes collateral value stochastic. Native-currency
  accounting is essential and cannot be replaced by a linear USD approximation.
- **No live-trading claim.** This is an offline research system. It does not route orders or manage
  liquidation, collateral or exchange operational risk.

A negative result is a valid result: if deviations fail to overcome two crossings and fees, the
project quantifies how much apparent midpoint alpha was microstructure noise.

## References

- Jim Gatheral and Antoine Jacquier, [Arbitrage-free SVI volatility
  surfaces](https://arxiv.org/abs/1204.0646).
- Douglas Breeden and Robert Litzenberger, [Prices of State-Contingent Claims Implicit in Option
  Prices](https://www.jstor.org/stable/1926731).
- Peter Carr and Dilip Madan, [Towards a Theory of Volatility
  Trading](https://www.cambridge.org/core/books/abs/handbooks-in-mathematical-finance/towards-a-theory-of-volatility-trading/5BF36E8C020553D95F22CF846F63EC63).
- Deribit, [Inverse Options](https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options)
  and [Fees](https://support.deribit.com/hc/en-us/articles/25944746248989-Fees).
- Tardis.dev, [Deribit data coverage](https://docs.tardis.dev/historical-data-details/deribit) and
  [downloadable CSV schema](https://docs.tardis.dev/downloadable-csv-files/data-types).
