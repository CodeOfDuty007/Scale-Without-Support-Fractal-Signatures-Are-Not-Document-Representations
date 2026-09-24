"""Scaling-exponent estimators for a token-level surprisal trajectory.

A document's per-token log-probabilities s_1..s_N are z-scored and cumulatively
summed into a walk X (``generate_trajectory``). Two log-log slope estimators are
then fit over the fixed scale grid ``scale``:

* ``calc_H`` -- Hurst exponent by rescaled-range (R/S) analysis on the increments.
* ``calc_S`` -- Hoelder (self-similarity) exponent from the return probability
  p_eps(tau) = P(|X_{t+tau} - X_t| < eps). Start offsets are enumerated
  exhaustively whenever n - tau <= n_boot (every document in the corpus), which
  makes the estimate exact and deterministic; a seeded sampler is used only for
  longer inputs.

Both accept an explicit ``scale_param`` so windowed sub-document estimates use
only the scales the window can support, and ``return_fit`` exposes the
per-scale points used by the residual features.
"""
import numpy as np
from scipy.stats import linregress

scale = [8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320]
#Epsilon is NOT a fixed constant handed down by any single reference --
#three sources give three different values: andsteing/gagle's decompiled
#bundle.js defaults to 1e-2 (exposed as a UI slider, so not fixed there
#either); the GAGLE_Dataset.ipynb demo uses 0.1, noting explicitly this is
#"larger than in the paper" because its concatenation method differs; and
#the ORIGINAL paper's own colab (google-research/fractals_language/
#fractal_patterns_in_language.ipynb, arXiv:2402.01825) uses 5e-2 as an
#"illustration only" default on 896-1024 token documents with scales up to
#300 -- and explicitly states its own defaults differ from what the actual
#paper experiments used. Epsilon has to be calibrated to the corpus's own
#document length / scale-range ratio, not copied from any one source.
#Calibrated empirically here via a sweep against real GAGLE corpus-level
#fits (see reports/ for the sweep result and justification).
EPSILON = 1e-2
#Per-scale bootstrap draw count. NOT the same thing as the "bootstrap-10" in
#the project execution plan, which refers to outer CI-resampling
#repeats of the whole estimation, not this. Verified against synthetic fBm
#with known Hurst exponent: at n_boot=10 on a pooled/long series, bias
#ranged -0.25 to -0.31 with R^2 0.13-0.65 (pure sampling noise -- each
#p_epsilon(tau) is estimated from as few as 10 draws). Sweeping n_boot
#10->100->1000->5000 on the same series collapsed bias to -0.005..-0.06
#with R^2 0.89-1.0. 2000 is comfortably in the region where bias is
#sampling-noise-dominated no longer, without the cost of 5000.
N_BOOT = 2000

#F-4: fallback seed for the sampling branch of calc_S, which now only fires on
#pooled trajectories longer than n_boot + tau. Fixed so that even that branch is
#reproducible; pass seed=... to override per call. The per-document path does not
#consult it at all -- it enumerates exhaustively and is exact.
S_BOOTSTRAP_SEED = 20260823

def generate_trajectory(RawScores):
    scores=np.array(RawScores)
    mu=np.mean(scores)
#calc standard deviation
    sigma=np.std(scores)
    #z-score before cumsumming: verified byte-for-byte against the reference
    #andsteing/gagle Space's `computeXx` ((x-mean)/std then cumsum). An
    #earlier revision mistakenly "fixed" this by removing the /sigma, on the
    #theory that z-scoring decoupled a fixed epsilon from the trajectory's
    #real scale -- but the reference relies on exactly this normalization to
    #make a single fixed epsilon meaningful across documents of different
    #raw score scale. That earlier change is reverted here.
    increment=(scores-mu)/sigma
    #below line of code creates the random walk using the cumsum of increment
    trajectory=np.cumsum(increment)
    return trajectory,increment

def calc_S(x, scale_param=None, epsilon=EPSILON, n_boot=N_BOOT, return_std_err=False,
           seed=None, return_fit=False):
    """x: a single trajectory (1D array-like), OR a list of trajectories from
    multiple documents to pool into one corpus-level estimate. The paper is
    explicit that S/H describe a stochastic process and are meant to be
    estimated from a corpus of many documents, not a single ~400-token
    realization. This list-of-trajectories form matches the ORIGINAL
    paper's own reference code (google-research/fractals_language/
    fractal_patterns_in_language.ipynb, arXiv:2402.01825 -- the source the
    GAGLE dataset's own notebook says it "copied/modified" its version
    from): get_self_sim_exponent there takes a 2D (docs x tokens) array and
    draws random offsets WITHIN each document row, never crossing document
    boundaries. GAGLE_Dataset.ipynb's own demo instead concatenates all
    documents into one flat 1D array before sampling, which lets a random
    (offset, offset+tau) pair straddle a document boundary and measure a
    meaningless displacement -- a shortcut deviation from the original
    method, not something to replicate.
    """
    current_scale = scale_param if scale_param is not None else scale
    # A 2-D ndarray (docs x tokens) fails `isinstance(list)` and would be taken
    # as ONE trajectory whose rows are time steps -- returning a plausible
    # number instead of raising. The docstring below points readers at exactly
    # such a reference implementation, so this trap is easy to walk into.
    # Audit 2026-08-24 demonstrated it: a 2500x400 array returned (0.0142,
    # R2=0.277) rather than failing. Raise instead of guessing.
    if isinstance(x, np.ndarray) and x.ndim > 1:
        raise ValueError(
            f"calc_S received a {x.ndim}-D ndarray. Pass a LIST of 1-D "
            f"trajectories for pooled mode, or a single 1-D array. A 2-D array "
            f"would be silently treated as one trajectory with rows as time steps.")
    trajectories = x if (isinstance(x, list) and len(x) > 0 and hasattr(x[0], "__len__")) else [x]
    trajectories = [np.asarray(t) for t in trajectories]

    flat = np.concatenate(trajectories) if len(trajectories) > 1 else trajectories[0]
    if len(flat) < 2:
        return (float("nan"), 0.0, float("nan")) if return_std_err else (float("nan"), 0.0)
    #second normalization, distinct from generate_trajectory's raw-score z-score:
    #divide the trajectory itself by its own (pooled) stddev, matching the
    #reference's `l = a(s); r = r.map(row => row.map(v => v/l))`.
    traj_std = np.std(flat)
    if traj_std > 0:
        trajectories = [t / traj_std for t in trajectories]

    #Bootstrap p_epsilon(tau) = P(|X_{t+tau}-X_t| < epsilon), matching the
    #reference exactly: n_boot RANDOM start offsets per scale per trajectory,
    #not the exhaustive sliding-window enumeration a single ~400-token
    #document doesn't have enough independent samples to support at large tau
    #(observed directly: 3-of-272 non-overlapping-equivalent hits at tau=128).
    #Vectorized (draw all n_boot offsets for a (trajectory, tau) pair in one
    #np.random.randint call, then fancy-index all displacements at once)
    #rather than a Python-level loop of n_boot individual randint() calls --
    #same sampling distribution, ~identical statistical behavior, but P1
    #calls this ~1.8M times (vs. O(10^2-10^3) calls in P0/fBm validation),
    #where the per-call Python-loop overhead alone was projected at ~10
    #hours just for this step.
    #F-4 (2026-08-23): this loop used to call np.random.randint with NO seed
    #anywhere in the module, so S was a fresh Monte-Carlo draw on every run --
    #three consecutive calls on one array gave S = 0.121 / 0.052 / 0.162, and
    #on another 0.052 / -0.008 / -0.005 (a sign flip), against a corpus sd of
    #~0.28. calc_H was deterministic throughout, which is what exposed it.
    #
    #EXHAUSTIVE WHERE POSSIBLE. Documents are clipped to ~400 tokens, so the
    #number of DISTINCT start offsets (n - tau) is smaller than n_boot at every
    #scale: 392 at tau=8 down to 80 at tau=320, against 2000 draws -- i.e. we
    #were sampling 5x to 25x WITH REPLACEMENT from a pool we could just
    #enumerate. Enumerating is simultaneously exact (zero MC error),
    #deterministic, and 5-25x cheaper.
    #
    #CORRECTION (independent audit, 2026-08-24). An earlier version of this
    #comment claimed sampling "is kept only for P0's long pooled trajectories,
    #where n - tau genuinely exceeds n_boot". THAT IS FALSE. `n` below is the
    #length of each INDIVIDUAL trajectory -- the loop is `for traj in
    #trajectories` -- so list mode does not lengthen anything: every element is
    #one document. Measured over all 445,738 score arrays in the corpus, array
    #length runs 320..463, so n_starts <= 455 against n_boot=2000, a 4.4x
    #margin. THE SAMPLING BRANCH IS DEAD CODE ACROSS THE ENTIRE PIPELINE,
    #including P0's pooled call. It is retained only as a guard for
    #hypothetical inputs longer than n_boot + tau. Do not repeat the old claim
    #in a methods section.
    #
    #WEIGHTING IS UNCHANGED, and that is load-bearing. The old code drew
    #exactly n_boot displacements from EVERY trajectory and took one pooled
    #mean; with equal counts per trajectory that pooled mean is identically the
    #mean of per-trajectory means. Computing p0 as an explicit mean-of-means
    #therefore reproduces the old estimator's expectation exactly, including in
    #list mode where trajectories differ in length (320-463 tokens after the
    #short-window branch). Pooling raw displacements after enumeration would
    #NOT have been equivalent -- it would silently re-weight long documents
    #more heavily. Verified before changing.
    rng = np.random.default_rng(S_BOOTSTRAP_SEED if seed is None else seed)
    p0 = []
    # Tracks, per scale, whether per_traj ended up empty PURELY because every
    # trajectory was too short for this tau (n <= tau for all of them) -- a
    # length/data problem -- as opposed to per_traj having real per-
    # trajectory statistics that happened to average to 0.0 (an epsilon-
    # calibration problem, the case the reason string below was designed
    # for). Both produce the same p0 value (0.0), so this is the only way to
    # tell them apart afterward. See the reason classification below.
    too_short_at_scale = []
    for tau in current_scale:
        per_traj = []
        for traj in trajectories:
            n = len(traj)
            if n <= tau:
                continue
            n_starts = n - tau
            if n_starts <= n_boot:
                k = np.arange(n_starts)          # exhaustive: exact, deterministic
            else:
                k = rng.integers(0, n_starts, size=n_boot)
            d = traj[k + tau] - traj[k]
            per_traj.append(float(np.mean(np.abs(d) < epsilon)))
        too_short_at_scale.append(len(per_traj) == 0)
        p0.append(float(np.mean(per_traj)) if per_traj else 0.0)

    #log-log regression, filtering non-positive scale/prob pairs (matching the
    #reference's power-law fit helper, which applies the same filter).
    scale_arr = np.asarray(current_scale, dtype=float)
    p0_arr = np.asarray(p0, dtype=float)
    mask = (scale_arr > 0) & (p0_arr > 0)
    if mask.sum() < 2:
        #Audit 2026-08-28 (~0.05% of documents in both GAGLE and RAID):
        #distinguish "p0 is exactly 0 at every scale" (this document's own
        #z-scored/cumsummed trajectory has too much net drift relative to the
        #fixed EPSILON for any displacement to ever land within epsilon at
        #any tested lag) from a genuinely too-short/insufficient-data input.
        #Confirmed via direct re-scoring from gagle.zip that the former is
        #NOT a data-quality issue (raw scores are well-formed, high-variance,
        #high H_R2 on the SAME trajectory, since H doesn't depend on
        #EPSILON at all) -- it is EPSILON's corpus-level calibration failing
        #on an atypical individual document's drift range. Holder_PerDoc_
        #Unreliable already flags per-document S as unreliable for EVERY
        #row; this reason string distinguishes a narrower, additional
        #silent-total-failure mode within that already-known-unreliable
        #estimator, so a future consumer of return_fit doesn't conflate the
        #two silent-NaN causes the way both the numeric return already did.
        #2026-08-29: the ternary above conflated a THIRD cause into
        #"p0_all_zero_at_every_scale" -- every trajectory being too short for
        #every tau also drives p0 to 0.0 at every scale (see
        #too_short_at_scale above), which is a length/data problem, not the
        #epsilon-calibration problem this reason string documents. Found by
        #code review, confirmed dormant for every current caller (see the
        #ledger), fixed as a 3-way classification so a future caller with
        #short trajectories doesn't get a mislabeled reason.
        too_short_arr = np.asarray(too_short_at_scale, dtype=bool)
        n_positive = int((p0_arr > 0).sum())
        if n_positive == 0 and bool(too_short_arr.all()):
            # every scale failed purely because every trajectory was too
            # short for that tau -- distinct from the epsilon-calibration
            # case below.
            reason = "trajectories_too_short_at_every_scale"
        elif n_positive == 0:
            # at least one scale had real per-trajectory statistics (not
            # skipped for length) and they still averaged to 0 -- the
            # documented 2026-08-28 epsilon-calibration failure mode. A
            # scale that mixes some too-short trajectories with some
            # real-but-zero ones also lands here, by design: it isn't
            # purely a length failure, so it keeps this label rather than
            # inventing a 4th category.
            reason = "p0_all_zero_at_every_scale"
        else:
            reason = "insufficient_scale_points"
        empty = {"scales": [], "log_scale": [], "log_p0": [], "resid": [],
                 "slope": float("nan"), "intercept": float("nan"), "reason": reason}
        if return_fit:
            return ((float("nan"), 0.0, float("nan"), empty) if return_std_err
                    else (float("nan"), 0.0, empty))
        return (float("nan"), 0.0, float("nan")) if return_std_err else (float("nan"), 0.0)
    slope, intercept, r_value, p_value, std_err = linregress(np.log(scale_arr[mask]), np.log(p0_arr[mask]))
    s = -slope
    #Coefficient of Determination) tells you how well your data actually fits the mathematical model you are trying to use.
    #R^2 represents the proportion of the variance in the dependent variable
    #t tells you how much the text actually behaves like a fractal.
    #std_err here is linregress's standard error of the SLOPE estimate --
    #a legitimate, free-of-charge OLS uncertainty on Hat{S} itself (S = -slope),
    #used by build_p1_feature_store.py as a per-document CI half-width
    #(1.96*std_err for a 95% CI) instead of an expensive repeated-bootstrap
    #outer loop, since the inner n_boot resampling already dominates cost.
    if return_fit:
        # P3 residual features (D-13) need the INDIVIDUAL per-scale points, not
        # just the fitted slope. Exposed here rather than recomputed in
        # build_features.py: a second implementation of this fit would be a
        # fourth copy of estimator logic in this project, and the previous three
        # copies each drifted from the original and caused a defect (D-07's key,
        # the v1 cap generator, reconcile_p1_against_p0's human key). One
        # estimator, one fit, one set of points.
        lx = np.log(scale_arr[mask])
        ly = np.log(p0_arr[mask])
        fit = {"scales": scale_arr[mask].tolist(), "log_scale": lx.tolist(),
               "log_p0": ly.tolist(),
               "resid": (ly - (slope * lx + intercept)).tolist(),
               "slope": float(slope), "intercept": float(intercept)}
        if return_std_err:
            return s, r_value**2, std_err, fit
        return s, r_value**2, fit
    if return_std_err:
        return s, r_value**2, std_err
    return s, r_value**2



def calc_H(increment, scale_param=None, return_std_err=False, return_fit=False):
    """scale_param: explicit scale list to sweep, defaulting to the module-level
    `scale`. Needed for windowed sub-document H (P1 Family A): a fixed-size
    window (e.g. 200 tokens) can't support the full 11-value scale up to 320
    -- large scales silently drop out (len//n == 0 gives zero chunks for that
    n), and building log_n from a front-truncated slice of the GLOBAL scale
    list (the previous behavior) then pairs the surviving R/S values against
    the WRONG scales whenever a scale is skipped for any reason other than
    being the largest ones. This only ever worked by luck on 320-400 token
    docs, where the dropped scales happened to be exactly the largest, at the
    end of the list. Explicit scale_param + building log_n from the actually-
    used scales (not a prefix slice) fixes both a real bug and a latent one.
    """
    current_scale = scale_param if scale_param is not None else scale
    used_scales = []
    avg_rs_values = []
    for n in current_scale:
        chunks=len(increment)//n
        rs_ratio=[]
        for i in range(chunks):
            #divide an article into smaller chunks
            chk=increment[i*n:(i+1)*n]
            z = np.cumsum(chk - np.mean(chk))
            R = np.max(z) - np.min(z)
            S = np.std(chk)
            if S > 0:
                rs_ratio.append(R / S)
        if len(rs_ratio) > 0:
            avg_rs_values.append(np.mean(rs_ratio))
            used_scales.append(n)

    if len(avg_rs_values) < 2:
        # Arity must match calc_S's degenerate path. Independent audit
        # 2026-08-24 found this returned a 3-tuple while calc_S returned a
        # 4-tuple, so build_features.py's 4-way unpack raised ValueError --
        # which build_features then swallowed into a truncated *_ERROR string,
        # silently dropping an entire scorer's features for that row.
        # Unreachable on this corpus (Mistral_H NaN = 0, Gemma_H NaN = 1930 =
        # exactly the null-array count), fixed anyway because the failure mode
        # is silent rather than loud.
        _empty = {"scales": [], "log_scale": [], "log_rs": [], "resid": [],
                  "slope": float("nan"), "intercept": float("nan")}
        if return_fit:
            return ((float("nan"), 0.0, float("nan"), _empty) if return_std_err
                    else (float("nan"), 0.0, _empty))
        return (float("nan"), 0.0, float("nan")) if return_std_err else (float("nan"), 0.0)
    log_n = np.log(used_scales)
    log_rs = np.log(avg_rs_values)
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_rs)
    # "scales" mirrors calc_S's fit dict. Without it build_features.py had to
    # reconstruct scales as int(round(exp(log_scale))) -- a lossy round-trip
    # through logs to recover integers the function already knew.
    _fit = {"scales": [int(round(float(np.exp(v)))) for v in log_n],
            "log_scale": list(map(float, log_n)), "log_rs": list(map(float, log_rs)),
            "resid": [float(y - (slope * xx + intercept)) for xx, y in zip(log_n, log_rs)],
            "slope": float(slope), "intercept": float(intercept)}
    H = slope
    #std_err: OLS standard error of the slope (=H) from the same regression --
    #see calc_S's return_std_err for why this is used as a free CI half-width
    #instead of a repeated-bootstrap outer loop.
    if return_fit:
        # See calc_S's return_fit note: P3 residuals (D-13) need the per-scale
        # points, and a second copy of this fit elsewhere would be a fourth
        # duplicated estimator in this project. One fit, one set of points.
        if return_std_err:
            return H, r_value**2, std_err, _fit
        return H, r_value**2, _fit
    if return_std_err:
        return H, r_value**2, std_err
    return H, r_value**2
