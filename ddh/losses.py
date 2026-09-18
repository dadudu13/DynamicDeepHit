import torch

SURVIVAL_FLOOR = 1e-3

def log_likelihood_per_sample(outcomes, cif, t, e, s=None):
    """
        Per-sample log likelihood contribution, shape [N].

        Uncensored (e_i = k + 1) : log P(T = t_i, K = k)
        Censored   (e_i = 0)     : log P(T > t_i) = log(1 - sum_k F_k(t_i))

        When `s` is given, - log S(s_i) is added for patients with s_i > 0 so the
        contribution becomes the *conditional* log likelihood given survival past
        the landmark, log P(T = t_i, K = k | T > s_i).

        Splitting the per-sample term out of negative_log_likelihood lets the
        all-landmark objective weight each row (see landmark_nll_loss) instead of
        summing them all with equal weight.
    """
    n = t.shape[0]
    rows = torch.arange(n, device = t.device)
    log_lik = torch.zeros(n, dtype = cif[0].dtype, device = cif[0].device)

    # Censored: probability of surviving beyond t_i
    censored = (e == 0)
    if censored.any():
        total_cif = torch.stack([c[rows, t] for c in cif]).sum(0)
        log_lik = torch.where(censored, torch.log(1 - total_cif + 1e-10), log_lik)

    # Uncensored: probability mass on the observed (time, cause) bin
    for k, ok in enumerate(outcomes):
        selection = (e == k + 1)
        if selection.any():
            log_lik = torch.where(selection, torch.log(ok[rows, t] + 1e-10), log_lik)

    # Landmark conditioning: subtract log S(s_i) only for patients with s > 0.
    # Patients at s=0 have no prior observation window — conditioning on bin 0
    # adds no information and creates a -log S(0) gradient that explodes when
    # many single-plan patients are present, causing model collapse.
    if s is not None:
        valid = s > 0
        if valid.any():
            survival_at_s = (
                1.0 - torch.stack([c[rows, s] for c in cif]).sum(0)
            ).clamp(min = SURVIVAL_FLOOR)
            log_lik = log_lik - torch.where(
                valid, torch.log(survival_at_s), torch.zeros_like(survival_at_s)
            )

    return log_lik

def negative_log_likelihood(outcomes, cif, t, e, s=None):
    """
        Compute the log likelihood loss
        This function is used to compute the survival loss

        s: optional landmark bin index per patient (LongTensor, shape N).
           When provided, subtracts log S(s_i) from each patient's contribution
           so the loss becomes P(T=t, K=k | T > s_i) rather than P(T=t, K=k).

        NB: the reduction is a *sum* over the batch divided by the number of
        causes, not a mean over patients — the term therefore scales with batch
        size, which is what alpha/beta in total_loss were tuned against.
    """
    return - log_likelihood_per_sample(outcomes, cif, t, e, s = s).sum() / len(outcomes)

def build_landmark_prefixes(x, s_all, visit_days=None, decay_days=None,
                            entry_weight=None):
    """
        Expand one row per patient into one row per (patient, visit) landmark.

        A patient with V_i observed visits becomes V_i rows: row v keeps visits
        1..v+1 and NaNs out everything after, exactly the input the model would
        receive if that visit were the patient's last observation.  This is the
        training-time counterpart of build_intermediate_landmark_dataset in the
        DDH_implement pipeline, which does the same truncation at evaluation time.

        Args:
            x      : [N, T, D] padded features, NaN after each patient's last visit.
            s_all  : [N, T] landmark bin index of every timestep (LongTensor).

        Returns:
            x_exp       : [R, T', D] truncated sequences, R = sum_i V_i.
            s_exp       : [R] landmark bin of each row.
            patient_idx : [R] index into the original batch.
            weight      : [R] 1 / V_i, so every patient contributes weight 1 in
                          total regardless of how many visits they have.
    """
    lengths = (~torch.isnan(x[:, :, 0])).sum(dim = 1)
    t_max = int(lengths.max().item())
    if s_all.shape[1] < t_max:
        raise ValueError(
            f"s_all covers {s_all.shape[1]} timesteps but the batch has {t_max} visits"
        )
    x = x[:, :t_max]

    device = x.device
    patient_idx = torch.repeat_interleave(
        torch.arange(x.shape[0], device = device), lengths
    )
    offsets = torch.cumsum(lengths, 0) - lengths
    visit_idx = torch.arange(len(patient_idx), device = device) - offsets[patient_idx]

    x_exp = x[patient_idx].clone()
    positions = torch.arange(t_max, device = device).unsqueeze(0)
    x_exp[positions > visit_idx.unsqueeze(1)] = float('nan')

    s_exp = s_all[patient_idx, visit_idx]

    if decay_days is not None and visit_days is not None:
        # Exponential decay in landmark DAY, renormalised within patient.
        #
        # Day, not visit index: a weekly-seen patient reaches visit 5 at day 35
        # and a quarterly one at day 450, so indexing by visit confounds clinical
        # timing with visit frequency.
        #
        # Renormalising per patient keeps the 1/V_i design's guarantee that every
        # patient contributes total weight 1 -- this only redistributes attention
        # *within* a patient, never between them.  As decay_days -> inf every
        # exponent is 0 and the weights collapse back to a uniform 1/V_i.
        days = torch.nan_to_num(
            visit_days[:, :t_max].to(x_exp.dtype)[patient_idx, visit_idx], nan = 0.0
        ).clamp(min = 0.0)
        w = torch.exp(-days / float(decay_days))
        totals = torch.zeros(x.shape[0], dtype = w.dtype, device = device)
        totals = totals.index_add(0, patient_idx, w)
        weight = w / totals[patient_idx].clamp(min = 1e-12)
    else:
        weight = (1.0 / lengths.double())[patient_idx]

    if entry_weight is not None:
        # Applied OUTSIDE the per-patient normalisation, deliberately.  Folded
        # inside, a per-patient constant divides straight back out and does
        # nothing at all -- a bug that looks exactly like "the entry weight had
        # no effect".  Out here it scales the patient's TOTAL contribution, which
        # is the point: it reweights *between* patients, while the decay term
        # reweights *within* one.
        weight = weight * entry_weight.to(weight.dtype)[patient_idx]

    return x_exp, s_exp, patient_idx, weight

def landmark_nll_loss(model, x_exp, t_exp, e_exp, s_exp, weight,
                      nll_weight=1.0, horizon_spec=None, gamma=0.0,
                      components=None):
    """
        Conditional survival NLL evaluated at every landmark, averaged within a
        patient before being summed across patients:

            L = (1/K) * sum_i (1 / V_i) * sum_v - log P(T_i, K_i | T_i > s_iv, X_i,<=v)

        `weight` carries the 1 / V_i factor (see build_landmark_prefixes), so this
        reduces to negative_log_likelihood exactly when every patient contributes
        a single (final-visit) landmark.  Rows whose landmark bin is at or beyond
        the patient's event bin carry no information about the remaining horizon
        and are dropped; the 1 / V_i weight is deliberately left as-is, so those
        few patients (0.5% of rows on the VTE cohort) count slightly under 1.

        With `horizon_spec` and gamma > 0 the same forward pass also supplies
        horizon_likelihood_loss, and the returned scalar is
        nll_weight * nll + gamma * horizon.
        Sharing the forward is what keeps the extra term nearly free: it reuses
        `outcomes` and `cif`, adding only two gathers per horizon.  Pass a dict as
        `components` to read the two pieces back for logging.
    """
    _, outcomes = model(x_exp)
    cif = [torch.cumsum(ok, 1) for ok in outcomes]
    log_lik = log_likelihood_per_sample(outcomes, cif, t_exp, e_exp, s = s_exp)

    usable = (s_exp < t_exp).double()
    nll = - (log_lik * weight * usable).sum() / len(outcomes)

    # gamma=0 or no spec returns exactly the previous scalar at the default
    # nll_weight=1, so every arm trained before this term existed is an exact
    # control.  nll_weight carries beta INSIDE, so that the caller does not
    # multiply the horizon term by beta as well.
    if horizon_spec is None or gamma == 0:
        if components is not None:
            components["nll"] = float(nll.item())
            components["horizon"] = 0.0
        return nll_weight * nll

    hz = horizon_likelihood_loss(
        outcomes, cif, t_exp, e_exp, s_exp, weight,
        day = horizon_spec["day"],
        split_time = horizon_spec["split_time"],
        horizons = horizon_spec["horizons"],
        g_bins = horizon_spec["g_bins"],
        ipcw_clip = horizon_spec.get("ipcw_clip", 10.0),
        event_of_interest = horizon_spec.get("event_of_interest", 0),
    )
    if components is not None:
        components["nll"] = float(nll.item())
        components["horizon"] = float(hz.item())
    return nll_weight * nll + gamma * hz

def ranking_loss(cif, t, e, sigma, s=None, cause_weights=None):
    """
        Penalize wrong ordering of probability
        Equivalent to a C Index
        This function is used to penalize wrong ordering in the survival prediction

        s: optional per-patient landmark bin index (LongTensor, shape N).
           When provided, comparisons and CIF indexing use remaining time
           r_i = t_i - s_i rather than absolute event time t_i, matching
           the DDH paper which ranks patients by time elapsed since last
           measurement rather than by absolute event time.

        cause_weights: list of K non-negative floats (alpha_k in the DDH paper)
           scaling each cause's ranking contribution. Defaults to equal weights
           1/K. Increase a cause's weight to prioritise its ranking accuracy
           (e.g. [0.8, 0.2] to penalise wrong VTE ordering more than death).

        Uses torch.sum over valid pairs per the DDH paper (Σ_{i≠j}).
        Because the sum scales with batch size, set alpha in total_loss small
        (e.g. 0.01-0.05) or increase sigma to keep the total loss stable.
    """
    K = len(cif)
    if cause_weights is None:
        cause_weights = [1.0 / K] * K

    # Remaining time per patient: r_i = tau_i - t_last_visit_i
    # When s is None all landmarks are 0 so r == t and behaviour is unchanged.
    r = (t - s) if s is not None else t

    loss = 0
    for k, cifk in enumerate(cif):
        mask = e - 1 == k
        cause_loss = 0
        for ci, ti, ri in zip(cifk[mask], t[mask], r[mask]):
            # Eligible comparators: all patients whose remaining time exceeds r_i
            eligible = r > ri
            if eligible.sum() > 0:
                if s is not None:
                    # For each eligible patient j, evaluate their conditional CIF at
                    # the absolute bin corresponding to remaining time r_i:
                    #   abs_bin_j = s_j + r_i
                    # This maps "r_i units after j's landmark" into j's absolute-time CIF.
                    col_idx = (s[eligible] + ri).clamp(0, cifk.shape[1] - 1)
                    risk_j = cifk[eligible][torch.arange(eligible.sum()), col_idx]
                else:
                    risk_j = cifk[eligible][torch.arange(eligible.sum()), ti]
                # Sum over all valid pairs per DDH paper formulation.
                cause_loss += torch.sum(torch.exp((risk_j - ci[ti]) / sigma))
        loss += cause_weights[k] * cause_loss

    return loss

def longitudinal_loss(longitudinal_prediction, x, cancer_id_col=None):
    """
        Penalize error in the longitudinal predictions
        This function is used to compute the error made by the RNN

        NB: In the paper, they seem to use different losses for continuous and categorical
        But this was not reflected in the code associated (therefore we compute MSE for all)

        NB: Original paper mentions possibility of different alphas for each risk
        But take same for all (for ranking loss)
    """
    length = (~torch.isnan(x[:,:,0])).sum(axis = 1) - 1
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")

    # Create a grid of the column index
    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

    # Select all predictions until the last observed
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))

    # Select all observations that can be predicted
    observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
    observation_mask[:, 0] = False # Remove first observation

    # A baseline-only dataset has no next visit to reconstruct.  Returning a
    # differentiable zero keeps the survival objectives active without asking
    # MSELoss to reduce an empty tensor (which otherwise returns NaN).
    if not observation_mask.any():
        return longitudinal_prediction.sum() * 0.0

    if cancer_id_col is not None:
        x_target = torch.cat([x[:, :, :cancer_id_col], x[:, :, cancer_id_col + 1:]], dim=2)
    else:
        x_target = x
    return torch.nn.MSELoss(reduction = 'mean')(longitudinal_prediction[prediction_mask], x_target[observation_mask])

def total_loss(model, x, t, e, alpha, beta, sigma, s=None, cause_weights=None, include_nll=True):
    longitudinal_prediction, outcomes = model(x)
    t, e = t.long(), e.int()

    # Compute cumulative function from prediced outcomes
    cif = [torch.cumsum(ok, 1) for ok in outcomes]

    # Ranking loss uses the conditional CIF F̂_k for patients with s > 0, and
    # the raw unconditional CIF for patients with s = 0 (no prior window).
    # Applying conditional CIF at s=0 risks dividing by near-zero survival
    # during early training and creates the same collapse as in the NLL term.
    if s is not None:
        valid = s > 0
        if valid.any():
            rows = torch.arange(cif[0].shape[0])
            valid_rows = rows[valid]
            survival_at_s = (
                1.0 - torch.stack([c[valid_rows, s[valid]] for c in cif]).sum(0)
            ).clamp(min=1e-8)
            cif_ranking = []
            for c in cif:
                c_rank = c.clone()
                c_rank[valid] = (
                    (c[valid] - c[valid_rows, s[valid]][:, None]) / survival_at_s[:, None]
                ).clamp(0.0, 1.0)
                cif_ranking.append(c_rank)
        else:
            cif_ranking = cif
    else:
        cif_ranking = cif

    cancer_id_col = model.cancer_id_col if model.use_cancer_embedding else None
    loss = (1 - alpha - beta) * longitudinal_loss(longitudinal_prediction, x, cancer_id_col=cancer_id_col) +\
              alpha * ranking_loss(cif_ranking, t, e, sigma, s=s, cause_weights=cause_weights)

    # include_nll=False keeps the longitudinal/ranking weights untouched while the
    # survival term is supplied elsewhere — used by the all-landmark objective,
    # which computes the NLL over every (patient, visit) landmark instead of only
    # the final visit represented in this batch.
    if include_nll:
        loss = loss + beta * negative_log_likelihood(outcomes, cif, t, e, s=s)
    return loss


# ---------------------------------------------------------------------------
# Horizon-specific likelihood
#
# WHY THIS TERM EXISTS.  The all-landmark NLL is a multinomial likelihood over
# (cause, bin): it scores the probability mass the model puts on the ONE bin an
# event actually landed in, spread over ~100 bins.  Nothing in it constrains the
# CUMULATIVE probability at the horizons the model is deployed at.  The
# parameterisation probe showed the consequence directly: at landmark 0 the
# model over-predicted the 90-day level by as much on the data it was fitted
# on as out of sample.  That is not a generalisation failure, it is the
# objective not being asked about the level.
#
# This term asks.  For every (patient, visit) landmark row and every deployment
# horizon h it forms the exact quantity the model is scored on,
#
#     p = F_1(day_s + h | T > day_s, X_<=v) = (F_1(b_h) - F_1(s)) / S(s)
#
# and adds the IPCW-weighted binary log likelihood of the observed status at h.
# Binary log likelihood is a strictly proper scoring rule, so its minimiser is
# the true conditional probability at h -- the level, not just the ordering.
#
# LABELS AND WEIGHTS mirror V3's evaluation code exactly (Graf/Gerds-Schumacher):
#     T <= h, cause 1   -> y = 1, weight 1 / G(T-)
#     T <= h, cause 2   -> y = 0, weight 1 / G(T-)   (death without VTE is a true 0)
#     T >  h            -> y = 0, weight 1 / G(h)
#     censored before h -> weight 0                  (status unknown)
# Matching the metric matters more than picking a different defensible estimator:
# a training term and a reported metric that disagree cannot be read together.
#
# NOT CARVED OUT OF THE EXISTING WEIGHTS.  The term is added with its own gamma
# on top of (1-alpha-beta)/alpha/beta rather than taking a share of them, so
# gamma=0 reproduces the previous objective bit for bit and every existing arm
# stays a valid control.

CENSOR_FLOOR = 1e-2
HORIZON_EPS = 1e-5


def censoring_survival_by_bin(t, e, n_bins):
    """
        Kaplan-Meier estimate of the censoring distribution G on the bin grid.

        Fitted once per training fold from that fold's own (t, e), so no
        held-out information reaches the weights.  Returned as a length-n_bins
        vector so a weight lookup inside the training loop is an index, not a fit.
    """
    t = t.detach().long().reshape(-1).cpu().clamp(0, int(n_bins) - 1)
    e = e.detach().int().reshape(-1).cpu()
    n_bins = int(n_bins)
    counts = torch.bincount(t, minlength = n_bins).double()
    at_risk = counts.flip(0).cumsum(0).flip(0)                 # #{T >= b}
    cens = torch.bincount(t[e == 0], minlength = n_bins).double()
    frac = (1.0 - cens / at_risk.clamp(min = 1.0)).clamp(min = 0.0)
    return torch.cumprod(frac, 0).clamp(min = CENSOR_FLOOR)


def landmark_row_days(x_exp, patient_idx, visit_days):
    """
        Landmark day of every expanded prefix row.

        build_landmark_prefixes NaNs out everything after visit v, so the number
        of observed timesteps left in a row IS v+1; that recovers the visit index
        without changing the helper's signature or its tested return shape.
    """
    visit_idx = (~torch.isnan(x_exp[:, :, 0])).sum(dim = 1).long() - 1
    visit_idx = visit_idx.clamp(min = 0, max = visit_days.shape[1] - 1)
    days = visit_days[patient_idx, visit_idx]
    return torch.nan_to_num(days, nan = 0.0).clamp(min = 0.0)


def horizon_likelihood_loss(outcomes, cif, t, e, s, weight, day, split_time,
                            horizons, g_bins, ipcw_clip = 10.0,
                            event_of_interest = 0):
    """
        IPCW-weighted binary log likelihood of the landmark-conditioned CIF at
        each deployment horizon, summed over rows exactly the way
        negative_log_likelihood sums over patients (so gamma is on the same
        scale as beta) and averaged over horizons (so gamma does not change
        meaning when a horizon is added or removed).

        Args:
            day        : [R] landmark day of each expanded row.
            split_time : [n_bins+1] bin edges, as a tensor.
            g_bins     : [n_bins] censoring survival, from censoring_survival_by_bin.
            ipcw_clip  : cap on 1/G.  Late landmarks push b_h deep into follow-up
                         where G is small; uncapped, a handful of rows would carry
                         most of the gradient.
    """
    n_bins = cif[0].shape[1]
    device = cif[0].device
    rows = torch.arange(len(t), device = device)
    g_bins = g_bins.to(device)

    F_s = torch.stack([c[rows, s] for c in cif])
    S_s = (1.0 - F_s.sum(0)).clamp(min = SURVIVAL_FLOOR)
    F_eoi_s = F_s[event_of_interest]

    # Rows whose landmark is at or past the event bin carry no information about
    # the remaining horizon -- the same rule landmark_nll_loss applies.
    alive_at_s = (s < t).to(F_s.dtype)

    total = torch.zeros((), dtype = cif[0].dtype, device = device)
    for h in horizons:
        # Matches ddh_api._visit_day_bins: np.digitize(right=True) - 1 is
        # torch.bucketize(right=False) - 1, NOT bucketize(right=True).
        b_h = (torch.bucketize(day + float(h), split_time, right = False) - 1)
        b_h = b_h.clamp(0, n_bins - 1)

        p = ((cif[event_of_interest][rows, b_h] - F_eoi_s) / S_s).clamp(
            HORIZON_EPS, 1.0 - HORIZON_EPS)

        y = ((t <= b_h) & (e == event_of_interest + 1)).to(p.dtype)
        known = ((t > b_h) | (e != 0)).to(p.dtype)
        # A horizon landing inside the landmark's own bin is not a prediction.
        resolvable = (b_h > s).to(p.dtype)

        w_ipcw = torch.where(
            t <= b_h,
            1.0 / g_bins[(t - 1).clamp(min = 0)],
            1.0 / g_bins[b_h],
        ).clamp(max = float(ipcw_clip))

        w = weight.to(p.dtype) * w_ipcw * known * resolvable * alive_at_s
        total = total - (w * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum()

    return total / len(outcomes) / max(len(horizons), 1)
