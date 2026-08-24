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

def build_landmark_prefixes(x, s_all):
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
    weight = (1.0 / lengths.double())[patient_idx]

    return x_exp, s_exp, patient_idx, weight

def landmark_nll_loss(model, x_exp, t_exp, e_exp, s_exp, weight):
    """
        Conditional survival NLL evaluated at every landmark, averaged within a
        patient before being summed across patients:

            L = (1/K) * sum_i (1 / V_i) * sum_v - log P(T_i, K_i | T_i > s_iv, X_i,<=v)

        `weight` carries the 1 / V_i factor (see build_landmark_prefixes), so this
        reduces to negative_log_likelihood exactly when every patient contributes
        a single (final-visit) landmark.  Rows whose landmark bin is at or beyond
        the patient's event bin carry no information about the remaining horizon
        and are dropped.
    """
    _, outcomes = model(x_exp)
    cif = [torch.cumsum(ok, 1) for ok in outcomes]
    log_lik = log_likelihood_per_sample(outcomes, cif, t_exp, e_exp, s = s_exp)

    usable = (s_exp < t_exp).double()
    return - (log_lik * weight * usable).sum() / len(outcomes)

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
