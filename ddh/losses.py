import torch

def negative_log_likelihood(outcomes, cif, t, e, s=None):
    """
        Compute the log likelihood loss
        This function is used to compute the survival loss

        s: optional landmark bin index per patient (LongTensor, shape N).
           When provided, subtracts log S(s_i) from each patient's contribution
           so the loss becomes P(T=t, K=k | T > s_i) rather than P(T=t, K=k).
    """
    loss, censored_cif = 0, 0
    for k, ok in enumerate(outcomes):
        # Censored cif
        censored_cif += cif[k][e == 0][torch.arange((e == 0).sum()), t[e == 0]]
        # Uncensored
        selection = e == (k + 1)
        loss += torch.sum(torch.log(ok[selection][torch.arange((selection).sum()), t[selection]] + 1e-10))

    # Censored loss
    loss += torch.sum(torch.log(1 - censored_cif + 1e-10))

    # Landmark conditioning: subtract log S(s_i) only for patients with s > 0.
    # Patients at s=0 have no prior observation window — conditioning on bin 0
    # adds no information and creates a -log S(0) gradient that explodes when
    # many single-plan patients are present, causing model collapse.
    if s is not None:
        valid = s > 0
        if valid.any():
            valid_rows = torch.arange(len(s))[valid]
            survival_at_s = (
                1.0 - torch.stack([cif[k][valid_rows, s[valid]] for k in range(len(cif))]).sum(0)
            ).clamp(min=1e-10)
            loss -= torch.sum(torch.log(survival_at_s))

    return - loss / len(outcomes)

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

    if cancer_id_col is not None:
        x_target = torch.cat([x[:, :, :cancer_id_col], x[:, :, cancer_id_col + 1:]], dim=2)
    else:
        x_target = x
    return torch.nn.MSELoss(reduction = 'mean')(longitudinal_prediction[prediction_mask], x_target[observation_mask])

def total_loss(model, x, t, e, alpha, beta, sigma, s=None, cause_weights=None):
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
    return (1 - alpha - beta) * longitudinal_loss(longitudinal_prediction, x, cancer_id_col=cancer_id_col) +\
              alpha * ranking_loss(cif_ranking, t, e, sigma, s=s, cause_weights=cause_weights) +\
              beta * negative_log_likelihood(outcomes, cif, t, e, s=s)