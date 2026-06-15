import torch

def negative_log_likelihood(outcomes, cif, t, e):
    """
        Compute the log likelihood loss 
        This function is used to compute the survival loss
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
    return - loss / len(outcomes)

def ranking_loss(cif, t, e, sigma, cause_weights=None):
    """
        Penalize wrong ordering of probability
        Equivalent to a C Index
        This function is used to penalize wrong ordering in the survival prediction

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

    loss = 0
    for k, cifk in enumerate(cif):
        mask = e - 1 == k
        cause_loss = 0
        for ci, ti in zip(cifk[mask], t[mask]):
            # Eligible comparators: all patients still event-free at t_i
            eligible = t > ti
            if eligible.sum() > 0:
                risk_j = cifk[eligible][torch.arange(eligible.sum()), ti]
                # Sum over all valid pairs per DDH paper formulation.
                cause_loss += torch.sum(torch.exp((risk_j - ci[ti]) / sigma))
        loss += cause_weights[k] * cause_loss

    return loss

def longitudinal_loss(longitudinal_prediction, x):
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

    return torch.nn.MSELoss(reduction = 'mean')(longitudinal_prediction[prediction_mask], x[observation_mask])

def total_loss(model, x, t, e, alpha, beta, sigma, cause_weights=None):
    longitudinal_prediction, outcomes = model(x)
    t, e = t.long(), e.int()

    # Compute cumulative function from prediced outcomes
    cif = [torch.cumsum(ok, 1) for ok in outcomes]

    return (1 - alpha - beta) * longitudinal_loss(longitudinal_prediction, x) +\
              alpha * ranking_loss(cif, t, e, sigma, cause_weights=cause_weights) +\
              beta * negative_log_likelihood(outcomes, cif, t, e)