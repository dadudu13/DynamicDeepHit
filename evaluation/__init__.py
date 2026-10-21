"""How the landmark Dynamic-DeepHit models are trained and evaluated.

Scripts (run with python -m evaluation.<name>; each has --help):
    make_synthetic    a synthetic cohort in the expected layout
    train             train one configuration on every split
    apply_external    score an external / temporal cohort with each split's model
    evaluate          compare configurations on landmarks, visits and the external cohort
    history_ablation  within-patient test of whether earlier visits are used
"""
