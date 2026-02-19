import torch


def build_phase1_configs(device=torch.device("cpu")):
    """
    Builds full Phase 1 configuration tensor.

    Returns:
        configs: Tensor [N, 10]
    """

    # --------------------------------------------------------
    # Base Axes
    # --------------------------------------------------------

    families  = torch.arange(5, device=device)                # 0–4
    lobes     = torch.arange(4, 13, device=device)            # 4–12
    orders    = torch.arange(4, 13, device=device)            # 4–12
    scaling   = torch.arange(4, device=device)                # 0–3
    precision = torch.arange(2, device=device)                # 0–1
    whitening = torch.tensor([0, 1], device=device)           # 0–1

    base_axes = torch.cartesian_prod(
        families,
        lobes,
        orders,
        scaling,
        precision,
        whitening
    )
    # Shape: [6480, 6]

    # --------------------------------------------------------
    # Domain Assignment
    # --------------------------------------------------------

    sigma_vals = torch.arange(6.0, 12.0 + 0.5, 0.5, device=device)

    # All valid (min, max) pairs
    pairs = torch.combinations(sigma_vals, r=2)  # [78, 2]

    # Independent wide/narrow combinations
    wide  = pairs.unsqueeze(1)    # [78,1,2]
    narrow = pairs.unsqueeze(0)   # [1,78,2]

    wide_exp   = wide.expand(-1, pairs.shape[0], -1)
    narrow_exp = narrow.expand(pairs.shape[0], -1, -1)

    # Dominance constraint: wide_max > narrow_max
    mask = wide_exp[:, :, 1] > narrow_exp[:, :, 1]

    valid_wide   = wide_exp[mask]
    valid_narrow = narrow_exp[mask]

    domain_tensor = torch.cat([valid_wide, valid_narrow], dim=1)
    # Shape: [3003, 4]
    # [wide_min, wide_max, narrow_min, narrow_max]

    # --------------------------------------------------------
    # Combine Base Axes + Domains
    # --------------------------------------------------------

    B = base_axes.shape[0]
    D = domain_tensor.shape[0]

    base_exp   = base_axes.unsqueeze(1).expand(B, D, -1)
    domain_exp = domain_tensor.unsqueeze(0).expand(B, D, -1)

    configs = torch.cat([base_exp, domain_exp], dim=2)
    configs = configs.reshape(-1, configs.shape[-1])

    # Cast to float32 for storage efficiency
    configs = configs.to(torch.float32)

    return configs


if __name__ == "__main__":
    configs = build_phase1_configs(device=torch.device("cpu"))
    print("Total configs:", configs.shape[0])
