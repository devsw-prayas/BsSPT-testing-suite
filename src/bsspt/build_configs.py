import torch


def build_phase1_configs(device=torch.device("cpu")):
    """
    Builds full Phase 1 configuration tensor.

    Column order matches schema.CONFIG_COLUMNS:
        family_id, K, order, scaling_id, precision_id, whitened,
        wide_min, wide_max, narrow_min, narrow_max

    Returns
    -------
    configs : Tensor [N, 10], float64

    Previously cast to float32 which risks rounding errors on the
    0.5 nm sigma increments and makes integer columns (family_id etc.)
    less safe for large values. Using float64 throughout.
    """

    # --------------------------------------------------------
    # Base Axes
    # --------------------------------------------------------

    families  = torch.arange(5,  device=device, dtype=torch.int64)     # 0-4
    lobes     = torch.arange(4, 13, device=device, dtype=torch.int64)  # 4-12
    orders    = torch.arange(4, 13, device=device, dtype=torch.int64)  # 4-12
    scaling   = torch.arange(4,  device=device, dtype=torch.int64)     # 0-3
    precision = torch.arange(2,  device=device, dtype=torch.int64)     # 0-1
    whitening = torch.tensor([0, 1], device=device, dtype=torch.int64) # 0-1

    base_axes = torch.cartesian_prod(
        families, lobes, orders, scaling, precision, whitening
    ).to(torch.float64)
    # Shape: [6480, 6]

    # --------------------------------------------------------
    # Domain Assignment
    # --------------------------------------------------------

    sigma_vals = torch.arange(6.0, 12.5, 0.5, device=device, dtype=torch.float64)
    # 13 values: 6.0, 6.5, ..., 12.0

    pairs = torch.combinations(sigma_vals, r=2)     # [78, 2]  (min < max by construction)

    wide_exp   = pairs.unsqueeze(1).expand(-1, pairs.shape[0], -1)  # [78, 78, 2]
    narrow_exp = pairs.unsqueeze(0).expand(pairs.shape[0], -1, -1)  # [78, 78, 2]

    # Dominance constraint: wide_max > narrow_max
    mask = wide_exp[:, :, 1] > narrow_exp[:, :, 1]

    valid_wide   = wide_exp[mask]    # [D, 2]
    valid_narrow = narrow_exp[mask]  # [D, 2]

    domain_tensor = torch.cat([valid_wide, valid_narrow], dim=1)
    # Shape: [D, 4] — columns: wide_min, wide_max, narrow_min, narrow_max

    # --------------------------------------------------------
    # Combine Base Axes + Domains
    # --------------------------------------------------------

    B = base_axes.shape[0]
    D = domain_tensor.shape[0]

    base_exp   = base_axes.unsqueeze(1).expand(B, D, -1)
    domain_exp = domain_tensor.unsqueeze(0).expand(B, D, -1)

    configs = torch.cat([base_exp, domain_exp], dim=2)
    configs = configs.reshape(-1, configs.shape[-1])
    # Shape: [B*D, 10], float64

    return configs


if __name__ == "__main__":
    configs = build_phase1_configs(device=torch.device("cpu"))
    print(f"Total configs : {configs.shape[0]:,}")
    print(f"Config shape  : {configs.shape}")
    print(f"Dtype         : {configs.dtype}")
