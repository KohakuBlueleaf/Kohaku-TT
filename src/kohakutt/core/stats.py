"""Error statistics of a result against a reference (typically CPU float64).

Statistics are reported, not judged: abs / rel error max / median / mean and cosine.
"""

import torch


def diff_stats(result, reference):
    """Elementwise error of `result` against `reference` (any shapes that broadcast,
    any dtypes; compared in float64): abs error max / median / mean, rel error
    |result - reference| / |reference| max / median / mean over positions where the
    reference is nonzero, cosine similarity, and the count of non-finite results."""
    a = result.detach().double().flatten().cpu()
    b = reference.detach().double().flatten().cpu()
    finite = torch.isfinite(a) & torch.isfinite(b)
    err = (a - b).abs()[finite]
    nonzero = finite & (b != 0)
    rel = (a - b).abs()[nonzero] / b.abs()[nonzero]

    def summary(t):
        if t.numel() == 0:
            return (0.0, 0.0, 0.0)
        return (t.max().item(), t.median().item(), t.mean().item())

    abs_max, abs_median, abs_mean = summary(err)
    rel_max, rel_median, rel_mean = summary(rel)
    af, bf = a[finite], b[finite]
    denom = af.norm() * bf.norm()
    cosine = (af @ bf / denom).item() if denom > 0 else 1.0
    return {
        "abs_max": abs_max,
        "abs_median": abs_median,
        "abs_mean": abs_mean,
        "rel_max": rel_max,
        "rel_median": rel_median,
        "rel_mean": rel_mean,
        "cosine": cosine,
        "nonfinite": int((~torch.isfinite(a)).sum()),
    }


def format_stats(stats):
    """One line: abs max / median / mean, rel max / median / mean, cosine, nonfinite."""
    return (
        f"abs max {stats['abs_max']:.3e} med {stats['abs_median']:.3e} mean "
        f"{stats['abs_mean']:.3e} | rel max {stats['rel_max']:.3e} med "
        f"{stats['rel_median']:.3e} mean {stats['rel_mean']:.3e} | cos "
        f"{stats['cosine']:.6f} | nonfinite {stats['nonfinite']}"
    )
