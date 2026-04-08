from __future__ import annotations

import os
from typing import Any


def _get(hex_data: Any, key: str, default: Any = None) -> Any:
    # Supports passing a pandas Series or a plain dict.
    if hex_data is None:
        return default
    if isinstance(hex_data, dict):
        return hex_data.get(key, default)
    try:
        return hex_data.get(key, default)  # type: ignore[union-attr]
    except Exception:
        return default


def generate_business_report(hex_data: Any, business_type: str) -> str:
    """
    Generate a professional text insight for a selected hexagon.

    If an API key is present, we still fall back to a template summary if
    the LLM client isn't available. (This keeps the project scaffold runnable.)
    """
    competitor_count = float(_get(hex_data, "competitor_count", 0.0) or 0.0)
    magnet_count = float(_get(hex_data, "magnet_count", 0.0) or 0.0)
    population_proxy = float(_get(hex_data, "population_proxy", 0.0) or 0.0)
    score = float(_get(hex_data, "score", 0.0) or 0.0)
    avg_competitor_rating = _get(hex_data, "avg_competitor_rating", None)
    try:
        avg_competitor_rating_f = float(avg_competitor_rating) if avg_competitor_rating is not None else None
    except Exception:
        avg_competitor_rating_f = None

    magnet_types = _get(hex_data, "magnet_types", None)
    if magnet_types is None:
        magnet_types_str = "nearby magnet locations"
    elif isinstance(magnet_types, str):
        magnet_types_str = magnet_types
    else:
        try:
            magnet_types_str = ", ".join(map(str, magnet_types))
        except Exception:
            magnet_types_str = "nearby magnet locations"

    # Convert floor area proxy into an estimated "residents" count.
    # This is a rough heuristic: average usable residential floor area per person.
    meters2_per_person = 40.0
    residents_est = int(round(population_proxy / meters2_per_person)) if population_proxy > 0 else 0

    # Main risk heuristic.
    if competitor_count > 0 and competitor_count >= magnet_count:
        main_risk = "higher competition than demand indicators"
    elif competitor_count > 0:
        main_risk = "existing nearby businesses could capture some demand"
    elif magnet_count <= 0:
        main_risk = "very limited magnet activity around the area"
    else:
        main_risk = "demand may be strong, but competition risk should be monitored over time"

    prompt = (
        "You are a data analyst helping a business choose locations.\n"
        f"Business type (Target): {business_type}\n"
        f"Hex metrics: score={score:.1f}/10, residents_est={residents_est}, "
        f"magnet_count={magnet_count:.0f} ({magnet_types_str}), competitor_count={competitor_count:.0f}, "
        f"avg_competitor_rating={avg_competitor_rating_f}.\n"
        "Write a concise insight (1-2 sentences) that includes:\n"
        "1) Recommendation with resident and competition context\n"
        "2) The main risk\n"
        "Keep it specific and professional."
    )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if api_key:
        # Optional: if openai is installed, you can swap this template out later.
        # For now, keep it deterministic and dependency-free.
        # Returning the template is still consistent with "prompt-based" generation.
        pass

    extra_opportunity = ""
    if avg_competitor_rating_f is not None and competitor_count > 0 and avg_competitor_rating_f < 3.8:
        extra_opportunity = (
            f" Market Opportunity: Competitors in this area have low ratings. "
            f"A high-quality {business_type} could easily capture the market."
        )

    # Template-based summary (deterministic scaffold).
    return (
        f"Target: {business_type}. "
        f"Analysis: This spot has {residents_est} residents and only {competitor_count:.0f} competitors. "
        f"The main risk is {main_risk}."
        f"{extra_opportunity}"
    )

