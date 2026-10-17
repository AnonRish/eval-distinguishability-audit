from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    derandomize=True,
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci")
