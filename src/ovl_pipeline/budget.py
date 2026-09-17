"""Conservative forecast arithmetic, not a provider guard or spend attestation.

All money uses integer micro-US dollars. Runtime estimates round upward and
include the required 25% margin. A forecast is never authority to skip actual
provider guards, singleton-resource reconciliation or end-to-end trust gates.
"""
from decimal import Decimal, InvalidOperation
import re

from .canonical import EvidenceError, digest, require_digest
from .schema import fields, integer

CAP = 100_000_000
OPERATING_LIMIT = 90_000_000
MIN_PILOT_MS = 600_000


def money(value):
    if type(value) is not str or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,6})?", value):
        raise EvidenceError("money must be a nonnegative decimal USD string with at most six decimals")
    try:
        amount = int(Decimal(value) * 1_000_000)
    except (InvalidOperation, ValueError) as e:
        raise EvidenceError("invalid money") from e
    integer(amount, 0, 2**53 - 1, "money")
    return amount


def ceil_div(a, b):
    return (a + b - 1) // b


def forecast(value):
    """Forecast full remaining mandatory work from already measured pilot inputs.

    A throughput measurement includes representative checkpoint/log overhead.
    Inputs must name the measurement and exact recipe/corpus digests. This
    calculator does not authenticate those reports: the production controller
    must verify them and must ensure the target counts are the complete streams.
    Fixed costs cover setup, data preparation/reconstruction, export, retained
    storage and all mandatory work outside measured GPU training/replay.

    v1 is retained for historical arithmetic reports only. v2 estimates complete
    update counts from an independently validated stream census, avoiding a false
    assumption that pilot and full-corpus windows have the same padding/masks.
    Its denominator counts only full-shape pilot batches; elapsed time includes
    all pilot work, including partial batches. Every remaining production update,
    including a final short batch, is charged at that conservative measured rate.
    Production callers must require v2 and authenticate the cited census/pilot.
    """
    fields(value, "schema spent_usd committed_future_usd hourly_usd fixed_remaining_usd phases", "forecast")
    version = value["schema"]
    if version not in ("ovl.cost-forecast-input.v1", "ovl.cost-forecast-input.v2"):
        raise EvidenceError("unsupported cost forecast")
    by_updates = version == "ovl.cost-forecast-input.v2"
    spent, committed, hourly, fixed = [money(value[k]) for k in
        ("spent_usd", "committed_future_usd", "hourly_usd", "fixed_remaining_usd")]
    if hourly <= 0:
        raise EvidenceError("positive all-in compute hourly quote required")
    phases = value["phases"]
    if type(phases) is not dict or set(phases) != {"wikipedia", "conversation"}:
        raise EvidenceError("both complete phases required in forecast")
    estimates = {}
    for name, phase in phases.items():
        work, measured = ("updates", "measured_full_batch_updates") if by_updates else ("targets", "measured_targets")
        identity_fields = "measurement_sha256 recipe_sha256 stream_sha256" + (" schedule_sha256" if by_updates else "")
        fields(phase, f"{work} training_completed replay_completed {measured} measured_ms warmup_excluded overhead_included {identity_fields}", "phase forecast")
        for key in identity_fields.split():
            require_digest(phase[key])
        for key in (work, measured):
            integer(phase[key], 1, 2**53 - 1, key)
        integer(phase["measured_ms"], MIN_PILOT_MS, 2**53 - 1, "pilot duration")
        if phase["warmup_excluded"] is not True or phase["overhead_included"] is not True:
            raise EvidenceError("pilot must exclude warmup and include representative checkpoint/log overhead")
        for key in ("training_completed", "replay_completed"):
            integer(phase[key], 0, phase[work], key)
        if phase["replay_completed"] > phase["training_completed"]:
            raise EvidenceError("replay cannot precede recorded training")
        remaining = 2 * phase[work] - phase["training_completed"] - phase["replay_completed"]
        # Include full sequential replay; never estimate a sampled audit.
        ms = ceil_div(remaining * phase["measured_ms"], phase[measured])
        estimates[name] = {f"remaining_{work}_including_replay": remaining,
                           "remaining_ms_with_margin": ceil_div(ms * 5, 4)}
    total_ms = sum(p["remaining_ms_with_margin"] for p in estimates.values())
    compute = ceil_div(total_ms * hourly, 3_600_000)
    total = spent + committed + fixed + compute
    headroom = OPERATING_LIMIT - spent - committed - fixed
    return {
        "schema": "ovl.cost-forecast.v2" if by_updates else "ovl.cost-forecast.v1", "input_sha256": digest(value),
        "scope": "forecast-arithmetic-only-not-production-admission",
        "result": "FITS_OPERATING_LIMIT" if total <= OPERATING_LIMIT else "STOP",
        "cap_micro_usd": CAP, "operating_limit_micro_usd": OPERATING_LIMIT,
        "protected_reserve_micro_usd": CAP - OPERATING_LIMIT,
        "projected_total_micro_usd": total, "remaining_compute_micro_usd": compute,
        "phases": estimates,
        # This is an upper bound to convert to a provider deadline before rental,
        # subtracting setup/checkpoint latency and rechecking billed elapsed time.
        "maximum_affordable_compute_ms": max(0, headroom * 3_600_000 // hourly),
        "provider_guard": "NOT_RUN", "actual_spend_reconciliation": "NOT_RUN",
    }
