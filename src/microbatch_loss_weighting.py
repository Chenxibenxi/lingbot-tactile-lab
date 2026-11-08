"""Benchmark-only accumulation; never imported by frozen training runners."""
import math


def weighted_micro_loss(total_loss, action_loss, *, valid_coordinates,
                        global_valid_coordinates, samples, global_samples,
                        alignment_disabled, sequence_mode, equal_router_tokens):
    """Preserve coordinate-mean action and sample-mean auxiliary objectives.

    Counts must be computed over the entire optimizer step before backward.
    Caller must verify router tokens/sample and keep all returned losses attached.
    """
    if not alignment_disabled or sequence_mode != 'per_sequence' or not equal_router_tokens:
        raise ValueError('unsupported auxiliary reduction contract')
    for n in (valid_coordinates, global_valid_coordinates, samples, global_samples):
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n <= 0 or int(n) != n:
            raise ValueError('counts must be finite positive integers')
    if valid_coordinates > global_valid_coordinates or samples > global_samples:
        raise ValueError('micro counts exceed optimizer-step counts')
    return (action_loss * (valid_coordinates / global_valid_coordinates)
            + (total_loss - action_loss) * (samples / global_samples))
