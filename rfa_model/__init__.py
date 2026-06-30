"""
rfa_model

Python model for RFA electron trajectory, emission, cascade, and current accounting.
"""

__version__ = "0.1.0"

from .io import load_field_npz, save_field_npz

from .fields import (
    attach_default_owner_name_map,
    build_field_interpolators,
    build_potential_interpolator,
)

from .geometry import (
    default_sample_parts,
    load_and_align_sample_assembly,
    load_and_align_grid_frames,
    sample_bounds,
)

from .samplers import (
    load_default_surface_models,
)

from .cascade import (
    run_cascade_batch_parallel,
    print_cascade_batch_summary,
    save_cascade_batch_tables,
)

from .accounting import (
    summarize_cascade_accounting,
    add_per_primary_to_current_counts,
)