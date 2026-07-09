"""
rfa_model

Python model for RFA electron trajectory, emission, cascade, and current accounting.
"""

__version__ = "0.1.0"

from .io import load_field_npz, save_field_npz

from .fields import (
    OWNER_ID,
    OWNER_NAME,
    attach_default_owner_name_map,
    make_empty_field_grid,
    build_rfa_field,
    calculate_electric_field,
    compute_electric_field,
    build_field_interpolators,
    build_potential_interpolator,
    evaluate_field,
    evaluate_potential,
    E_at_point,
    potential_at_point,
    classify_grid_point,
    mark_analytic_rfa_surfaces,
    mark_named_meshes,
    solve_laplace_red_black_sor,
    solve_laplace_jacobi,
)

from .geometry import (
    default_sample_parts,
    load_and_align_sample_assembly,
    load_and_align_grid_frames,
    sample_bounds,
    build_collision_mesh_dict,
)

from .samplers import (
    load_default_surface_models,
    surface_family,
    sample_surface_event,
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

from .collisions import (
    build_stl_intersector,
    build_stl_bounding_boxes
)

from .plotting import (
    plot_trajectories_3d,
    plot_trajectory_projections,
    plot_hit_points,
    plot_current_balance,
    plot_terminal_counts,
    plot_meshes_3d,
    plot_fixed_voxels_3d,
    plot_owner_slice,
    plot_potential_slice,
)
