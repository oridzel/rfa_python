# Finite sample collision fix

Back up your current rfa_model directory, then copy the four files in this
package's rfa_model directory over the corresponding existing modules.
Restart the notebook kernel before rerunning the simulation.

Changes:
- Primary and emitted paths include sample STL side/back collisions, including
  voxel-bypass paths. Front-plane and STL hits compete by distance.
- STL sample entries preserve the local face normal; gun emissions use it.
- The analytical front rectangle uses the actual STL front-face center and
  extents, rather than an independently prescribed pivot. Zero beam offsets
  therefore refer to the STL front-face center (a 120 micrometre tangential
  shift for the supplied 89 degree mesh). No mesh or field is transformed.
- Primary CSV adds sample_face, terminal_owner and front_face_angle_deg.
  incidence_angle_deg is now relative to the impacted face, including sides.
- Mismatched sample mechanical angle and dominant STL face normal (>0.1 deg)
  raises ValueError before running.

Validation performed:
python tests/check_saved_impacts.py
Ten saved trajectory collision neighborhoods checked with the patched primary
tracker, a zero-field step and an independent triangle intersector. Five side
entries and five front entries terminate correctly. STL geometry inference
and the 85/89 mismatch guard pass. All four modules pass Python syntax checks.
This is targeted validation, not a full electrostatic/cascade simulation;
field, samplers and remaining package dependencies were not supplied.

Next run:
Use the same energy, seed, count and 89 degree field/mesh, sample_theta_deg=89,
track_points=True, track_stride=1, track_primary_only=True and the same ten
tracked_primary_indices. Export the new primary CSV, tracks and geometry.
Inspect df_primary.groupby(["reason", "sample_face"], dropna=False).size().
Side angles should use side normals (not approximately 88 degrees). Separate
front and side impacts in yield analysis; do not reuse old correction factors.
Side surfaces use the existing sample material models at their local incidence
angle. This assumes those models apply to the sides; surface finish differences
are not modeled. Endpoint/catalog sampler limitations may need further work
if a side incidence lies outside the configured sampler range.

Returning emitted STL impacts retain reason=hit_stl and owner=sample so the
existing owner-based cascade accounting handles them as sample re-impacts.
