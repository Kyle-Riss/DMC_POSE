# Temporal Fall Label Protocol v1

This protocol defines the temporal boundaries used to train and evaluate the
central 20 Hz fall detector. Automatic proposals are review aids only; they are
never ground truth without visual confirmation.

## Unit of annotation

- One physical event is a recording_id.
- Synchronized camera views of the same event share the same physical boundary
  times, allowing at most one frame of view-specific decoding uncertainty.
- All views of one event must stay in the same train/validation/test split.

## Boundaries

All boundaries are zero-based decoded frame indices.

- fall_onset_frame: first frame showing a sustained, irreversible transition
  toward the fall. Preparatory rolling, ordinary repositioning, or clip-start
  motion is not onset.
- impact_frame: first frame where the falling body makes its principal contact
  with the floor or other receiving surface. Reaching the bed edge is not impact.
- post_fall_stable_frame: first frame after impact from which the body remains
  substantially in its post-fall pose for at least 0.5 seconds (10 frames at
  20 Hz). Small limb or head movements are allowed.
- fall_end_frame: last frame belonging to the fall/post-fall episode. If the
  person remains down through the clip, this is the last decoded frame. If a
  recovery begins, it is the frame immediately before recovery starts.
- onset_earliest_frame and onset_latest_frame: uncertainty interval enclosing
  every defensible onset frame. A clear onset may use the same value for both.

Required ordering:

    onset_earliest <= fall_onset <= onset_latest
    fall_onset <= impact <= post_fall_stable <= fall_end

## Review procedure

1. Review the complete clip to understand the action.
2. Compare every synchronized camera view.
3. Inspect onset and impact neighborhoods at one-frame resolution.
4. Confirm post-fall stability using at least the next 0.5 seconds.
5. Save complete only when every boundary is visually supported.
6. Save needs_adjudication when occlusion, ambiguous contact, synchronization,
   or disagreement prevents a defensible boundary.

## Confidence

- high: all boundaries are directly visible in at least two synchronized views.
- medium: the event is clear but one boundary requires a one-to-three-frame
  inference because of occlusion or view angle.
- low: do not mark complete; use needs_adjudication.

## Manifest disposition

- complete: include as a reviewed positive only after subject/session identity and
  a leakage-safe split are assigned.
- excluded: never include as a positive. A verified normal bed exit may be
  curated separately as a hard negative under a different manifest contract.
- needs_adjudication: hold out from training and frozen evaluation. Occluded or
  gradual-impact events remain useful for a later onset-only/bed-exit-fall task,
  but must not receive an invented impact boundary.
- unreviewed or in_progress: block manifest generation.
- All synchronized views of one recording must have the same disposition.

## Prohibited shortcuts

- Do not copy motion-proposal boundaries into ground truth.
- Do not infer subject identity from appearance.
- Do not count synchronized views as independent events.
- Do not synthesize, repeat, or zero-fill missing pose observations.
- Do not use proposal-derived diagnostic metrics as production performance.
