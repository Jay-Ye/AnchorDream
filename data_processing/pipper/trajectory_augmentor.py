import numpy as np
from typing import List, Iterable, Optional, Tuple, Callable

def augment_trajectory_xy_progressive(
    poses: np.ndarray,
    cut_frames: Iterable[int],
    moving_offsets: Optional[Iterable[Tuple[float, float]]] = None,
    *,
    offset_sampler: Optional[Callable[[int], np.ndarray]] = None,
    in_place: bool = False,
) -> np.ndarray:
    """
    Augment a trajectory with alternating segments: moving -> skill -> moving -> ...
    - Only x,y are changed; z and orientation (w,x_r,y_r,z_r) are preserved.
    - For each *moving* segment, apply a final XY offset at the segment's *end boundary frame*
      (i.e., the cut), and distribute that total offset *backwards* progressively over all
      frames in the moving segment.
    - The subsequent *skill* segment is translated by the cumulative moved end position
      (i.e., sum of all previous moving offsets), so the skill runs at the new location.
    - Works for any number of stages; first segment is treated as moving.

    Parameters
    ----------
    poses : (T,7) array-like
        Trajectory poses per timestep as [x, y, z, w, x_r, y_r, z_r].
    cut_frames : iterable of int
        Strictly increasing frame indices that segment the trajectory into states.
        Example: [6,25,30,40] with T timesteps yields:
          moving: [0,6) ; skill: [6,25) ; moving: [25,30) ; skill: [30,40) ; moving: [40,T)
        IMPORTANT: we will also apply the *full* offset to the pose at each boundary index `cut`
        (if 0 <= cut < T) to make the end of moving land exactly on the offset.
    moving_offsets : iterable of (dx, dy), optional
        XY offsets for each moving segment, in order. If None, you must provide `offset_sampler`
        to generate them. The number of moving segments is
          n_moving = ceil((n_segments)/2) where n_segments = len(cuts)+1.
    offset_sampler : callable(n_moving) -> (n_moving,2) array, optional
        If provided and `moving_offsets` is None, used to sample per-moving offsets.
    in_place : bool
        If True, modify `poses` in place; otherwise operate on a copy.

    Returns
    -------
    out : (T,7) np.ndarray
        Augmented poses.
    """
    poses = poses if in_place else np.array(poses, copy=True)
    assert poses.ndim == 2 and poses.shape[1] == 7, "poses must be (T,7)"
    T = poses.shape[0]

    # Normalize and validate cuts
    cuts = list(cut_frames)
    if any(c < 0 or c > T for c in cuts) or any(cuts[i] >= cuts[i+1] for i in range(len(cuts)-1)):
        raise ValueError("cut_frames must be strictly increasing and within [0, T].")

    # Build segment boundaries as half-open intervals [start, end)
    # Example with T=... and cuts=[c1,c2,...]: segments are [0,c1), [c1,c2), ..., [c_last,T)
    seg_starts = [0] + cuts
    seg_ends   = cuts + [T]
    n_segments = len(seg_starts)

    # Determine number of moving segments (odd indices are skills; moving starts first)
    moving_indices = [i for i in range(n_segments) if i % 2 == 0]  # 0,2,4,...
    n_moving = len(moving_indices)

    # Prepare offsets for moving segments
    if moving_offsets is None:
        if offset_sampler is None:
            # default: zeros (no-op) to be explicit
            moving_offsets_arr = np.zeros((n_moving, 2), dtype=np.float64)
        else:
            moving_offsets_arr = np.asarray(offset_sampler(n_moving), dtype=np.float64)
            if moving_offsets_arr.shape != (n_moving, 2):
                raise ValueError("offset_sampler must return shape (n_moving,2)")
    else:
        moving_offsets_arr = np.asarray(list(moving_offsets), dtype=np.float64)
        if moving_offsets_arr.shape != (n_moving, 2):
            raise ValueError(f"moving_offsets must have shape ({n_moving}, 2)")

    # We'll accumulate the net XY translation applied up to the *end* of the last moving segment
    cumulative_xy = np.zeros(2, dtype=np.float64)

    # Helper to apply a constant XY translation to a span [a,b)
    def translate_span_xy(a: int, b: int, delta_xy: np.ndarray):
        if a < b:
            poses[a:b, 0] += delta_xy[0]
            poses[a:b, 1] += delta_xy[1]

    # Iterate segments
    moving_idx_ptr = 0
    for seg_i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
        is_moving = (seg_i % 2 == 0)

        if is_moving:
            # Progressive interpolation within [s, e):
            # - distribute from 0 at index s to ~almost full at index e-1
            # - also set the boundary pose at `e` (if e<T) to full offset
            dx, dy = moving_offsets_arr[moving_idx_ptr]
            moving_idx_ptr += 1

            L = e - s
            if L > 0:
                # For i in [s, e): factor = (i - s + 1) / (L + 1)
                # This ensures strictly increasing progression that reaches exactly full at the boundary frame `e`.
                idxs = np.arange(s, e, dtype=np.int64)
                factors = (idxs - s + 1) / (L + 1)
                poses[idxs, 0] += factors * dx
                poses[idxs, 1] += factors * dy

            # Apply full offset to the boundary pose at e, if it exists.
            if e < T:
                poses[e, 0] += dx
                poses[e, 1] += dy

            # Update cumulative offset; subsequent skill segment will be translated by this.
            cumulative_xy += np.array([dx, dy], dtype=np.float64)

        else:
            # Skill segment: translate entire [s, e) by the *current* cumulative_xy.
            # Note: The boundary pose at index s (which equals previous segment's end)
            # has already been moved to the correct end position above.
            # We translate [s, e) so the whole skill sits at that location.
            translate_span_xy(s, e, cumulative_xy)

    return poses
