# Project
You are the lead developer for this project. Our goal is to reproduce the results from the paper: "1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities". We then want to check if the results are replicable with a transformers model instead of an MLP.

## Technical Environment
- You are running inside of a devcontainer docker
- Use `uv` for all package management.
- Report any issues you may have with the technical environment back to the user.

## Coding Rules
- Make sure that code is as simple as possible, and try to keep changes minimal whenever possible.
- Don't edit any code unrelated to the task you were given.
- Make sure that you look at the bigger picture and that your code is compatible with the rest of the codebase.
- Don't leave comments in the code unless an unobvious decision was made or the code is extremely complex.
- Make sure all functions have docstrings in a consistant format.

## Editing this file
- If you struggle with a task, or spot a repeating pattern, feel free to edit this AGENTS.md file, and treat is as a persistent place for notes, and important observations.

## Notes & Observations

### `negative_mode == "cross"` performance (2026-06-26)
- `cross` mode requires `hybrid_goal_encoder=True`. In `loss.py`, the critic
  loss needs composite (raw_a, text_b) embeddings, where each composite is a
  distinct input to the *nonlinear* backbone (MLP layer 2 `swish`, or the
  transformer) and so cannot be obtained by gathering per-goal embeddings.
- Two cost drivers: (1) the g-encoder forward over composites, (2) the
  `(B, n_neg)` logits matrix. The logits matrix is the same size in both
  modes; the slowdown was entirely in the g-encoder forward.
- **Why standard negatives are cheap:** they reuse the `B` per-goal embeddings
  already computed for the positives (pure gather, no extra forward). Cross
  composites can't be gathered this way because the backbone is nonlinear.
- **Key fact:** `ant_u4_maze` has only `N=10` distinct goals (env samples
  goals by index from `possible_goals`). So there are only `N^2 = 100`
  distinct composites that could ever exist — recomputing them per batch
  sample was pure waste (the same composite ran ~B/N times per step).
- **Fix: N^2-composite caching.** `HybridGoalEncoder.__call__(grid=True)`
  computes the full `(N, N, D)` grid of all composites once per step in
  `O(N^2)` encoder work (MLP: outer-sum of the two linear halves + one
  `swish`/matmul over `N^2` rows; semantic: one backbone call on `N^2`
  token-pairs). `loss.py` then maps batch goals → indices via
  `goal_indices` (nearest-neighbour, matching the frozen-text encoder's
  internal mapping) and gathers `grid[a, b]`. This is **mathematically
  identical** to per-sample forwards (`grid[a,a] == standard(a)` verified),
  but reduces encoder work from `O(B*K)` to `O(N^2)` per step. For u4_maze
  that's ~100 vs ~33k-174k forwards.
- `cross_negative_count` (`K`, default 32) now caps cross negatives per row
  at `K`, with the remaining `N-1-2K` filled by cheap standard grid gathers,
  keeping total negatives per row at exactly `N-1`.
- Semantic backbone: all three paths (standard / pairwise / grid) use
  `nn.remat(TransformerBackbone)` under one name so params are shared and
  activation memory under `value_and_grad` is bounded.
- Cross/std partner goal indices are disjoint (mod N) so no negative is
  duplicated.
- **Caveat:** the cache is per-step only (weights change each SGD step); it
  does not persist across steps.