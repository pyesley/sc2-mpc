"""
Bio cost — variant v2a: FORMATION-LOCK.

Builds on v1b. Hypothesis: v1b's marauder_front penalty (W=4) is too
weak — losses correlate with formation breaks during opening contact.
Crank the penalty 3.5× and shift the candidate-mode mix sharply toward
SHIELD so the MPC actually has formation-preserving candidates to pick
from in the first few timesteps.

Edits vs cost_bio (baseline) and v1b:
  - v1b's marauder_front hook + W=4  →  W=14   (3.5× stronger)
  - mpc_vectorized_bio.MODE_PROBS:
      INDEPENDENT 30% → 10%   (fewer chaotic candidates)
      SHIELD      18% → 38%   (more formation candidates)
"""

import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio
import mpc_vectorized_bio

# Reuse v1b's marauder_front hook
from cost_bio_v1b import (
    _marauder_front_single, _marauder_front_batch,
)
import cost_bio_v1b
cost_bio_v1b.W_MARAUDER_FRONT = 14.0   # bumped from 4
cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch

# Shift sampler toward SHIELD candidates
mpc_vectorized_bio.MODE_PROBS = np.array([0.10, 0.16, 0.20, 0.38, 0.16])
