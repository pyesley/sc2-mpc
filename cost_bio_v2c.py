"""
Bio cost — variant v2c: EARLY HOLD-AND-SHOOT.

Builds on v1b. Hypothesis: the losses happen in the OPENING (bio
scrambles forward 0-5s before formation sets, gets stalker-shot
during the move). Bias the candidate set toward HOLD so units stay
put while marauders catch up and form the shield.

Edits vs v1b:
  - mpc_vectorized_bio.MODE_PROBS:
      HOLD        14% → 32%   (much more hold candidates)
      INDEPENDENT 30% → 16%
      SHIELD      18% → 22%
  - keeps v1b's marauder_front hook unchanged
"""

import numpy as np
from cost_bio import compute_cost, compute_cost_batch
import cost_bio
import mpc_vectorized_bio
from cost_bio_v1b import _marauder_front_single, _marauder_front_batch

cost_bio.EXTRA_COST_FN = _marauder_front_single
cost_bio.EXTRA_COST_FN_BATCH = _marauder_front_batch

mpc_vectorized_bio.MODE_PROBS = np.array([0.16, 0.16, 0.14, 0.22, 0.32])
