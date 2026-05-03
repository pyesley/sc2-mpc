# Macro + SMPC Integration Plan

A focused analysis of `sc2_macro_strategy.md` and concrete next steps to
turn the SMPC work in this repo into a full SC2 bot. Written for an AI
agent (or human) picking this up cold.

---

## 1. Executive Summary

`sc2_macro_strategy.md` specifies an AlphaStar-style macro neural network
(transformer entity encoder + CNN spatial encoder + MLP scalar encoder +
LSTM core + auto-regressive action heads) trained via supervised
behavioral cloning on Blizzard replays followed by V-trace self-play RL.
It explicitly references "SMPC micro" (this repo) as the combat layer
and defines a clean `MacroMicroInterface` for the handoff.

The architectures are compatible by design — what's missing is a
**bridge layer** that turns our scenario-specific SMPC implementations
into something a macro process can call for arbitrary mid-game
compositions, plus enough of a macro to actually produce game state for
the SMPC to act on.

**Recommended path**: start with a hardcoded heuristic macro + the
existing SMPC. Get something playing the AI in two weeks. Replace
pieces from there. Do NOT start with the LSTM — the AlphaStar
architecture is months of training infrastructure before it does
anything.

---

## 2. What's Working in This Repo (Current State)

### 2.1 Architecture today

```
scenario_<X>.py         ←  python-sc2 BotAI; spawns units, builds
                            scenario State each step, calls MPC, executes
                            chosen actions on units
       │
       ▼
mpc_vectorized_<X>.py   ←  Batched stochastic MPC: samples N×S candidate
                            action sequences, simulates each forward H
                            steps with stochastic dynamics, scores by
                            cost, picks best by CVaR
       │
       ▼
cost_<X>.py             ←  Cost function (single-state spec form +
cost_primitives.py          batched form). Variants override module-level
                            weights (W_*). EXTRA_COST_FN(_BATCH) hooks
                            allow plugging in new components without
                            touching the composer.
       │
       ▼
eureka_iterate.py       ←  Runs K cost variants × M games in parallel
                            (multiprocessing + ProcessPoolExecutor + one
                            python-sc2 game per process), aggregates
                            win rate + per-component cost stats per
                            variant
```

### 2.2 File-by-file inventory (relevant to a full-game bot)

#### Scenarios (each is one combat situation we've solved)

| File | Composition | Status / Win rate |
|---|---|---|
| `micro_scenario.py` + `cost_function.py` + `mpc_vectorized.py` | 2 Marines vs 1 Zealot | ~100% (existing) |
| `scenario_3v2.py` + `cost_3v2.py` | 3 Marines vs Zealot+Stalker | ~ winning, not measured at N=30 |
| `scenario_drop.py` + `cost_drop.py` | Medivac+4M vs Zealot+Stalker (drop micro) | working |
| `stim_kite_scenario.py` + `stim_kite_cost.py` | Stim Marine vs Zealot | working |
| `scenario_circle.py` + `cost_circle.py` (+ `mpc_vectorized_circle.py`) | 3 Marines vs 2 Zealots | 90% (cost_circle), confirmed at N≥30 |
| **`scenario_bio.py` + `cost_bio.py` + `mpc_vectorized_bio.py`** | **6 Marine + 2 Marauder + 1 Medivac vs 3 Zealot + 2 Stalker** | **100%** with `cost_bio_v6a` (90 wins / 0 losses across 30+60 game confirmation) |

#### Bio cost variants (Eureka iteration history)

| Variant | Edit | Win rate |
|---|---|---|
| `cost_bio.py` | baseline (compositional primitives + composer) | 47% (N=30, OLD spawn position) → 97% (NEW spawn) |
| `cost_bio_v1a` | survival-first (death penalties 3x) | 12% — backfired |
| `cost_bio_v1b` | marauder-front formation cost | 50% (N=30, OLD spawn) |
| `cost_bio_v1c` | aggressive shooter pinning | 50% (N=30) |
| `cost_bio_v2a` | formation-lock (stronger marauder_front + SHIELD-heavy) | 42% (N=12) |
| `cost_bio_v2b` | protect-marines (heavy survival weights) | 8% — disaster |
| `cost_bio_v2c` | early-hold-and-shoot (HOLD mode 14% → 32%) | 57% (N=30) |
| `cost_bio_v3a/b/c` | various combinations | 42-67% (N=12, no real edge) |
| `cost_bio_v4a/b/c` | longer MPC horizon, more candidates | **8-67%** — bigger MPC was WORSE |
| `cost_bio_v5a/b/c` | shorter horizon + focus_fire boost | 25-92% (N=12 noise) |
| **`cost_bio_v6a`** | v2c + W_COHESION=5 + spawn back 6 units | **100%** — the winner |

#### Core infrastructure

- `cost_primitives.py` — type-pair matchup library (kite_marine_vs_zealot,
  engage_marine_vs_stalker, marauder vs stalker/zealot, medivac
  safety/heal). Each primitive is a small numpy function that
  vectorizes naturally. **This is the foundation for compositional
  cost in arbitrary compositions.**
- `eureka_iterate.py` — multi-variant Eureka harness. Defaults to N=30
  per variant, parallel cap of 20 (avoids SC2 process contention on a
  22-core machine). Reads `COST_MODULE` env var, `SCENARIO` env var,
  `GAMES_PER_VARIANT`, `MAX_PARALLEL`.
- `run_parallel.py` — simpler version, runs same scenario × N games.
- `visualizer.py` (2v1) and `visualizer_bio.py` (bio) — matplotlib
  TkAgg live viz. Type-distinguishing markers, HP labels, damage
  flashes, heal indicator line, formation trails.

### 2.3 Performance numbers worth knowing

- Vectorized MPC step: **~16-30 ms** (bio with team-mode sampler).
  Original loop-based MPC was ~700 ms — the vectorization is a 30-60×
  win.
- One bio game wall clock: **~17-25 s** (decisive wins) up to 60 s
  (timeout cap). Game time is shorter than wall clock because SC2 runs
  faster than realtime in headless mode but is throttled by our MPC
  step latency.
- 30 parallel games of bio: **~30-60 s wall clock** with parallel cap
  of 20.
- 60 games of v6a confirmation: **~80 s wall clock**.

### 2.4 Key lessons from this session (DO NOT RELEARN)

1. **Small-batch (N=12) win-rate measurements are useless for ranking
   variants.** 95% Wilson CI for 9/12 wins is roughly [40%, 90%] —
   wide enough that "the iter-2 winner" was indistinguishable from
   baseline. We chased noise for several iterations. Always use N≥30
   per variant for variant comparisons. The harness now defaults to
   this.

2. **Scenario design dominates cost iteration.** Five Eureka iterations
   moving cost weights bought us +10 percentage points (47% → 57%). A
   single 15-minute change moving the bio spawn 6 units further back
   bought us +40 points (47% → 97%). Always check the scenario before
   blaming the cost.

3. **The action sampler is often the bottleneck before the cost.**
   Independent per-unit sampling could not produce coordinated team
   moves. Adding 5 team modes (independent / retreat / focus_fire /
   shield / hold) doubled win rates on circle (25% → 60%+). Variants
   with new cost components only worked once the sampler could
   actually produce the behavior the cost rewarded (`marauder_front`
   went from inert to load-bearing this way).

4. **Bigger MPC search can be WORSE.** Doubling the horizon from 8 to
   12 dropped bio win rate from 75% to 8%. Longer horizon plans
   commit to predicted futures the simulator gets wrong; the
   simulator-vs-real-game accuracy gap is the hard limit. Improving
   simulator fidelity is more valuable than expanding search.

5. **Simulator constants matter.** Bio's `ZEALOT_DPS` was 26.3 (real
   value 18.6, off by 41%) — MPC over-feared zealots. `MARAUDER_DPS`
   was a single fixed value (9.3) missing the +10 bonus vs armored
   stalker (real DPS vs stalker: 17.76, almost 2× higher). MPC didn't
   know marauders were the right answer to the stalker problem. Fix
   these BEFORE iterating cost weights on top.

6. **Process-level oversubscription degrades win rates.** Running
   30 SC2 processes on 22 cores depressed measured win rate from ~75%
   to ~50%. Cap parallel game count below CPU count.

7. **The user's strategic observations are usable as cost components.**
   "If the enemy splits the group it tends to hunt them down" turned
   directly into `W_COHESION` (penalty on max pairwise bio distance > 6),
   which moved v6a from 97% to 100% — a reliable +3 points.

---

## 3. Analysis of `sc2_macro_strategy.md`

### 3.1 What's well-designed

- **Clean separation of macro vs micro responsibilities.** The doc
  explicitly says SMPC handles unit-level combat, macro handles
  production/economy/timing. This is the right factoring — these
  decisions operate at very different time scales (macro: every 1-12
  seconds of game time; micro: every game frame ≈ 0.04 s).
- **Auto-regressive action decomposition.** Splitting the action into
  (action_type → delay → spatial_target) is the AlphaStar pattern and
  it works.
- **Explicit `EngagementRequest` / `EngagementResult` dataclasses
  (section 7).** These define the bridge well. They're what we should
  actually implement.
- **Action masking for training stability (section 5).** Critical and
  often forgotten. Don't let the network output illegal actions.
- **Hard-coded safety rails (section 10).** Smart — the NN doesn't
  need to learn "build supply before getting blocked" from scratch.
- **Hyperparameter table (section 12).** Concrete starting numbers,
  not just hand-waving.

### 3.2 What naturally fits with our SMPC

The doc's `MacroMicroInterface.should_engage_smpc()` triggers a handoff
to "the SMPC controller". Our `mpc_select_action_vectorized()` IS that
controller. The mapping:

| Doc concept (section 7) | Our equivalent | Notes |
|---|---|---|
| `EngagementRequest{army_unit_tags, objective, target_position, enemy_composition_estimate}` | scenario `BioState` / `StateCircle` | our state is type-separated arrays per scenario; needs generalization |
| `smpc_controller.step(obs)` | `mpc_select_action_vectorized(state)` | wrap with PySC2-obs → state conversion |
| `EngagementResult{outcome, units_lost, units_killed_estimate, remaining_army_supply}` | not produced today | trivially computable from before/after state diff |
| 5 team modes in our sampler | implicit in macro's "objective" | macro's "attack" vs "defend" vs "retreat" maps roughly to FOCUS_FIRE/SHIELD/RETREAT |

### 3.3 Integration gaps (what the doc doesn't solve)

1. **Generic compositional state.** The doc assumes there's a single
   `EngagementRequest` for arbitrary unit lists. Our `BioState` has
   `marine_positions`, `marauder_positions`, etc. as separate fields —
   tied to a fixed composition. Need an `EngagementState` with
   `my_units: List[UnitInfo]` / `enemy_units: List[UnitInfo]`.

2. **Cost-variant classifier.** The doc never says HOW SMPC picks
   which cost function to use. With our library
   (`cost_bio_v6a`, `cost_circle`, `cost_3v2`, `cost_drop`,
   `cost_function`, `stim_kite_cost`), there's no automatic selection.
   Need a function `(my_units, enemy_units) → variant_name`. Rule
   table is fine for v1.

3. **Multi-engagement handling.** A real game can have multiple
   simultaneous fights (main army + runby). Need to spatially cluster
   controlled units, instantiate one SMPC per cluster.

4. **Engagement detection beyond "enemy within 15".** Real triggers:
   our army arriving at attack target, base under attack, scout
   dying, opportunity for picking off isolated enemy units.

5. **Returning a real `EngagementResult`.** Today our SMPC just
   terminates with a per-game WIN/LOSS print. Need to track
   per-engagement deltas (units lost, units killed, time elapsed,
   HP delta) and return them so macro's value function can learn.

6. **Library API mismatch.** The doc uses **PySC2** (Blizzard wrapper);
   we use **python-sc2** (community fork by burnysc2). Different
   `obs` and action interfaces. **Recommendation: stay on python-sc2**
   — that's where our scenario code already runs, and python-sc2 is
   easier to use. The macro doc's PySC2 references can be ported by
   updating the `obs.observation.raw_data.units` style accesses to
   `self.units` / `self.enemy_units` style.

---

## 4. Recommended Development Path

Three phases. **Each phase produces something that runs.** Don't try
to build the AlphaStar LSTM as the first step — without macro
infrastructure to feed it observations and replay parsing to train
it, you'll spend months before you have a bot you can play.

### Phase 1 — Heuristic macro + existing SMPC (~1-2 weeks)

Build a hardcoded macro (200-400 lines) that does a stock build order
and triggers SMPC at army-attack times. Use our existing
`cost_bio_v6a` style cost functions for combat. Test vs Computer AI.

**Deliverable**: `bot_full_game.py` that beats `Computer(VeryEasy)`
consistently and `Computer(Easy)` sometimes. Prove the
macro-micro handoff works end-to-end.

### Phase 2 — Compositional cost library (~1-2 weeks)

Generalize cost from per-scenario to per-composition. Add type-pair
primitives for the unit types that appear in real games (~15 more
primitives covers most of TvP/TvZ early-mid). Build the
cost-variant classifier (rule table). Now SMPC handles arbitrary
army compositions.

**Deliverable**: bot still beats VeryEasy and now beats Easy, Medium,
sometimes Hard.

### Phase 3 — Trained NN macro (~months, optional)

Replace heuristic macro with the LSTM from `sc2_macro_strategy.md`.
Use heuristic macro from phase 1 as the supervised target (replay
parsing is months of work; bootstrapping from your own heuristic is
faster). Eventually self-play RL via V-trace.

**Deliverable**: a learning bot. This is the big-budget item.

---

## 5. Phase 1 Detail: Building `bot_full_game.py`

This is the most important section. **Read this carefully before
starting.**

### 5.1 File structure for phase 1

```
sc2-mpc/
├── (existing scenario_*.py, cost_*.py, mpc_vectorized_*.py)
│
├── full_game/
│   ├── bot_full_game.py        ← entry point: BotAI subclass
│   ├── engagement_state.py     ← generic EngagementState dataclass
│   ├── unit_info.py            ← UnitInfo dataclass
│   ├── heuristic_macro.py      ← hardcoded build order + production logic
│   ├── engagement_detector.py  ← when to call SMPC
│   ├── unit_clusterer.py       ← spatial clustering for multi-engagements
│   ├── cost_classifier.py      ← composition → cost variant name
│   ├── smpc_runner.py          ← wraps mpc_select_action_vectorized
│   ├── obs_to_state.py         ← python-sc2 obs → EngagementState
│   └── execute_actions.py      ← MPC actions (np arrays) → unit commands
└── run_full_game.py            ← launches one game
```

### 5.2 Generic state dataclass (start here)

```python
# unit_info.py
from dataclasses import dataclass
import numpy as np

@dataclass
class UnitInfo:
    tag: int                    # python-sc2 unit tag (stable id)
    type_id: int                # UnitTypeId.value
    type_name: str              # "Marine", "Marauder", etc.
    pos: np.ndarray             # shape (2,) — x, y
    hp: float
    hp_max: float
    shield: float
    shield_max: float
    energy: float
    energy_max: float
    weapon_ready: bool          # weapon_cooldown == 0
    is_flying: bool
    is_alive: bool              # always True when constructed; flips when killed

# engagement_state.py
from dataclasses import dataclass, field
from typing import List
from .unit_info import UnitInfo

@dataclass
class EngagementState:
    my_units: List[UnitInfo]    # variable composition
    enemy_units: List[UnitInfo]
    step: int
    time: float

    def my_by_type(self, type_name: str) -> List[UnitInfo]:
        return [u for u in self.my_units if u.type_name == type_name]

    def enemy_by_type(self, type_name: str) -> List[UnitInfo]:
        return [u for u in self.enemy_units if u.type_name == type_name]

    def composition_key(self) -> tuple:
        """Hashable tuple of (type_name, count) sorted by name. Used by
        the cost classifier."""
        from collections import Counter
        my = tuple(sorted(Counter(u.type_name for u in self.my_units).items()))
        enemy = tuple(sorted(Counter(u.type_name for u in self.enemy_units).items()))
        return (my, enemy)
```

### 5.3 obs → EngagementState (the bridge)

```python
# obs_to_state.py
import numpy as np
from .unit_info import UnitInfo
from .engagement_state import EngagementState

def obs_to_engagement_state(bot_ai, my_unit_tags, enemy_unit_tags, step, time):
    """python-sc2 BotAI obs → generic EngagementState.

    bot_ai: the BotAI instance (has self.units, self.enemy_units)
    my_unit_tags: list of int tags for units to control
    enemy_unit_tags: list of int tags for enemy units in this engagement
    """
    my = []
    for tag in my_unit_tags:
        u = bot_ai.units.find_by_tag(tag)
        if u is None:
            continue        # died
        my.append(UnitInfo(
            tag=u.tag,
            type_id=int(u.type_id.value),
            type_name=u.type_id.name,
            pos=np.array([u.position.x, u.position.y], dtype=np.float64),
            hp=float(u.health),
            hp_max=float(u.health_max),
            shield=float(u.shield),
            shield_max=float(u.shield_max),
            energy=float(u.energy),
            energy_max=float(u.energy_max),
            weapon_ready=(u.weapon_cooldown == 0),
            is_flying=u.is_flying,
            is_alive=True,
        ))

    enemy = []
    for tag in enemy_unit_tags:
        u = bot_ai.enemy_units.find_by_tag(tag)
        if u is None:
            continue
        enemy.append(UnitInfo(... same fields ...))

    return EngagementState(my_units=my, enemy_units=enemy, step=step, time=time)
```

### 5.4 Cost classifier (rule table is fine for v1)

```python
# cost_classifier.py
from collections import Counter

def pick_cost_variant(state):
    """Return the COST_MODULE name to use for this engagement."""
    my = Counter(u.type_name for u in state.my_units)
    enemy = Counter(u.type_name for u in state.enemy_units)

    has_medivac = my.get("Medivac", 0) > 0
    has_marauder = my.get("Marauder", 0) > 0
    only_marines_my = (set(my.keys()) - {"Marine"}) == set()
    only_zealots_enemy = (set(enemy.keys()) - {"Zealot"}) == set()
    has_zealot_enemy = enemy.get("Zealot", 0) > 0
    has_stalker_enemy = enemy.get("Stalker", 0) > 0

    # Rules (most specific first)
    if has_medivac and has_marauder and (has_zealot_enemy or has_stalker_enemy):
        return "cost_bio_v6a"

    if my.get("Marine", 0) == 3 and enemy.get("Zealot", 0) == 2 and len(enemy) == 1:
        return "cost_circle"

    if my.get("Marine", 0) == 3 and has_zealot_enemy and has_stalker_enemy:
        return "cost_3v2"

    if my.get("Marine", 0) == 2 and only_zealots_enemy and enemy.get("Zealot", 0) == 1:
        return "cost_function"     # the original 2v1

    if my.get("Marine", 0) == 1 and enemy.get("Zealot", 0) == 1:
        return "stim_kite_cost"

    # Fallback: bio cost handles arbitrary bio-vs-protoss reasonably
    return "cost_bio_v6a"
```

### 5.5 SMPC runner (composition-agnostic wrapper)

This is the most important piece because it has to bridge our
type-separated MPC implementations to the generic `EngagementState`.

**Approach**: maintain a dispatcher per cost variant that knows how to
build the variant's specific State dataclass from the generic
`EngagementState`. The MPC functions stay scenario-specific; only the
dispatcher is new code.

```python
# smpc_runner.py
import importlib
import numpy as np

class SMPCRunner:
    """Dispatches to the right scenario's MPC based on a chosen variant."""

    def __init__(self):
        self._cache = {}

    def step(self, engagement_state, variant_name):
        """Returns dict {unit_tag: np.ndarray(2) action_or_None_for_attack}."""
        builder = self._dispatchers.get(variant_name)
        if builder is None:
            raise ValueError(f"No dispatcher for variant {variant_name}")
        scenario_state, unit_order = builder(engagement_state)
        actions, components = self._mpc_for(variant_name)(scenario_state)
        return {tag: actions[i] for i, tag in enumerate(unit_order)}

    def _mpc_for(self, variant_name):
        # Each variant_name maps to a (scenario_module, mpc_function) pair
        # e.g. cost_bio_v6a → (scenario_bio.BioState, mpc_vectorized_bio.mpc_select_action_vectorized)
        mapping = {
            "cost_bio_v6a": ("scenario_bio", "BioState",
                              "mpc_vectorized_bio", "mpc_select_action_vectorized"),
            "cost_circle":   ("scenario_circle", "StateCircle",
                              "mpc_vectorized_circle", "mpc_select_action_vectorized"),
            # ... add more as you bring scenarios online
        }
        scen_mod, _, mpc_mod, mpc_fn = mapping[variant_name]
        return getattr(importlib.import_module(mpc_mod), mpc_fn)

    @property
    def _dispatchers(self):
        return {
            "cost_bio_v6a": _build_bio_state,
            "cost_circle":   _build_circle_state,
            # ...
        }


def _build_bio_state(es):
    """EngagementState → BioState. Returns (state, unit_order_for_action_decoding)."""
    from scenario_bio import BioState
    # Filter to expected types in expected counts (BioState has fixed shape)
    marines = [u for u in es.my_units if u.type_name == "Marine"][:6]
    marauders = [u for u in es.my_units if u.type_name == "Marauder"][:2]
    medivacs = [u for u in es.my_units if u.type_name == "Medivac"][:1]
    zealots = [u for u in es.enemy_units if u.type_name == "Zealot"][:3]
    stalkers = [u for u in es.enemy_units if u.type_name == "Stalker"][:2]

    # Pad to expected counts (with hp=0 sentinel = dead)
    while len(marines) < 6:    marines.append(_dead_marine())
    while len(marauders) < 2:  marauders.append(_dead_marauder())
    # ... etc

    state = BioState(
        marine_positions=[u.pos for u in marines],
        marine_hps=[u.hp for u in marines],
        marine_weapon_ready=[u.weapon_ready for u in marines],
        marauder_positions=[u.pos for u in marauders],
        # ... etc all fields
        n_marines=6, n_marauders=2, n_medivacs=1, n_zealots=3, n_stalkers=2,
        step=es.step, time=es.time,
    )
    unit_order = [u.tag for u in marines + marauders + medivacs]
    return state, unit_order
```

Note: this still has the rigidity of BioState's fixed counts. If the
real composition is 4 Marines + 1 Marauder + 1 Medivac, you'll either
pad with dead sentinels or reach for a different cost variant. Phase 2
fixes this with truly variable-shape compositional cost.

### 5.6 Heuristic macro (basic Protoss build for v1)

Don't overthink this. Use a known stock build. `python-sc2` has tons
of working examples — start by reading
`/home/pyesley/sc2-mpc/test_bot.py` for the obvious pattern.

**Suggested first build (Protoss Stalker push)**:
1. Always train probes from each Nexus until 16 per base
2. Build a Pylon when supply_left < 4 and minerals >= 100
3. Build a Gateway when no Gateway exists and a Pylon is ready
4. Build an Assimilator when Gateway exists and < 1 gas per base
5. Build a CyberneticsCore when Gateway is ready
6. Build 2nd Gateway when 1 exists
7. Train Stalkers from each Gateway when affordable
8. When ≥6 Stalkers, attack-move to enemy main base center
9. SMPC takes over when enemy combat units within 15 of our army
   centroid

This is ~150 lines. It loses to most ladder players but beats
VeryEasy/Easy AI. It's enough to test the SMPC handoff.

```python
# heuristic_macro.py — sketch
from sc2.ids.unit_typeid import UnitTypeId

class HeuristicMacro:
    def __init__(self):
        self.attack_launched = False

    async def step(self, bot):
        # Probes
        for nexus in bot.townhalls.ready.idle:
            if bot.minerals >= 50 and bot.supply_left >= 1:
                nexus.train(UnitTypeId.PROBE)

        # Pylon when supply low
        if (bot.supply_left < 4
            and bot.minerals >= 100
            and not bot.already_pending(UnitTypeId.PYLON)):
            await bot.build(UnitTypeId.PYLON, near=bot.townhalls.first)

        # Gateway
        if (bot.structures(UnitTypeId.PYLON).ready.exists
            and bot.structures(UnitTypeId.GATEWAY).amount < 2
            and bot.minerals >= 150):
            await bot.build(UnitTypeId.GATEWAY,
                            near=bot.structures(UnitTypeId.PYLON).ready.first)

        # Cybercore
        if (bot.structures(UnitTypeId.GATEWAY).ready.exists
            and not bot.structures(UnitTypeId.CYBERNETICSCORE).exists
            and bot.minerals >= 150):
            await bot.build(UnitTypeId.CYBERNETICSCORE,
                            near=bot.structures(UnitTypeId.PYLON).ready.first)

        # Assimilator
        if (bot.structures(UnitTypeId.CYBERNETICSCORE).exists
            and bot.gas_buildings.amount < 2 * bot.townhalls.amount
            and bot.minerals >= 75):
            for nexus in bot.townhalls.ready:
                vgs = bot.vespene_geyser.closer_than(15.0, nexus)
                for vg in vgs:
                    if not bot.gas_buildings.closer_than(1.0, vg):
                        await bot.build(UnitTypeId.ASSIMILATOR, vg)
                        break

        # Train Stalkers
        for gw in bot.structures(UnitTypeId.GATEWAY).ready.idle:
            if (bot.minerals >= 125 and bot.vespene >= 50
                and bot.supply_left >= 2
                and bot.structures(UnitTypeId.CYBERNETICSCORE).ready.exists):
                gw.train(UnitTypeId.STALKER)

        # Attack
        stalkers = bot.units(UnitTypeId.STALKER)
        if stalkers.amount >= 6 and not self.attack_launched:
            target = bot.enemy_start_locations[0]
            for s in stalkers:
                s.attack(target)
            self.attack_launched = True
```

### 5.7 Engagement detection (where to call SMPC)

```python
# engagement_detector.py
def detect_engagements(bot, fight_radius=12.0):
    """Returns list of dicts: {my_tags, enemy_tags, centroid}."""
    import numpy as np
    from sklearn.cluster import DBSCAN  # or roll a simple distance-based clusterer

    my_combat = [u for u in bot.units if u.can_attack and not u.is_structure]
    enemy_combat = [u for u in bot.enemy_units if u.can_attack and not u.is_structure]
    if not my_combat or not enemy_combat:
        return []

    my_pos = np.array([[u.position.x, u.position.y] for u in my_combat])
    enemy_pos = np.array([[u.position.x, u.position.y] for u in enemy_combat])

    # For each enemy unit, find closest of our units. If distance <
    # fight_radius, that enemy is "in this engagement".
    engagements = []

    # Cluster our units (by position) into engagement groups
    # Simple: agglomerative clustering with linkage threshold = fight_radius
    clusters = simple_distance_cluster(my_pos, threshold=fight_radius * 1.5)

    for cluster_indices in clusters:
        my_cluster = [my_combat[i] for i in cluster_indices]
        cluster_center = my_pos[cluster_indices].mean(axis=0)
        # Enemies within fight_radius of this cluster's centroid
        enemy_in_fight = [
            e for e in enemy_combat
            if np.linalg.norm(np.array([e.position.x, e.position.y]) - cluster_center) < fight_radius
        ]
        if enemy_in_fight:
            engagements.append({
                "my_tags": [u.tag for u in my_cluster],
                "enemy_tags": [e.tag for e in enemy_in_fight],
                "centroid": cluster_center,
            })
    return engagements
```

### 5.8 Top-level bot loop

```python
# bot_full_game.py
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.player import Bot, Computer

from full_game.heuristic_macro import HeuristicMacro
from full_game.engagement_detector import detect_engagements
from full_game.cost_classifier import pick_cost_variant
from full_game.obs_to_state import obs_to_engagement_state
from full_game.smpc_runner import SMPCRunner
from full_game.execute_actions import execute_smpc_actions

class FullGameBot(BotAI):
    def __init__(self):
        super().__init__()
        self.macro = HeuristicMacro()
        self.smpc = SMPCRunner()
        self.step_count = 0

    async def on_step(self, iteration):
        self.step_count += 1

        # Macro layer: produce + build (always)
        await self.macro.step(self)

        # Detect engagements
        engagements = detect_engagements(self)

        for eng in engagements:
            es = obs_to_engagement_state(self, eng["my_tags"], eng["enemy_tags"],
                                          self.step_count, self.time)
            variant = pick_cost_variant(es)
            tag_to_action = self.smpc.step(es, variant)
            execute_smpc_actions(self, tag_to_action, eng["enemy_tags"])

# run_full_game.py
def main():
    run_game(
        maps.get("AcropolisLE"),     # any standard ladder map
        [
            Bot(Race.Protoss, FullGameBot()),
            Computer(Race.Protoss, Difficulty.VeryEasy),
        ],
        realtime=False,
    )

if __name__ == "__main__":
    main()
```

### 5.9 Phase 1 milestones (validate as you go)

1. **Macro alone runs.** Comment out engagements; let macro build
   stalkers and a-move them. Verify it actually plays a game (might
   lose, that's fine).
2. **Engagement detection works.** Print detected engagements every
   10 steps. Verify it picks up enemy contact.
3. **One scenario integrates.** Hard-code variant = "cost_bio_v6a"
   and test only on a fight that has a medivac. Verify SMPC runs
   without crashing.
4. **Classifier picks correctly.** Test classifier on known
   compositions; verify it returns the right name.
5. **End-to-end win.** Bot beats Computer(VeryEasy). Wins should be
   reproducible.

### 5.10 Watch out for (phase 1 pitfalls)

- **python-sc2 MPC step latency stalls real-time games.** With our
  ~25 ms MPC step and a real game running at ~22 fps, an engagement
  with 4 simultaneous SMPC instances stalls badly. Mitigation:
  reduce `MPC_N_CANDIDATES` (currently 64) to 32 for in-game use, or
  add a min_step_interval that re-uses last step's actions if MPC is
  taking too long. Realtime games are FAR more demanding than the
  headless scenarios.
- **Composition-mismatch edge cases.** Classifier returns a variant
  whose State dataclass expects 6 marines but you only have 4. The
  dispatcher pads with "dead sentinels". Verify the cost function
  treats those as dead (hp=0 → alive=False in our composer). Some
  cost components might still touch them.
- **Unit dies between obs and action.** A marine is in your
  `my_unit_tags` at obs time but dies during the engagement. Your
  action for that unit is a no-op. Don't crash; filter `is None`
  results from `find_by_tag`.
- **Multiple engagements modifying the same units.** Engagement
  detector might place a unit in two engagements if it's between two
  enemy clusters. Pick one (closest enemy cluster) and stick with it.

---

## 6. Phase 2 Detail: Compositional Cost Library

### 6.1 New type-pair primitives to add to `cost_primitives.py`

The minimum set for TvP/TvZ early-mid:

| Primitive | Why |
|---|---|
| `engage_marine_vs_zergling` | TvZ ling defense — marines stay tight, hold position |
| `kite_marine_vs_roach` | roach has range 4, marine 5 — marines kite at edge |
| `engage_marine_vs_baneling` | banelings 1-shot marines on contact; spread is critical |
| `engage_marauder_vs_roach` | marauder slows, roach is armored — marauder beats roach 1v1 |
| `engage_ghost_vs_caster` | EMP and Snipe priority — ghost stays behind army |
| `engage_siege_tank_setup` | tank in siege mode is stationary AOE — costs around tank position |
| `kite_stalker_vs_marauder` | mirror of marauder-vs-stalker; stalker should kite |
| `engage_zealot_vs_zealot` | symmetric melee — concave matters |
| `engage_voidray_vs_armored` | voidray prismatic alignment — bonus vs armored |
| `engage_phoenix_vs_ground` | phoenix lift mechanic; not relevant for v1 |

For each, follow the pattern in `cost_primitives.py`:
- numpy function taking distance array `d` and returning cost array
- distance bands set from real SC2 weapon ranges and unit speeds
- comment line citing the source data

### 6.2 Truly compositional cost composer

Replace `cost_bio.py`'s hand-written composer (per-marine loop, per-marauder
loop, etc.) with a generic one that:

```python
# cost_compositional.py — replaces cost_bio.py for arbitrary armies
def compute_cost(state):
    components = {}

    # 1. enemy_hp drive (universal)
    components['enemy_hp'] = ...

    # 2. survival per-type weighted (universal — sum over types)
    components['survival'] = sum(
        TYPE_WEIGHT[u.type_name] * (1 - u.hp / u.hp_max)
        for u in state.my_units
        if u.is_alive
    )

    # 3. matchup costs — sum over (my_unit, nearest_enemy) pairs,
    #    look up the right primitive based on type pair
    matchup = 0.0
    for my in state.my_units:
        if not my.is_alive: continue
        nearest = nearest_alive_enemy(my, state)
        if nearest is None: continue
        d = distance(my.pos, nearest.pos)
        primitive = MATCHUP_TABLE.get((my.type_name, nearest.type_name))
        if primitive is None: continue
        matchup += primitive(d)
    components['matchup'] = matchup

    # 4. focus_fire bonus (universal)
    weakest = min(state.enemy_units, key=lambda e: e.hp + e.shield)
    in_range = sum(1 for u in state.my_units
                   if u.is_alive and distance(u.pos, weakest.pos) <= weapon_range(u.type_name))
    components['focus_fire'] = -2.5 * in_range

    # 5. medivac heal proximity (only when medivacs in army)
    medivacs = [u for u in state.my_units if u.type_name == "Medivac"]
    if medivacs:
        ...

    # 6. cohesion (anti-split, the v6a finding)
    bio = [u for u in state.my_units if u.type_name in ("Marine", "Marauder") and u.is_alive]
    if len(bio) >= 2:
        max_pair = max(distance(a.pos, b.pos) for a in bio for b in bio)
        if max_pair > COHESION_THRESHOLD:
            components['cohesion'] = W_COHESION * (max_pair - COHESION_THRESHOLD)

    return sum(components.values()), components
```

The vectorized form is harder because variable-length arrays don't
batch as cleanly. **Two options**:

a) **Pad to MAX_UNITS per side.** Maintain `(N, MAX_M, 2)` position
arrays with an alive mask. Wastes compute on dead slots but vectorizes
naturally. Probably best for phase 2.

b) **JIT/numba compile the per-unit loop.** Single-state cost is fast
enough if numba'd. Skip vectorization, but lose 10-30× speedup vs
batched numpy.

Recommend (a) for phase 2; the wasted compute is small and the API
stays clean.

### 6.3 Multi-engagement handling

Each engagement gets its own SMPC instance. Run them in sequence per
on_step (Python is GIL-bound so threading doesn't help; per-engagement
they're fast). If too slow:

- Cache cost-classifier output per (composition_key, recent steps) —
  classification is deterministic per composition
- Stagger MPC: only re-plan engagement N every K steps; reuse previous
  action otherwise
- Reduce per-engagement candidates if N_engagements > 2

### 6.4 EngagementResult tracking

Wrap each engagement with a tracker that records initial unit
HPs/positions and reports the deltas at the end:

```python
class EngagementTracker:
    def __init__(self, engagement_state):
        self.start_hp_my = sum(u.hp for u in engagement_state.my_units)
        self.start_hp_enemy = sum(u.hp + u.shield for u in engagement_state.enemy_units)
        self.start_time = engagement_state.time
        self.start_my_tags = set(u.tag for u in engagement_state.my_units)
        self.start_enemy_tags = set(u.tag for u in engagement_state.enemy_units)

    def finalize(self, final_state):
        my_lost = self.start_my_tags - set(u.tag for u in final_state.my_units)
        enemy_lost = self.start_enemy_tags - set(u.tag for u in final_state.enemy_units)
        return EngagementResult(
            outcome="won" if not my_lost else ("traded" if enemy_lost else "lost"),
            units_lost=list(my_lost),
            units_killed_estimate=len(enemy_lost),
            duration=final_state.time - self.start_time,
            hp_lost=self.start_hp_my - sum(u.hp for u in final_state.my_units),
            hp_dealt=self.start_hp_enemy - sum(u.hp + u.shield for u in final_state.enemy_units),
        )
```

Phase 1 doesn't strictly need this. Phase 3 (NN training) absolutely
needs it.

---

## 7. Suggested New MPC Scenarios

The library's value is having one cost variant per major combat
situation. From the user's earlier scenario list (in conversation
history), prioritize these:

### 7.1 Add new primitives (highest leverage)

1. **Round 21 — 1 Banshee vs Marines (kite at edge)**: introduces a
   *flying air unit kiting ground* primitive. New mechanic: hold-position
   firing. Cost components: edge proximity penalty (don't fall off
   platform), kite distance from marines.

2. **Round 22 — Ghosts + Marines vs Zerglings + Banelings (Snipe)**:
   introduces *spell casting*. Ghost has Snipe ability (450-range
   single-target). Cost components: snipe target priority (banelings
   first), ghost safety (stay behind marines), energy management.

3. **Round 30 — 1 Tank + 1 Medivac vs 3 Stalkers (pickup-during-projectile)**:
   introduces the *pickup-fire-drop* timing trick. Medivac picks up
   tank during stalker projectile flight, projectile misses (tank is
   "in cargo"), drop tank, fire siege shot, repeat. Tests whether MPC
   can plan 2-step ability sequences with timing constraints.

4. **Storm dodging (vs High Templar)**: 8 marines vs 2 Templars + Zealots.
   Storm is 4-radius AOE, 2.85 sec duration, 80 dmg total. Marines
   must spread immediately when storm cast triggers. Tests *reactive
   spread* under time pressure.

### 7.2 Test compositional generalization (later)

- 12 Marines + 4 Marauders + 2 Medivac vs 6 Zealots + 4 Stalkers
  (large bio army — same primitives as bio_v6a, larger scale)
- Bio + Tank (split-army cost — tank in back, bio in front)
- Marine + Liberator vs Mutalisks (air defense kite)

### 7.3 Process for adding a new scenario (use the bio scenario as template)

1. Copy `scenario_bio.py` → `scenario_X.py`. Update spawn block
   (composition + positions).
2. Copy `cost_bio.py` → `cost_X.py`. Strip components that don't
   apply, add scenario-specific ones.
3. Copy `mpc_vectorized_bio.py` → `mpc_vectorized_X.py`. Update
   action-space size, mode templates, simulator mechanics for new
   unit types.
4. Run via `python scenario_X.py` (single game) and verify
   compile + reasonable behavior.
5. Run baseline 30 games via
   `GAMES_PER_VARIANT=30 SCENARIO=scenario_X.py python eureka_iterate.py cost_X`.
6. **Iterate cost in batches of N=30 per variant. Do NOT use N=12.**
7. Add the variant + classifier rule to phase-1's `cost_classifier.py`.
8. Commit each iteration with the win-rate result in the commit message.

---

## 8. Pitfalls and Anti-Patterns

Curated from this session:

1. **Don't trust N=12 win rates.** Confidence interval is too wide.
   Always N=30+ for variant comparisons.
2. **Don't iterate cost weights before checking simulator constants.**
   Bio's wrong `ZEALOT_DPS` constant invalidated multiple cost
   iterations. Audit `cost_primitives.py` and any DPS/HP constants
   when adding a new scenario.
3. **Don't increase MPC horizon assuming bigger = better.** Past ~3 sec
   prediction window, simulator errors compound and longer horizon
   makes things WORSE.
4. **Don't add cost components without verifying the sampler can
   produce the rewarded behavior.** If your sampler can't propose
   "marauder advances while marines hold", a `marauder_front` cost is
   inert. Audit the sampler first.
5. **Don't oversubscribe SC2 processes vs CPU cores.** Cap at
   ~max_cores - 2. Above that, win rate measurements degrade.
6. **Don't write 1000 lines of variant code when 5 lines work.**
   Every cost variant in this repo is either a full file copy (use
   only when adding new logic) or a 5-line module-attr override.
   Default to the override pattern; refactor cost code to expose
   weights when the override pattern would be cleaner.
7. **Don't kill your own structures in `setup_scenario`.** Killing
   the starting Command Center / Nexus triggers an instant Defeat
   if you have no other units yet (lost ~20 minutes of debugging
   to this in scenario_scv_escape.py before deleting).

---

## 9. Reference: Files and Commits

### Key files to read first

```
mpc_vectorized_bio.py         ← reference vectorized SMPC
cost_primitives.py            ← compositional cost building blocks
cost_bio.py                   ← reference cost composer (with variant
                                weights as module attrs + EXTRA_COST_FN
                                hooks)
cost_bio_v6a.py               ← 100% champion variant — model for new
                                weight-override variants
eureka_iterate.py             ← multi-variant Eureka harness (study
                                this before iterating any cost)
scenario_bio.py               ← reference scenario file (BotAI
                                subclass, debug spawn, on_step loop)
visualizer_bio.py             ← matplotlib live viz (TkAgg)
sc2_macro_strategy.md         ← the macro NN spec (referenced here)
```

### Commits worth knowing (in this session)

```
82530fb  Bio scenario: spawn 6 units further back + new W_COHESION cost
         → THE 100% commit. Spawn fix dominated cost iteration.
5d372a2  Sim overhaul: target-aware DPS with armor + marauder armored bonus
         → Fixed ZEALOT_DPS 26.3 → 18.6, added marauder +10 vs armored.
758a611  Bio iter-2..5 + viz; honest: small-batch variance hid true ~50% rate
         → Lessons learned commit. Read the message body.
d41a5be  Bio sampler: team-mode candidates
         → 5 team modes added. The sampler upgrade that unlocked cost
           iteration to actually have effect.
3f30610  Bio cost iter-1: 3 variants tested, all flat or worse than baseline
         → Confirms cost weights alone hit a ceiling with bad sampler.
f1588f5  Bio + Medivac vs Mixed Toss — compositional cost architecture
         → cost_primitives.py introduction.
cba1cad  Round 29 (3 Marines vs 2 Zealots) + Eureka multi-variant harness
         → Eureka harness introduction.
```

### Useful invocations

```bash
# Run a single game with a specific cost variant
SC2PATH=/home/pyesley/StarCraftII COST_MODULE=cost_bio_v6a python scenario_bio.py

# Run a single game with the matplotlib visualizer
SC2PATH=/home/pyesley/StarCraftII COST_MODULE=cost_bio_v6a python scenario_bio.py --vis

# Eureka multi-variant comparison (default N=30, parallel cap 20)
SC2PATH=/home/pyesley/StarCraftII SCENARIO=scenario_bio.py python eureka_iterate.py \
    cost_bio cost_bio_v2c cost_bio_v6a

# Eureka with specific game count
GAMES_PER_VARIANT=60 python eureka_iterate.py cost_bio_v6a

# Larger parallelism cap (only if you have >20 cores AND CPU-bound)
MAX_PARALLEL=30 python eureka_iterate.py ...

# Run a single scenario in parallel via run_parallel.py
SC2PATH=/home/pyesley/StarCraftII python run_parallel.py 16 scenario_bio.py
```

---

## 10. Glossary

- **MPC** — Model Predictive Control. Plan ahead H steps using a
  simulator, pick the action that minimizes predicted cost. Re-plan
  every step.
- **SMPC** — Stochastic MPC. Same as MPC but accounts for uncertain
  futures (e.g., zealot target switching is stochastic). Score
  candidate actions by CVaR (Conditional Value at Risk) over a
  distribution of future scenarios rather than a single deterministic
  rollout.
- **CVaR** — Conditional Value at Risk. Average of the worst α-fraction
  of outcomes. Robust to bad-luck scenarios, not just mean-reward.
- **Eureka loop** — LLM-driven cost-function iteration (paper:
  arxiv.org/abs/2310.12931). LLM proposes cost variants, eval each
  via training/simulation, LLM reflects on results, proposes
  improvements. In our adaptation: human (or Claude) plays the LLM
  role; SMPC + parallel SC2 games plays the eval role.
- **Team mode** — In our action sampler, each candidate trajectory
  commits to one of 5 coordinated tactics (independent / retreat /
  focus_fire / shield / hold) for the entire planning horizon.
  Replaces independent per-unit sampling.
- **Cohesion** — Cost component penalizing the bio army being stretched
  out (max pairwise distance > 6). Encodes the user's observation
  that the enemy splits the army to pick off stragglers.
- **Marauder front (shield) formation** — Marauders advance toward
  enemy centroid, marines + medivac stay behind. Marauders are
  tankier (125 HP, 1 armor, +10 from medivac heal) so they should
  absorb stalker fire while marines deal damage from behind.

---

*Doc written 2026-05-02 in this Eureka session. Reading time ~15 min.
Implementation time for phase 1 ~1-2 weeks of focused work.*
