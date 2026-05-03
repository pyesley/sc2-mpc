"""
Round 29: 3 Marines vs 2 Zealots — all-melee opponents.

Tests the "circular kite" cost (cost_circle.py) where two outer marines
orbit on opposite sides of a stationary center marine and split the
zealots' aggro between them. The center marine deals continuous damage
without moving.

Adapted from scenario_3v2.py with two changes:
  - 2 Zealots instead of Zealot + Stalker (no ranged enemy)
  - State carries lists of zealot positions / HPs / alive flags so
    multiple identical enemies are first-class
"""

import os
import sys
import importlib
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.unit import Unit

# Cost module is selectable via env var so the Eureka harness can run
# many variants concurrently (each subprocess sets COST_MODULE).
COST_MODULE_NAME = os.environ.get("COST_MODULE", "cost_circle")
_cost_mod = importlib.import_module(COST_MODULE_NAME)
compute_cost = _cost_mod.compute_cost


# ─── Constants ───────────────────────────────────────────────
MARINE_RANGE = 5.0
MARINE_SPEED = 2.25
MARINE_HP = 45
MARINE_DPS = 9.8

ZEALOT_SPEED = 2.25
ZEALOT_HP = 100
ZEALOT_SHIELDS = 50
ZEALOT_HP_MAX = ZEALOT_HP + ZEALOT_SHIELDS
ZEALOT_DPS = 26.3                # 16 dmg / 0.61 s

N_MARINES = 3
N_ZEALOTS = 2

ZEALOT_SWITCH_SHARPNESS = 3.0    # softmax sharpness for stochastic target choice
ZEALOT_PURSUIT_NOISE = 0.15
MARINE_EXEC_NOISE = 0.05

MAP_CENTER = Point2((32, 32))
TIME_LIMIT = 60.0


# ─── State ───────────────────────────────────────────────────
@dataclass
class StateCircle:
    marine_positions: List[np.ndarray]
    marine_hps: List[float]
    marine_weapon_ready: List[bool]
    zealot_positions: List[np.ndarray]
    zealot_hps: List[float]
    zealot_alive: List[bool]
    n_marines: int
    n_zealots: int
    step: int
    time: float


# ─── Dynamics ────────────────────────────────────────────────
def _zealot_pick_target(z_pos, marine_positions, live_marines, rng=None):
    """Stochastic softmax target selection across live marines."""
    if not live_marines:
        return None
    dists = np.array([np.linalg.norm(z_pos - marine_positions[i]) for i in live_marines])
    if rng is None:
        return live_marines[int(np.argmin(dists))]
    # Softmax over -k * d  (closer = higher prob)
    logits = -ZEALOT_SWITCH_SHARPNESS * dists
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = rng.choice(len(live_marines), p=probs)
    return live_marines[idx]


def simulate_circle(state, marine_actions, dt=0.4, rng=None):
    """Roll out an action sequence. marine_actions: List[List[np.array(2)]]
    indexed [marine_idx][step]."""
    trajectory: List[StateCircle] = []
    m_pos = [p.copy() for p in state.marine_positions]
    m_hp = list(state.marine_hps)
    z_pos = [p.copy() for p in state.zealot_positions]
    z_hp = list(state.zealot_hps)
    z_alive = list(state.zealot_alive)

    n_m = state.n_marines
    n_z = state.n_zealots
    horizon = len(marine_actions[0]) if marine_actions else 0

    for step_i in range(horizon):
        live_marines = [i for i in range(n_m) if m_hp[i] > 0]
        if not live_marines:
            break

        # Marine movement (with execution noise if stochastic)
        moved = [False] * n_m
        for i in live_marines:
            a = marine_actions[i][step_i]
            n = float(np.linalg.norm(a))
            if n > 0.1:
                if rng is not None:
                    m_pos[i] = m_pos[i] + (a / n) * MARINE_SPEED * dt + rng.normal(0, MARINE_EXEC_NOISE, 2)
                else:
                    m_pos[i] = m_pos[i] + (a / n) * MARINE_SPEED * dt
                moved[i] = True

        # Zealot movement: each picks a target, chases
        zealot_targets = [None] * n_z
        for j in range(n_z):
            if not z_alive[j]:
                continue
            tgt = _zealot_pick_target(z_pos[j], m_pos, live_marines, rng=rng)
            zealot_targets[j] = tgt
            if tgt is None:
                continue
            r = m_pos[tgt] - z_pos[j]
            d = float(np.linalg.norm(r))
            if d < 0.1:
                continue
            direction = r / d
            if rng is not None:
                ang = rng.normal(0, ZEALOT_PURSUIT_NOISE)
                ca, sa = np.cos(ang), np.sin(ang)
                direction = np.array([
                    ca * direction[0] - sa * direction[1],
                    sa * direction[0] + ca * direction[1],
                ])
            move_d = min(ZEALOT_SPEED * dt, d)
            z_pos[j] = z_pos[j] + direction * move_d

        # Marine shooting: stationary marines shoot closest live zealot in range
        for i in live_marines:
            if moved[i]:
                continue
            best_j, best_d = None, 1e9
            for j in range(n_z):
                if not z_alive[j]:
                    continue
                d = float(np.linalg.norm(m_pos[i] - z_pos[j]))
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= MARINE_RANGE:
                z_hp[best_j] -= MARINE_DPS * dt

        # Zealot melee damage: each zealot hits its current target if in melee
        for j in range(n_z):
            if not z_alive[j]:
                continue
            tgt = zealot_targets[j]
            if tgt is None:
                continue
            d = float(np.linalg.norm(m_pos[tgt] - z_pos[j]))
            if d < 1.0:
                m_hp[tgt] -= ZEALOT_DPS * dt

        # Death checks
        for j in range(n_z):
            if z_alive[j] and z_hp[j] <= 0:
                z_alive[j] = False
                z_hp[j] = 0.0
        for i in range(n_m):
            m_hp[i] = max(0.0, m_hp[i])

        weapon_ready = [not moved[i] for i in range(n_m)]

        trajectory.append(StateCircle(
            marine_positions=[p.copy() for p in m_pos],
            marine_hps=list(m_hp),
            marine_weapon_ready=weapon_ready,
            zealot_positions=[p.copy() for p in z_pos],
            zealot_hps=list(z_hp),
            zealot_alive=list(z_alive),
            n_marines=n_m,
            n_zealots=n_z,
            step=state.step + step_i + 1,
            time=state.time + (step_i + 1) * dt,
        ))

    return trajectory


# ─── Action sampling ─────────────────────────────────────────
def sample_action_circle(state, marine_idx):
    """Sample one marine's next action.

    Distance-based, no role labels — the cost function induces roles.
    Bias near-zealot marines toward tangential motion (kite); bias
    far-zealot marines toward holding (shoot).
    """
    m_pos = state.marine_positions[marine_idx]
    alive_z = [j for j in range(state.n_zealots) if state.zealot_alive[j]]
    if not alive_z:
        return np.zeros(2)

    dists = [np.linalg.norm(m_pos - state.zealot_positions[j]) for j in alive_z]
    nearest_local = int(np.argmin(dists))
    z_near = state.zealot_positions[alive_z[nearest_local]]
    d_near = dists[nearest_local]

    away = m_pos - z_near
    d = float(np.linalg.norm(away))
    if d > 0.1:
        away_n = away / d
    else:
        ang = np.random.uniform(0, 2 * np.pi)
        away_n = np.array([np.cos(ang), np.sin(ang)])
    tangent = np.array([-away_n[1], away_n[0]])
    if np.random.random() < 0.5:
        tangent = -tangent

    if d_near < 1.8:
        # Melee danger — flee
        w_away = np.random.uniform(0.7, 1.0)
        w_tang = np.random.uniform(-0.4, 0.4)
    elif d_near < 3.5:
        # Kite range — tangential heavy
        w_away = np.random.uniform(-0.1, 0.4)
        w_tang = np.random.uniform(0.5, 1.0)
    elif d_near <= MARINE_RANGE:
        # In firing range, safe distance — likely shoot (hold)
        r = np.random.random()
        if r < 0.55:
            return np.zeros(2)
        elif r < 0.8:
            w_away = np.random.uniform(-0.3, 0.3)
            w_tang = np.random.uniform(-0.5, 0.5)
        else:
            w_away = np.random.uniform(-0.2, 0.5)
            w_tang = np.random.uniform(-0.4, 0.4)
    elif d_near < 7.0:
        # Slightly out of range — close in or hold
        if np.random.random() < 0.35:
            return np.zeros(2)
        w_away = np.random.uniform(-0.7, -0.2)
        w_tang = np.random.uniform(-0.4, 0.4)
    else:
        # Very far — approach
        w_away = np.random.uniform(-1.0, -0.5)
        w_tang = np.random.uniform(-0.3, 0.3)

    direction = w_away * away_n + w_tang * tangent + 0.08 * np.random.randn(2)
    n = float(np.linalg.norm(direction))
    if n > 0.1:
        return direction / n
    return np.zeros(2)


def mpc_select_action(state, n_candidates=128, n_scenarios=6, horizon=8,
                       dt=0.4, cvar_alpha=0.3):
    n_m = state.n_marines
    best_cvar = float('inf')
    best_actions = [np.zeros(2) for _ in range(n_m)]
    best_components: Dict[str, float] = {}

    for _ in range(n_candidates):
        all_actions = []
        for i in range(n_m):
            seq = [sample_action_circle(state, i) for _ in range(horizon)]
            all_actions.append(seq)

        scenario_costs = []
        scenario_comps = []
        for _ in range(n_scenarios):
            rng = np.random.RandomState(seed=None)
            traj = simulate_circle(state, all_actions, dt=dt, rng=rng)
            total = 0.0
            comps: Dict[str, float] = {}
            for sim in traj:
                c, co = compute_cost(sim)
                total += c
                for k, v in co.items():
                    comps[k] = comps.get(k, 0.0) + v
            scenario_costs.append(total)
            scenario_comps.append(comps)

        arr = np.array(scenario_costs)
        sorted_idx = np.argsort(arr)[::-1]
        n_tail = max(1, int(np.ceil(n_scenarios * cvar_alpha)))
        cvar = float(np.mean(arr[sorted_idx[:n_tail]]))

        if cvar < best_cvar:
            best_cvar = cvar
            best_actions = [all_actions[i][0] for i in range(n_m)]
            best_components = scenario_comps[sorted_idx[n_scenarios // 2]]

    return best_actions, best_components


# ─── SC2 Bot ─────────────────────────────────────────────────
class BotCircle(BotAI):
    def __init__(self, visualize=False, use_slow=False):
        super().__init__()
        self.scenario_started = False
        self.setup_done = False
        self.marine_tags: List[int] = []
        self.zealot_tags: List[int] = []
        self.step_count = 0
        self.total_components: Dict[str, float] = {}
        self.game_over = False
        self.visualize = visualize
        self.use_slow = use_slow

    async def on_step(self, iteration: int):
        if self.game_over:
            return

        if not self.scenario_started:
            await self.setup_scenario()
            self.scenario_started = True
            return

        if not self.setup_done:
            marines = self.units(UnitTypeId.MARINE)
            zealots = self.enemy_units(UnitTypeId.ZEALOT)
            if marines.amount >= N_MARINES and zealots.amount >= N_ZEALOTS:
                self.marine_tags = [m.tag for m in marines]
                self.zealot_tags = [z.tag for z in zealots]
                self.setup_done = True
                print(f"Scenario ready: {N_MARINES} Marines vs {N_ZEALOTS} Zealots")
            return

        # Resolve units
        marines = [self.units.find_by_tag(t) for t in self.marine_tags]
        marines = [m for m in marines if m is not None]
        if not marines:
            remaining = self.units(UnitTypeId.MARINE)
            if not remaining:
                self.end_game("LOSS", "All marines dead")
                return
            marines = list(remaining)
            self.marine_tags = [m.tag for m in marines]

        zealots_now = []
        for t in self.zealot_tags:
            z = self.enemy_units.find_by_tag(t)
            if z is not None:
                zealots_now.append(z)

        if not zealots_now:
            # try any remaining zealots
            remaining = self.enemy_units(UnitTypeId.ZEALOT)
            if not remaining:
                self.end_game("WIN", "All zealots killed")
                return
            zealots_now = list(remaining)
            self.zealot_tags = [z.tag for z in zealots_now]

        if self.time > TIME_LIMIT:
            zsummary = ", ".join(f"z{i}={z.health + z.shield:.0f}"
                                  for i, z in enumerate(zealots_now))
            self.end_game("TIMEOUT", zsummary)
            return

        # Build state
        n_z_total = len(self.zealot_tags)
        z_positions, z_hps, z_alive = [], [], []
        for t in self.zealot_tags:
            z = self.enemy_units.find_by_tag(t)
            if z is not None:
                z_positions.append(np.array([z.position.x, z.position.y]))
                z_hps.append(z.health + z.shield)
                z_alive.append(True)
            else:
                z_positions.append(np.array([999.0, 999.0]))
                z_hps.append(0.0)
                z_alive.append(False)

        state = StateCircle(
            marine_positions=[np.array([m.position.x, m.position.y]) for m in marines],
            marine_hps=[m.health for m in marines],
            marine_weapon_ready=[m.weapon_cooldown == 0 for m in marines],
            zealot_positions=z_positions,
            zealot_hps=z_hps,
            zealot_alive=z_alive,
            n_marines=len(marines),
            n_zealots=n_z_total,
            step=self.step_count,
            time=self.time,
        )

        self.step_count += 1
        if self.use_slow:
            actions, components = mpc_select_action(state)
        else:
            from mpc_vectorized_circle import mpc_select_action_vectorized
            actions, components = mpc_select_action_vectorized(state)
        for k, v in components.items():
            self.total_components[k] = self.total_components.get(k, 0.0) + v

        # Pick a priority target for "attack" fallback (closest alive zealot to centroid)
        live_zealots = [z for z in zealots_now if z.health + z.shield > 0]
        if live_zealots:
            marine_centroid = np.mean(
                [np.array([m.position.x, m.position.y]) for m in marines], axis=0)
            priority_target = min(live_zealots,
                                   key=lambda z: ((z.position.x - marine_centroid[0])**2 +
                                                  (z.position.y - marine_centroid[1])**2))
        else:
            priority_target = None

        for i, marine in enumerate(marines):
            action = actions[i] if i < len(actions) else np.zeros(2)
            if float(np.linalg.norm(action)) > 0.1:
                target = Point2((
                    marine.position.x + action[0] * 3,
                    marine.position.y + action[1] * 3,
                ))
                marine.move(target)
            else:
                if priority_target is not None:
                    marine.attack(priority_target)

        if self.step_count % 20 == 0:
            alive_m = sum(1 for m in marines if m.health > 0)
            z_hp_str = ", ".join(
                f"z{i}={'DEAD' if not state.zealot_alive[i] else f'{state.zealot_hps[i]:.0f}'}"
                for i in range(state.n_zealots))
            m_hp_str = ",".join(f"{m.health:.0f}" for m in marines)
            print(f"  Step {self.step_count:4d} | t={self.time:5.1f}s | "
                  f"{z_hp_str} | marines={alive_m} alive [{m_hp_str}]")

    async def setup_scenario(self):
        # Keep starting townhalls alive (avoid auto-Defeat); only kill units.
        if self.units:
            await self.client.debug_kill_unit(self.units)
        if self.enemy_units:
            await self.client.debug_kill_unit(self.enemy_units)

        c = MAP_CENTER
        # 3 Marines on the left
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((c.x - 5, c.y - 2)), 1],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((c.x - 5, c.y)), 1],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((c.x - 5, c.y + 2)), 1],
        ])
        # 2 Zealots on the right
        await self.client.debug_create_unit([
            [UnitTypeId.ZEALOT, 1, Point2((c.x + 5, c.y + 1.5)), 2],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.ZEALOT, 1, Point2((c.x + 5, c.y - 1.5)), 2],
        ])
        print(f"Spawned: {N_MARINES} Marines vs {N_ZEALOTS} Zealots")

    def end_game(self, result: str, reason: str):
        self.game_over = True
        print(f"\n{'='*60}")
        print(f"GAME OVER: {result} — {reason}")
        print(f"Steps: {self.step_count}, Game time: {self.time:.1f}s")
        print("Accumulated cost components:")
        for k, v in sorted(self.total_components.items()):
            print(f"  {k:30s}: {v:10.2f}")
        print(f"{'='*60}\n")

    async def on_end(self, game_result):
        pass


def main():
    visualize = '--vis' in sys.argv
    use_slow = '--slow' in sys.argv
    mode = "loop-MPC (slow)" if use_slow else "vectorized SMPC"
    print(f"Round 29: 3 Marines vs 2 Zealots — circular kite [{mode}, cost={COST_MODULE_NAME}]")
    run_game(
        maps.get("Flat32"),
        [
            Bot(Race.Terran, BotCircle(visualize=visualize, use_slow=use_slow)),
            Computer(Race.Protoss, Difficulty.VeryEasy),
        ],
        realtime=False,
    )


if __name__ == "__main__":
    main()
