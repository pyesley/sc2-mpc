"""
Scenario 3: 3 Marines vs Zealot + Stalker.
Focus-fire the stalker first (outranges and outruns marines),
kite the zealot, then finish the zealot.

Key challenge: stalker has range 6 (>marine 5) and speed 2.95 (>marine 2.25).
Marines CANNOT kite stalkers. Must engage and burst down the stalker
while staying spread from the zealot.
"""

import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.unit import Unit

from cost_3v2 import compute_cost

# ─── Constants ───────────────────────────────────────────────
MARINE_RANGE = 5.0
MARINE_SPEED = 2.25
MARINE_HP = 45
MARINE_DPS = 9.8

STALKER_RANGE = 6.0
STALKER_SPEED = 2.95
STALKER_HP = 80
STALKER_SHIELDS = 80
STALKER_DPS = 9.7  # 13 dmg / 1.34s

ZEALOT_SPEED = 2.25
ZEALOT_HP = 100
ZEALOT_SHIELDS = 50
ZEALOT_DPS = 26.3  # 16 dmg / 0.61s (high burst)

MAP_CENTER = Point2((32, 32))


@dataclass
class State3v2:
    marine_positions: List[np.ndarray]   # list of [x,y] for each marine
    marine_hps: List[float]
    marine_weapon_ready: List[bool]
    zealot_pos: np.ndarray
    zealot_hp: float
    stalker_pos: np.ndarray
    stalker_hp: float
    stalker_alive: bool
    zealot_alive: bool
    n_marines: int
    step: int
    time: float


# ─── Dynamics ────────────────────────────────────────────────
def simulate_3v2(state, marine_actions, dt=0.4):
    """Simulate forward. marine_actions is list of N x [move_dir(2)].
    Marines auto-attack priority target (stalker > zealot) when stationary and in range.
    """
    trajectory = []
    m_pos = [p.copy() for p in state.marine_positions]
    m_hp = list(state.marine_hps)
    z_pos = state.zealot_pos.copy()
    z_hp = state.zealot_hp
    s_pos = state.stalker_pos.copy()
    s_hp = state.stalker_hp
    s_alive = state.stalker_alive
    z_alive = state.zealot_alive
    n = state.n_marines

    for step_i in range(len(marine_actions[0]) if marine_actions else 0):
        live_marines = [i for i in range(n) if m_hp[i] > 0]
        if not live_marines:
            break

        # Move marines
        moving = []
        for i in live_marines:
            action = marine_actions[i][step_i]
            if np.linalg.norm(action) > 0.1:
                m_pos[i] = m_pos[i] + (action / np.linalg.norm(action)) * MARINE_SPEED * dt
                moving.append(True)
            else:
                moving.append(False)

        # Move enemies toward closest marine
        if z_alive and live_marines:
            closest = min(live_marines, key=lambda i: np.linalg.norm(m_pos[i] - z_pos))
            r = m_pos[closest] - z_pos
            d = np.linalg.norm(r)
            if d > 0.1:
                z_pos = z_pos + (r / d) * min(ZEALOT_SPEED * dt, d)

        if s_alive and live_marines:
            # Stalker: move to attack range of closest marine
            closest = min(live_marines, key=lambda i: np.linalg.norm(m_pos[i] - s_pos))
            r = m_pos[closest] - s_pos
            d = np.linalg.norm(r)
            if d > STALKER_RANGE:
                s_pos = s_pos + (r / d) * min(STALKER_SPEED * dt, d - STALKER_RANGE + 0.5)

        # Marines damage enemies (priority: stalker first)
        for idx, i in enumerate(live_marines):
            if moving[idx]:
                continue
            d_stalker = np.linalg.norm(m_pos[i] - s_pos) if s_alive else 999
            d_zealot = np.linalg.norm(m_pos[i] - z_pos) if z_alive else 999

            if s_alive and d_stalker <= MARINE_RANGE:
                s_hp -= MARINE_DPS * dt
            elif z_alive and d_zealot <= MARINE_RANGE:
                z_hp -= MARINE_DPS * dt

        # Stalker damages closest marine in range
        if s_alive:
            for i in live_marines:
                d = np.linalg.norm(m_pos[i] - s_pos)
                if d <= STALKER_RANGE:
                    m_hp[i] -= STALKER_DPS * dt
                    break  # stalker only hits one target

        # Zealot damages closest marine in melee
        if z_alive:
            for i in live_marines:
                d = np.linalg.norm(m_pos[i] - z_pos)
                if d < 1.0:
                    m_hp[i] -= ZEALOT_DPS * dt
                    break

        # Check deaths
        if s_hp <= 0:
            s_alive = False
            s_hp = 0
        if z_hp <= 0:
            z_alive = False
            z_hp = 0
        for i in range(n):
            m_hp[i] = max(0, m_hp[i])

        live_marines_now = [i for i in range(n) if m_hp[i] > 0]
        weapon_ready = [not (moving[live_marines.index(i)] if i in live_marines else True)
                        for i in range(n)]

        trajectory.append(State3v2(
            marine_positions=[p.copy() for p in m_pos],
            marine_hps=list(m_hp),
            marine_weapon_ready=weapon_ready,
            zealot_pos=z_pos.copy(),
            zealot_hp=z_hp,
            stalker_pos=s_pos.copy(),
            stalker_hp=s_hp,
            stalker_alive=s_alive,
            zealot_alive=z_alive,
            n_marines=n,
            step=state.step + step_i + 1,
            time=state.time + (step_i + 1) * dt,
        ))

    return trajectory


# ─── Action Sampling ─────────────────────────────────────────
def sample_action_3v2(state, marine_idx):
    """Sample action for one marine based on current state."""
    m_pos = state.marine_positions[marine_idx]
    z_pos = state.zealot_pos
    s_pos = state.stalker_pos

    d_zealot = np.linalg.norm(m_pos - z_pos) if state.zealot_alive else 999
    d_stalker = np.linalg.norm(m_pos - s_pos) if state.stalker_alive else 999

    if state.stalker_alive:
        # Phase 1: Stalker alive — focus stalker, avoid zealot
        away_z = (m_pos - z_pos)
        away_z_norm = away_z / max(np.linalg.norm(away_z), 0.1)
        toward_s = (s_pos - m_pos)
        toward_s_norm = toward_s / max(np.linalg.norm(toward_s), 0.1)

        if d_zealot < 2.0:
            # Zealot in melee range — flee from zealot
            direction = away_z_norm + 0.2 * np.random.randn(2)
        elif d_stalker <= MARINE_RANGE and d_zealot > 3.0:
            # In range of stalker, safe from zealot — hold and shoot
            if np.random.random() < 0.7:
                return np.zeros(2)
            direction = 0.3 * np.random.randn(2)
        elif d_stalker > MARINE_RANGE + 1.0:
            # Too far from stalker — approach it (but watch zealot)
            direction = toward_s_norm * 0.7 + away_z_norm * 0.3 + 0.2 * np.random.randn(2)
        else:
            # Near stalker range — hold or slight adjustment
            if np.random.random() < 0.5:
                return np.zeros(2)
            direction = 0.3 * away_z_norm + 0.2 * np.random.randn(2)
    else:
        # Phase 2: Stalker dead — kite zealot (same as 2v1 problem)
        away_z = (m_pos - z_pos)
        d = np.linalg.norm(away_z)
        if d > 0.1:
            away_z_norm = away_z / d
            tangent = np.array([-away_z_norm[1], away_z_norm[0]])
            if np.random.random() < 0.5:
                tangent = -tangent
        else:
            away_z_norm = np.array([1.0, 0.0])
            tangent = np.array([0.0, 1.0])

        if d_zealot < 2.0:
            direction = away_z_norm + 0.2 * np.random.randn(2)
        elif d_zealot < 3.5:
            w_away = np.random.uniform(-0.1, 0.5)
            w_tang = np.random.uniform(0.3, 1.0)
            direction = w_away * away_z_norm + w_tang * tangent
        elif d_zealot <= MARINE_RANGE:
            if np.random.random() < 0.6:
                return np.zeros(2)
            direction = 0.2 * np.random.randn(2)
        else:
            direction = -0.3 * away_z_norm + 0.2 * np.random.randn(2)

    norm = np.linalg.norm(direction)
    if norm > 0.1:
        return direction / norm
    return np.zeros(2)


def mpc_select_action(state, n_candidates=128, n_scenarios=6, horizon=8, dt=0.4):
    """Stochastic MPC for 3v2."""
    n = state.n_marines
    best_cvar = float('inf')
    best_actions = [np.zeros(2) for _ in range(n)]
    best_components = {}

    for _ in range(n_candidates):
        all_marine_actions = []
        for i in range(n):
            actions = [sample_action_3v2(state, i) for _ in range(horizon)]
            all_marine_actions.append(actions)

        scenario_costs = []
        scenario_comps = []

        for s in range(n_scenarios):
            # Add noise for stochastic scenarios
            noisy = []
            for i in range(n):
                noisy_i = []
                for a in all_marine_actions[i]:
                    if np.linalg.norm(a) > 0.1:
                        na = a + 0.05 * np.random.randn(2)
                        norm = np.linalg.norm(na)
                        noisy_i.append(na / norm if norm > 0.1 else a)
                    else:
                        noisy_i.append(a)
                noisy.append(noisy_i)

            trajectory = simulate_3v2(state, noisy, dt)
            total = 0.0
            comps = {}
            for sim_state in trajectory:
                c, co = compute_cost(sim_state)
                total += c
                for k, v in co.items():
                    comps[k] = comps.get(k, 0.0) + v
            scenario_costs.append(total)
            scenario_comps.append(comps)

        arr = np.array(scenario_costs)
        sorted_idx = np.argsort(arr)[::-1]
        n_tail = max(1, int(np.ceil(n_scenarios * 0.3)))
        cvar = np.mean(arr[sorted_idx[:n_tail]])

        if cvar < best_cvar:
            best_cvar = cvar
            best_actions = [all_marine_actions[i][0] for i in range(n)]
            median_idx = sorted_idx[n_scenarios // 2]
            best_components = scenario_comps[median_idx]

    return best_actions, best_components


# ─── SC2 Bot ─────────────────────────────────────────────────
class Bot3v2(BotAI):
    def __init__(self, visualize=False):
        super().__init__()
        self.scenario_started = False
        self.setup_done = False
        self.marine_tags = []
        self.zealot_tag = None
        self.stalker_tag = None
        self.step_count = 0
        self.total_components: Dict[str, float] = {}
        self.game_over = False
        self.visualize = visualize

    async def on_step(self, iteration):
        if self.game_over:
            return

        if not self.scenario_started:
            await self.setup_scenario()
            self.scenario_started = True
            return

        if not self.setup_done:
            marines = self.units(UnitTypeId.MARINE)
            enemies = self.enemy_units
            if marines.amount >= 3 and enemies.amount >= 2:
                self.marine_tags = [m.tag for m in marines]
                zealots = enemies(UnitTypeId.ZEALOT)
                stalkers = enemies(UnitTypeId.STALKER)
                if zealots and stalkers:
                    self.zealot_tag = zealots.first.tag
                    self.stalker_tag = stalkers.first.tag
                    self.setup_done = True
                    print(f"Scenario ready: 3 Marines vs Zealot + Stalker")
            return

        # Get units
        marines = [self.units.find_by_tag(t) for t in self.marine_tags]
        marines = [m for m in marines if m is not None]

        if not marines:
            remaining = self.units(UnitTypeId.MARINE)
            if not remaining:
                self.end_game("LOSS", "All marines dead")
                return
            marines = list(remaining)
            self.marine_tags = [m.tag for m in marines]

        zealot = self.enemy_units.find_by_tag(self.zealot_tag) if self.zealot_tag else None
        stalker = self.enemy_units.find_by_tag(self.stalker_tag) if self.stalker_tag else None

        # Check if enemies refreshed tags
        if not zealot:
            zealots = self.enemy_units(UnitTypeId.ZEALOT)
            if zealots:
                zealot = zealots.first
                self.zealot_tag = zealot.tag
        if not stalker:
            stalkers = self.enemy_units(UnitTypeId.STALKER)
            if stalkers:
                stalker = stalkers.first
                self.stalker_tag = stalker.tag

        if not zealot and not stalker:
            self.end_game("WIN", "All enemies killed")
            return

        if self.time > 60:
            z_hp = (zealot.health + zealot.shield) if zealot else 0
            s_hp = (stalker.health + stalker.shield) if stalker else 0
            self.end_game("TIMEOUT", f"Zealot HP: {z_hp:.0f}, Stalker HP: {s_hp:.0f}")
            return

        # Default positions for dead enemies
        z_pos = np.array([zealot.position.x, zealot.position.y]) if zealot else np.array([999, 999])
        s_pos = np.array([stalker.position.x, stalker.position.y]) if stalker else np.array([999, 999])

        state = State3v2(
            marine_positions=[np.array([m.position.x, m.position.y]) for m in marines],
            marine_hps=[m.health for m in marines],
            marine_weapon_ready=[m.weapon_cooldown == 0 for m in marines],
            zealot_pos=z_pos,
            zealot_hp=(zealot.health + zealot.shield) if zealot else 0,
            stalker_pos=s_pos,
            stalker_hp=(stalker.health + stalker.shield) if stalker else 0,
            stalker_alive=stalker is not None,
            zealot_alive=zealot is not None,
            n_marines=len(marines),
            step=self.step_count,
            time=self.time,
        )

        self.step_count += 1
        actions, cost_components = mpc_select_action(state)

        for k, v in cost_components.items():
            self.total_components[k] = self.total_components.get(k, 0.0) + v

        # Execute
        # Priority target for attack commands
        priority_target = stalker if stalker else zealot
        for i, marine in enumerate(marines):
            if i < len(actions):
                action = actions[i]
            else:
                action = np.zeros(2)

            if np.linalg.norm(action) > 0.1:
                target = Point2((
                    marine.position.x + action[0] * 3,
                    marine.position.y + action[1] * 3,
                ))
                marine.move(target)
            else:
                if priority_target and marine.distance_to(priority_target) <= MARINE_RANGE:
                    marine.attack(priority_target)
                elif priority_target:
                    marine.attack(priority_target)

        if self.step_count % 20 == 0:
            alive = sum(1 for m in marines if m.health > 0)
            s_str = f"stalker={state.stalker_hp:.0f}" if state.stalker_alive else "stalker=DEAD"
            z_str = f"zealot={state.zealot_hp:.0f}" if state.zealot_alive else "zealot=DEAD"
            print(f"  Step {self.step_count:4d} | t={self.time:5.1f}s | "
                  f"{s_str} | {z_str} | "
                  f"marines={alive} alive, "
                  f"hp=[{','.join(f'{m.health:.0f}' for m in marines)}]")

    async def setup_scenario(self):
        if self.units:
            await self.client.debug_kill_unit(self.units)
        if self.enemy_units:
            await self.client.debug_kill_unit(self.enemy_units)

        center = MAP_CENTER
        # Marines on the left
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((center.x - 5, center.y - 2)), 1],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((center.x - 5, center.y)), 1],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((center.x - 5, center.y + 2)), 1],
        ])
        # Enemies on the right
        await self.client.debug_create_unit([
            [UnitTypeId.ZEALOT, 1, Point2((center.x + 5, center.y + 1)), 2],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.STALKER, 1, Point2((center.x + 5, center.y - 1)), 2],
        ])
        print("Spawned: 3 Marines vs Zealot + Stalker")

    def end_game(self, result, reason):
        self.game_over = True
        print(f"\n{'='*60}")
        print(f"GAME OVER: {result} — {reason}")
        print(f"Steps: {self.step_count}, Game time: {self.time:.1f}s")
        for k, v in sorted(self.total_components.items()):
            print(f"  {k:30s}: {v:10.2f}")
        print(f"{'='*60}\n")

    async def on_end(self, game_result):
        pass


def main():
    visualize = '--vis' in sys.argv
    run_game(
        maps.get("Flat32"),
        [
            Bot(Race.Terran, Bot3v2(visualize=visualize)),
            Computer(Race.Protoss, Difficulty.VeryEasy),
        ],
        realtime=False,
    )

if __name__ == "__main__":
    main()
