"""
2 Marines vs 1 Zealot micro challenge — role-free MPC.

No fixed bait/shooter roles. The MPC controls both marines generically.
The cost function dynamically assigns 'near' and 'far' roles based on
who is currently closer to the zealot. This lets roles swap naturally
when the zealot switches targets.

Uses debug API to spawn units on Flat32 map. Runs headless.
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.unit import Unit

from cost_function import compute_cost


# ─── Game Constants ──────────────────────────────────────────
MARINE_RANGE = 5.0
MARINE_SPEED = 2.25
ZEALOT_SPEED = 2.25
MARINE_HP = 45
ZEALOT_HP = 100
ZEALOT_SHIELDS = 50
MARINE_DPS = 9.8  # 6 damage / 0.61s cooldown

MAP_CENTER = Point2((32, 32))


# ─── MPC State (role-free) ───────────────────────────────────
@dataclass
class MicroState:
    """Snapshot of the engagement. Marines are m1/m2, no role labels."""
    m1_pos: np.ndarray
    m2_pos: np.ndarray
    zealot_pos: np.ndarray
    m1_hp: float
    m2_hp: float
    zealot_hp: float
    zealot_hp_max: float
    m1_weapon_ready: bool
    m2_weapon_ready: bool
    dist_m1_zealot: float
    dist_m2_zealot: float
    step: int
    time: float


# ─── Simple Dynamics Model ───────────────────────────────────
def predict_zealot_move(zealot_pos, m1_pos, m2_pos, dt):
    """Zealot chases the closest marine."""
    d1 = np.linalg.norm(zealot_pos - m1_pos)
    d2 = np.linalg.norm(zealot_pos - m2_pos)
    target = m1_pos if d1 <= d2 else m2_pos
    direction = target - zealot_pos
    dist = np.linalg.norm(direction)
    if dist < 0.1:
        return zealot_pos.copy()
    return zealot_pos + (direction / dist) * min(ZEALOT_SPEED * dt, dist)


def simulate_trajectory(state, m1_actions, m2_actions, dt=0.5):
    """Roll out trajectory for both marines. No role assumptions."""
    trajectory = []
    m1_pos = state.m1_pos.copy()
    m2_pos = state.m2_pos.copy()
    zealot_pos = state.zealot_pos.copy()
    zealot_hp = state.zealot_hp
    m1_hp = state.m1_hp
    m2_hp = state.m2_hp

    for i in range(len(m1_actions)):
        m1_move = m1_actions[i]
        m2_move = m2_actions[i]

        m1_pos = m1_pos + m1_move * MARINE_SPEED * dt
        m2_pos = m2_pos + m2_move * MARINE_SPEED * dt
        zealot_pos = predict_zealot_move(zealot_pos, m1_pos, m2_pos, dt)

        d1 = float(np.linalg.norm(zealot_pos - m1_pos))
        d2 = float(np.linalg.norm(zealot_pos - m2_pos))

        # Marines shoot if in range and stationary
        m1_moving = np.linalg.norm(m1_move) > 0.1
        m2_moving = np.linalg.norm(m2_move) > 0.1
        if d1 <= MARINE_RANGE and not m1_moving:
            zealot_hp -= MARINE_DPS * dt
        if d2 <= MARINE_RANGE and not m2_moving:
            zealot_hp -= MARINE_DPS * dt

        # Zealot damages marine in melee range
        if d1 < 1.0:
            m1_hp -= 16 * dt
        if d2 < 1.0:
            m2_hp -= 16 * dt

        trajectory.append(MicroState(
            m1_pos=m1_pos.copy(), m2_pos=m2_pos.copy(),
            zealot_pos=zealot_pos.copy(),
            m1_hp=max(0, m1_hp), m2_hp=max(0, m2_hp),
            zealot_hp=max(0, zealot_hp), zealot_hp_max=state.zealot_hp_max,
            m1_weapon_ready=not m1_moving, m2_weapon_ready=not m2_moving,
            dist_m1_zealot=d1, dist_m2_zealot=d2,
            step=state.step + i + 1, time=state.time + (i + 1) * dt,
        ))

    return trajectory


# ─── Action Sampling (distance-based, not role-based) ────────
def sample_marine_action(marine_pos, zealot_pos, dist_to_zealot):
    """Sample a candidate action for a marine based on its distance to zealot.

    Close marines get kiting-biased actions. Far marines get hold-and-shoot
    or retreat actions. The cost function decides which behavior is optimal.
    """
    away = marine_pos - zealot_pos
    d = np.linalg.norm(away)

    if d < 0.1:
        return np.random.randn(2) / np.linalg.norm(np.random.randn(2) + 1e-6)

    away_norm = away / d
    tangent = np.array([-away_norm[1], away_norm[0]])
    if np.random.random() < 0.5:
        tangent = -tangent

    if dist_to_zealot < 2.0:
        # DANGER: flee
        w_away = np.random.uniform(0.6, 1.0)
        w_tang = np.random.uniform(-0.4, 0.4)
    elif dist_to_zealot < 3.5:
        # Close: kite tangentially or flee
        w_away = np.random.uniform(-0.1, 0.5)
        w_tang = np.random.uniform(0.3, 1.0)
    elif dist_to_zealot < 5.0:
        # Medium: could be near-marine kiting or far-marine shooting
        r = np.random.random()
        if r < 0.4:
            # Hold and shoot
            return np.zeros(2)
        elif r < 0.7:
            # Tangential adjustment
            w_away = np.random.uniform(-0.3, 0.3)
            w_tang = np.random.uniform(-0.5, 0.5)
        else:
            # Approach or retreat slightly
            w_away = np.random.uniform(-0.5, 0.5)
            w_tang = np.random.uniform(-0.3, 0.3)
    elif dist_to_zealot < 7.0:
        # Far: approach to get in range, or hold
        r = np.random.random()
        if r < 0.5:
            # Approach zealot
            w_away = np.random.uniform(-1.0, -0.3)
            w_tang = np.random.uniform(-0.3, 0.3)
        elif r < 0.8:
            return np.zeros(2)  # hold
        else:
            w_away = np.random.uniform(-0.5, 0.3)
            w_tang = np.random.uniform(-0.5, 0.5)
    else:
        # Very far: approach
        w_away = np.random.uniform(-1.0, -0.5)
        w_tang = np.random.uniform(-0.3, 0.3)

    direction = w_away * away_norm + w_tang * tangent + 0.1 * np.random.randn(2)
    norm = np.linalg.norm(direction)
    if norm > 0.1:
        return direction / norm
    return np.zeros(2)


def generate_candidate_actions(n_candidates, horizon, state):
    """Generate candidate action sequences for both marines.
    Actions are distance-based — no role assignment here."""
    candidates = []

    for _ in range(n_candidates):
        m1_actions = []
        m2_actions = []
        for _ in range(horizon):
            m1_actions.append(sample_marine_action(
                state.m1_pos, state.zealot_pos, state.dist_m1_zealot))
            m2_actions.append(sample_marine_action(
                state.m2_pos, state.zealot_pos, state.dist_m2_zealot))
        candidates.append((m1_actions, m2_actions))

    return candidates


def mpc_select_action(state, n_candidates=128, horizon=8, dt=0.5):
    """Run MPC: generate candidates, simulate, evaluate, pick best."""
    candidates = generate_candidate_actions(n_candidates, horizon, state)

    best_cost = float('inf')
    best_m1_action = np.zeros(2)
    best_m2_action = np.zeros(2)
    best_components = {}

    for m1_actions, m2_actions in candidates:
        trajectory = simulate_trajectory(state, m1_actions, m2_actions, dt)

        total_cost = 0.0
        all_components = {}

        for sim_state in trajectory:
            cost, components = compute_cost(sim_state)
            total_cost += cost
            for k, v in components.items():
                all_components[k] = all_components.get(k, 0.0) + v

        if total_cost < best_cost:
            best_cost = total_cost
            best_m1_action = m1_actions[0]
            best_m2_action = m2_actions[0]
            best_components = all_components

    return best_m1_action, best_m2_action, best_components


# ─── SC2 Bot ─────────────────────────────────────────────────
class MicroBot(BotAI):
    """Role-free 2v1 micro controller."""

    def __init__(self, visualize=False):
        super().__init__()
        self.scenario_started = False
        self.setup_done = False
        self.m1_tag: Optional[int] = None
        self.m2_tag: Optional[int] = None
        self.zealot_tag: Optional[int] = None
        self.step_count = 0
        self.total_cost_components: Dict[str, float] = {}
        self.game_over = False
        self.visualize = visualize
        self.vis = None

    async def on_step(self, iteration: int):
        if self.game_over:
            return

        if not self.scenario_started:
            await self.setup_scenario()
            self.scenario_started = True
            return

        if not self.setup_done:
            if self.units.amount >= 2 and self.enemy_units.amount >= 1:
                marines = self.units(UnitTypeId.MARINE)
                self.m1_tag = marines[0].tag
                self.m2_tag = marines[1].tag
                zealot = self.enemy_units(UnitTypeId.ZEALOT).first
                self.zealot_tag = zealot.tag
                self.setup_done = True
                print(f"Scenario ready. M1={self.m1_tag}, M2={self.m2_tag}, Zealot={self.zealot_tag}")
            return

        # Find units
        m1 = self.units.find_by_tag(self.m1_tag)
        m2 = self.units.find_by_tag(self.m2_tag)
        zealot = self.enemy_units.find_by_tag(self.zealot_tag) if self.zealot_tag else None

        if not zealot:
            zealots = self.enemy_units(UnitTypeId.ZEALOT)
            if not zealots:
                self.end_game("WIN", "Zealot killed")
                return
            zealot = zealots.first
            self.zealot_tag = zealot.tag

        # Count surviving marines
        marines = self.units(UnitTypeId.MARINE)
        if marines.amount == 0:
            self.end_game("LOSS", "Both marines dead")
            return

        if self.time > 60:
            self.end_game("TIMEOUT", f"Zealot HP: {zealot.health + zealot.shield:.0f}")
            return

        # Handle 1-marine case: just kite and shoot
        if marines.amount == 1:
            sole = marines.first
            state = self.build_state_1marine(sole, zealot)
            m1_action, _, components = mpc_select_action(state)
            self.execute_single(sole, zealot, m1_action)
            self.step_count += 1
            if self.step_count % 20 == 0:
                print(f"  Step {self.step_count:4d} | t={self.time:5.1f}s | "
                      f"zealot_hp={zealot.health + zealot.shield:5.1f} | "
                      f"marine_hp={sole.health:3.0f} d={sole.distance_to(zealot):4.1f} | "
                      f"[1 marine remaining]")
            return

        # Ensure m1/m2 are still alive — reassign if one died
        if not m1 or not m2:
            self.m1_tag = marines[0].tag
            self.m2_tag = marines[1].tag if marines.amount > 1 else marines[0].tag
            m1 = self.units.find_by_tag(self.m1_tag)
            m2 = self.units.find_by_tag(self.m2_tag)

        # Build state and run MPC
        state = self.build_state(m1, m2, zealot)
        self.step_count += 1

        m1_action, m2_action, cost_components = mpc_select_action(state)

        for k, v in cost_components.items():
            self.total_cost_components[k] = self.total_cost_components.get(k, 0.0) + v

        self.execute_actions(m1, m2, zealot, m1_action, m2_action)

        # Live visualization
        if self.visualize and self.step_count % 2 == 0:
            if self.vis is None:
                from visualizer import MicroVisualizer
                self.vis = MicroVisualizer()
            self.vis.update(state, self.step_count, self.time)

        # Log
        if self.step_count % 20 == 0:
            # Determine current roles for display
            near_label = "m1" if state.dist_m1_zealot <= state.dist_m2_zealot else "m2"
            far_label = "m2" if near_label == "m1" else "m1"
            print(f"  Step {self.step_count:4d} | t={state.time:5.1f}s | "
                  f"zealot_hp={state.zealot_hp:5.1f} | "
                  f"near({near_label})_hp={state.m1_hp if near_label=='m1' else state.m2_hp:3.0f} "
                  f"d={min(state.dist_m1_zealot, state.dist_m2_zealot):4.1f} | "
                  f"far({far_label})_hp={state.m1_hp if far_label=='m1' else state.m2_hp:3.0f} "
                  f"d={max(state.dist_m1_zealot, state.dist_m2_zealot):4.1f}")

    async def setup_scenario(self):
        if self.units:
            await self.client.debug_kill_unit(self.units)
        if self.enemy_units:
            await self.client.debug_kill_unit(self.enemy_units)

        center = MAP_CENTER
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((center.x - 3, center.y)), 1],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((center.x - 5, center.y + 2)), 1],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.ZEALOT, 1, Point2((center.x + 5, center.y)), 2],
        ])
        print("Spawned: 2 Marines vs 1 Zealot")

    def build_state(self, m1: Unit, m2: Unit, zealot: Unit) -> MicroState:
        m1_pos = np.array([m1.position.x, m1.position.y])
        m2_pos = np.array([m2.position.x, m2.position.y])
        z_pos = np.array([zealot.position.x, zealot.position.y])
        return MicroState(
            m1_pos=m1_pos, m2_pos=m2_pos, zealot_pos=z_pos,
            m1_hp=m1.health, m2_hp=m2.health,
            zealot_hp=zealot.health + zealot.shield,
            zealot_hp_max=ZEALOT_HP + ZEALOT_SHIELDS,
            m1_weapon_ready=m1.weapon_cooldown == 0,
            m2_weapon_ready=m2.weapon_cooldown == 0,
            dist_m1_zealot=float(np.linalg.norm(m1_pos - z_pos)),
            dist_m2_zealot=float(np.linalg.norm(m2_pos - z_pos)),
            step=self.step_count, time=self.time,
        )

    def build_state_1marine(self, marine: Unit, zealot: Unit) -> MicroState:
        """Build state with only 1 marine (m2 is a ghost far away)."""
        m_pos = np.array([marine.position.x, marine.position.y])
        z_pos = np.array([zealot.position.x, zealot.position.y])
        ghost = np.array([999.0, 999.0])
        return MicroState(
            m1_pos=m_pos, m2_pos=ghost, zealot_pos=z_pos,
            m1_hp=marine.health, m2_hp=0,
            zealot_hp=zealot.health + zealot.shield,
            zealot_hp_max=ZEALOT_HP + ZEALOT_SHIELDS,
            m1_weapon_ready=marine.weapon_cooldown == 0,
            m2_weapon_ready=False,
            dist_m1_zealot=float(np.linalg.norm(m_pos - z_pos)),
            dist_m2_zealot=999.0,
            step=self.step_count, time=self.time,
        )

    def execute_actions(self, m1: Unit, m2: Unit, zealot: Unit,
                        m1_action: np.ndarray, m2_action: np.ndarray):
        for marine, action in [(m1, m1_action), (m2, m2_action)]:
            if np.linalg.norm(action) > 0.1:
                target = Point2((
                    marine.position.x + action[0] * 3,
                    marine.position.y + action[1] * 3,
                ))
                marine.move(target)
            else:
                if marine.distance_to(zealot) <= MARINE_RANGE:
                    marine.attack(zealot)
                else:
                    marine.attack(zealot)  # approach to range

    def execute_single(self, marine: Unit, zealot: Unit, action: np.ndarray):
        if np.linalg.norm(action) > 0.1:
            target = Point2((
                marine.position.x + action[0] * 3,
                marine.position.y + action[1] * 3,
            ))
            marine.move(target)
        else:
            marine.attack(zealot)

    def end_game(self, result: str, reason: str):
        self.game_over = True
        print(f"\n{'='*60}")
        print(f"GAME OVER: {result} — {reason}")
        print(f"Steps: {self.step_count}, Game time: {self.time:.1f}s")
        print(f"Accumulated cost components:")
        for k, v in sorted(self.total_cost_components.items()):
            print(f"  {k:30s}: {v:10.2f}")
        print(f"{'='*60}\n")
        if self.vis:
            self.vis.show_result(result, reason)

    async def on_end(self, game_result):
        pass


def main():
    visualize = '--vis' in sys.argv
    run_game(
        maps.get("Flat32"),
        [
            Bot(Race.Protoss, MicroBot(visualize=visualize)),
            Computer(Race.Protoss, Difficulty.VeryEasy),
        ],
        realtime=False,
    )


if __name__ == "__main__":
    main()
