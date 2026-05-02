"""
Round 3: 1 Stimmed Marine vs 1 Zealot.
Stutter-step kite the zealot, refreshing stim as needed without
overlapping stims. Tests MPC's ability to handle:
  - Stutter-step timing (attack while stationary, move during cooldown)
  - Stim management (discrete decision: when to stim, avoid double-stim)
  - HP resource management (stim costs 10 HP)
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.unit import Unit

from stim_kite_cost import compute_cost


# ─── Constants ───────────────────────────────────────────────
MARINE_RANGE = 5.0
MARINE_SPEED_NORMAL = 2.25
MARINE_SPEED_STIM = 4.73
MARINE_HP = 45
MARINE_DMG = 6.0
MARINE_COOLDOWN_NORMAL = 0.61   # seconds between attacks
MARINE_COOLDOWN_STIM = 0.41     # 50% faster with stim
STIM_COST_HP = 10
STIM_DURATION = 11.0  # seconds

ZEALOT_SPEED = 2.25
ZEALOT_HP = 100
ZEALOT_SHIELDS = 50
ZEALOT_DMG = 16  # 8x2 per hit

MAP_CENTER = Point2((32, 32))


@dataclass
class StimState:
    marine_pos: np.ndarray
    zealot_pos: np.ndarray
    marine_hp: float
    zealot_hp: float
    zealot_hp_max: float
    dist: float
    marine_speed: float
    is_stimmed: bool
    stim_remaining: float      # seconds of stim left
    weapon_ready: bool
    weapon_cooldown: float     # seconds until next shot
    step: int
    time: float


# ─── Dynamics Model ──────────────────────────────────────────
def simulate_stim_trajectory(state, actions, dt=0.3):
    """Simulate forward. Each action is (move_dir[2], do_stim: bool).
    Marine shoots when stationary AND weapon ready AND in range.
    """
    trajectory = []
    m_pos = state.marine_pos.copy()
    z_pos = state.zealot_pos.copy()
    m_hp = state.marine_hp
    z_hp = state.zealot_hp
    is_stimmed = state.is_stimmed
    stim_remaining = state.stim_remaining
    weapon_cd = state.weapon_cooldown

    for i, (move_dir, do_stim) in enumerate(actions):
        # Apply stim if requested
        if do_stim and not is_stimmed and m_hp > STIM_COST_HP:
            is_stimmed = True
            stim_remaining = STIM_DURATION
            m_hp -= STIM_COST_HP

        # Current speed
        speed = MARINE_SPEED_STIM if is_stimmed else MARINE_SPEED_NORMAL
        atk_cd = MARINE_COOLDOWN_STIM if is_stimmed else MARINE_COOLDOWN_NORMAL

        # Move marine
        moving = np.linalg.norm(move_dir) > 0.1
        if moving:
            norm = np.linalg.norm(move_dir)
            m_pos = m_pos + (move_dir / norm) * speed * dt

        # Move zealot toward marine
        r = m_pos - z_pos
        d = np.linalg.norm(r)
        if d > 0.1:
            z_pos = z_pos + (r / d) * min(ZEALOT_SPEED * dt, d)

        dist = float(np.linalg.norm(m_pos - z_pos))

        # Marine shoots if stationary, in range, weapon ready
        weapon_cd = max(0, weapon_cd - dt)
        if not moving and dist <= MARINE_RANGE and weapon_cd <= 0:
            z_hp -= MARINE_DMG
            weapon_cd = atk_cd

        # Zealot hits if in melee
        if dist < 1.0:
            m_hp -= ZEALOT_DMG * dt / 0.5  # approximate DPS scaled by dt

        # Update stim timer
        if is_stimmed:
            stim_remaining -= dt
            if stim_remaining <= 0:
                is_stimmed = False
                stim_remaining = 0.0

        trajectory.append(StimState(
            marine_pos=m_pos.copy(),
            zealot_pos=z_pos.copy(),
            marine_hp=max(0, m_hp),
            zealot_hp=max(0, z_hp),
            zealot_hp_max=state.zealot_hp_max,
            dist=dist,
            marine_speed=speed,
            is_stimmed=is_stimmed,
            stim_remaining=stim_remaining,
            weapon_ready=weapon_cd <= 0,
            weapon_cooldown=weapon_cd,
            step=state.step + i + 1,
            time=state.time + (i + 1) * dt,
        ))

    return trajectory


# ─── Action Sampling ─────────────────────────────────────────
def sample_stim_action(state, dist):
    """Sample an action: (move_direction[2], do_stim: bool).

    Stutter-step logic:
    - If weapon ready and in range → hold (shoot)
    - If weapon on cooldown → move away (kite)
    - If too close → run away
    - Stim decision based on current state
    """
    m_pos = state.marine_pos
    z_pos = state.zealot_pos
    away = m_pos - z_pos
    d = np.linalg.norm(away)

    if d > 0.1:
        away_norm = away / d
    else:
        away_norm = np.array([1.0, 0.0])

    tangent = np.array([-away_norm[1], away_norm[0]])
    if np.random.random() < 0.5:
        tangent = -tangent

    do_stim = False

    if dist < 1.5:
        # DANGER: run away immediately
        direction = away_norm + 0.2 * np.random.randn(2)
        # Stim if not already stimmed and have HP
        if not state.is_stimmed and state.marine_hp > STIM_COST_HP + 5:
            do_stim = np.random.random() < 0.8
    elif dist < MARINE_RANGE and state.weapon_ready:
        # In range, weapon ready → STOP AND SHOOT
        if np.random.random() < 0.85:
            direction = np.zeros(2)  # hold
        else:
            direction = 0.2 * np.random.randn(2)
    elif state.is_stimmed:
        # Stimmed, weapon on cooldown → kite away
        if dist < 4.0:
            # Move away during cooldown
            direction = away_norm + 0.3 * tangent * np.random.randn()
        elif dist < MARINE_RANGE:
            # Good range, can hold or adjust
            if np.random.random() < 0.4:
                direction = np.zeros(2)
            else:
                direction = 0.3 * away_norm + 0.3 * np.random.randn(2)
        else:
            # Too far, approach slightly
            direction = -0.3 * away_norm + 0.2 * np.random.randn(2)
    else:
        # Not stimmed
        if dist < 3.0:
            # Close without stim — dangerous, consider stimming
            direction = away_norm + 0.2 * np.random.randn(2)
            if state.marine_hp > STIM_COST_HP + 5:
                do_stim = np.random.random() < 0.6
        elif dist < MARINE_RANGE and state.weapon_ready:
            direction = np.zeros(2)
        elif dist < MARINE_RANGE:
            # In range but weapon on cooldown
            direction = 0.3 * away_norm + 0.2 * np.random.randn(2)
        else:
            direction = -0.3 * away_norm + 0.2 * np.random.randn(2)

    norm = np.linalg.norm(direction)
    if norm > 0.1:
        direction = direction / norm
    else:
        direction = np.zeros(2)

    # Random stim exploration
    if not do_stim and np.random.random() < 0.05:
        if not state.is_stimmed and state.marine_hp > STIM_COST_HP + 5:
            do_stim = True

    return direction, do_stim


def mpc_select_action(state, n_candidates=128, n_scenarios=6, horizon=10, dt=0.3):
    """Stochastic MPC for stutter-step kiting."""
    best_cvar = float('inf')
    best_action = (np.zeros(2), False)
    best_components = {}

    for _ in range(n_candidates):
        # Generate action sequence
        actions = []
        for h in range(horizon):
            actions.append(sample_stim_action(state, state.dist))

        scenario_costs = []
        scenario_components = []

        for s in range(n_scenarios):
            # Add stochastic noise to zealot movement
            noisy_actions = []
            for move_dir, do_stim in actions:
                # Add execution noise to movement
                if np.linalg.norm(move_dir) > 0.1:
                    noise = 0.05 * np.random.randn(2)
                    noisy_move = move_dir + noise
                    norm = np.linalg.norm(noisy_move)
                    if norm > 0.1:
                        noisy_move = noisy_move / norm
                    else:
                        noisy_move = move_dir
                else:
                    noisy_move = move_dir
                noisy_actions.append((noisy_move, do_stim))

            trajectory = simulate_stim_trajectory(state, noisy_actions, dt)
            total_cost = 0.0
            all_components = {}
            for sim_state in trajectory:
                cost, components = compute_cost(sim_state)
                total_cost += cost
                for k, v in components.items():
                    all_components[k] = all_components.get(k, 0.0) + v

            scenario_costs.append(total_cost)
            scenario_components.append(all_components)

        # CVaR (worst 30%)
        arr = np.array(scenario_costs)
        sorted_idx = np.argsort(arr)[::-1]
        n_tail = max(1, int(np.ceil(n_scenarios * 0.3)))
        cvar = np.mean(arr[sorted_idx[:n_tail]])

        if cvar < best_cvar:
            best_cvar = cvar
            best_action = actions[0]
            median_idx = sorted_idx[n_scenarios // 2]
            best_components = scenario_components[median_idx]

    return best_action, best_components


# ─── SC2 Bot ─────────────────────────────────────────────────
class StimKiteBot(BotAI):
    def __init__(self, visualize=False):
        super().__init__()
        self.scenario_started = False
        self.setup_done = False
        self.stim_researched = False
        self.marine_tag = None
        self.zealot_tag = None
        self.step_count = 0
        self.total_components: Dict[str, float] = {}
        self.game_over = False
        self.visualize = visualize
        self.vis = None
        self.stim_expire_time = 0.0

    async def on_step(self, iteration):
        if self.game_over:
            return

        if not self.scenario_started:
            await self.setup_scenario()
            self.scenario_started = True
            return

        if not self.setup_done:
            # Wait for stim research + units
            marines = self.units(UnitTypeId.MARINE)
            zealots = self.enemy_units(UnitTypeId.ZEALOT)
            if marines.amount >= 1 and zealots.amount >= 1:
                if not self.stim_researched:
                    # Force-research stim via debug
                    await self.client.debug_upgrade()
                    self.stim_researched = True
                    print("Researched all upgrades (including stim)")
                    return

                self.marine_tag = marines.first.tag
                self.zealot_tag = zealots.first.tag
                self.setup_done = True
                print(f"Scenario ready. Marine={self.marine_tag}, Zealot={self.zealot_tag}")
            return

        marine = self.units.find_by_tag(self.marine_tag)
        zealot = self.enemy_units.find_by_tag(self.zealot_tag)

        if not marine:
            marines = self.units(UnitTypeId.MARINE)
            if not marines:
                self.end_game("LOSS", "Marine dead")
                return
            marine = marines.first
            self.marine_tag = marine.tag

        if not zealot:
            zealots = self.enemy_units(UnitTypeId.ZEALOT)
            if not zealots:
                self.end_game("WIN", "Zealot killed")
                return
            zealot = zealots.first
            self.zealot_tag = zealot.tag

        if self.time > 60:
            self.end_game("TIMEOUT", f"Zealot HP: {zealot.health + zealot.shield:.0f}")
            return

        # Build state
        m_pos = np.array([marine.position.x, marine.position.y])
        z_pos = np.array([zealot.position.x, zealot.position.y])
        dist = float(np.linalg.norm(m_pos - z_pos))

        # Check stim status from buffs
        is_stimmed = BuffId.STIMPACK in marine.buffs
        if is_stimmed:
            stim_remaining = max(0, self.stim_expire_time - self.time)
        else:
            stim_remaining = 0.0

        state = StimState(
            marine_pos=m_pos,
            zealot_pos=z_pos,
            marine_hp=marine.health,
            zealot_hp=zealot.health + zealot.shield,
            zealot_hp_max=ZEALOT_HP + ZEALOT_SHIELDS,
            dist=dist,
            marine_speed=MARINE_SPEED_STIM if is_stimmed else MARINE_SPEED_NORMAL,
            is_stimmed=is_stimmed,
            stim_remaining=stim_remaining,
            weapon_ready=marine.weapon_cooldown == 0,
            weapon_cooldown=marine.weapon_cooldown / 22.4,  # convert frames to seconds
            step=self.step_count,
            time=self.time,
        )

        # MPC
        self.step_count += 1
        (move_dir, do_stim), cost_components = mpc_select_action(state)

        for k, v in cost_components.items():
            self.total_components[k] = self.total_components.get(k, 0.0) + v

        # Execute
        self.execute_action(marine, zealot, move_dir, do_stim, state)

        # Visualize
        if self.visualize and self.step_count % 2 == 0:
            if self.vis is None:
                from stim_kite_vis import StimKiteVisualizer
                self.vis = StimKiteVisualizer()
            self.vis.update(state, self.step_count, self.time)

        # Log
        if self.step_count % 20 == 0:
            stim_str = f"STIM({stim_remaining:.1f}s)" if is_stimmed else "no-stim"
            print(f"  Step {self.step_count:4d} | t={self.time:5.1f}s | "
                  f"zealot_hp={state.zealot_hp:5.1f} | "
                  f"marine_hp={state.marine_hp:3.0f} d={dist:4.1f} | "
                  f"{stim_str} | wcd={state.weapon_cooldown:.2f}s")

    def execute_action(self, marine, zealot, move_dir, do_stim, state):
        # Stim
        if do_stim and not state.is_stimmed and state.marine_hp > STIM_COST_HP:
            marine(AbilityId.EFFECT_STIM_MARINE)
            self.stim_expire_time = self.time + STIM_DURATION

        # Move or attack
        if np.linalg.norm(move_dir) > 0.1:
            target = Point2((
                marine.position.x + move_dir[0] * 3,
                marine.position.y + move_dir[1] * 3,
            ))
            marine.move(target)
        else:
            if marine.distance_to(zealot) <= MARINE_RANGE:
                marine.attack(zealot)
            else:
                marine.attack(zealot)

    async def setup_scenario(self):
        if self.units:
            await self.client.debug_kill_unit(self.units)
        if self.enemy_units:
            await self.client.debug_kill_unit(self.enemy_units)

        center = MAP_CENTER
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 1, Point2((center.x - 5, center.y)), 1],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.ZEALOT, 1, Point2((center.x + 5, center.y)), 2],
        ])
        # We'll research stim on the next step via debug_upgrade
        print("Spawned: 1 Marine vs 1 Zealot (stim kite challenge)")

    def end_game(self, result, reason):
        self.game_over = True
        print(f"\n{'='*60}")
        print(f"GAME OVER: {result} — {reason}")
        print(f"Steps: {self.step_count}, Game time: {self.time:.1f}s")
        print(f"Cost components:")
        for k, v in sorted(self.total_components.items()):
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
            Bot(Race.Terran, StimKiteBot(visualize=visualize)),
            Computer(Race.Protoss, Difficulty.VeryEasy),
        ],
        realtime=False,
    )


if __name__ == "__main__":
    main()
