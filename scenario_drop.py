"""
Scenario 4: Medivac + 4 Marines drop micro vs 1 Zealot + 1 Stalker.
Boost in, drop marines near the enemy, marines shoot, pick up marines
before enemy closes, boost out. Repeat until enemies die.

Tests: load/unload timing, boost management, hit-and-run pattern.
The MPC must coordinate the medivac and marines as a unit.
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
from sc2.ids.buff_id import BuffId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.unit import Unit

from cost_drop import compute_cost

# ─── Constants ───────────────────────────────────────────────
MARINE_RANGE = 5.0
MARINE_SPEED = 2.25
MARINE_DPS = 9.8

MEDIVAC_SPEED = 2.5
MEDIVAC_BOOST_SPEED = 4.25
MEDIVAC_BOOST_DURATION = 4.25
MEDIVAC_BOOST_COOLDOWN = 8.57
MEDIVAC_CARGO = 8

STALKER_RANGE = 6.0
STALKER_SPEED = 2.95
ZEALOT_SPEED = 2.25

MAP_CENTER = Point2((32, 32))


@dataclass
class DropState:
    medivac_pos: np.ndarray
    marine_positions: List[np.ndarray]  # empty if loaded in medivac
    marine_hps: List[float]
    marines_loaded: bool                 # are marines inside medivac?
    n_marines: int
    zealot_pos: np.ndarray
    zealot_hp: float
    stalker_pos: np.ndarray
    stalker_hp: float
    zealot_alive: bool
    stalker_alive: bool
    boost_available: bool
    boost_active: bool
    boost_remaining: float
    boost_cooldown: float
    dist_medivac_enemies: float          # min dist to any enemy
    step: int
    time: float


# ─── Dynamics ────────────────────────────────────────────────
def simulate_drop(state, actions, dt=0.4):
    """Simulate drop micro. Actions: list of (phase, move_dir, do_boost).
    phase: 'approach' | 'drop' | 'fighting' | 'pickup' | 'retreat'

    Simplified: action is (medivac_move_dir[2], marine_move_dir[2],
                           do_drop: bool, do_pickup: bool, do_boost: bool)
    """
    trajectory = []
    med_pos = state.medivac_pos.copy()
    m_pos = [p.copy() for p in state.marine_positions]
    m_hp = list(state.marine_hps)
    loaded = state.marines_loaded
    z_pos = state.zealot_pos.copy()
    z_hp = state.zealot_hp
    s_pos = state.stalker_pos.copy()
    s_hp = state.stalker_hp
    z_alive = state.zealot_alive
    s_alive = state.stalker_alive
    boost_active = state.boost_active
    boost_remaining = state.boost_remaining
    boost_cd = state.boost_cooldown

    n = state.n_marines

    for i, (med_move, mar_move, do_drop, do_pickup, do_boost) in enumerate(actions):
        # Boost management
        if do_boost and not boost_active and boost_cd <= 0:
            boost_active = True
            boost_remaining = MEDIVAC_BOOST_DURATION
            boost_cd = MEDIVAC_BOOST_COOLDOWN

        med_speed = MEDIVAC_BOOST_SPEED if boost_active else MEDIVAC_SPEED

        if boost_active:
            boost_remaining -= dt
            if boost_remaining <= 0:
                boost_active = False
                boost_remaining = 0
        boost_cd = max(0, boost_cd - dt)

        # Move medivac
        if np.linalg.norm(med_move) > 0.1:
            med_move_n = med_move / np.linalg.norm(med_move)
            med_pos = med_pos + med_move_n * med_speed * dt

        # Drop marines
        if do_drop and loaded:
            loaded = False
            # Place marines near medivac
            for j in range(n):
                angle = 2 * np.pi * j / n
                m_pos[j] = med_pos + 1.5 * np.array([np.cos(angle), np.sin(angle)])

        # Pickup marines
        if do_pickup and not loaded:
            # Check if medivac is close enough to marines
            all_close = all(np.linalg.norm(m_pos[j] - med_pos) < 3.0 for j in range(n) if m_hp[j] > 0)
            if all_close:
                loaded = True

        # Move marines (only if not loaded)
        if not loaded:
            for j in range(n):
                if m_hp[j] <= 0:
                    continue
                if np.linalg.norm(mar_move) > 0.1:
                    mar_n = mar_move / np.linalg.norm(mar_move)
                    m_pos[j] = m_pos[j] + mar_n * MARINE_SPEED * dt

        # Move enemies
        if z_alive:
            if not loaded:
                # Zealot chases closest marine
                live = [j for j in range(n) if m_hp[j] > 0]
                if live:
                    closest = min(live, key=lambda j: np.linalg.norm(m_pos[j] - z_pos))
                    r = m_pos[closest] - z_pos
                    d = np.linalg.norm(r)
                    if d > 0.1:
                        z_pos = z_pos + (r / d) * min(ZEALOT_SPEED * dt, d)
            # If loaded, zealot stands still (no ground target)

        if s_alive:
            if not loaded:
                live = [j for j in range(n) if m_hp[j] > 0]
                if live:
                    closest = min(live, key=lambda j: np.linalg.norm(m_pos[j] - s_pos))
                    r = m_pos[closest] - s_pos
                    d = np.linalg.norm(r)
                    if d > STALKER_RANGE:
                        s_pos = s_pos + (r / d) * min(STALKER_SPEED * dt, d - STALKER_RANGE + 0.5)

        # Marines shoot (priority: stalker first) when not loaded and not moving
        if not loaded and np.linalg.norm(mar_move) < 0.1:
            for j in range(n):
                if m_hp[j] <= 0:
                    continue
                if s_alive and np.linalg.norm(m_pos[j] - s_pos) <= MARINE_RANGE:
                    s_hp -= MARINE_DPS * dt
                elif z_alive and np.linalg.norm(m_pos[j] - z_pos) <= MARINE_RANGE:
                    z_hp -= MARINE_DPS * dt

        # Enemy damage
        if not loaded:
            if s_alive:
                for j in range(n):
                    if m_hp[j] > 0 and np.linalg.norm(m_pos[j] - s_pos) <= STALKER_RANGE:
                        m_hp[j] -= 9.7 * dt
                        break
            if z_alive:
                for j in range(n):
                    if m_hp[j] > 0 and np.linalg.norm(m_pos[j] - z_pos) < 1.0:
                        m_hp[j] -= 26.3 * dt
                        break

        if s_hp <= 0: s_alive = False; s_hp = 0
        if z_hp <= 0: z_alive = False; z_hp = 0
        for j in range(n): m_hp[j] = max(0, m_hp[j])

        # Compute min distance to any enemy
        dists = []
        if z_alive: dists.append(np.linalg.norm(med_pos - z_pos))
        if s_alive: dists.append(np.linalg.norm(med_pos - s_pos))
        min_d = min(dists) if dists else 999

        trajectory.append(DropState(
            medivac_pos=med_pos.copy(),
            marine_positions=[p.copy() for p in m_pos],
            marine_hps=list(m_hp),
            marines_loaded=loaded, n_marines=n,
            zealot_pos=z_pos.copy(), zealot_hp=z_hp,
            stalker_pos=s_pos.copy(), stalker_hp=s_hp,
            zealot_alive=z_alive, stalker_alive=s_alive,
            boost_available=boost_cd <= 0 and not boost_active,
            boost_active=boost_active,
            boost_remaining=boost_remaining, boost_cooldown=boost_cd,
            dist_medivac_enemies=min_d,
            step=state.step + i + 1, time=state.time + (i + 1) * dt,
        ))

    return trajectory


# ─── Action Sampling ─────────────────────────────────────────
def sample_drop_action(state):
    """Sample a drop micro action based on game phase."""
    med = state.medivac_pos
    z = state.zealot_pos if state.zealot_alive else np.array([999, 999])
    s = state.stalker_pos if state.stalker_alive else np.array([999, 999])

    # Find enemy centroid
    enemies = []
    if state.zealot_alive: enemies.append(z)
    if state.stalker_alive: enemies.append(s)
    if not enemies:
        return np.zeros(2), np.zeros(2), False, False, False

    enemy_center = np.mean(enemies, axis=0)
    to_enemy = enemy_center - med
    d_enemy = np.linalg.norm(to_enemy)
    to_enemy_norm = to_enemy / max(d_enemy, 0.1)
    away_enemy = -to_enemy_norm

    do_drop = False
    do_pickup = False
    do_boost = False

    if state.marines_loaded:
        # Marines inside medivac — decide: approach, drop, or retreat
        if d_enemy > 8.0:
            # Far away — approach (boost if available)
            med_move = to_enemy_norm + 0.1 * np.random.randn(2)
            if state.boost_available and np.random.random() < 0.5:
                do_boost = True
        elif d_enemy > 4.0:
            # Getting close — approach or drop
            if np.random.random() < 0.6:
                # Drop marines!
                do_drop = True
                med_move = to_enemy_norm * 0.3 + 0.2 * np.random.randn(2)
            else:
                med_move = to_enemy_norm + 0.1 * np.random.randn(2)
        else:
            # Very close — drop immediately
            do_drop = True
            med_move = 0.3 * np.random.randn(2)

        mar_move = np.zeros(2)  # loaded, doesn't matter

    else:
        # Marines on ground — they fight while medivac hovers nearby
        live_marines = [i for i in range(state.n_marines) if state.marine_hps[i] > 0]
        if not live_marines:
            # All dead, flee
            med_move = away_enemy + 0.2 * np.random.randn(2)
            mar_move = np.zeros(2)
        else:
            # Marines: stand and shoot, or micro away from zealot
            closest_to_zealot = min(
                (np.linalg.norm(state.marine_positions[i] - z) for i in live_marines),
                default=999
            )

            if closest_to_zealot < 2.5:
                # Zealot close — pickup time!
                if np.random.random() < 0.7:
                    do_pickup = True
                    # Medivac moves to marines
                    marine_center = np.mean([state.marine_positions[i] for i in live_marines], axis=0)
                    med_move = (marine_center - med)
                    d = np.linalg.norm(med_move)
                    med_move = med_move / max(d, 0.1)
                    mar_move = np.zeros(2)  # stand for pickup
                else:
                    # Marines run from zealot
                    away_z = np.mean([state.marine_positions[i] - z for i in live_marines], axis=0)
                    mar_move = away_z / max(np.linalg.norm(away_z), 0.1)
                    med_move = mar_move  # medivac follows
            else:
                # Marines are safe — hold and shoot
                if np.random.random() < 0.6:
                    mar_move = np.zeros(2)  # hold fire
                else:
                    mar_move = 0.2 * np.random.randn(2)

                # Medivac hovers near marines
                if live_marines:
                    marine_center = np.mean([state.marine_positions[i] for i in live_marines], axis=0)
                    med_to_marines = marine_center - med
                    med_move = med_to_marines / max(np.linalg.norm(med_to_marines), 0.1) * 0.5
                else:
                    med_move = away_enemy

            # After pickup, boost away
            if do_pickup and state.boost_available and np.random.random() < 0.8:
                do_boost = True

    # Normalize
    for v in [med_move, mar_move]:
        n = np.linalg.norm(v)
        if n > 1.0:
            v[:] = v / n

    return med_move, mar_move, do_drop, do_pickup, do_boost


def mpc_select_action(state, n_candidates=96, n_scenarios=4, horizon=8, dt=0.4):
    """Stochastic MPC for drop micro."""
    best_cvar = float('inf')
    best_action = (np.zeros(2), np.zeros(2), False, False, False)
    best_components = {}

    for _ in range(n_candidates):
        actions = [sample_drop_action(state) for _ in range(horizon)]

        scenario_costs = []
        scenario_comps = []
        for s in range(n_scenarios):
            trajectory = simulate_drop(state, actions, dt)
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
            best_action = actions[0]
            best_components = scenario_comps[sorted_idx[n_scenarios // 2]]

    return best_action, best_components


# ─── SC2 Bot ─────────────────────────────────────────────────
class DropBot(BotAI):
    def __init__(self, visualize=False):
        super().__init__()
        self.scenario_started = False
        self.setup_done = False
        self.medivac_tag = None
        self.marine_tags = []
        self.zealot_tag = None
        self.stalker_tag = None
        self.step_count = 0
        self.total_components: Dict[str, float] = {}
        self.game_over = False
        self.marines_loaded = True  # start loaded
        self.last_boost_time = -999

    async def on_step(self, iteration):
        if self.game_over:
            return

        if not self.scenario_started:
            await self.setup_scenario()
            self.scenario_started = True
            return

        if not self.setup_done:
            medivacs = self.units(UnitTypeId.MEDIVAC)
            enemies = self.enemy_units
            if medivacs.amount >= 1 and enemies.amount >= 2:
                self.medivac_tag = medivacs.first.tag
                zealots = enemies(UnitTypeId.ZEALOT)
                stalkers = enemies(UnitTypeId.STALKER)
                if zealots and stalkers:
                    self.zealot_tag = zealots.first.tag
                    self.stalker_tag = stalkers.first.tag
                    self.setup_done = True
                    print("Scenario ready: Medivac(4 marines) vs Zealot + Stalker")
            return

        medivac = self.units.find_by_tag(self.medivac_tag)
        if not medivac:
            medivacs = self.units(UnitTypeId.MEDIVAC)
            if not medivacs:
                self.end_game("LOSS", "Medivac destroyed")
                return
            medivac = medivacs.first
            self.medivac_tag = medivac.tag

        zealot = self.enemy_units.find_by_tag(self.zealot_tag) if self.zealot_tag else None
        stalker = self.enemy_units.find_by_tag(self.stalker_tag) if self.stalker_tag else None

        if not zealot:
            zealots = self.enemy_units(UnitTypeId.ZEALOT)
            zealot = zealots.first if zealots else None
            if zealot: self.zealot_tag = zealot.tag
        if not stalker:
            stalkers = self.enemy_units(UnitTypeId.STALKER)
            stalker = stalkers.first if stalkers else None
            if stalker: self.stalker_tag = stalker.tag

        if not zealot and not stalker:
            self.end_game("WIN", "All enemies killed")
            return

        if self.time > 90:
            self.end_game("TIMEOUT", "Time limit")
            return

        # Determine marine state
        marines = self.units(UnitTypeId.MARINE)
        loaded = medivac.cargo_used > 0 if medivac else False
        self.marines_loaded = loaded

        if not loaded and marines.amount == 0 and medivac.cargo_used == 0:
            self.end_game("LOSS", "All marines dead")
            return

        # Build state
        med_pos = np.array([medivac.position.x, medivac.position.y])
        z_pos = np.array([zealot.position.x, zealot.position.y]) if zealot else np.array([999, 999])
        s_pos = np.array([stalker.position.x, stalker.position.y]) if stalker else np.array([999, 999])

        if loaded:
            m_positions = [med_pos.copy() for _ in range(4)]
            m_hps = [45.0] * 4  # approximate
        else:
            m_positions = [np.array([m.position.x, m.position.y]) for m in marines]
            m_hps = [m.health for m in marines]
            while len(m_positions) < 4:
                m_positions.append(np.array([999, 999]))
                m_hps.append(0)

        boost_active = BuffId.MEDIVACSPEEDBOOST in medivac.buffs if medivac else False
        boost_cd = max(0, self.last_boost_time + MEDIVAC_BOOST_COOLDOWN - self.time)

        dists = []
        if zealot: dists.append(medivac.distance_to(zealot))
        if stalker: dists.append(medivac.distance_to(stalker))

        state = DropState(
            medivac_pos=med_pos,
            marine_positions=m_positions,
            marine_hps=m_hps,
            marines_loaded=loaded,
            n_marines=min(4, marines.amount + medivac.cargo_used),
            zealot_pos=z_pos, zealot_hp=(zealot.health + zealot.shield) if zealot else 0,
            stalker_pos=s_pos, stalker_hp=(stalker.health + stalker.shield) if stalker else 0,
            zealot_alive=zealot is not None, stalker_alive=stalker is not None,
            boost_available=not boost_active and boost_cd <= 0,
            boost_active=boost_active,
            boost_remaining=0, boost_cooldown=boost_cd,
            dist_medivac_enemies=min(dists) if dists else 999,
            step=self.step_count, time=self.time,
        )

        self.step_count += 1
        action, cost_components = mpc_select_action(state)
        med_move, mar_move, do_drop, do_pickup, do_boost = action

        for k, v in cost_components.items():
            self.total_components[k] = self.total_components.get(k, 0.0) + v

        # Execute
        priority_target = stalker if stalker else zealot

        if do_boost and state.boost_available:
            medivac(AbilityId.EFFECT_MEDIVACIGNITEAFTERBURNERS)
            self.last_boost_time = self.time

        if do_drop and loaded:
            # Unload all
            medivac(AbilityId.UNLOADALLAT_MEDIVAC, medivac.position)
        elif do_pickup and not loaded and marines:
            # Load marines
            for m in marines:
                if medivac.distance_to(m) < 4:
                    medivac(AbilityId.LOAD_MEDIVAC, m)

        # Move medivac
        if np.linalg.norm(med_move) > 0.1:
            target = Point2((med_pos[0] + med_move[0] * 4, med_pos[1] + med_move[1] * 4))
            medivac.move(target)

        # Marines: attack or move
        if not loaded and marines:
            if np.linalg.norm(mar_move) > 0.1:
                for m in marines:
                    target = Point2((m.position.x + mar_move[0] * 3, m.position.y + mar_move[1] * 3))
                    m.move(target)
            elif priority_target:
                for m in marines:
                    if m.distance_to(priority_target) <= MARINE_RANGE:
                        m.attack(priority_target)
                    else:
                        m.attack(priority_target)

        if self.step_count % 20 == 0:
            s_str = f"stalker={state.stalker_hp:.0f}" if state.stalker_alive else "stalker=DEAD"
            z_str = f"zealot={state.zealot_hp:.0f}" if state.zealot_alive else "zealot=DEAD"
            load_str = "LOADED" if loaded else f"{marines.amount} on ground"
            boost_str = "BOOST" if boost_active else ""
            print(f"  Step {self.step_count:4d} | t={self.time:5.1f}s | "
                  f"{s_str} | {z_str} | {load_str} {boost_str}")

    async def setup_scenario(self):
        if self.units:
            await self.client.debug_kill_unit(self.units)
        if self.enemy_units:
            await self.client.debug_kill_unit(self.enemy_units)

        center = MAP_CENTER
        # Medivac with marines loaded (spawn medivac, then load)
        await self.client.debug_create_unit([
            [UnitTypeId.MEDIVAC, 1, Point2((center.x - 12, center.y)), 1],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, 4, Point2((center.x - 12, center.y)), 1],
        ])
        # Enemies
        await self.client.debug_create_unit([
            [UnitTypeId.ZEALOT, 1, Point2((center.x + 5, center.y + 1)), 2],
        ])
        await self.client.debug_create_unit([
            [UnitTypeId.STALKER, 1, Point2((center.x + 5, center.y - 1)), 2],
        ])
        print("Spawned: Medivac + 4 Marines vs Zealot + Stalker")

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
            Bot(Race.Terran, DropBot(visualize=visualize)),
            Computer(Race.Protoss, Difficulty.VeryEasy),
        ],
        realtime=False,
    )

if __name__ == "__main__":
    main()
