"""
Bio + Medivac vs Mixed Protoss.

  Us:    6 Marines + 2 Marauders + 1 Medivac
  Them:  3 Zealots + 2 Stalkers

Compositional cost (cost_bio.py + cost_primitives.py) so the same
matchup library scales to any bio-vs-protoss skirmish — adding more
marines or swapping a marauder for a ghost is a state-shape change,
not a fresh cost-function rewrite.
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


# Cost module via env var (Eureka harness compat)
COST_MODULE_NAME = os.environ.get("COST_MODULE", "cost_bio")
_cost_mod = importlib.import_module(COST_MODULE_NAME)
compute_cost = _cost_mod.compute_cost


# ─── Composition (fixed at scenario) ─────────────────────────
N_MARINES = 6
N_MARAUDERS = 2
N_MEDIVACS = 1
N_ZEALOTS = 3
N_STALKERS = 2

MARINE_RANGE = 5.0
MARAUDER_RANGE = 6.0

MAP_CENTER = Point2((32, 32))
TIME_LIMIT = 60.0


# ─── State ───────────────────────────────────────────────────
@dataclass
class BioState:
    marine_positions: List[np.ndarray]
    marine_hps: List[float]
    marine_weapon_ready: List[bool]
    marauder_positions: List[np.ndarray]
    marauder_hps: List[float]
    marauder_weapon_ready: List[bool]
    medivac_positions: List[np.ndarray]
    medivac_hps: List[float]
    zealot_positions: List[np.ndarray]
    zealot_hps: List[float]
    zealot_alive: List[bool]
    stalker_positions: List[np.ndarray]
    stalker_hps: List[float]
    stalker_alive: List[bool]
    n_marines: int
    n_marauders: int
    n_medivacs: int
    n_zealots: int
    n_stalkers: int
    step: int
    time: float


# ─── SC2 Bot ─────────────────────────────────────────────────
class BotBio(BotAI):
    def __init__(self, visualize=False):
        super().__init__()
        self.scenario_started = False
        self.setup_done = False
        self.marine_tags: List[int] = []
        self.marauder_tags: List[int] = []
        self.medivac_tags: List[int] = []
        self.zealot_tags: List[int] = []
        self.stalker_tags: List[int] = []
        self.step_count = 0
        self.total_components: Dict[str, float] = {}
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
            marines = self.units(UnitTypeId.MARINE)
            marauders = self.units(UnitTypeId.MARAUDER)
            medivacs = self.units(UnitTypeId.MEDIVAC)
            zealots = self.enemy_units(UnitTypeId.ZEALOT)
            stalkers = self.enemy_units(UnitTypeId.STALKER)

            ready = (marines.amount >= N_MARINES
                     and marauders.amount >= N_MARAUDERS
                     and medivacs.amount >= N_MEDIVACS
                     and zealots.amount >= N_ZEALOTS
                     and stalkers.amount >= N_STALKERS)
            if ready:
                self.marine_tags = [u.tag for u in marines][:N_MARINES]
                self.marauder_tags = [u.tag for u in marauders][:N_MARAUDERS]
                self.medivac_tags = [u.tag for u in medivacs][:N_MEDIVACS]
                self.zealot_tags = [u.tag for u in zealots][:N_ZEALOTS]
                self.stalker_tags = [u.tag for u in stalkers][:N_STALKERS]
                self.setup_done = True
                print(f"Bio scenario ready: "
                      f"{N_MARINES}M+{N_MARAUDERS}MM+{N_MEDIVACS}Mv vs "
                      f"{N_ZEALOTS}Z+{N_STALKERS}S")
            return

        # Resolve current units (still alive)
        def _resolve(tags, container):
            return [container.find_by_tag(t) for t in tags]

        marines = _resolve(self.marine_tags, self.units)
        marauders = _resolve(self.marauder_tags, self.units)
        medivacs = _resolve(self.medivac_tags, self.units)
        zealots = _resolve(self.zealot_tags, self.enemy_units)
        stalkers = _resolve(self.stalker_tags, self.enemy_units)

        # Win/loss checks
        n_bio_alive = sum(1 for u in marines + marauders if u is not None)
        n_mv_alive = sum(1 for u in medivacs if u is not None)
        n_e_alive = sum(1 for u in zealots + stalkers if u is not None)

        if n_bio_alive == 0 and n_mv_alive == 0:
            self.end_game("LOSS", "Whole army destroyed")
            return
        if n_bio_alive == 0 and n_mv_alive > 0:
            # Medivac with no bio left = effective loss (no DPS)
            self.end_game("LOSS", "All bio dead, medivac alone")
            return
        if n_e_alive == 0:
            self.end_game("WIN", "All enemies killed")
            return
        if self.time > TIME_LIMIT:
            e_summary = ", ".join(
                [f"z{i}={u.health + u.shield:.0f}" for i, u in enumerate(zealots) if u]
              + [f"s{i}={u.health + u.shield:.0f}" for i, u in enumerate(stalkers) if u]
            )
            self.end_game("TIMEOUT", e_summary)
            return

        # Build state (dead units → ghost position 999, hp 0)
        def _pos(u, default=(999.0, 999.0)):
            return np.array([u.position.x, u.position.y]) if u else np.array(default)

        marine_positions = [_pos(u) for u in marines]
        marine_hps = [u.health if u else 0.0 for u in marines]
        marine_ready = [(u.weapon_cooldown == 0) if u else False for u in marines]

        marauder_positions = [_pos(u) for u in marauders]
        marauder_hps = [u.health if u else 0.0 for u in marauders]
        marauder_ready = [(u.weapon_cooldown == 0) if u else False for u in marauders]

        medivac_positions = [_pos(u) for u in medivacs]
        medivac_hps = [u.health if u else 0.0 for u in medivacs]

        zealot_positions = [_pos(u) for u in zealots]
        zealot_hps = [(u.health + u.shield) if u else 0.0 for u in zealots]
        zealot_alive = [u is not None for u in zealots]

        stalker_positions = [_pos(u) for u in stalkers]
        stalker_hps = [(u.health + u.shield) if u else 0.0 for u in stalkers]
        stalker_alive = [u is not None for u in stalkers]

        state = BioState(
            marine_positions=marine_positions, marine_hps=marine_hps,
            marine_weapon_ready=marine_ready,
            marauder_positions=marauder_positions, marauder_hps=marauder_hps,
            marauder_weapon_ready=marauder_ready,
            medivac_positions=medivac_positions, medivac_hps=medivac_hps,
            zealot_positions=zealot_positions, zealot_hps=zealot_hps,
            zealot_alive=zealot_alive,
            stalker_positions=stalker_positions, stalker_hps=stalker_hps,
            stalker_alive=stalker_alive,
            n_marines=N_MARINES, n_marauders=N_MARAUDERS, n_medivacs=N_MEDIVACS,
            n_zealots=N_ZEALOTS, n_stalkers=N_STALKERS,
            step=self.step_count, time=self.time,
        )
        self.step_count += 1

        # MPC
        from mpc_vectorized_bio import mpc_select_action_vectorized
        actions, components = mpc_select_action_vectorized(state)
        for k, v in components.items():
            self.total_components[k] = self.total_components.get(k, 0.0) + v

        # Execute (priority target = lowest-HP alive enemy in range)
        live_enemies = [u for u in zealots + stalkers if u is not None]
        if live_enemies:
            priority = min(live_enemies, key=lambda u: u.health + u.shield)
        else:
            priority = None

        # Unit ordering in actions: [marines..., marauders..., medivac]
        units_in_order = marines + marauders + medivacs
        for i, u in enumerate(units_in_order):
            if u is None or i >= len(actions):
                continue
            a = actions[i]
            if float(np.linalg.norm(a)) > 0.1:
                tgt = Point2((u.position.x + a[0] * 3,
                              u.position.y + a[1] * 3))
                u.move(tgt)
            else:
                # Hold position and attack (medivac stays where it is)
                if u.type_id == UnitTypeId.MEDIVAC:
                    u.move(u.position)   # hover
                elif priority is not None:
                    u.attack(priority)

        if self.visualize and self.step_count % 2 == 0:
            if self.vis is None:
                from visualizer_bio import BioVisualizer
                self.vis = BioVisualizer()
            self.vis.update(state, self.step_count, self.time)

        if self.step_count % 20 == 0:
            ehp = sum(u.health + u.shield for u in live_enemies)
            mhp = sum(u.health for u in marines + marauders if u is not None)
            print(f"  Step {self.step_count:4d} | t={self.time:5.1f}s | "
                  f"bio_alive={n_bio_alive}/{N_MARINES + N_MARAUDERS} "
                  f"mhp={mhp:5.0f} | "
                  f"e_alive={n_e_alive}/{N_ZEALOTS + N_STALKERS} ehp={ehp:5.0f}")

    async def setup_scenario(self):
        # Keep starting townhalls; clear units only.
        if self.units:
            await self.client.debug_kill_unit(self.units)
        if self.enemy_units:
            await self.client.debug_kill_unit(self.enemy_units)

        c = MAP_CENTER
        # Bio army on the left, further back to give time for the
        # formation to set up before contact (was c.x-8, now c.x-14).
        for i in range(N_MARINES):
            dx = -14 + (i % 3) * 1.2
            dy = -2 + (i // 3) * 1.5
            await self.client.debug_create_unit([
                [UnitTypeId.MARINE, 1, Point2((c.x + dx, c.y + dy)), 1],
            ])
        for i in range(N_MARAUDERS):
            await self.client.debug_create_unit([
                [UnitTypeId.MARAUDER, 1, Point2((c.x - 13, c.y + (-1 + 2*i))), 1],
            ])
        for i in range(N_MEDIVACS):
            await self.client.debug_create_unit([
                [UnitTypeId.MEDIVAC, 1, Point2((c.x - 15, c.y)), 1],
            ])
        # Protoss army on the right
        for i in range(N_ZEALOTS):
            await self.client.debug_create_unit([
                [UnitTypeId.ZEALOT, 1, Point2((c.x + 6, c.y - 2 + 2*i)), 2],
            ])
        for i in range(N_STALKERS):
            await self.client.debug_create_unit([
                [UnitTypeId.STALKER, 1, Point2((c.x + 7, c.y - 1.5 + 3*i)), 2],
            ])
        print(f"Spawned: {N_MARINES}M+{N_MARAUDERS}MM+{N_MEDIVACS}Mv vs "
              f"{N_ZEALOTS}Z+{N_STALKERS}S")

    def end_game(self, result: str, reason: str):
        self.game_over = True
        print(f"\n{'='*60}")
        print(f"GAME OVER: {result} — {reason}")
        print(f"Steps: {self.step_count}, Game time: {self.time:.1f}s")
        print("Accumulated cost components:")
        for k, v in sorted(self.total_components.items()):
            print(f"  {k:30s}: {v:10.2f}")
        print(f"{'='*60}\n")
        if self.vis:
            self.vis.show_result(result, reason)

    async def on_end(self, game_result):
        pass


def main():
    visualize = '--vis' in sys.argv
    print(f"Bio scenario [{COST_MODULE_NAME}]: "
          f"{N_MARINES}M+{N_MARAUDERS}MM+{N_MEDIVACS}Mv vs "
          f"{N_ZEALOTS}Z+{N_STALKERS}S")
    run_game(
        maps.get("Flat32"),
        [
            Bot(Race.Terran, BotBio(visualize=visualize)),
            Computer(Race.Protoss, Difficulty.VeryEasy),
        ],
        realtime=False,
    )


if __name__ == "__main__":
    main()
