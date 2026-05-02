"""
Minimal test bot to verify the SC2 environment works.
Runs headless on Linux. Creates a single zealot and attacks.
"""

import sys
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer


class TestBot(BotAI):
    async def on_step(self, iteration: int):
        if iteration == 0:
            print(f"Game started. Map: {self.game_info.map_name}")
            print(f"Our race: {self.race}")
            print(f"Starting units: {self.units}")
            print(f"Starting structures: {self.structures}")

        # Basic macro: build workers and supply
        if self.can_afford(UnitTypeId.PROBE) and self.townhalls.ready:
            for nexus in self.townhalls.ready:
                if nexus.is_idle:
                    nexus.train(UnitTypeId.PROBE)

        # Build a pylon if supply is low
        if (
            self.supply_left < 5
            and self.already_pending(UnitTypeId.PYLON) == 0
            and self.can_afford(UnitTypeId.PYLON)
        ):
            await self.build(UnitTypeId.PYLON, near=self.townhalls.first)

        # Build a gateway
        if (
            self.structures(UnitTypeId.GATEWAY).amount < 2
            and self.can_afford(UnitTypeId.GATEWAY)
            and self.structures(UnitTypeId.PYLON).ready
        ):
            await self.build(
                UnitTypeId.GATEWAY,
                near=self.structures(UnitTypeId.PYLON).ready.first,
            )

        # Train zealots
        for gw in self.structures(UnitTypeId.GATEWAY).ready:
            if gw.is_idle and self.can_afford(UnitTypeId.ZEALOT):
                gw.train(UnitTypeId.ZEALOT)

        # Attack with zealots when we have 3+
        zealots = self.units(UnitTypeId.ZEALOT)
        if zealots.amount >= 3:
            target = self.enemy_start_locations[0]
            for zealot in zealots:
                zealot.attack(target)

        # Print status every 100 frames
        if iteration % 100 == 0 and iteration > 0:
            print(
                f"Step {iteration}: "
                f"workers={self.workers.amount}, "
                f"zealots={zealots.amount}, "
                f"supply={self.supply_used}/{self.supply_cap}"
            )

    async def on_end(self, game_result):
        print(f"Game ended: {game_result}")


def main():
    run_game(
        maps.get("Simple64"),
        [
            Bot(Race.Protoss, TestBot()),
            Computer(Race.Terran, Difficulty.Easy),
        ],
        realtime=False,
        sc2_version="4.10",
    )


if __name__ == "__main__":
    main()
