"""
Live 2D visualization for the 2 Marines vs 1 Zealot micro scenario.
Shows unit positions, health bars, movement trails, role labels, distance rings,
and combat feedback (yellow flash = dealing damage, red flash = taking damage,
bullet lines from shooter to target).
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque


class MicroVisualizer:
    def __init__(self, map_size=32):
        self.map_size = map_size
        self.trail_len = 80

        # Movement trails
        self.m1_trail = deque(maxlen=self.trail_len)
        self.m2_trail = deque(maxlen=self.trail_len)
        self.zealot_trail = deque(maxlen=self.trail_len)

        # Previous frame state for detecting HP changes
        self.prev_m1_hp = None
        self.prev_m2_hp = None
        self.prev_zealot_hp = None

        # Flash timers (count down frames to show the flash effect)
        self.m1_flash = 0       # >0 means flashing
        self.m2_flash = 0
        self.zealot_flash = 0
        self.m1_flash_type = None   # 'dealing' or 'taking'
        self.m2_flash_type = None
        self.zealot_flash_type = None
        self.FLASH_DURATION = 3  # frames to hold flash

        # Set up the figure
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.fig.canvas.manager.set_window_title('SC2 Micro: 2 Marines vs Zealot')
        self.ax.set_xlim(15, 50)
        self.ax.set_ylim(15, 50)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1a1a2e')
        self.ax.grid(True, alpha=0.15, color='white')
        self.ax.set_xlabel('X', color='white')
        self.ax.set_ylabel('Y', color='white')
        self.fig.patch.set_facecolor('#0a0a1a')
        self.ax.tick_params(colors='white')

        # Unit dots
        self.m1_dot, = self.ax.plot([], [], 'o', color='#00aaff', markersize=10, zorder=5)
        self.m2_dot, = self.ax.plot([], [], 'o', color='#00ddff', markersize=10, zorder=5)
        self.zealot_dot, = self.ax.plot([], [], 'o', color='#ff4444', markersize=12, zorder=5)

        # Flash rings (larger glowing circles behind the unit dot)
        self.m1_flash_ring, = self.ax.plot([], [], 'o', color='#ffdd00', markersize=18,
                                            alpha=0, zorder=4, markeredgewidth=0)
        self.m2_flash_ring, = self.ax.plot([], [], 'o', color='#ffdd00', markersize=18,
                                            alpha=0, zorder=4, markeredgewidth=0)
        self.zealot_flash_ring, = self.ax.plot([], [], 'o', color='#ff0000', markersize=20,
                                                alpha=0, zorder=4, markeredgewidth=0)

        # Trails
        self.m1_trail_line, = self.ax.plot([], [], '-', color='#00aaff', alpha=0.3, linewidth=1)
        self.m2_trail_line, = self.ax.plot([], [], '-', color='#00ddff', alpha=0.3, linewidth=1)
        self.zealot_trail_line, = self.ax.plot([], [], '-', color='#ff4444', alpha=0.3, linewidth=1)

        # Range circles
        self.range_circle_attack = None
        self.range_circle_melee = None

        # Text annotations
        self.m1_label = self.ax.text(0, 0, '', fontsize=8, color='#00aaff',
                                     ha='center', va='bottom', fontweight='bold')
        self.m2_label = self.ax.text(0, 0, '', fontsize=8, color='#00ddff',
                                     ha='center', va='bottom', fontweight='bold')
        self.zealot_label = self.ax.text(0, 0, '', fontsize=8, color='#ff4444',
                                         ha='center', va='bottom', fontweight='bold')

        # Status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                         fontsize=10, color='white', va='top',
                                         fontfamily='monospace',
                                         bbox=dict(boxstyle='round', facecolor='#1a1a2e',
                                                   alpha=0.8, edgecolor='#444'))

        # Ephemeral artists (bullet lines, damage numbers) cleared each frame
        self.ephemeral = []

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, state, step, game_time):
        """Update visualization with current game state."""
        m1_pos = state.m1_pos
        m2_pos = state.m2_pos
        z_pos = state.zealot_pos

        # ── Detect combat events ──
        m1_dealing = False
        m2_dealing = False
        m1_taking = False
        m2_taking = False
        zealot_taking = False
        zealot_dealing = False

        if self.prev_zealot_hp is not None:
            zealot_hp_delta = state.zealot_hp - self.prev_zealot_hp
            m1_hp_delta = state.m1_hp - self.prev_m1_hp
            m2_hp_delta = state.m2_hp - self.prev_m2_hp

            # Zealot took damage => one or both marines are dealing
            if zealot_hp_delta < -0.5:
                zealot_taking = True
                # Figure out which marine(s) are shooting
                # Marine deals damage when: in range AND weapon on cooldown (just fired)
                if state.dist_m1_zealot <= 5.5 and not state.m1_weapon_ready:
                    m1_dealing = True
                if state.dist_m2_zealot <= 5.5 and not state.m2_weapon_ready:
                    m2_dealing = True
                # If neither detected as firing, credit the closer in-range one
                if not m1_dealing and not m2_dealing:
                    if state.dist_m1_zealot <= 5.5:
                        m1_dealing = True
                    elif state.dist_m2_zealot <= 5.5:
                        m2_dealing = True

            # Marines took damage => zealot is dealing
            if m1_hp_delta < -0.5:
                m1_taking = True
                zealot_dealing = True
            if m2_hp_delta < -0.5:
                m2_taking = True
                zealot_dealing = True

        self.prev_m1_hp = state.m1_hp
        self.prev_m2_hp = state.m2_hp
        self.prev_zealot_hp = state.zealot_hp

        # ── Update flash timers ──
        if m1_dealing:
            self.m1_flash = self.FLASH_DURATION
            self.m1_flash_type = 'dealing'
        elif m1_taking:
            self.m1_flash = self.FLASH_DURATION
            self.m1_flash_type = 'taking'

        if m2_dealing:
            self.m2_flash = self.FLASH_DURATION
            self.m2_flash_type = 'dealing'
        elif m2_taking:
            self.m2_flash = self.FLASH_DURATION
            self.m2_flash_type = 'taking'

        if zealot_taking:
            self.zealot_flash = self.FLASH_DURATION
            self.zealot_flash_type = 'taking'
        elif zealot_dealing:
            self.zealot_flash = self.FLASH_DURATION
            self.zealot_flash_type = 'dealing'

        # ── Clear ephemeral artists ──
        for a in self.ephemeral:
            a.remove()
        self.ephemeral = []

        # ── Update trails ──
        self.m1_trail.append(m1_pos.copy())
        self.m2_trail.append(m2_pos.copy())
        self.zealot_trail.append(z_pos.copy())

        # Determine roles
        d1 = state.dist_m1_zealot
        d2 = state.dist_m2_zealot
        m1_role = "NEAR" if d1 <= d2 else "FAR"
        m2_role = "NEAR" if d2 < d1 else "FAR"

        # ── Unit dot colors (flash override) ──
        m1_color = '#00aaff'
        m2_color = '#00ddff'
        zealot_color = '#ff4444'

        if self.m1_flash > 0:
            m1_color = '#ffdd00' if self.m1_flash_type == 'dealing' else '#ff2200'
            self.m1_flash -= 1
        if self.m2_flash > 0:
            m2_color = '#ffdd00' if self.m2_flash_type == 'dealing' else '#ff2200'
            self.m2_flash -= 1
        if self.zealot_flash > 0:
            zealot_color = '#ffdd00' if self.zealot_flash_type == 'dealing' else '#ff2200'
            self.zealot_flash -= 1

        self.m1_dot.set_color(m1_color)
        self.m2_dot.set_color(m2_color)
        self.zealot_dot.set_color(zealot_color)

        # Flash rings (glow behind unit when dealing/taking)
        if self.m1_flash > 0:
            ring_color = '#ffdd00' if self.m1_flash_type == 'dealing' else '#ff2200'
            self.m1_flash_ring.set_data([m1_pos[0]], [m1_pos[1]])
            self.m1_flash_ring.set_color(ring_color)
            self.m1_flash_ring.set_alpha(0.4)
        else:
            self.m1_flash_ring.set_alpha(0)

        if self.m2_flash > 0:
            ring_color = '#ffdd00' if self.m2_flash_type == 'dealing' else '#ff2200'
            self.m2_flash_ring.set_data([m2_pos[0]], [m2_pos[1]])
            self.m2_flash_ring.set_color(ring_color)
            self.m2_flash_ring.set_alpha(0.4)
        else:
            self.m2_flash_ring.set_alpha(0)

        if self.zealot_flash > 0:
            ring_color = '#ffdd00' if self.zealot_flash_type == 'dealing' else '#ff2200'
            self.zealot_flash_ring.set_data([z_pos[0]], [z_pos[1]])
            self.zealot_flash_ring.set_color(ring_color)
            self.zealot_flash_ring.set_alpha(0.4)
        else:
            self.zealot_flash_ring.set_alpha(0)

        # ── Bullet lines (bright yellow from marine to zealot when dealing damage) ──
        if m1_dealing:
            bullet, = self.ax.plot([m1_pos[0], z_pos[0]], [m1_pos[1], z_pos[1]],
                                    '-', color='#ffdd00', alpha=0.7, linewidth=2, zorder=6)
            self.ephemeral.append(bullet)
        if m2_dealing:
            bullet, = self.ax.plot([m2_pos[0], z_pos[0]], [m2_pos[1], z_pos[1]],
                                    '-', color='#ffdd00', alpha=0.7, linewidth=2, zorder=6)
            self.ephemeral.append(bullet)

        # Zealot melee attack line (red from zealot to marine when dealing)
        if zealot_dealing:
            # Draw to whichever marine is taking damage
            if m1_taking:
                slash, = self.ax.plot([z_pos[0], m1_pos[0]], [z_pos[1], m1_pos[1]],
                                      '-', color='#ff2200', alpha=0.8, linewidth=3, zorder=6)
                self.ephemeral.append(slash)
            if m2_taking:
                slash, = self.ax.plot([z_pos[0], m2_pos[0]], [z_pos[1], m2_pos[1]],
                                      '-', color='#ff2200', alpha=0.8, linewidth=3, zorder=6)
                self.ephemeral.append(slash)

        # ── Damage numbers ──
        if self.prev_zealot_hp is not None:
            if zealot_taking:
                dmg = abs(state.zealot_hp - (self.prev_zealot_hp or state.zealot_hp))
                if dmg > 0:
                    txt = self.ax.text(z_pos[0] + 0.5, z_pos[1] - 0.8,
                                       f'-{dmg:.0f}', fontsize=7, color='#ffdd00',
                                       ha='center', fontweight='bold', alpha=0.9)
                    self.ephemeral.append(txt)
            if m1_taking:
                txt = self.ax.text(m1_pos[0] + 0.5, m1_pos[1] - 0.8,
                                   f'-{abs(state.m1_hp - (self.prev_m1_hp or state.m1_hp)):.0f}',
                                   fontsize=7, color='#ff2200',
                                   ha='center', fontweight='bold', alpha=0.9)
                self.ephemeral.append(txt)
            if m2_taking:
                txt = self.ax.text(m2_pos[0] + 0.5, m2_pos[1] - 0.8,
                                   f'-{abs(state.m2_hp - (self.prev_m2_hp or state.m2_hp)):.0f}',
                                   fontsize=7, color='#ff2200',
                                   ha='center', fontweight='bold', alpha=0.9)
                self.ephemeral.append(txt)

        # ── Update positions ──
        self.m1_dot.set_data([m1_pos[0]], [m1_pos[1]])
        self.m2_dot.set_data([m2_pos[0]], [m2_pos[1]])
        self.zealot_dot.set_data([z_pos[0]], [z_pos[1]])

        # Trails
        if self.m1_trail:
            trail = np.array(self.m1_trail)
            self.m1_trail_line.set_data(trail[:, 0], trail[:, 1])
        if self.m2_trail:
            trail = np.array(self.m2_trail)
            self.m2_trail_line.set_data(trail[:, 0], trail[:, 1])
        if self.zealot_trail:
            trail = np.array(self.zealot_trail)
            self.zealot_trail_line.set_data(trail[:, 0], trail[:, 1])

        # Labels
        self.m1_label.set_position((m1_pos[0], m1_pos[1] + 0.8))
        self.m1_label.set_text(f'M1 [{m1_role}] {state.m1_hp:.0f}hp d={d1:.1f}')

        self.m2_label.set_position((m2_pos[0], m2_pos[1] + 0.8))
        self.m2_label.set_text(f'M2 [{m2_role}] {state.m2_hp:.0f}hp d={d2:.1f}')

        self.zealot_label.set_position((z_pos[0], z_pos[1] + 0.8))
        self.zealot_label.set_text(f'Zealot {state.zealot_hp:.0f}hp')

        # Range circles
        if self.range_circle_attack:
            self.range_circle_attack.remove()
        if self.range_circle_melee:
            self.range_circle_melee.remove()

        self.range_circle_attack = plt.Circle(
            (z_pos[0], z_pos[1]), 5.0, fill=False,
            color='#00ff00', alpha=0.2, linestyle='--', linewidth=1)
        self.ax.add_patch(self.range_circle_attack)

        self.range_circle_melee = plt.Circle(
            (z_pos[0], z_pos[1]), 1.5, fill=False,
            color='#ff0000', alpha=0.3, linestyle='-', linewidth=1)
        self.ax.add_patch(self.range_circle_melee)

        # Distance lines
        line1, = self.ax.plot([z_pos[0], m1_pos[0]], [z_pos[1], m1_pos[1]],
                               '--', color='#00aaff', alpha=0.15, linewidth=1)
        line2, = self.ax.plot([z_pos[0], m2_pos[0]], [z_pos[1], m2_pos[1]],
                               '--', color='#00ddff', alpha=0.15, linewidth=1)
        self.ephemeral.extend([line1, line2])

        # Status text
        near_d = min(d1, d2)
        far_d = max(d1, d2)
        sep = far_d - near_d
        self.status_text.set_text(
            f'Step: {step:4d}  Time: {game_time:5.1f}s\n'
            f'Zealot HP: {state.zealot_hp:5.1f} / {state.zealot_hp_max:.0f}\n'
            f'M1 HP: {state.m1_hp:3.0f}  M2 HP: {state.m2_hp:3.0f}\n'
            f'Near d: {near_d:4.1f}  Far d: {far_d:4.1f}\n'
            f'Separation: {sep:4.1f}'
        )

        # Auto-adjust view
        all_x = [m1_pos[0], m2_pos[0], z_pos[0]]
        all_y = [m1_pos[1], m2_pos[1], z_pos[1]]
        cx, cy = np.mean(all_x), np.mean(all_y)
        spread = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 12)
        margin = spread * 0.6 + 3
        self.ax.set_xlim(cx - margin, cx + margin)
        self.ax.set_ylim(cy - margin, cy + margin)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def show_result(self, result, reason):
        color = '#00ff00' if result == 'WIN' else '#ff0000'
        self.ax.text(0.5, 0.5, f'{result}\n{reason}',
                     transform=self.ax.transAxes, fontsize=24,
                     color=color, ha='center', va='center',
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                               alpha=0.9, edgecolor=color))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(3)

    def close(self):
        plt.close(self.fig)
