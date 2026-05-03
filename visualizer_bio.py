"""
Live matplotlib visualization for the Bio + Medivac vs Mixed Protoss
scenario. Shows all units with type-specific colors / shapes, HP labels,
movement trails for the key support units (marauders + medivac),
damage flashes, and a status panel.

Color / marker key:
  Marines      light blue circles
  Marauders    dark blue squares (larger)
  Medivac      teal star (largest)
  Zealots      red diamonds
  Stalkers     orange triangles
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque


COLOR_MARINE = '#00aaff'
COLOR_MARAUDER = '#0066cc'
COLOR_MEDIVAC = '#00ddaa'
COLOR_ZEALOT = '#ff4444'
COLOR_STALKER = '#ff8800'

FLASH_DURATION = 3   # frames


class BioVisualizer:
    def __init__(self, trail_len=80):
        self.trail_len = trail_len

        # Per-unit trails (key units only)
        self.marauder_trails = None     # initialized on first frame
        self.medivac_trails = None

        # HP tracking for damage-flash detection
        self.prev_m_hps = None
        self.prev_mm_hps = None
        self.prev_mv_hps = None
        self.prev_z_hps = None
        self.prev_s_hps = None

        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 9))
        self.fig.canvas.manager.set_window_title(
            'SC2 Bio: 6M+2MM+1Mv vs 3Z+2S')
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1a1a2e')
        self.ax.grid(True, alpha=0.15, color='white')
        self.fig.patch.set_facecolor('#0a0a1a')
        self.ax.tick_params(colors='white')

        # Unit artists (will be rebuilt each frame for varying counts)
        self.unit_artists = []
        self.label_artists = []
        self.trail_artists = []
        self.ephemeral = []

        # Status text
        self.status_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=10, color='white', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e',
                      alpha=0.85, edgecolor='#444'))

        # Legend
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MARINE,
                   markersize=9, label='Marine', linestyle=''),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MARAUDER,
                   markersize=11, label='Marauder', linestyle=''),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=COLOR_MEDIVAC,
                   markersize=15, label='Medivac', linestyle=''),
            Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_ZEALOT,
                   markersize=10, label='Zealot', linestyle=''),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=COLOR_STALKER,
                   markersize=11, label='Stalker', linestyle=''),
        ]
        self.ax.legend(handles=legend_handles, loc='lower right',
                       facecolor='#1a1a2e', edgecolor='#444',
                       labelcolor='white', fontsize=8)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def _init_trails(self, state):
        """Lazy init of trail deques sized to actual unit counts."""
        self.marauder_trails = [deque(maxlen=self.trail_len)
                                  for _ in range(state.n_marauders)]
        self.medivac_trails = [deque(maxlen=self.trail_len)
                                 for _ in range(state.n_medivacs)]

    def _clear_artists(self):
        for a in self.unit_artists + self.label_artists + self.trail_artists \
                  + self.ephemeral:
            try:
                a.remove()
            except Exception:
                pass
        self.unit_artists = []
        self.label_artists = []
        self.trail_artists = []
        self.ephemeral = []

    def update(self, state, step, game_time):
        if self.marauder_trails is None:
            self._init_trails(state)

        self._clear_artists()

        # ── Detect damage events ──
        m_hp = list(state.marine_hps)
        mm_hp = list(state.marauder_hps)
        mv_hp = list(state.medivac_hps)
        z_hp = list(state.zealot_hps)
        s_hp = list(state.stalker_hps)

        z_taking = [False] * state.n_zealots
        s_taking = [False] * state.n_stalkers
        m_taking = [False] * state.n_marines
        mm_taking = [False] * state.n_marauders
        mv_taking = [False] * state.n_medivacs

        if self.prev_z_hps is not None:
            z_taking = [(z_hp[j] - self.prev_z_hps[j]) < -0.5
                        for j in range(state.n_zealots)]
            s_taking = [(s_hp[j] - self.prev_s_hps[j]) < -0.5
                        for j in range(state.n_stalkers)]
            m_taking = [(m_hp[i] - self.prev_m_hps[i]) < -0.5
                        for i in range(state.n_marines)]
            mm_taking = [(mm_hp[i] - self.prev_mm_hps[i]) < -0.5
                         for i in range(state.n_marauders)]
            mv_taking = [(mv_hp[i] - self.prev_mv_hps[i]) < -0.5
                         for i in range(state.n_medivacs)]

        self.prev_m_hps = m_hp
        self.prev_mm_hps = mm_hp
        self.prev_mv_hps = mv_hp
        self.prev_z_hps = z_hp
        self.prev_s_hps = s_hp

        def _draw_unit(pos, color, marker, size, hp, hp_max, taking_dmg, label):
            edge = '#ff2200' if taking_dmg else 'white'
            edge_w = 2.5 if taking_dmg else 0.6
            artist, = self.ax.plot([pos[0]], [pos[1]], marker=marker,
                                    color=color, markersize=size, zorder=5,
                                    markeredgecolor=edge, markeredgewidth=edge_w,
                                    linestyle='')
            self.unit_artists.append(artist)
            # HP label (above)
            txt = self.ax.text(pos[0], pos[1] + 0.6, f'{hp:.0f}',
                                fontsize=7, color=color, ha='center',
                                fontweight='bold')
            self.label_artists.append(txt)

        # ── Draw bio (marines, marauders, medivac) ──
        for i in range(state.n_marines):
            if m_hp[i] > 0:
                _draw_unit(state.marine_positions[i], COLOR_MARINE, 'o', 9,
                           m_hp[i], 45, m_taking[i], f'M{i}')
        for i in range(state.n_marauders):
            if mm_hp[i] > 0:
                _draw_unit(state.marauder_positions[i], COLOR_MARAUDER, 's', 12,
                           mm_hp[i], 125, mm_taking[i], f'MM{i}')
                self.marauder_trails[i].append(np.array(state.marauder_positions[i]))
        for i in range(state.n_medivacs):
            if mv_hp[i] > 0:
                _draw_unit(state.medivac_positions[i], COLOR_MEDIVAC, '*', 17,
                           mv_hp[i], 150, mv_taking[i], f'Mv{i}')
                self.medivac_trails[i].append(np.array(state.medivac_positions[i]))

        # ── Draw enemies ──
        for j in range(state.n_zealots):
            if state.zealot_alive[j]:
                _draw_unit(state.zealot_positions[j], COLOR_ZEALOT, 'D', 11,
                           z_hp[j], 150, z_taking[j], f'Z{j}')
        for j in range(state.n_stalkers):
            if state.stalker_alive[j]:
                _draw_unit(state.stalker_positions[j], COLOR_STALKER, '^', 12,
                           s_hp[j], 160, s_taking[j], f'S{j}')

        # ── Trails (marauders + medivac only — too many marines) ──
        for trail in self.marauder_trails:
            if len(trail) > 1:
                pts = np.array(trail)
                ln, = self.ax.plot(pts[:, 0], pts[:, 1], '-',
                                    color=COLOR_MARAUDER, alpha=0.35, linewidth=1)
                self.trail_artists.append(ln)
        for trail in self.medivac_trails:
            if len(trail) > 1:
                pts = np.array(trail)
                ln, = self.ax.plot(pts[:, 0], pts[:, 1], '-',
                                    color=COLOR_MEDIVAC, alpha=0.4, linewidth=1.5)
                self.trail_artists.append(ln)

        # ── Heal indicator: green line from medivac to nearest injured bio ──
        if state.n_medivacs and mv_hp[0] > 0:
            mv_p = np.array(state.medivac_positions[0])
            best = None
            best_dmg = 0
            for i in range(state.n_marines):
                if m_hp[i] > 0 and (45 - m_hp[i]) > best_dmg:
                    d = float(np.linalg.norm(mv_p - np.array(state.marine_positions[i])))
                    if d <= 4.0:
                        best_dmg = 45 - m_hp[i]
                        best = state.marine_positions[i]
            for i in range(state.n_marauders):
                if mm_hp[i] > 0 and (125 - mm_hp[i]) > best_dmg:
                    d = float(np.linalg.norm(mv_p - np.array(state.marauder_positions[i])))
                    if d <= 4.0:
                        best_dmg = 125 - mm_hp[i]
                        best = state.marauder_positions[i]
            if best is not None:
                ln, = self.ax.plot([mv_p[0], best[0]], [mv_p[1], best[1]],
                                    '-', color='#88ff66', alpha=0.7, linewidth=2,
                                    zorder=4)
                self.ephemeral.append(ln)

        # ── Status panel ──
        n_m_alive = sum(1 for h in m_hp if h > 0)
        n_mm_alive = sum(1 for h in mm_hp if h > 0)
        n_mv_alive = sum(1 for h in mv_hp if h > 0)
        n_z_alive = sum(state.zealot_alive)
        n_s_alive = sum(state.stalker_alive)
        bio_total_hp = sum(h for h in m_hp + mm_hp if h > 0)
        e_total_hp = sum(z_hp[j] for j in range(state.n_zealots) if state.zealot_alive[j])
        e_total_hp += sum(s_hp[j] for j in range(state.n_stalkers) if state.stalker_alive[j])
        self.status_text.set_text(
            f'Step: {step:4d}  Time: {game_time:5.1f}s\n'
            f'Bio   : {n_m_alive}M  {n_mm_alive}MM  {n_mv_alive}Mv   '
            f'HP={bio_total_hp:5.0f}\n'
            f'Enemy : {n_z_alive}Z  {n_s_alive}S            '
            f'HP={e_total_hp:5.0f}'
        )

        # ── Auto-zoom to all live units ──
        all_x, all_y = [], []
        for i in range(state.n_marines):
            if m_hp[i] > 0:
                all_x.append(state.marine_positions[i][0])
                all_y.append(state.marine_positions[i][1])
        for i in range(state.n_marauders):
            if mm_hp[i] > 0:
                all_x.append(state.marauder_positions[i][0])
                all_y.append(state.marauder_positions[i][1])
        for j in range(state.n_zealots):
            if state.zealot_alive[j]:
                all_x.append(state.zealot_positions[j][0])
                all_y.append(state.zealot_positions[j][1])
        for j in range(state.n_stalkers):
            if state.stalker_alive[j]:
                all_x.append(state.stalker_positions[j][0])
                all_y.append(state.stalker_positions[j][1])
        if all_x and all_y:
            cx, cy = np.mean(all_x), np.mean(all_y)
            spread = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 12)
            margin = spread * 0.55 + 3
            self.ax.set_xlim(cx - margin, cx + margin)
            self.ax.set_ylim(cy - margin, cy + margin)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def show_result(self, result, reason):
        color = '#00ff66' if result == 'WIN' else '#ff5555'
        self.ax.text(0.5, 0.5, f'{result}\n{reason}',
                     transform=self.ax.transAxes, fontsize=22,
                     color=color, ha='center', va='center',
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                               alpha=0.9, edgecolor=color))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(4)

    def close(self):
        plt.close(self.fig)
