"""
Nonlinear Stochastic MPC via Sequential Quadratic Programming (SQP).

Solves the multi-scenario MPC by iteratively:
  1. Linearize dynamics around the current predicted trajectory
  2. Solve a QP for the optimal actions
  3. Re-linearize around the new predicted trajectory
  4. Repeat for N_SQP iterations

This handles the nonlinearity of zealot pursuit dynamics much better
than a single linearization, since each horizon step is linearized
at the predicted state for that step, not the current state.

State: x = [m1_x, m1_y, m2_x, m2_y, z_x, z_y]  (6-dim)
Action: u = [u_m1_x, u_m1_y, u_m2_x, u_m2_y]    (4-dim)
"""

import numpy as np
import osqp
from scipy import sparse

MARINE_SPEED = 2.25
ZEALOT_SPEED = 2.25
MARINE_RANGE = 5.0

D_NEAR_REF = 4.0    # push near ref out — give more buffer
D_FAR_REF = 5.0

Q_NEAR = 25.0       # much higher weight for near marine tracking
Q_FAR = 8.0
R_ACTION = 0.05     # lower action penalty — let marines move freely
R_MOVE_FAR = 2.0    # but still penalize far marine moving

N_SQP = 3

ZEALOT_SWITCH_SHARPNESS = 3.0


def zealot_target_probability(d1, d2):
    gap = ZEALOT_SWITCH_SHARPNESS * (d2 - d1)
    return 1.0 / (1.0 + np.exp(-np.clip(gap, -20, 20)))


def simulate_zealot_step(z, m_target, dt):
    """Exact (nonlinear) zealot step: chase m_target."""
    r = m_target - z
    d = np.linalg.norm(r)
    if d < 0.1:
        return z.copy()
    return z + (r / d) * min(ZEALOT_SPEED * dt, d)


def linearize_zealot(m_target, z_pos, dt):
    """Linearize zealot pursuit around (m_target, z_pos)."""
    r = m_target - z_pos
    d = np.linalg.norm(r)
    if d < 0.3:
        return np.eye(2), np.zeros((2, 2)), z_pos.copy()

    r_hat = r / d
    I2 = np.eye(2)
    P = (I2 - np.outer(r_hat, r_hat)) / d
    v_dt = ZEALOT_SPEED * dt

    A_zz = I2 - v_dt * P
    A_zm = v_dt * P
    z_next = z_pos + min(v_dt, d) * r_hat
    c_z = z_next - A_zz @ z_pos - A_zm @ m_target

    return A_zz, A_zm, c_z


def build_dynamics_at_state(m1, m2, z, target_idx, dt):
    """Build A, B, c for dynamics linearized at a specific state."""
    m_target = m1 if target_idx == 0 else m2
    A_zz, A_zm, c_z = linearize_zealot(m_target, z, dt)

    A = np.eye(6)
    A[4:6, 4:6] = A_zz
    if target_idx == 0:
        A[4:6, 0:2] = A_zm
    else:
        A[4:6, 2:4] = A_zm

    speed_dt = MARINE_SPEED * dt
    B = np.zeros((6, 4))
    B[0, 0] = speed_dt
    B[1, 1] = speed_dt
    B[2, 2] = speed_dt
    B[3, 3] = speed_dt

    c = np.zeros(6)
    c[4:6] = c_z
    return A, B, c


def compute_reference(m_near, m_far, z, target_idx):
    """Compute reference positions for near and far marines."""
    # Near: on circle of D_NEAR_REF, tangentially offset for kiting
    r_near = m_near - z
    d_near = np.linalg.norm(r_near)
    if d_near > 0.1:
        r_hat = r_near / d_near
        tangent = np.array([-r_hat[1], r_hat[0]])
        ref_dir = 0.7 * r_hat + 0.3 * tangent
        ref_dir = ref_dir / np.linalg.norm(ref_dir)
        near_ref = z + D_NEAR_REF * ref_dir
    else:
        near_ref = z + np.array([D_NEAR_REF, 0.0])

    # Far: on circle of D_FAR_REF, current direction from zealot
    r_far = m_far - z
    d_far = np.linalg.norm(r_far)
    if d_far > 0.1:
        far_ref = z + D_FAR_REF * (r_far / d_far)
    else:
        far_ref = z + np.array([0.0, D_FAR_REF])

    # Ensure angular separation
    v_n = near_ref - z
    v_f = far_ref - z
    nn, nf = np.linalg.norm(v_n), np.linalg.norm(v_f)
    if nn > 0.1 and nf > 0.1:
        cos_a = np.clip(np.dot(v_n, v_f) / (nn * nf), -1, 1)
        if np.arccos(cos_a) < np.pi / 4:
            rot = np.pi / 3
            c, s = np.cos(rot), np.sin(rot)
            v_rot = np.array([c * v_f[0] - s * v_f[1], s * v_f[0] + c * v_f[1]])
            far_ref = z + D_FAR_REF * v_rot / np.linalg.norm(v_rot)

    if target_idx == 0:
        return np.concatenate([near_ref, far_ref, z])
    else:
        return np.concatenate([far_ref, near_ref, z])


def forward_simulate(x0, actions, target_idx, dt, H):
    """Forward simulate the nonlinear dynamics to get a predicted trajectory.
    Used to get the linearization points for SQP.
    """
    trajectory = [x0.copy()]
    x = x0.copy()
    for t in range(H):
        if t < len(actions):
            u = actions[t]
        else:
            u = np.zeros(4)

        # Marine movement
        x[0:2] += MARINE_SPEED * dt * u[0:2]
        x[2:4] += MARINE_SPEED * dt * u[2:4]

        # Zealot movement (nonlinear)
        m_target = x[0:2] if target_idx == 0 else x[2:4]
        x[4:6] = simulate_zealot_step(x[4:6], m_target, dt)

        trajectory.append(x.copy())

    return trajectory


def solve_sqp_smpc(state, horizon=6, dt=0.5):
    """Nonlinear SMPC via SQP with N_SQP re-linearizations."""
    S = 2
    H = horizon
    nx, nu = 6, 4

    n_u_total = nu * H
    n_x_total = nx * H * S
    n_vars = n_u_total + n_x_total

    d1 = state.dist_m1_zealot
    d2 = state.dist_m2_zealot
    p1 = zealot_target_probability(d1, d2)
    weights = [p1, 1.0 - p1]

    x0 = np.array([
        state.m1_pos[0], state.m1_pos[1],
        state.m2_pos[0], state.m2_pos[1],
        state.zealot_pos[0], state.zealot_pos[1],
    ])

    # Initialize with zero actions
    current_actions = [np.zeros(nu) for _ in range(H)]

    def u_idx(t):
        return t * nu
    def x_idx(s, t):
        return n_u_total + s * (nx * H) + (t - 1) * nx

    best_actions = current_actions

    for sqp_iter in range(N_SQP):
        # ── Step 1: Forward simulate to get linearization points ──
        trajectories = []
        for s in range(S):
            traj = forward_simulate(x0, current_actions, target_idx=s, dt=dt, H=H)
            trajectories.append(traj)

        # ── Step 2: Linearize dynamics at each (scenario, timestep) ──
        # and compute reference positions
        dynamics_data = []  # [(A_st, B_st, c_st)] for each (s, t)
        references = []     # [x_ref_s] for each s

        for s in range(S):
            traj = trajectories[s]
            scenario_dynamics = []
            for t in range(H):
                x_t = traj[t]
                m1_t = x_t[0:2]
                m2_t = x_t[2:4]
                z_t = x_t[4:6]
                A_st, B_st, c_st = build_dynamics_at_state(m1_t, m2_t, z_t, s, dt)
                scenario_dynamics.append((A_st, B_st, c_st))
            dynamics_data.append(scenario_dynamics)

            # Compute per-timestep references along the predicted trajectory
            # so the near marine's reference ORBITS the zealot over the horizon
            step_refs = []
            for t in range(H + 1):
                x_t = traj[t]
                m1_t, m2_t, z_t = x_t[0:2], x_t[2:4], x_t[4:6]
                if s == 0:
                    ref_t = compute_reference(m1_t, m2_t, z_t, target_idx=0)
                else:
                    ref_t = compute_reference(m2_t, m1_t, z_t, target_idx=1)
                step_refs.append(ref_t)
            references.append(step_refs)

        # ── Step 3: Build and solve QP ──
        P = np.zeros((n_vars, n_vars))
        q_vec = np.zeros(n_vars)

        for s in range(S):
            w = weights[s]
            step_refs = references[s]

            if s == 0:
                near_idx = [0, 1]
                far_idx = [2, 3]
            else:
                near_idx = [2, 3]
                far_idx = [0, 1]

            Q_diag = np.zeros(nx)
            Q_diag[near_idx[0]] = Q_NEAR
            Q_diag[near_idx[1]] = Q_NEAR
            Q_diag[far_idx[0]] = Q_FAR
            Q_diag[far_idx[1]] = Q_FAR
            Q_mat = np.diag(Q_diag)

            for t in range(1, H + 1):
                xi = x_idx(s, t)
                x_ref_t = step_refs[t]  # per-timestep reference
                P[xi:xi + nx, xi:xi + nx] += 2.0 * w * Q_mat
                q_vec[xi:xi + nx] += -2.0 * w * Q_mat @ x_ref_t

        # Safety penalty: for each scenario, add large cost if near marine
        # is predicted to be below d=2.5 at any step.
        # Linearized: d_near ≈ d_near_pred + g_near' @ (x - x_pred)
        # Penalty: Q_SAFETY * max(0, d_min - d_near)^2
        # ≈ Q_SAFETY * max(0, d_min - d_near_pred - g_near'@δx)^2
        Q_SAFETY = 50.0
        D_MIN = 2.5
        for s in range(S):
            w = weights[s]
            traj = trajectories[s]
            near_marine_idx = [0, 1] if s == 0 else [2, 3]

            for t in range(1, H + 1):
                x_pred = traj[t]
                z_pred = x_pred[4:6]
                m_near_pred = x_pred[near_marine_idx[0]:near_marine_idx[0] + 2]

                r = m_near_pred - z_pred
                d_pred = np.linalg.norm(r)

                if d_pred < D_MIN + 1.0:  # only add penalty when close to danger
                    if d_pred > 0.1:
                        r_hat = r / d_pred
                    else:
                        r_hat = np.array([1.0, 0.0])

                    # Gradient of distance w.r.t. state
                    g = np.zeros(nx)
                    g[near_marine_idx[0]] = r_hat[0]
                    g[near_marine_idx[0] + 1] = r_hat[1]
                    g[4] = -r_hat[0]
                    g[5] = -r_hat[1]

                    # How far below safety threshold
                    violation = D_MIN - d_pred
                    if violation > 0:
                        # Already below threshold — strong quadratic + linear push
                        xi = x_idx(s, t)
                        P[xi:xi + nx, xi:xi + nx] += 2.0 * w * Q_SAFETY * np.outer(g, g)
                        q_vec[xi:xi + nx] += 2.0 * w * Q_SAFETY * violation * (-g)
                    elif violation > -1.0:
                        # Approaching threshold — softer penalty
                        soft_w = Q_SAFETY * 0.3 * (1.0 + violation)  # ramps from 0 to 0.3*Q_SAFETY
                        xi = x_idx(s, t)
                        P[xi:xi + nx, xi:xi + nx] += 2.0 * w * soft_w * np.outer(g, g)

        # Action regularization
        for t in range(H):
            ui = u_idx(t)
            P[ui:ui + nu, ui:ui + nu] += 2.0 * R_ACTION * np.eye(nu)
            # Extra penalty for far marine moving
            if d1 > d2:
                P[ui, ui] += 2.0 * R_MOVE_FAR
                P[ui + 1, ui + 1] += 2.0 * R_MOVE_FAR
            else:
                P[ui + 2, ui + 2] += 2.0 * R_MOVE_FAR
                P[ui + 3, ui + 3] += 2.0 * R_MOVE_FAR

        P = 0.5 * (P + P.T)

        # Dynamics constraints (re-linearized at predicted trajectory)
        n_eq = nx * H * S
        A_eq = np.zeros((n_eq, n_vars))
        b_eq = np.zeros(n_eq)

        row = 0
        for s in range(S):
            for t in range(H):
                A_st, B_st, c_st = dynamics_data[s][t]
                ui = u_idx(t)
                xi_next = x_idx(s, t + 1)

                A_eq[row:row + nx, ui:ui + nu] = B_st
                A_eq[row:row + nx, xi_next:xi_next + nx] = -np.eye(nx)

                if t == 0:
                    b_eq[row:row + nx] = -c_st - A_st @ x0
                else:
                    xi_curr = x_idx(s, t)
                    A_eq[row:row + nx, xi_curr:xi_curr + nx] = A_st
                    b_eq[row:row + nx] = -c_st

                row += nx

        # Solve
        P_sparse = sparse.csc_matrix(P)
        A_eq_sparse = sparse.csc_matrix(A_eq)
        A_bounds = sparse.eye(n_vars, format='csc')
        A_full = sparse.vstack([A_eq_sparse, A_bounds], format='csc')

        l_bounds = -np.inf * np.ones(n_vars)
        u_bounds = np.inf * np.ones(n_vars)
        for t in range(H):
            ui = u_idx(t)
            l_bounds[ui:ui + nu] = -1.0
            u_bounds[ui:ui + nu] = 1.0

        l_full = np.concatenate([b_eq, l_bounds])
        u_full = np.concatenate([b_eq, u_bounds])

        solver = osqp.OSQP()
        solver.setup(P_sparse, q_vec, A_full, l_full, u_full,
                     verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                     max_iter=4000, warm_start=True, polish=True)
        result = solver.solve()

        if result.info.status in ('solved', 'solved_inaccurate'):
            # Extract actions for next SQP iteration
            current_actions = []
            for t in range(H):
                ui = u_idx(t)
                u_t = result.x[ui:ui + nu]
                current_actions.append(u_t)
            best_actions = current_actions
        else:
            break  # QP failed, use previous best

    # Extract first action
    u0 = best_actions[0]
    m1_action = u0[0:2].copy()
    m2_action = u0[2:4].copy()

    for action in [m1_action, m2_action]:
        n = np.linalg.norm(action)
        if n > 1.0:
            action[:] = action / n

    info = {
        'qp_status': result.info.status if result else 'no_solution',
        'qp_obj': result.info.obj_val if result else 0.0,
        'p_chase_m1': p1,
        'sqp_iters': sqp_iter + 1,
    }

    return m1_action, m2_action, info


def mpc_select_action_qp(state):
    """Drop-in replacement for mpc_select_action using SQP-based SMPC."""
    m1_action, m2_action, info = solve_sqp_smpc(state, horizon=6, dt=0.5)

    from cost_function import compute_cost
    from micro_scenario import simulate_trajectory

    trajectory = simulate_trajectory(state, [m1_action], [m2_action], dt=0.5)
    if trajectory:
        _, components = compute_cost(trajectory[0])
    else:
        components = {}

    components['qp_obj'] = info.get('qp_obj', 0.0)
    components['p_chase_m1'] = info.get('p_chase_m1', 0.5)

    return m1_action, m2_action, components
