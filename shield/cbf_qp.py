import numpy as np

try:
    import osqp
    from scipy import sparse  # <-- CSC matrisler için

    HAVE_OSQP = True
except Exception:
    HAVE_OSQP = False


class CBFShield:
    """
    h(x) = dist_obs(x) - d_min >= 0
    QP:  min ||u - u_p||^2   s.t.  a u + alpha*h >= 0
    OSQP varsa CSC (sparse) matrislerle çözer; yoksa half-space projeksiyonuna düşer.
    """

    def __init__(self, d_min=0.08, alpha=2.0, qdot_max=1.2, dt=0.02):
        self.d_min = float(d_min)
        self.alpha = float(alpha)
        self.qdot_max = float(qdot_max)
        self.dt = float(dt)

    def _jacobian_approx(self, q):
        # ee konumunun q'ya sayısal jacobian'ı
        eps = 1e-4
        base = forward_kinematics(q)
        J = np.zeros((2, 3), dtype=np.float64)
        for i in range(3):
            dq = np.zeros_like(q)
            dq[i] = eps
            p = forward_kinematics(q + dq)
            J[:, i] = (p - base) / eps
        return J

    def project(self, state, u_policy):
        q = state[:3].astype(np.float64)
        ee = state[-4:-2].astype(np.float64)
        obs = state[-2:].astype(np.float64)
        to_obs = ee - obs
        dist = np.linalg.norm(to_obs)
        h = dist - self.d_min
        # dh/dt ~ n^T J qdot
        n = np.array([1.0, 0.0], dtype=np.float64) if dist < 1e-9 else (to_obs / dist)
        J = self._jacobian_approx(q)
        a = (n @ J).reshape(1, 3)  # (1x3)

        u_p = np.clip(u_policy, -self.qdot_max, self.qdot_max).astype(np.float64)

        if HAVE_OSQP:
            # Quadratic cost: (u - u_p)^T (u - u_p) = 0.5 u^T (2I) u + (-2 u_p)^T u + const
            P = sparse.csc_matrix(2.0 * np.eye(3))
            qv = -2.0 * u_p
            A = sparse.csc_matrix(a)  # (1x3) CSC
            l_bound = np.array([-self.alpha * h], dtype=np.float64)
            u = np.array([np.inf], dtype=np.float64)

            prob = osqp.OSQP()
            prob.setup(P=P, q=qv, A=A, l=l_bound, u=u, verbose=False, polish=False)
            res = prob.solve()
            u_sol = res.x if res.x is not None else u_p  # emniyet için fallback
        else:
            # Half-space projeksiyonu: {a u >= -alpha h}
            a_v = a.flatten()
            if a_v @ u_p + self.alpha * h >= 0:
                u_sol = u_p
            else:
                u_sol = (
                    u_p
                    + ((-self.alpha * h - a_v @ u_p) / (np.dot(a_v, a_v) + 1e-12)) * a_v
                )

        return np.clip(u_sol, -self.qdot_max, self.qdot_max).astype(np.float32)


# env'in FK fonksiyonunu tekrar kullan
from envs.franka_kinematic_env import FrankaKinematicEnv as _E


def forward_kinematics(q):
    e = _E()
    e.q = q.astype(np.float32)
    return e.fk(e.q).astype(np.float64)
