import time

import numpy as np

try:
    import osqp
    from scipy import sparse

    HAVE_OSQP = True
except Exception:
    HAVE_OSQP = False

from envs.franka_kinematic_env import FrankaKinematicEnv as _E


# --- Yardımcılar ---
def fk(q):
    e = _E()
    e.q = q.astype(np.float32)
    return e.fk(e.q).astype(np.float64)


def jacobian_num(q):
    eps = 1e-4
    base = fk(q)
    J = np.zeros((2, 3), dtype=np.float64)
    for i in range(3):
        dq = np.zeros_like(q)
        dq[i] = eps
        p = fk(q + dq)
        J[:, i] = (p - base) / eps
    return J


class MPCShield:
    """
    QP tabanlı MPC Shield (H ufuk):
      q_{k+1} = q_k + u_k*dt
      ee_{k+1} ≈ ee_k + J(q_k) u_k dt
    Güvenlik kısıtı (her k): n_k^T J(q_k) u_k + alpha * h_k >= 0
      h_k = ||ee_k - obs|| - d_min, n_k = (ee_k - obs)/||ee_k - obs||
    Maliyet: Σ ||u_k - u_ref||^2 + rho Σ ||u_k - u_{k-1}||^2
    Özellikler:
      - Çok-adımlı "referans yörünge" propagasyonu (u_ref ile)
      - OSQP warm-start
      - Çözüm süresi (ms) ölçümü + dışarıdan erişim
    """

    def __init__(
        self, d_min=0.08, alpha=2.0, qdot_max=1.2, dt=0.02, horizon=10, rho=0.0
    ):
        self.d_min = float(d_min)
        self.alpha = float(alpha)
        self.qdot_max = float(qdot_max)
        self.dt = float(dt)
        self.H = int(horizon)
        self.rho = float(rho)
        self._last_solve_ms = 0.0
        self._warm_U = None  # warm-start vektörü (3H,)
        try:
            from .cbf_qp import CBFShield

            self.cbf = CBFShield(d_min=d_min, alpha=alpha, qdot_max=qdot_max, dt=dt)
        except Exception:
            self.cbf = None

    # --- referans yörünge oluşturma (u_ref ile ileri besleme) ---
    def _rollout_ref(self, q0, ee0, obs, u_ref):
        q = q0.astype(np.float64).copy()
        ee = ee0.astype(np.float64).copy()
        a_list, h_list = [], []
        for _ in range(self.H):
            to_obs = ee - obs
            dist = float(np.linalg.norm(to_obs) + 1e-12)
            h = dist - self.d_min
            n = to_obs / dist
            J = jacobian_num(q)
            a = (n @ J).reshape(3)
            a_list.append(a)
            h_list.append(h)
            # referans propagasyon: bir adım ilerlet
            ee = ee + (J @ u_ref) * self.dt
            q = q + u_ref * self.dt
        return a_list, h_list

    def get_last_solve_ms(self):
        return float(self._last_solve_ms)

    def project(self, state, u_policy):
        q = state[:3].astype(np.float64)
        ee = state[-4:-2].astype(np.float64)
        obs = state[-2:].astype(np.float64)
        u_ref = np.clip(u_policy, -self.qdot_max, self.qdot_max).astype(np.float64)

        if not HAVE_OSQP:
            return (
                self.cbf.project(state, u_policy)
                if self.cbf is not None
                else u_ref.astype(np.float32)
            )

        # --- Maliyet ---
        nvar = 3 * self.H
        P = sparse.csc_matrix(2.0 * sparse.eye(nvar))
        qv = np.zeros(nvar, dtype=np.float64)
        for k in range(self.H):
            qv[3 * k : 3 * (k + 1)] += -2.0 * u_ref
        if self.rho > 0 and self.H > 1:
            rows = []
            for k in range(1, self.H):
                for j in range(3):
                    row = np.zeros(nvar)
                    row[3 * k + j] = 1.0
                    row[3 * (k - 1) + j] = -1.0
                    rows.append(row)
            if rows:
                D = sparse.csc_matrix(np.vstack(rows))
                P = P + 2.0 * self.rho * (D.T @ D)

        # --- Kısıtlar (A*U ∈ [l,u]) ---
        rows_idx, cols_idx, data_val = [], [], []
        l_list, u_list = [], []
        row = 0

        # Çok-adımlı referans yörüngeye göre lineerleştir
        a_list, h_list = self._rollout_ref(q, ee, obs, u_ref)

        # 1) Güvenlik satırları
        for k in range(self.H):
            a = a_list[k]
            for j in range(3):
                rows_idx.append(row)
                cols_idx.append(3 * k + j)
                data_val.append(a[j])
            l_list.append(-self.alpha * h_list[k])
            u_list.append(np.inf)
            row += 1

        # 2) Kutulama: |u_i| <= qmax
        qmax = self.qdot_max
        for i in range(nvar):
            rows_idx.append(row)
            cols_idx.append(i)
            data_val.append(1.0)
            l_list.append(-np.inf)
            u_list.append(qmax)
            row += 1
            rows_idx.append(row)
            cols_idx.append(i)
            data_val.append(-1.0)
            l_list.append(-qmax)
            u_list.append(np.inf)
            row += 1

        m = row
        A = sparse.coo_matrix((data_val, (rows_idx, cols_idx)), shape=(m, nvar)).tocsc()
        l_bound = np.asarray(l_list, dtype=np.float64)
        u = np.asarray(u_list, dtype=np.float64)

        # --- Çöz (warm-start + süre ölçümü) ---
        prob = osqp.OSQP()
        prob.setup(P=P, q=qv, A=A, l=l_bound, u=u, verbose=False, polish=False)
        if self._warm_U is not None and self._warm_U.shape == (nvar,):
            try:
                prob.warm_start(x=self._warm_U)
            except Exception:
                pass

        t0 = time.perf_counter()
        res = prob.solve()
        self._last_solve_ms = (time.perf_counter() - t0) * 1000.0

        # --- Sağlam çıktı ---
        if res.x is None:
            # infeasible/diğer: CBF fallback -> yoksa u_ref
            out = self.cbf.project(state, u_policy) if self.cbf is not None else u_ref
            return np.clip(out, -self.qdot_max, self.qdot_max).astype(np.float32)

        u0 = np.asarray(res.x[:3])
        if u0.dtype == object or not np.all(np.isfinite(u0)):
            out = self.cbf.project(state, u_policy) if self.cbf is not None else u_ref
            return np.clip(out, -self.qdot_max, self.qdot_max).astype(np.float32)

        # warm-start vektörünü kaydet (ileri kaydır)
        U = np.asarray(res.x).astype(np.float64)
        self._warm_U = np.roll(U, -3)
        self._warm_U[-3:] = U[-3:]

        return np.clip(u0.astype(np.float64), -self.qdot_max, self.qdot_max).astype(
            np.float32
        )
