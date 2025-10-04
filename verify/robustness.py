# verify/robustness.py
import numpy as np

from specs.specs import stl_robustness, violation_from_robustness


class EpisodeTracer:
    def __init__(self):
        self.buf = []

    def add(self, info: dict, qdot=None):
        # info sözlük değilse boş al
        rec = dict(info) if isinstance(info, dict) else {}

        # qdot parametresi yoksa info içinden dene
        if qdot is None:
            qdot = rec.get("qdot", None)

        # Güvenli maksimum |qdot| hesabı
        if qdot is None:
            max_abs = 0.0
        else:
            qdot_arr = np.asarray(qdot, dtype=np.float32)
            max_abs = float(np.max(np.abs(qdot_arr))) if qdot_arr.size > 0 else 0.0

        rec["qdot_inf"] = max_abs
        self.buf.append(rec)

    def clear(self):
        # Bölüm bitince buffer’ı sıfırla
        self.buf.clear()

    def summary(self, d_min: float, qdot_max: float):
        rob = stl_robustness(self.buf, float(d_min), float(qdot_max))
        return {
            "robustness": rob,
            "violation": violation_from_robustness(rob),
        }
