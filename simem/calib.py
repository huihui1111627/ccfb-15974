"""模型参数在线校准。

原理: 网格模型的输出(进风温度场)是参数 p 的非线性函数。用有限差分雅可比 +
阻尼高斯-牛顿(Levenberg-Marquardt)最小化"进风侧传感器读数 - 模型预测"的残差,
带物理边界约束与阻尼,防止过拟合个别热点。

校准消费传感器读数,只更新 `RoomModel` 的参数;预测用的"数字孪生"模型用同一套
参数,因此校准的好坏直接决定预测精度。
"""
from __future__ import annotations

import numpy as np

from . import layout as L

# 只校准对进风温度最敏感、可辨识的几个参数(避免 8 参数在有限观测下欠定)。
# h_mix/h_side 对进风温度影响弱且与 gap_flow 强相关, 保持标称值。
CALIB_SPEC = {
    'gap_flow':  (0.03, 0.90, L.GAP_FLOW),
    'recirc':    (0.005, 0.22, L.RECIRC),
    'v_kw':      (0.010, 0.150, L.V_CAB_KW),
    'h_wall':    (0.002, 0.06, 0.014),
}


class Calibrator:
    """在线校准器。假设传感器读数与模型预测之差是参数误差引起。"""

    def __init__(self, room, sensors, spec=None):
        self.room = room
        self.sensors = sensors
        self.spec = spec or CALIB_SPEC
        self.keys = list(self.spec)
        self.theta = np.array([self.spec[k][2] for k in self.keys])
        self.lo = np.array([self.spec[k][0] for k in self.keys])
        self.hi = np.array([self.spec[k][1] for k in self.keys])
        self.lam = 1e-2
        self.history = []
        # 进风侧传感器 -> 代表机柜(取该网格处所有机柜的中位)
        tp = sensors.tp
        rack_of_cell = {}
        for n, c in enumerate(tp.rack_front):
            rack_of_cell.setdefault(int(c), []).append(n)
        self.inlet_idx = []
        for mask, c in zip(sensors.inlet_mask, sensors.cells):
            if not mask:
                continue
            cand = rack_of_cell.get(int(c), [])
            if cand:
                self.inlet_idx.append(int(cand[len(cand) // 2]))
        self.inlet_idx = np.asarray(self.inlet_idx, np.int32)

    # ------------------------------------------------------------ 参数
    def _apply(self, theta):
        p = {k: float(v) for k, v in zip(self.keys, theta)}
        self.room.apply_params(p)

    def param_dict(self):
        return {k: float(v) for k, v in zip(self.keys, self.theta)}

    def _roll(self, state, n, dt_s, args):
        rack_kw, T_sup, v_crah, crah_on, tile, t_out, q_misc = args
        self.room.T = state.copy()
        for _ in range(n):
            self.room.step(dt_s, rack_kw, T_sup, v_crah, crah_on, tile,
                           t_out, q_misc=q_misc, iters=30, tol=2e-4)
        return self.room.rack_inlet()[self.inlet_idx]

    def calibrate(self, state, dt_s, args, y_meas, iters=4, warm=8):
        """state: 当前温度场;args: (rack_kw,T_sup,v_crah,crah_on,tile,t_out,q_misc);
        y_meas: 进风侧传感器读数(与 sensors.inlet_mask 对齐)。"""
        saved = self.room.T.copy()
        best = self.theta.copy()
        self._apply(self.theta)
        pred = self._roll(state, warm, dt_s, args)
        resid = pred - y_meas
        cost = float(np.mean(resid ** 2))
        base = cost
        n_par = len(self.theta)
        hstep = 1e-3 * np.maximum(np.abs(self.theta), 5e-3)
        for it in range(iters):
            J = np.zeros((len(resid), n_par))
            for k in range(n_par):
                th = self.theta.copy()
                # 中心化步长, 保证不因触界退化为 0
                dk = hstep[k]
                th[k] = np.clip(self.theta[k] + dk, self.lo[k], self.hi[k])
                if th[k] == self.theta[k]:
                    th[k] = np.clip(self.theta[k] - dk, self.lo[k], self.hi[k])
                denom = th[k] - self.theta[k]
                self._apply(th)
                pk = self._roll(state, warm, dt_s, args)
                if abs(denom) < 1e-12:
                    J[:, k] = 0.0
                else:
                    d = (pk - pred) / denom
                    J[:, k] = np.clip(d, -1e3, 1e3)
            self._apply(self.theta)
            J = np.nan_to_num(J, nan=0.0, posinf=1e3, neginf=-1e3)
            A = J.T @ J
            g = J.T @ resid
            diag = np.maximum(np.diag(A), 1e-8)
            accepted = False
            for _ in range(5):
                M = A + self.lam * np.diag(diag)
                try:
                    dth = -np.linalg.solve(M, g)
                except np.linalg.LinAlgError:
                    break
                th_new = np.clip(self.theta + dth, self.lo, self.hi)
                if np.allclose(th_new, self.theta, atol=1e-8):
                    break
                self._apply(th_new)
                pred2 = self._roll(state, warm, dt_s, args)
                cost2 = float(np.mean((pred2 - y_meas) ** 2))
                if cost2 < cost * (1 - 1e-6):
                    self.theta = th_new
                    pred, resid, cost = pred2, pred2 - y_meas, cost2
                    self.lam = max(self.lam * 0.4, 1e-6)
                    accepted = True
                    break
                self.lam = min(self.lam * 6.0, 1e4)
            if not accepted:
                break
        self.room.T = saved
        self._apply(self.theta)
        self.history.append(dict(theta=self.theta.copy(), rms=float(np.sqrt(cost)),
                                 rms0=float(np.sqrt(base))))
        return self.param_dict(), float(np.sqrt(cost)), float(np.sqrt(base))
