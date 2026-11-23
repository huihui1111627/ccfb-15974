"""精密空调(CRAH)模型: 一阶响应、启停延迟、制冷量上限、能耗与 COP。"""
from __future__ import annotations

import numpy as np

from . import layout as L


class CRAH:
    """单台精密空调。

    控制量: 送风温度设定 t_set 与风机风量 v(0 表示停机)。
    执行层一阶滞后: 送风温度、风量向设定值指数逼近(时间常数 TAU, 单位 分钟)。
    启停有固定延迟: 指令开机后需 DELAY_START 分钟才建立风量/制冷。
    制冷量由风量能输送的最大温差决定, 并受 Q_MAX 限制。
    """

    def __init__(self, idx, dt_s):
        d = L.CRAHS[idx]
        self.idx = idx
        self.name = d['name']
        self.x, self.y = d['x'], d['y']
        self.dt = dt_s
        self.cmd_on = False
        self.cmd_tset = 22.0
        self.cmd_vfrac = L.V_MIN / L.V_RATED
        self.on = False
        self.tset = 22.0
        self.v = 0.0
        self.vfrac = 0.0
        self.q = 0.0
        self.t_sup = 22.0
        self.t_ret = 28.0
        self.fault = False
        self.wait = 0.0
        self.t_out = L.T_AMB_REF
        self.energy_kwh = self.chiller_kwh = self.fan_kwh = 0.0
        self.q_delivered = 0.0
        self.starts = 0
        self.hours = 0.0

    # ------------------------------------------------------------ 指令
    def command(self, on, tset, vfrac):
        on = bool(on)
        if on != self.cmd_on and not self.fault:
            self.wait = L.DELAY_START if on else L.DELAY_STOP
            if on:
                self.starts += 1
        self.cmd_on = on
        self.cmd_tset = float(np.clip(tset, L.T_SUP_MIN, L.T_SUP_MAX))
        self.cmd_vfrac = float(np.clip(vfrac, L.V_MIN / L.V_RATED, 1.0))

    # ------------------------------------------------------------ 执行
    def advance(self, t_ret):
        dt = self.dt
        self.t_ret = float(t_ret)
        dm = dt / 60.0
        if self.fault:
            self.on = False
            self.v = 0.0
            self.vfrac = 0.0
            self.q = 0.0
            self.t_sup = self.t_ret
            self._energy()
            return
        if self.wait > 0:
            self.wait = max(0.0, self.wait - dm)
            if self.wait == 0:
                self.on = self.cmd_on
        else:
            self.on = self.cmd_on
        a_v = 1.0 - np.exp(-dm / L.TAU_V)
        target = self.cmd_vfrac if (self.on or self.wait > 0) else 0.0
        self.vfrac += (target - self.vfrac) * a_v
        self.v = self.vfrac * L.V_RATED
        if self.v < L.V_MIN * 0.6:
            self.v = 0.0
        a_t = 1.0 - np.exp(-dm / L.TAU_TSET)
        self.tset += (self.cmd_tset - self.tset) * a_t
        if self.on and self.v > 1e-6:
            # 送风温度设定决定机组出力: q = rho*cp*V*(T_ret - T_set)
            q_req = 1.206 * self.v * (self.t_ret - self.tset)
            self.q = float(np.clip(q_req, 0.0, L.Q_MAX))
            self.t_sup = self.t_ret - self.q / (1.206 * self.v)
        else:
            self.q = 0.0
            self.t_sup = self.t_ret
        self.t_sup = float(np.clip(self.t_sup, L.T_SUP_MIN, L.T_SUP_MAX))
        self._energy()

    # ------------------------------------------------------------ 能耗
    def cop(self, t_sup=None, t_out=None):
        t_sup = self.t_sup if t_sup is None else t_sup
        t_out = self.t_out if t_out is None else t_out
        cop = L.COP0
        cop *= 1.0 + L.COP_TSUP_SLOPE * (t_sup - L.COP_TSUP_REF)
        cop *= 1.0 - L.COP_TOUT_SLOPE * (t_out - L.T_AMB_REF)
        return float(np.clip(cop, 1.4, 6.5))

    def _energy(self):
        h = self.dt / 3600.0
        self.hours += h
        if self.v < 1e-6 or self.fault:
            # 停机时风机/水泵全停; 开机准备期间仍算最低风机
            if self.wait > 0 and self.cmd_on:
                fan = L.FAN_KW * 0.25
                self.fan_kwh += fan * h
                self.energy_kwh += fan * h
            return
        fan = L.FAN_KW * (self.v / L.V_RATED) ** 3
        cop = self.cop()
        chiller = self.q / cop
        pump = self.q * 0.012
        chiller *= (1.0 + L.CHW_LOSS)
        self.fan_kwh += fan * h
        self.chiller_kwh += (chiller + pump) * h
        self.energy_kwh += (fan + chiller + pump) * h
        self.q_delivered += self.q * h

    # ------------------------------------------------------------ 快照
    def snapshot(self):
        return dict(name=self.name, x=self.x, y=self.y, on=bool(self.on),
                    fault=bool(self.fault), cmd_on=bool(self.cmd_on),
                    cmd_tset=round(self.cmd_tset, 2), tset=round(self.tset, 2),
                    cmd_vfrac=round(self.cmd_vfrac, 3), v=round(self.v, 3),
                    vfrac=round(self.vfrac, 3), q=round(self.q, 2),
                    t_sup=round(self.t_sup, 2), t_ret=round(self.t_ret, 2),
                    energy_kwh=round(self.energy_kwh, 3),
                    cop=round(self.cop(), 2), starts=self.starts)

    def reset_energy(self):
        self.energy_kwh = self.chiller_kwh = self.fan_kwh = 0.0
        self.q_delivered = 0.0
        self.hours = 0.0


class Plant:
    """全部空调组成的冷冻/送风系统。"""

    def __init__(self, dt_s):
        self.dt = dt_s
        self.units = [CRAH(k, dt_s) for k in range(L.NC)]
        self.t_out = L.T_AMB_REF

    def command(self, on, tset, vfrac):
        for u, a, b, c in zip(self.units, on, tset, vfrac):
            u.command(a, b, c)

    def advance(self, t_ret_vec):
        for u, r in zip(self.units, t_ret_vec):
            u.t_out = self.t_out
            u.advance(r)

    def cols(self):
        on = np.array([u.on for u in self.units])
        v = np.array([u.v for u in self.units])
        tsup = np.array([u.t_sup for u in self.units])
        q = np.array([u.q for u in self.units])
        fault = np.array([u.fault for u in self.units])
        return on, v, tsup, q, fault

    def cmd_cols(self):
        on = np.array([u.cmd_on for u in self.units])
        tset = np.array([u.cmd_tset for u in self.units])
        v = np.array([u.cmd_vfrac * L.V_RATED for u in self.units])
        return on, tset, v

    def total_kwh(self):
        return sum(u.energy_kwh for u in self.units)

    def snapshots(self):
        return [u.snapshot() for u in self.units]

    def reset_energy(self):
        for u in self.units:
            u.reset_energy()
