"""负载预测、温度预测与热点预警。"""
from __future__ import annotations

import numpy as np

from . import layout as L


class LoadForecaster:
    """在线学习日曲线与机柜基线,输出未来负载预测。

    只用控制器能看到的历史:
      - 房间总功率的 20% 分位数作为"夜间基准水平" ref(慢速更新);
      - 逐小时维护 curve[h] = 该小时总功率/ref 的滑动平均;
      - 逐机柜维护其占总功率的份额(base),用于把总量分解到机柜。
    预测 = ref × curve[h] × 机柜份额, 再向当前实测水平做 30min 回归。
    """

    def __init__(self, topo, nrack):
        self.tp = topo
        self.nrack = nrack
        self.curve = np.ones(24)
        self.curve_n = np.zeros(24)
        self.share = np.full(nrack, 1.0 / nrack)
        self.base_n = 0
        self.learned = 0
        self.hist_total = []
        self.ref = None

    def update(self, hour, rack_kw, keep_hist=12 * 60):
        h = int(hour) % 24
        total = float(rack_kw.sum())
        self.hist_total.append(total)
        if len(self.hist_total) > keep_hist:
            self.hist_total.pop(0)
        ref = float(np.percentile(self.hist_total, 20))
        self.ref = ref
        ratio = total / max(ref, 1e-9)
        self.curve_n[h] += 1
        a = 1.0 / self.curve_n[h]
        self.curve[h] = (1 - a) * self.curve[h] + a * ratio
        b = min(self.base_n + 1, 4000)
        self.base_n = b
        self.share = (1 - 1 / b) * self.share + (1 / b) * (rack_kw / max(total, 1e-9))
        self.learned += 1

    def ready(self):
        return self.learned >= 6 and self.ref is not None

    def forecast(self, hours_future, rack_kw_now, dt_min=1.0, blend=0.15):
        """预测未来每步的机柜负载 (nstep, nrack)。"""
        hf = np.mod(np.asarray(hours_future), 24).astype(int)
        ref = self.ref if self.ref is not None else float(rack_kw_now.sum())
        cf = self.curve[hf]
        share = self.share / max(float(self.share.sum()), 1e-9)
        pred_hist = (ref * cf)[:, None] * share[None, :]
        nst = len(hours_future)
        decay = np.exp(-np.arange(nst) * dt_min / 30.0)[:, None]
        cur_dev = rack_kw_now[None, :] - pred_hist[0:1, :]
        pred = pred_hist + decay * cur_dev
        return np.clip(pred, 0.1, 90.0)


class TemperaturePredictor:
    """基于数字孪生的温度场滚动预测。

    预测器内部维护一份执行层状态(各空调当前实际送风温度/风量/启停/剩余延迟),
    从给出的实测状态开始,沿预测负载序列推进。可选用"理想的即时响应"预测器
    (不考虑空调动态)做对照。
    """

    def __init__(self, room, topo):
        self.room = room
        self.tp = topo

    def predict(self, T0, dt_s, rack_kw_seq, crah_state, crah_cmd_seq,
                tile_open, t_out_seq, q_misc=L.MISC_KW, iters=25, tol=3e-4):
        """返回 (nstep, nrack) 进风温度预测。

        T0: 当前温度场副本。
        crah_state: 当前执行层状态, list of dict(on, v, tset, wait, q, t_sup)。
        crah_cmd_seq: (nstep, NC) 每步的 (on, tset, vfrac) 指令数组, 形状 (nstep,NC,3)。
        t_out_seq: (nstep,) 室外温度。
        """
        from .crah import CRAH
        nc = L.NC
        dt = dt_s
        T = T0.copy()
        saved_T = self.room.T.copy()
        saved_Tret = self.room.T_ret.copy()
        outlet = np.zeros((len(rack_kw_seq), self.tp.nrack), np.float32)
        # 执行层本地副本
        plants = []
        for k in range(nc):
            u = _StatefulCRAH(k, dt)
            s = crah_state[k]
            u.on = s['on']
            u.v = s['v']
            u.tset = s['tset']
            u.wait = s['wait']
            u.t_ret = s['t_ret']
            u.t_sup = s['t_sup']
            u.q = s['q']
            plants.append(u)
        for n in range(len(rack_kw_seq)):
            ret = self.room._predict_return(T)
            tsup, von, on = np.zeros(nc), np.zeros(nc), np.zeros(nc, bool)
            for k, u in enumerate(plants):
                onk, tsetk, vf = crah_cmd_seq[n, k]
                u.set_command(bool(onk), float(tsetk), float(vf))
                u.advance(float(ret[k]))
                tsup[k], von[k], on[k] = u.t_sup, u.v, u.on
            T = self.room.step(dt, rack_kw_seq[n], tsup, von, on, tile_open,
                               float(t_out_seq[n]), q_misc=q_misc,
                               iters=iters, tol=tol)
            outlet[n] = self.room.rack_inlet(T)
        self.room.T = saved_T
        self.room.T_ret = saved_Tret
        return outlet


class _StatefulCRAH:
    """预测器内部的空调执行状态(轻量副本)。"""

    def __init__(self, idx, dt_s):
        self.idx = idx
        self.dt = dt_s
        self.on = False
        self.cmd_on = False
        self.wait = 0.0
        self.tset = 22.0
        self.cmd_tset = 22.0
        self.v = 0.0
        self.cmd_vfrac = 0.3
        self.vfrac = 0.0
        self.q = 0.0
        self.t_sup = 22.0
        self.t_ret = 28.0

    def set_command(self, on, tset, vfrac):
        if on != self.cmd_on:
            self.wait = L.DELAY_START if on else L.DELAY_STOP
        self.cmd_on = on
        self.cmd_tset = float(np.clip(tset, L.T_SUP_MIN, L.T_SUP_MAX))
        self.cmd_vfrac = float(np.clip(vfrac, L.V_MIN / L.V_RATED, 1.0))

    def advance(self, t_ret):
        dm = self.dt / 60.0
        self.t_ret = t_ret
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
            q_req = 1.206 * self.v * (self.t_ret - self.tset)
            self.q = float(np.clip(q_req, 0.0, L.Q_MAX))
            self.t_sup = self.t_ret - self.q / (1.206 * self.v)
        else:
            self.q = 0.0
            self.t_sup = self.t_ret
        self.t_sup = float(np.clip(self.t_sup, L.T_SUP_MIN, L.T_SUP_MAX))


class HotspotAlert:
    """从预测轨迹提取热点告警。"""

    def __init__(self, topo, warn=L.T_WARN, crit=L.T_CRIT):
        self.tp = topo
        self.warn = warn
        self.crit = crit

    def detect(self, pred_inlet, t0_min, dt_min):
        """pred_inlet: (nstep, nrack)。返回告警列表。"""
        peak = pred_inlet.max(0)
        alerts = []
        for r in np.where(peak > self.warn)[0]:
            traj = pred_inlet[:, r]
            over = traj > self.crit
            first = int(np.argmax(over)) if over.any() else None
            alerts.append(dict(
                rack=list(self.tp.rack_label[r]), idx=int(r),
                peak=round(float(peak[r]), 2),
                lead_min=round(t0_min + first * dt_min, 1) if first is not None else None,
                t_warn=round(float(t0_min + np.argmax(traj > self.warn) * dt_min), 1),
                crit=bool(over.any()),
            ))
        alerts.sort(key=lambda a: -a['peak'])
        return alerts
