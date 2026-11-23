"""空调联动优化控制。

优化问题(每 10min 求解一次):
    决策: 每台空调 n 的送风温度设定 T_n ∈ [T_MIN,T_MAX] 与风量比 f_n ∈ [f_min,1]。
    目标:  Σ_n P_n(T_n,f_n,T_out)·H  +  w_crit·Σ_rack max(0,T_inlet-T_CRIT)²
           + w_warn·Σ_rack max(0,T_inlet-T_WARN)²  +  切换惩罚
    约束: 所有机柜预测进风温度 ≤ T_CRIT(用大罚函数实现硬约束)。
    求解: 坐标下降 + 层次网格(先粗后细), 用带执行层延迟的数字孪生评估每条候选。
          纯 NumPy, 单周期约 40 次孪生评估, <1s。

为何不用梯度: 温度场对设定的响应有延迟/饱和/切换不连续, 而且故障后可用空调集合
会变; 坐标下降天然适配"逐一重分配"的场景, 且无需推导梯度。
"""
from __future__ import annotations

import numpy as np

from . import layout as L


def crah_power(q, t_sup, t_out, vfrac):
    v = np.maximum(vfrac, 0.0) * L.V_RATED
    fan = L.FAN_KW * (v / L.V_RATED) ** 3
    cop = L.COP0 * (1 + L.COP_TSUP_SLOPE * (t_sup - L.COP_TSUP_REF)) \
        * (1 - L.COP_TOUT_SLOPE * (t_out - L.T_AMB_REF))
    cop = float(np.clip(cop, 1.4, 6.5))
    q = float(np.clip(q, 0.0, L.Q_MAX))
    return fan + q / cop * (1 + L.CHW_LOSS) + q * 0.012


class FixedMaxController:
    """对照基线 A: 全部空调固定最大出力(最低送风温度 + 最大风量)。"""

    def decide(self, obs, load_pred, T_out, fault):
        on = np.ones(L.NC, bool)
        on[fault] = False
        return dict(on=on, tset=np.full(L.NC, L.T_SUP_MIN),
                    vfrac=np.full(L.NC, 1.0))


class LPController:
    """对照基线 B: 按最热点统一调节。

    物理做法: 所有空调同一送风温度, 用最热机柜的进风温度反馈(比例+积分)。
    它能把最热点压到目标以下, 但所有机组都被拉到满足最热点的低送风温度,
    冷量分配不分区域, 因而能耗高于按区域联动的优化控制。
    """

    def __init__(self):
        self.integ = 0.0
        self.prev = None
        self.load_ref = None

    def decide(self, obs, load_pred, T_out, fault):
        err = obs['inlet_max'] - (L.T_TARGET - 0.5)
        self.integ = float(np.clip(self.integ + 0.25 * err, -4.0, 8.0))
        # 统一调节: 统一送风温度, 由最热点反馈决定。加一个简单负载前馈(总量),
        # 但仍对所有机组使用同一设定——这正是它无法按区域精确分配冷量的根源。
        load_kw = float(np.sum(obs['load'])) if obs.get('load') is not None else 0.0
        ff = -0.035 * (load_kw - 260.0)
        t_set = float(np.clip(L.T_TARGET - 2.5 * err - self.integ + ff,
                              L.T_SUP_MIN + 0.5, 23.0))
        on = np.ones(L.NC, bool)
        on[fault] = False
        return dict(on=on, tset=np.full(L.NC, t_set), vfrac=np.full(L.NC, 1.0))


class PredictiveController:
    """联动优化: 预测滚动 + 坐标下降。"""

    COARSE_T = np.array([11.0, 13.5, 16.0, 18.5, 21.0, 24.0])
    COARSE_V = np.array([0.7, 1.0])
    FINE_T = np.array([-1.5, 0.0, 1.5])
    FINE_V = np.array([0.0])

    def __init__(self, room, topo, horizon_min=30, dt_min=5.0, w_energy=1.0,
                 w_crit=80.0, w_warn=5.0, w_switch=0.6):
        self.room = room
        self.tp = topo
        self.horizon = horizon_min
        self.dt_min = dt_min
        self.w_energy = w_energy
        self.w_crit = w_crit
        self.w_warn = w_warn
        self.w_switch = w_switch
        self.last_info = {}
        self.eval_budget = 70
        self.safety_margin = 2.0
        # 自适应裕度: 当孪生近期残差(rmse)已知时, 裕度随模型误差缩放;
        # 误差小则可贴近上限以省电, 误差大则保守。
        self.adaptive_margin = True

    def _margin(self):
        base = self.safety_margin
        r = getattr(self, 'recent_rmse', None)
        if self.adaptive_margin and r is not None:
            # rmse 0.3K 以下 -> 1.0K 裕度; 1.2K 以上 -> 2.5K 裕度
            base = float(np.clip(0.8 + 1.4 * r, 1.0, 2.5))
        return base

    def optimize(self, T0, load_pred, t_out_seq, crah_state, tile_open,
                 fault, q_misc=L.MISC_KW, prev_cmd=None):
        nstep = len(load_pred)
        dt_s = self.dt_min * 60.0
        nc = L.NC
        evals = [0]
        budget = [int(getattr(self, 'eval_budget', 120))]

        def sim(cmd):
            evals[0] += 1
            return self._simulate(T0, load_pred, t_out_seq, crah_state, cmd,
                                  tile_open, fault, q_misc, dt_s, nstep)

        if prev_cmd is not None:
            on0 = prev_cmd['on'].copy()
            ts0 = prev_cmd['tset'].copy()
            vf0 = prev_cmd['vfrac'].copy()
        else:
            on0 = np.ones(nc, bool)
            ts0 = np.full(nc, 21.0)
            vf0 = np.full(nc, 0.85)
        on0[fault] = False
        cur = dict(on=on0, tset=ts0, vfrac=vf0)
        traj, e = sim(cur)
        best_cost = self._cost(traj, e, cur, prev_cmd)
        best = {k: v.copy() for k, v in cur.items()}

        # 阶段 0: 全局预设扫描。先试若干"统一设定", 避免坐标下降从全 21°C 出发时
        # 因需要"协作移动"而卡在局部最优。
        if prev_cmd is None:
            for ts in (11.5, 13.5, 16.0, 18.5, 21.0):
                for vf in (0.85, 1.0):
                    trial = dict(on=np.ones(nc, bool), tset=np.full(nc, ts),
                                 vfrac=np.full(nc, vf))
                    trial['on'][fault] = False
                    tr, ee = sim(trial)
                    c = self._cost(tr, ee, trial, None)
                    if c < best_cost - 1e-4:
                        best_cost, best = c, trial

        # 阶段 1: 粗网格坐标下降, 最多 2 轮
        for rnd in range(2):
            improved = False
            for k in range(nc):
                if fault[k] or evals[0] >= budget[0]:
                    continue
                base_cost = best_cost
                best_ts, best_vf = best['tset'][k], best['vfrac'][k]
                best_on = bool(best['on'][k])
                for ts in self.COARSE_T:
                    for vf in self.COARSE_V:
                        trial = {kk: vv.copy() for kk, vv in best.items()}
                        trial['on'][k] = True
                        trial['tset'][k] = ts
                        trial['vfrac'][k] = vf
                        tr, ee = sim(trial)
                        c = self._cost(tr, ee, trial, prev_cmd)
                        if c < base_cost - 1e-4:
                            base_cost, best_ts, best_vf, best_on = c, ts, vf, True
                if best['on'][k]:
                    trial = {kk: vv.copy() for kk, vv in best.items()}
                    trial['on'][k] = False
                    tr, ee = sim(trial)
                    c = self._cost(tr, ee, trial, prev_cmd)
                    if c < base_cost - 1e-4:
                        base_cost, best_on = c, False
                if base_cost < best_cost - 1e-4:
                    best['on'][k] = best_on
                    if best_on:
                        best['tset'][k] = best_ts
                        best['vfrac'][k] = best_vf
                    best_cost = base_cost
                    improved = True
            if not improved:
                break

        # 阶段 2: 局部细化
        for k in range(nc):
            if fault[k] or not best['on'][k] or evals[0] >= budget[0]:
                continue
            ts0 = best['tset'][k]
            vf0 = best['vfrac'][k]
            for dt_ in self.FINE_T:
                for dv in self.FINE_V:
                    ts = float(np.clip(ts0 + dt_, L.T_SUP_MIN + 1.0, L.T_SUP_MAX))
                    vf = float(np.clip(vf0 + dv, L.V_MIN / L.V_RATED, 1.0))
                    trial = {kk: vv.copy() for kk, vv in best.items()}
                    trial['tset'][k] = ts
                    trial['vfrac'][k] = vf
                    tr, ee = sim(trial)
                    c = self._cost(tr, ee, trial, prev_cmd)
                    if c < best_cost - 1e-5:
                        best, best_cost = trial, c

        traj, e = sim(best)
        # 能力受限保护: 若即便最优指令下预测仍超限, 说明系统冷量/风量不足,
        # 应把所有可用机组开到最大, 至少把温度压到能力极限。
        if float(traj.max()) > L.T_CRIT - 0.3:
            maxcmd = dict(on=np.ones(nc, bool),
                          tset=np.full(nc, L.T_SUP_MIN),
                          vfrac=np.ones(nc))
            maxcmd['on'][fault] = False
            tmax, emax = sim(maxcmd)
            if float(tmax.max()) <= float(traj.max()) + 0.05:
                best, traj, e = maxcmd, tmax, emax
                best_cost = self._cost(traj, e, best, prev_cmd)
        best['pred_inlet'] = traj
        best['pred_max'] = float(traj.max())
        best['cost'] = best_cost
        best['evals'] = evals[0]
        best['energy_kw'] = e
        self.last_info = dict(evals=evals[0], cost=best_cost, pred_max=best['pred_max'])
        return best

    def predict_fault_impact(self, T0, load_pred, t_out_seq, crah_state, tile_open,
                             fault, cmd, q_misc=L.MISC_KW):
        """故障后的超温时间预测。

        在故障参量 fault 下, 按给定指令 cmd(通常为故障后重新分配的最优指令)推进,
        返回 (首个超温时刻_min 或 None, 峰值, 轨迹)。若整段预测都不超温则返回 None,
        表示"在当前负载预测下不会超温"。
        """
        nstep = len(load_pred)
        dt_s = self.dt_min * 60.0
        traj, e = self._simulate(T0, load_pred, t_out_seq, crah_state, cmd,
                                 tile_open, fault, q_misc, dt_s, nstep)
        over = traj > L.T_CRIT
        if over.any():
            first = int(np.argmax(over.any(axis=1)))
            lead = (first + 1) * self.dt_min
        else:
            lead = None
        return dict(lead_min=lead, peak=float(traj.max()), traj=traj)

    # ------------------------------------------------------------ 仿真内核
    def _simulate(self, T0, load_pred, t_out_seq, crah_state, cmd, tile_open,
                  fault, q_misc, dt_s, nstep):
        from .forecast import _StatefulCRAH
        nc = L.NC
        saved_T = self.room.T.copy()
        saved_Tret = self.room.T_ret.copy()
        us = []
        for k in range(nc):
            u = _StatefulCRAH(k, dt_s)
            s = crah_state[k]
            u.on, u.v, u.tset, u.wait = s['on'], s['v'], s['tset'], s['wait']
            u.t_ret, u.t_sup, u.q = s['t_ret'], s['t_sup'], s['q']
            u.vfrac = s['v'] / L.V_RATED
            us.append(u)
        self.room.T = T0.copy()
        traj = np.empty((nstep, self.tp.nrack), np.float32)
        power = np.zeros(nstep)
        for n in range(nstep):
            tsup = np.zeros(nc)
            von = np.zeros(nc)
            onv = np.zeros(nc, bool)
            ret_all = self.room._predict_return(self.room.T)
            for k, u in enumerate(us):
                if fault[k]:
                    u.on = False
                    u.v = 0.0
                    continue
                u.set_command(bool(cmd['on'][k]), float(cmd['tset'][k]),
                              float(cmd['vfrac'][k]))
                u.advance(float(ret_all[k]))
                tsup[k], von[k], onv[k] = u.t_sup, u.v, u.on
                if u.v > 1e-6:
                    power[n] += crah_power(u.q, u.t_sup, t_out_seq[n], u.vfrac)
            self.room.step(dt_s, load_pred[n], tsup, von, onv, tile_open,
                           float(t_out_seq[n]), q_misc=q_misc, iters=18, tol=1e-3)
            traj[n] = self.room.rack_inlet()
        self.room.T = saved_T
        self.room.T_ret = saved_Tret
        return traj, float(power.mean())

    # ------------------------------------------------------------ 目标函数
    def _cost(self, traj, energy_kw, cmd, prev_cmd):
        # 孪生模型有残余误差, 用安全裕度把预测温度抬高后再判定超限, 避免"预测刚好
        # 不超、实际却超"。
        margin = self._margin()
        t_crit_eff = L.T_CRIT - margin
        t_warn_eff = L.T_WARN - margin
        over_crit = np.maximum(traj - t_crit_eff, 0.0)
        over_warn = np.maximum(traj - t_warn_eff, 0.0)
        wstep = np.linspace(0.6, 1.4, len(traj))[:, None]
        pen = self.w_crit * float((wstep * over_crit ** 2).sum()) \
            + self.w_warn * float((wstep * over_warn ** 2).sum())
        e = self.w_energy * energy_kw * (self.horizon / 60.0)
        sw = 0.0
        if prev_cmd is not None:
            sw = self.w_switch * float(np.abs(cmd['tset'] - prev_cmd['tset']).sum()
                                       + 8.0 * np.sum(cmd['on'] != prev_cmd['on']))
        return pen + e + sw
