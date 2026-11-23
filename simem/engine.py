"""仿真编排: 一个月闭环控制 vs 3 种对照策略。"""
from __future__ import annotations

import numpy as np

from . import layout as L
from .room import Topology, RoomModel
from .crah import Plant
from .scenario import Scenario
from .sim import Sensors, zone_means
from .forecast import LoadForecaster, HotspotAlert
from .control import FixedMaxController, LPController, PredictiveController


class Engine:
    def __init__(self, days=L.JULY_DAYS, seed=7, calib_every_h=1.5,
                 horizon_min=30, dt_min=2.0, policy='predictive', calibrate=True):
        self.tp = Topology()
        self.sc = Scenario(self.tp, seed=seed, days=days)
        # truth: 真实机房(参数未知给控制器), 产生传感器读数
        self.room = RoomModel(self.tp)
        self.plant = Plant(self.sc.dt_s)
        self.rng = np.random.default_rng(seed + 100)
        self.sensors = Sensors(self.tp, self.rng)
        self.dt_s = self.sc.dt_s
        self.calib_every = int(calib_every_h * 3600 / self.dt_s)
        self.horizon_min = horizon_min
        self.policy = policy
        self.nstep = self.sc.nstep
        self.ctrl_period = int(getattr(L, 'CTRL_EVERY_MIN', 10))
        self.ctrl_period_steps = int(self.ctrl_period * 60 / self.dt_s)
        self.pred_steps = int(horizon_min / 5.0)
        self.calibrate = calibrate
        # twin: 数字孪生(控制器用的模型), 初始参数偏离真值(模拟工程师凭经验建的
        # 粗略模型), 通过传感器读数在线校准逐步逼近。
        self.tw_params0 = dict(h_mix=L.H_MIX * 1.25, h_mix_hot=L.H_MIX_HOT * 0.85,
                               gap_flow=L.GAP_FLOW * 1.35, h_side=L.H_SIDE * 0.85,
                               h_wall=0.024, recirc=L.RECIRC * 0.55,
                               v_base=L.V_CAB_BASE * 1.15, v_kw=L.V_CAB_KW * 0.72)
        self.twin = RoomModel(self.tp, params=self.tw_params0)
        self.twin.T = self.room.T.copy()
        self.calib_hist = []
        self.calib_rmse = []

    # =============================================================
    def run(self, callbacks=None):
        tp = self.tp
        sc = self.sc
        room = self.room
        plant = self.plant
        # ---- 控制器
        if self.policy == 'fixed':
            ctrl = FixedMaxController()
        elif self.policy == 'lp':
            ctrl = LPController()
        else:
            # 控制器使用数字孪生(校准后的模型)做预测, 而非真实机房状态。
            ctrl = PredictiveController(self.twin, tp, horizon_min=self.horizon_min,
                                        dt_min=5.0)
        self.ctrl = ctrl
        lf = LoadForecaster(tp, tp.nrack)
        alerter = HotspotAlert(tp)

        # 记录
        N = self.nstep
        T_field = np.empty((N, tp.n), np.float32)
        inlet = np.empty((N, tp.nrack), np.float32)
        inlet_clean = np.empty((N, tp.nrack), np.float32)
        energy_seq = np.empty(N)
        cmd_log = []
        alerts_log = []
        conv = dict(alerts=0, over_steps=0, over_rack_steps=0, over_minutes=0.0,
                    peak=0.0, energy=0.0, starts=0)
        cmd = None
        prev_cmd = None
        over_t = []
        fault_seen = np.zeros(L.NC, bool)
        fault_log = []
        plant_units = plant.units
        t0 = 0
        for n in range(N):
            t = n * self.dt_s
            t_out = float(sc.t_out[n])
            kw = sc.kw[n]
            hour = (t / 3600.0) % 24
            lf.update(hour, kw)
            # 故障
            for k, u in enumerate(plant_units):
                u.fault = any(nm == u.name and s <= t < s + d for (s, d, nm) in sc.faults)
            fault = np.array([u.fault for u in plant_units], bool)
            plant.t_out = t_out

            # ---- 控制周期
            if n % self.ctrl_period_steps == 0:
                on, v, tsup, q, _ = plant.cols()
                T_meas = self.sensors.sample(room.T)
                inlet_rack = room.rack_inlet()
                inlet_max = float(inlet_rack.max())
                obs = dict(T=room.T, T_meas=T_meas, inlet_max=inlet_max,
                           inlet_rack=inlet_rack, load=kw,
                           sensors=self.sensors, crah_state=[
                               dict(on=u.on, v=u.v, tset=u.tset, wait=u.wait,
                                    t_ret=u.t_ret, t_sup=u.t_sup, q=u.q)
                               for u in plant_units])
                if self.policy == 'predictive':
                    pdt = int(5 * 60 / self.dt_s)   # 预测步对应的仿真步数
                    idx_seq = np.minimum(n + np.arange(self.pred_steps) * pdt, self.nstep - 1)
                    hf = (t + np.arange(self.pred_steps) * 300.0) / 3600.0 % 24
                    load_pred = lf.forecast(hf, kw, dt_min=5.0)
                    t_out_seq = sc.t_out[idx_seq]
                    tile_open = self._tile_open(kw)
                    cmd = ctrl.optimize(self.twin.T, load_pred, t_out_seq,
                                        obs['crah_state'], tile_open, fault,
                                        prev_cmd=prev_cmd)
                    # 热点预测与告警
                    al = alerter.detect(cmd['pred_inlet'], 0.0, self.dt_s / 60.0)
                    alerts_log.append((t, al, float(cmd['pred_max'])))
                    if cmd['pred_max'] > L.T_CRIT:
                        conv['alerts'] += 1
                    # ---- 故障: 记录故障后超温时间预测(故障后首次控制周期)
                    new_fault = fault & ~fault_seen
                    if new_fault.any():
                        H = 36  # 180 分钟故障影响预测窗口(36×5min)
                        pdt = int(5 * 60 / self.dt_s)
                        idx2 = np.minimum(n + np.arange(H) * pdt, self.nstep - 1)
                        # 故障影响预测用"理想负载预测"(即实际未来负载), 以单独衡量
                        # 模型的物理预测误差, 不含负载预测误差。
                        lp2 = sc.kw[idx2]
                        imp = ctrl.predict_fault_impact(self.twin.T, lp2, sc.t_out[idx2],
                                                        obs['crah_state'], tile_open,
                                                        fault, cmd)
                        fault_log.append(dict(
                            t=float(t), name=[plant_units[k].name
                                              for k in np.where(new_fault)[0]],
                            predicted_lead_min=imp['lead_min'],
                            predicted_peak=round(imp['peak'], 2),
                        ))
                    fault_seen = fault.copy()
                else:
                    cmd = ctrl.decide(obs, None, t_out, fault)
                prev_cmd = {k: np.asarray(vv).copy() for k, vv in cmd.items()
                            if k in ('on', 'tset', 'vfrac')}
                cmd_log.append((t, {k: np.asarray(vv).copy().tolist() if np.ndim(vv) else vv
                                    for k, vv in cmd.items() if k in ('on', 'tset', 'vfrac')}))
            # ---- 下发指令(每步都下发, 执行层有延迟/一阶)
            plant.command(cmd['on'], cmd['tset'], cmd['vfrac'])
            ret = room.T_ret
            plant.advance(ret)
            on, v, tsup, q, fault_now = plant.cols()
            tile_open = self._tile_open(kw)
            room.step(self.dt_s, kw, tsup, v, on, tile_open, t_out, q_misc=L.MISC_KW)
            # 数字孪生用相同输入推进(控制器维护自己的状态估计)
            if self.policy == 'predictive':
                self.twin.step(self.dt_s, kw, tsup, v, on, tile_open, t_out,
                               q_misc=L.MISC_KW)
                # 每个控制周期: 用传感器读数校正孪生状态(观测更新)
                if self.calibrate and n % self.ctrl_period_steps == 0:
                    self._correct_state()
                # 每隔较长时间: 校准模型参数
                if self.calibrate and n % self.calib_every == 0 and n > 0:
                    self._calibrate(n, kw, tsup, v, on, tile_open, t_out)

            # ---- 记录
            inl = room.rack_inlet()
            T_field[n] = room.T
            inlet[n] = inl
            inlet_clean[n] = inl  # 真值, 后续可加噪声
            e_kw = 0.0
            for u in plant_units:
                if u.fault or u.v < 1e-6:
                    continue
                e_kw += L.FAN_KW * (u.v / L.V_RATED) ** 3 + u.q / u.cop() * (1 + L.CHW_LOSS) + u.q * 0.012
            energy_seq[n] = e_kw
            over = inl > L.T_CRIT
            if over.any():
                conv['over_steps'] += 1
                conv['over_rack_steps'] += int(over.sum())
                over_t.append(t)
            conv['peak'] = max(conv['peak'], float(inl.max()))
            if callbacks and n % max(1, N // 200) == 0:
                callbacks(n, N, room, plant)
        conv['energy'] = plant.total_kwh()
        conv['starts'] = sum(u.starts for u in plant_units)
        conv['cmd_log'] = cmd_log
        conv['alerts'] = alerts_log
        # 超温时长: 按机柜统计
        over_mask = inlet > L.T_CRIT
        conv['over_rack_min'] = float(over_mask.sum() * self.dt_s / 60.0)
        conv['over_minutes_any'] = float(over_mask.any(1).sum() * self.dt_s / 60.0)
        conv['over_rack_count'] = int(over_mask.any(0).sum())
        conv['inlet'] = inlet
        conv['T_field'] = T_field
        conv['energy_seq'] = energy_seq
        conv['t'] = np.arange(N) * self.dt_s
        conv['over_mask'] = over_mask
        conv['fault_log'] = fault_log
        conv['twin_params'] = self.twin.p.copy()
        conv['calib_rmse'] = list(self.calib_rmse)
        return conv

    # ------------------------------------------------------------
    def _correct_state(self):
        """用传感器读数做状态观测更新(每个控制周期都做, 便宜)。

        孪生在两次测量间会因参数误差漂移; 把传感器位置的温度拉向实测值, 相当于
        一个简单的卡尔曼观测步, 让控制器的状态估计始终贴近真实机房。
        """
        y = self.sensors.sample(self.room.T)
        T = self.twin.T
        gain = 0.6
        T[self.sensors.cells] += gain * (y - T[self.sensors.cells])

    def _calibrate(self, n, kw, tsup, v, on, tile_open, t_out):
        """用进风侧传感器读数校准孪生模型参数。"""
        from .calib import Calibrator
        if not hasattr(self, '_calibrator'):
            self._calibrator = Calibrator(self.twin, self.sensors)
        y = self.sensors.sample(self.room.T)
        inlet_y, rack_ref = self.sensors.inlet_readings(y)
        args = (kw, tsup, v, on, tile_open, t_out, L.MISC_KW)
        try:
            _, rms, rms0 = self._calibrator.calibrate(self.twin.T.copy(), self.dt_s,
                                                      args, inlet_y, iters=2, warm=5)
            self.calib_rmse.append(round(float(rms), 3))
            self.calib_hist.append((n, rms, rms0, self._calibrator.param_dict()))
            # 把近期残差反馈给控制器, 用于自适应安全裕度
            if getattr(self, 'ctrl', None) is not None:
                self.ctrl.recent_rmse = float(rms)
        except Exception:
            pass

    # ------------------------------------------------------------
    def _tile_open(self, kw):
        """地板出风口开度: 与机柜功率相关(高功率机柜风口开大), 出厂设置不随控制改变。"""
        tp = self.tp
        ratio = kw / max(float(kw.mean()), 1e-6)
        return np.clip(0.55 + 0.65 * ratio, 0.35, 1.7)
