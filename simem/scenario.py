"""机房负载与室外气象场景。

负载: 每个机柜有基线功耗,乘 24h 日曲线和机柜级随机波动(平滑噪声),再叠加
训练服务器(白天持续高功耗,夜间有批处理波峰)、高密度推理机柜(白天波峰)。
室外: 7 月正弦日变化 + 数日热浪,逐分钟采样。
"""
from __future__ import annotations

import numpy as np

from . import layout as L

HOURS_PER_DAY = 24


def _smooth_noise(rng, n_days, sigma, corr_min=45.0, dt_min=1.0):
    """生成缓慢变化的随机序列(AR(1)),单位 K/kW 的相对波动。"""
    n = int(n_days * 24 * 60 / dt_min) + 2
    a = np.exp(-dt_min / corr_min)
    e = rng.normal(0, sigma * np.sqrt(1 - a * a), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for k in range(1, n):
        x[k] = a * x[k - 1] + e[k]
    return x


class Workload:
    """每天逐分钟生成所有机柜功率 (kW)。

    返回的数组形状 (n_minutes, nrack)。n_minutes 可跨多天。
    日曲线在白天(9-18 点)有业务高峰;训练集群在夜间跑批(23-5 点)。
    """

    def __init__(self, topo, seed=7):
        self.tp = topo
        self.rng = np.random.default_rng(seed)
        n = topo.nrack
        self.base = np.full(n, L.BASE_KW)
        lab = topo.rack_label
        dense = set(L.DENSE_RACKS)
        train = set(L.TRAINING_RACKS)
        for k, (row, i) in enumerate(lab):
            if (row, i) in train:
                self.base[k] = L.TRAIN_KW
            elif (row, i) in dense:
                self.base[k] = L.DENSE_BASE_KW
        self.base *= self.rng.uniform(0.9, 1.1, n)
        self.train_mask = np.array([(r, i) in train for r, i in lab])
        self.dense_mask = np.array([(r, i) in dense for r, i in lab])
        self.daily = L.DAY_KW if hasattr(L, 'DAY_KW') else None

    def hourly_curve(self, hours):
        """房间整体负载系数(业务曲线)。"""
        h = np.asarray(hours) % 24
        # 上午快速爬升, 下午平台, 夜间低谷
        return 0.62 + 0.30 * np.exp(-((h - 14.5) ** 2) / (2 * 4.2 ** 2)) \
                   + 0.08 * np.exp(-((h - 21.0) ** 2) / (2 * 2.0 ** 2))

    def train_curve(self, hours):
        """训练服务器夜间批处理曲线(23-5 点满负荷,白天 0.55)。"""
        h = np.asarray(hours) % 24
        night = np.exp(-((h - 3.0) ** 2) / (2 * 2.4 ** 2))
        return 0.55 + 0.45 * night

    def generate(self, seconds, dt_s, t0_seconds=0.0, heat_event=None):
        """生成 seconds 时长、步长 dt_s 的负载, 返回 (nstep, nrack) kW。"""
        nstep = int(round(seconds / dt_s))
        tsec = t0_seconds + np.arange(nstep) * dt_s
        hours = (tsec / 3600.0) % 24
        days = (tsec / 86400.0)
        curve = self.hourly_curve(hours)
        tr_curve = self.train_curve(hours)
        # 每机柜平滑波动(几分钟尺度)
        noise = np.empty((nstep, self.tp.nrack))
        for k in range(self.tp.nrack):
            nz = _smooth_noise(self.rng, seconds / 86400.0,
                               sigma=0.06, corr_min=35.0, dt_min=dt_s / 60)
            noise[:, k] = nz[:nstep]
        kw = self.base[None, :] * curve[:, None] * (1.0 + noise)
        # 训练服务器: 夜间批处理 + 白天持续
        kw[:, self.train_mask] *= tr_curve[:, None] / curve[:, None] * curve[:, None]
        kw[:, self.train_mask] = self.base[self.train_mask][None, :] * tr_curve[:, None] \
            * (1.0 + noise[:, self.train_mask])
        # 高密度推理: 白天波峰更强
        kw[:, self.dense_mask] *= (0.85 + 0.35 * curve[:, None] / max(curve.max(), 1e-6))
        if heat_event is not None:
            # 增加一次训练任务洪峰(指定时间窗内特定排功率抬升)
            t_lo, t_hi, rows, frac = heat_event
            rows = [self.tp.row_names.index(r) for r in rows]
            mask = (tsec >= t_lo) & (tsec <= t_hi)
            rowmask = np.isin(self.tp.rack_ri, rows)
            kw[np.ix_(mask, rowmask)] *= (1.0 + frac)
        kw = np.clip(kw, 0.15, 90.0)
        return kw


class Weather:
    """7 月室外气象。日变化用正弦(最低 5~6 点, 最高 15~16 点)。"""

    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        self.day_anom = self.rng.normal(0, 0.8, L.JULY_DAYS + 5)
        for d in range(L.HEAT_WAVE_DAYS[0], L.HEAT_WAVE_DAYS[1] + 1):
            self.day_anom[d] += 4.5

    def temperature(self, seconds):
        sec = np.asarray(seconds, float)
        day = sec / 86400.0
        hour = (sec / 3600.0) % 24
        di = np.clip(day.astype(int), 0, L.JULY_DAYS + 4)
        mean = L.WX_MEAN + self.day_anom[di]
        amp = L.WX_AMP + 0.15 * self.day_anom[di]
        shape = -np.cos((hour - 5.0) / 24.0 * 2 * np.pi) * 0.5 + 0.5
        t = mean - amp * 0.55 + amp * 1.1 * shape
        # 短时扰动
        t += 0.6 * np.sin(sec / 3600.0 * 1.7) * np.cos(sec / 3600.0 * 0.37)
        return t


class Scenario:
    """一个月完整场景: 时间轴、室外温度、机柜负载、空调故障时刻。"""

    def __init__(self, topo, seed=7, days=L.JULY_DAYS):
        self.tp = topo
        self.days = days
        self.dt_min = 1.0
        self.dt_s = self.dt_min * 60.0
        self.nstep = int(days * 24 * 60 / self.dt_min)
        self.seconds = np.arange(self.nstep) * self.dt_s
        self.hours = (self.seconds / 3600.0) % 24
        self.weather = Weather(seed + 3)
        self.t_out = self.weather.temperature(self.seconds)
        self.wl = Workload(topo, seed)
        # 负载洪峰: 第 12 天 9:00-13:00, R3/R6/R8 训练集群上量 40%
        lo = 12 * 86400 + 9 * 3600
        hi = 12 * 86400 + 13 * 3600
        self.kw = self.wl.generate(days * 86400, self.dt_s, 0.0,
                                   heat_event=(lo, hi, ['R3', 'R6', 'R8'], 0.45))
        self.faults = []   # (start_seconds, duration_seconds, crah_name)

    def add_fault(self, day, hour, dur_hours, name):
        self.faults.append((day * 86400 + hour * 3600, dur_hours * 3600, name))

    def fault_at(self, t_s, name):
        for (s, d, n) in self.faults:
            if n == name and s <= t_s < s + d:
                return True
        return False
