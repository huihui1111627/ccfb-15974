"""完整的机房仿真器: 场景 + 网格模型 + 空调系统 + 传感器 + 控制器。

每一步: 控制器(按周期)下发指令 -> 空调执行(延迟/一阶) -> 网格温度推进 ->
传感器采样 -> 记录。
"""
from __future__ import annotations

import numpy as np

from . import layout as L
from .room import Topology, RoomModel
from .crah import Plant
from .scenario import Scenario


class Sensors:
    """温度传感器布局与采样。

    进风: 每排机柜 7 个点(正面, 代表附近机柜);
    热通道: 每排 3 个点;侧通道: 8 个点。真值加高斯测量噪声。
    """

    def __init__(self, topo, rng, noise=L.SENSOR_NOISE):
        self.tp = topo
        self.noise = noise
        self.rng = rng
        cells, kind, rack_ref = [], [], []
        rack_of_cell = {}
        for n, c in enumerate(topo.rack_front):
            rack_of_cell.setdefault(int(c), []).append(n)
        for r_idx in range(len(L.ROW_NAMES)):
            front = topo.rack_front[topo.rack_ri == r_idx]
            picks = np.linspace(0, len(front) - 1, 7).astype(int)
            for p in picks:
                cells.append(int(front[p]))
                kind.append('inlet')
                rack_ref.append(int(np.where(topo.rack_ri == r_idx)[0][p]))
        for j in L.HOT_AISLE_J:
            for i in (8, 22, 36):
                k = topo.cell_at(i, j)
                if k is not None:
                    cells.append(k)
                    kind.append('hot')
                    rack_ref.append(-1)
        for j in (1, 9, 17, 21):
            for i in (1, 44):
                k = topo.cell_at(i, j)
                if k is not None:
                    cells.append(k)
                    kind.append('side')
                    rack_ref.append(-1)
        self.cells = np.asarray(cells, np.int32)
        self.kind = np.asarray(kind)
        self.rack_ref = np.asarray(rack_ref, np.int32)
        self.inlet_mask = self.kind == 'inlet'
        self.inlet_rack = self.rack_ref[self.inlet_mask]

    def sample(self, T):
        return T[self.cells] + self.rng.normal(0, self.noise, len(self.cells))

    def inlet_readings(self, y, with_noise=True):
        """从传感器读数 y 取出进风侧读数及其代表机柜。"""
        return y[self.inlet_mask], self.inlet_rack


def zone_means(topo, T):
    out = {}
    for r_idx, row in enumerate(L.ROW_NAMES):
        cells = topo.row_front_cells[r_idx]
        out[row] = float(T[cells].mean()) + L.TOP_INLET_PENALTY if len(cells) else float('nan')
    for a_idx, j in enumerate(L.HOT_AISLE_J):
        sel = topo.j == j
        out['H%d' % (a_idx + 1)] = float(T[sel].mean())
    return out
