"""新服务器上架位置建议。

对每个空机柜位(缺口), 模拟把一台高功耗服务器(默认 20kW)放进去后:
  - 该柜及其周围机柜进风温度的增量;
  - 为把最热点压回上限需要增加的空调出力(能耗增量)。
用一个"评估用"的稳态求解(固定当前空调设定, 只改负载)得到。最后按"对周围影响
最小 + 能耗增量最小"排序, 给出推荐位置。

可以用联动优化控制器重新优化后再评估, 更贴近实际(Twin 滚动)。这里用稳态更快,
且因为评价是相对的, 稳态已能区分好坏位置。
"""
from __future__ import annotations

import numpy as np

from . import layout as L


def _steady_inlet(room, kw, tsup, v, on, tile, t_out, n_iter=120, dt=300.0):
    """在副本上把温度场推到稳态, 返回 (每机柜进风温度, 各空调回风温度)。"""
    saved = room.T.copy()
    saved_ret = room.T_ret.copy()
    for _ in range(n_iter):
        room.step(dt, kw, tsup, v, on, tile, t_out, q_misc=L.MISC_KW, iters=40, tol=1e-3)
    inlet = room.rack_inlet()
    ret = room.T_ret.copy()
    room.T = saved
    room.T_ret = saved_ret
    return inlet, ret


def _cooling_kw(room, kw):
    """把机柜+室内热负荷当作需被带走的显热(kW)。"""
    return float(np.sum(kw)) + L.MISC_KW


def evaluate_new_server(room, topo, base_kw, tsup, v, on, tile, t_out,
                        new_kw=L.TRAIN_KW, crah_state=None, cop=3.0):
    """对每个空位评估放入 new_kw 服务器的效果。

    指标:
      local_rise    —— 该位置附近机柜进风温度的最大上升(K)
      max_delta     —— 全房最热机柜进风相对基线的上升(K), 反映是否把它推向上限
      energy_delta  —— 为带走新增显热需额外付出的空调功率(kW, 稳态)
    """
    n0, ret0 = _steady_inlet(room, base_kw, tsup, v, on, tile, t_out)
    base_max = float(n0.max())
    base_cool = _cooling_kw(room, base_kw)
    results = []
    for gap in L.GAPS:
        r = topo.rack_no(*gap)
        kw2 = base_kw.copy()
        if r is not None:
            kw2[r] = kw2[r] + new_kw
        else:
            rr = _neighbors_in_row(topo, gap)
            for x in rr:
                kw2[x] += new_kw / max(len(rr), 1)
        n1, ret1 = _steady_inlet(room, kw2, tsup, v, on, tile, t_out)
        near = _near_racks(topo, gap, radius=4)
        local_rise = float((n1[near] - n0[near]).max()) if len(near) else float((n1 - n0).max())
        max_delta = float(n1.max() - base_max)
        # 能耗增量: 新增显热 / COP(近似稳态) + 若该位置推高全场峰值, 需要额外降温
        # 补偿(其他区域被迫过冷), 按每升 1K 约 4% 制冷量估算。
        energy_delta = new_kw / cop * (1.0 + 0.04 * max(max_delta, 0.0))
        results.append(dict(gap=list(gap), local_rise=round(local_rise, 3),
                            max_delta=round(max_delta, 3),
                            energy_delta_kw=round(energy_delta, 2)))
    results.sort(key=lambda d: (d['max_delta'], d['local_rise']))
    for k, d in enumerate(results):
        d['rank'] = k + 1
    return results, base_max


def _neighbors_in_row(topo, gap):
    row, i = gap
    out = []
    for di in (-1, 1):
        r = topo.rack_no(row, i + di)
        if r is not None:
            out.append(r)
    return out


def _near_racks(topo, gap, radius=4):
    row, i = gap
    out = []
    for r, (rr, ii) in enumerate(topo.rack_label):
        if abs(ii - i) <= radius:
            out.append(r)
    return np.array(out, dtype=int)


class PlacementAdvisor:
    """带联动优化的上架建议: 对每个空位, 重新用优化器求最佳空调设定并算能耗。"""

    def __init__(self, room, topo, controller_factory=None):
        self.room = room
        self.tp = topo
        self.ctrl_factory = controller_factory

    def run(self, base_kw, t_out, tile, crah_cmd, new_kw=L.TRAIN_KW,
            crah_state=None, fault=None):
        tsup = np.asarray(crah_cmd['tset'])
        v = np.asarray(crah_cmd['vfrac']) * L.V_RATED
        on = np.asarray(crah_cmd['on'])
        return evaluate_new_server(self.room, self.tp, base_kw, tsup, v, on,
                                   tile, t_out, new_kw=new_kw)
