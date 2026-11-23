"""生成新服务器上架建议报告(results/placement.json), 供页面读取。

在"典型高负载时刻"(第 12 天 14:00 的工况)评估每个空位的上架影响。
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import layout as L
from .room import Topology, RoomModel
from .crah import Plant
from .scenario import Scenario
from .placement import evaluate_new_server

HERE = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(HERE, 'results')


def build(day=12, hour=14.0, seed=7, new_kw=None):
    new_kw = new_kw or L.TRAIN_KW
    tp = Topology()
    sc = Scenario(tp, days=day + 1, seed=seed)
    room = RoomModel(tp)
    plant = Plant(300.0)
    n = int((day * 24 + hour) * 60 / 5)   # 5min steps index
    n = min(n, sc.nstep - 1)
    # 用合理的固定空调设定推到稳态
    on = np.ones(L.NC, bool)
    for _ in range(200):
        plant.t_out = float(sc.t_out[n])
        plant.command(on, np.full(L.NC, 14.0), np.ones(L.NC))
        plant.advance(room.T_ret)
        o, v, tsup, q, f = plant.cols()
        tile = np.clip(0.55 + 0.65 * sc.kw[n] / sc.kw[n].mean(), 0.35, 1.7)
        room.step(300.0, sc.kw[n], tsup, v, o, tile, float(sc.t_out[n]), q_misc=L.MISC_KW)
    o, v, tsup, q, f = plant.cols()
    tile = np.clip(0.55 + 0.65 * sc.kw[n] / sc.kw[n].mean(), 0.35, 1.7)
    res, base_max = evaluate_new_server(room, tp, sc.kw[n], tsup, v, o, tile,
                                        float(sc.t_out[n]), new_kw=new_kw)
    out = dict(day=day, hour=hour, new_kw=new_kw,
               base_peak=round(base_max, 2),
               t_out=round(float(sc.t_out[n]), 1),
               load_kw=round(float(sc.kw[n].sum()), 1),
               gaps=L.GAPS, results=res,
               rack_labels=[[r, int(i)] for (r, i) in tp.rack_label])
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'placement.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False)
    return out


if __name__ == '__main__':
    o = build()
    print(f"base_peak={o['base_peak']} load={o['load_kw']} t_out={o['t_out']}")
    for d in o['results']:
        print(d)
