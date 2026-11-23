"""校准演示: 展示孪生模型从错误初值, 经过传感器读数在线校准后逼近真值的收敛过程。

跑 2 天, 记录每次参数校准的 RMSE 与参数估计, 输出 results/calibration.json。
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import layout as L
from .engine import Engine

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def main(days=2, seed=7):
    eng = Engine(days=days, policy='predictive', seed=seed, calibrate=True,
                 calib_every_h=0.75)
    # 演示: 故意用一个明显错误的初始模型(工程师凭经验但偏得较多), 展示校准收敛。
    eng.tw_params0 = dict(h_mix=L.H_MIX * 2.0, h_mix_hot=L.H_MIX_HOT * 0.5,
                          gap_flow=L.GAP_FLOW * 3.0, h_side=L.H_SIDE * 0.6,
                          h_wall=0.04, recirc=L.RECIRC * 2.5,
                          v_base=L.V_CAB_BASE * 1.6, v_kw=L.V_CAB_KW * 0.45)
    from .room import RoomModel
    eng.twin = RoomModel(eng.tp, params=eng.tw_params0)
    eng.twin.T = eng.room.T.copy()
    out = eng.run()
    truth = {k: float(eng.room.p.get(k)) for k in
             ('h_mix', 'h_mix_hot', 'gap_flow', 'h_side', 'h_wall', 'recirc',
              'v_base', 'v_kw')}
    init = {k: float(eng.tw_params0.get(k)) for k in truth}
    hist = []
    for (n, rms, rms0, params) in eng.calib_hist:
        hist.append(dict(t_h=round(n * eng.dt_s / 3600, 1), rms=round(float(rms), 3),
                         rms_before=round(float(rms0), 3),
                         params={k: round(float(v), 4) for k, v in params.items()}))
    final = {k: float(eng.twin.p.get(k)) for k in truth}
    res = dict(days=days, truth=truth, init=init, final=final, history=hist)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'calibration.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('init RMSE', hist[0]['rms'] if hist else None,
          '-> final', hist[-1]['rms'] if hist else None)
    print('truth', {k: round(v, 3) for k, v in truth.items()})
    print('final', {k: round(v, 3) for k, v in final.items()})
    return res


if __name__ == '__main__':
    main()
