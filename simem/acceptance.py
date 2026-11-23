"""验收: 跑一个月(含热浪与负载高峰), 对比三种空调运行方式的能耗/超温,
并单独关一台空调验证超温时间的预测误差。

输出 results/summary.json 与 run_*.json 供页面读取。
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import layout as L
from .engine import Engine

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def _count_events(over_mask):
    ev = 0
    for r in range(over_mask.shape[1]):
        col = over_mask[:, r].astype(int)
        ev += int(np.sum(np.diff(np.concatenate([[0], col])) == 1))
    return ev


def _metrics(out, days):
    inl = out['inlet']
    over = inl > L.T_CRIT
    return dict(
        energy_kwh=round(float(out['energy']), 1),
        peak_inlet=round(float(out['peak']), 2),
        over_minutes_any=round(float(out['over_minutes_any']), 1),
        over_rack_minutes=round(float(out['over_rack_min']), 1),
        over_rack_count=int(out['over_rack_count']),
        over_events=int(_count_events(over)),
        starts=int(out['starts']),
        mean_inlet=round(float(inl.mean()), 2),
        days=int(days),
    )


def run_policy(policy, days, seed=7, faults=None, callback=None):
    eng = Engine(days=days, policy=policy, seed=seed)
    if faults:
        for (day, hour, dur, name) in faults:
            eng.sc.add_fault(day, hour, dur, name)
    out = eng.run(callbacks=callback)
    return eng, out


def save_run(tag, eng, out, extra=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    tp = eng.tp
    stride = max(1, int(60 * 60 / eng.dt_s))   # 每 1 小时一帧(控制文件体积)
    frames = out['T_field'][::stride].astype(np.float32)
    times = out['t'][::stride]
    inlet = out['inlet'][::stride].astype(np.float32)
    r = dict(
        tag=tag, days=eng.sc.days,
        times=[round(float(x), 1) for x in times],
        frames=np.round(frames.reshape(len(frames), -1), 1).tolist(),
        nx=int(L.NX), ny=int(L.NY),
        cell_i=tp.i.tolist(), cell_j=tp.j.tolist(), cell_type=tp.type.tolist(),
        inlet=np.round(inlet, 1).tolist(),
        rack_labels=[[r_, int(i)] for (r_, i) in tp.rack_label],
        energy_seq=np.round(out['energy_seq'][::stride], 1).tolist(),
        t_out=np.round(eng.sc.t_out[::stride], 1).tolist(),
        cmd_log=[(round(float(t), 1), c) for (t, c) in out['cmd_log'][::4]],
        alerts=[(round(float(t), 1), a, float(m)) for (t, a, m) in out['alerts']],
        over_mask=out['over_mask'][::stride].astype(np.uint8).tolist(),
        fault_log=out.get('fault_log', []),
        calib_rmse=out.get('calib_rmse', []),
        twin_params=out.get('twin_params', {}),
    )
    if extra:
        r.update(extra)
    with open(os.path.join(RESULT_DIR, f'run_{tag}.json'), 'w') as f:
        json.dump(r, f)
    return r


def fault_diagnostics(eng, out, fault, window_min=240):
    """对比故障后首个超温时刻的实际值与故障发生时的预测值。

    fault: (day, hour, dur_h, name)
    """
    day, hour, dur_h, name = fault
    s0 = day * 86400 + hour * 3600
    dur = dur_h * 3600
    t = out['t']
    inl = out['inlet']
    fw = (t >= s0) & (t < s0 + dur)
    over = (inl > L.T_CRIT)
    actual_first_min = None
    first_rack = None
    if over[fw].any():
        rel = int(np.argmax(over[fw].any(axis=1)))
        actual_first_min = round((t[fw][rel] - s0) / 60.0, 1)
        rowmask = over[fw][rel]
        first_rack = list(eng.tp.rack_label[int(np.argmax(rowmask))])
    fault_peak = round(float(inl[fw].max()), 2) if fw.any() else None
    # 从引擎记录的故障预测里找该次故障
    flog = [f for f in out.get('fault_log', []) if name in f.get('name', [])]
    pred_lead = flog[0]['predicted_lead_min'] if flog else None
    pred_peak = flog[0]['predicted_peak'] if flog else None
    err_min = None
    if pred_lead is not None and actual_first_min is not None:
        err_min = round(pred_lead - actual_first_min, 1)
    err_peak = round(pred_peak - fault_peak, 2) if (pred_peak is not None and fault_peak is not None) else None
    return dict(
        policy=eng.policy, fault_crah=name, fault_day=day, fault_hour=hour, dur_h=dur_h,
        predicted_lead_min=pred_lead, actual_first_over_min=actual_first_min,
        lead_error_min=err_min, first_hot_rack=first_rack,
        predicted_peak=pred_peak, actual_peak=fault_peak, peak_error=err_peak,
    )


def main(days=None, seed=7, quick=False):
    days = days or L.JULY_DAYS
    os.makedirs(RESULT_DIR, exist_ok=True)
    summary = {'days': int(days), 'policies': {}, 'fault_test': None,
               'load': {}, 'config': {}}
    t_all = time.time()
    # 基准数据
    eng0 = Engine(days=days, policy='fixed', seed=seed)
    tp = eng0.tp
    tot = eng0.sc.kw.sum(1) + L.MISC_KW
    summary['load'] = dict(mean_kw=round(float(tot.mean()), 1),
                           peak_kw=round(float(tot.max()), 1),
                           t_out_max=round(float(eng0.sc.t_out.max()), 1))
    summary['config'] = dict(n_crah=L.NC, q_max_kw=L.Q_MAX, v_rated=L.V_RATED,
                             n_rack=tp.nrack, t_crit=L.T_CRIT, t_warn=L.T_WARN,
                             train_kw=L.TRAIN_KW)

    for pol in ('fixed', 'lp', 'predictive'):
        t0 = time.time()
        eng, out = run_policy(pol, days, seed)
        m = _metrics(out, days)
        m['wall_s'] = round(time.time() - t0, 1)
        summary['policies'][pol] = m
        save_run(pol, eng, out)
        print(f"[{pol}] {m}", flush=True)

    # ---- 故障测试: 第 8 天 01:00(夜间训练批处理高峰)关掉 S2, 持续 6h。
    # 用联动优化控制器(它可以重分配), 对比其故障后超温时间预测与仿真实际。
    # 故障测试放在热浪期下午最热时段(第 21 天 16:00, 室外约 36°C、负载高), 此时单台
    # 空调失效后其余机组能力不足, 会有机柜超温, 便于对比预测与实际超温时间。
    fday = 21 if days >= 23 else max(1, int(days) - 2)
    fault = (fday, 16.0, 4.0, 'S2')
    t0 = time.time()
    eng, out = run_policy('predictive', days, seed, faults=[fault])
    fr = fault_diagnostics(eng, out, fault)
    fr['wall_s'] = round(time.time() - t0, 1)
    summary['fault_test'] = fr
    save_run('predictive_fault', eng, out, extra={'fault': fr})
    print(f"[fault] {fr}", flush=True)

    summary['wall_total_s'] = round(time.time() - t_all, 1)
    with open(os.path.join(RESULT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    import sys
    d = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(days=d)
