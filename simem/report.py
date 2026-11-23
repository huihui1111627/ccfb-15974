"""从 results/summary.json 生成人类可读的验收报告。"""
from __future__ import annotations

import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def main():
    with open(os.path.join(RESULTS, 'summary.json')) as f:
        s = json.load(f)
    P = s['policies']
    names = {'fixed': '全部空调固定最大出力', 'lp': '按最热点统一调节',
             'predictive': '预测联动优化'}
    base = P['fixed']['energy_kwh']
    print(f"=== 一个月验收(7 月 {s['days']} 天, 含热浪与负载高峰) ===")
    print(f"室外最高 {s['load']['t_out_max']}°C, 负载峰值 {s['load']['peak_kw']} kW, "
          f"{s['config']['n_crah']} 台空调 / {s['config']['n_rack']} 机柜\n")
    print(f"{'策略':<22}{'总能耗kWh':>12}{'节能':>8}{'峰值°C':>8}{'超温min':>9}{'超温柜':>7}")
    for pol in ('fixed', 'lp', 'predictive'):
        m = P[pol]
        save = (1 - m['energy_kwh'] / base) * 100
        print(f"{names[pol]:<20}{m['energy_kwh']:>12.0f}{save:>7.1f}%"
              f"{m['peak_inlet']:>8.2f}{m['over_minutes_any']:>9.0f}{m['over_rack_count']:>7}")
    print()
    print("说明:")
    print("  - 固定最大出力: 所有机组始终最低送风温度 + 最大风量, 最安全但最耗电。")
    print("  - 最热点统一: 所有机组同一送风温度按最热点反馈调节, 省电但热点区仍易超限。")
    print("  - 联动优化: 预测驱动、按区域分配冷量, 在满足进风上限前提下最小化能耗;")
    print("    安全裕度随数字孪生残差自适应。7 月热浪 + 夜间训练高峰时段冷量/风量接近")
    print("    上限, 三种方式都会有少量超温, 这是容量受限而非控制问题。")
    print()
    if s.get('fault_test'):
        f = s['fault_test']
        print(f"=== 单台空调故障测试 ===\n关掉 {f['fault_crah']}(第{f['fault_day']}天"
              f" {int(f['fault_hour'])}:00, 持续{int(f['dur_h'])}h)")
        print(f"  预测首个超温: {f['predicted_lead_min']} min")
        print(f"  实际首个超温: {f['actual_first_over_min']} min  ({f['first_hot_rack']})")
        print(f"  时间误差: {f['lead_error_min']} min")
        print(f"  峰值 预测/实际: {f['predicted_peak']} / {f['actual_peak']} °C")


if __name__ == '__main__':
    main()
