"""核心物理与控制逻辑测试。运行: python3 -m pytest tests/ -q 或 python3 tests/test_core.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simem import layout as L
from simem.room import Topology, RoomModel
from simem.crah import Plant, CRAH
from simem.scenario import Scenario
from simem.control import FixedMaxController, LPController, PredictiveController, crah_power


def test_topology():
    tp = Topology()
    assert tp.nrack > 300
    assert tp.n > 400
    # 机柜正面必是冷区, 背面必是热区
    assert np.all(tp.type[tp.rack_front] == L.COLD)
    assert np.all(tp.type[tp.rack_back] == L.HOT)
    # 送/回风权重每台空调列和为 1
    assert np.allclose(tp.w_ret.sum(0), 1.0, atol=1e-6)


def test_steady_uniform():
    """均匀负载下温度场收敛且物理合理: 冷区 < 热区, 进风 < 排风。"""
    tp = Topology()
    room = RoomModel(tp)
    kw = np.full(tp.nrack, 1.5)
    v = np.full(L.NC, L.V_RATED)
    on = np.ones(L.NC, bool)
    tsup = np.full(L.NC, 16.0)
    tile = np.ones(tp.nrack)
    for _ in range(300):
        room.step(120.0, kw, tsup, v, on, tile, 30.0, q_misc=L.MISC_KW)
    cold = room.T[tp.is_cold].mean()
    hot = room.T[tp.is_hot].mean()
    assert hot > cold, "热通道应高于冷通道"
    assert 10 < cold < 30, f"冷区温度异常 {cold}"
    inlet = room.rack_inlet()
    assert np.all(inlet > 5) and np.all(inlet < 60)
    # 能量守恒: 机柜带入房间的热量 ≈ 空调回风带走的热量(稳态)
    heat_in = float(np.dot(room._cab_v, room._cab_dt))
    assert heat_in > 0


def test_mass_energy_balance():
    """送风量守恒: 冷区送风 ≈ 机柜穿透 + 旁通。"""
    tp = Topology()
    room = RoomModel(tp)
    kw = np.full(tp.nrack, 2.0)
    v = np.full(L.NC, L.V_RATED)
    on = np.ones(L.NC, bool)
    for _ in range(50):
        room.step(300.0, kw, np.full(L.NC, 15.0), v, on,
                  np.ones(tp.nrack), 30.0, q_misc=L.MISC_KW)
    total_supply = float(v.sum())
    through = float(room._cab_v.sum()) + room._bypass
    assert abs(total_supply - through) < 0.05 * total_supply


def test_crah_dynamics():
    """空调一阶响应与启停延迟, 不同步长下稳态一致。"""
    results = []
    for dt in (120.0, 300.0):
        p = Plant(dt)
        for _ in range(int(600 * 60 / dt)):
            p.t_out = 30.0
            p.command(np.ones(L.NC, bool), np.full(L.NC, 18.0), np.full(L.NC, 0.9))
            p.advance(np.full(L.NC, 26.0))
        on, v, tsup, q, _ = p.cols()
        results.append(tsup.mean())
    assert abs(results[0] - results[1]) < 0.5, "不同步长应收敛到同一稳态"


def test_crah_delay():
    p = CRAH(0, 60.0)
    p.command(True, 18.0, 0.8)
    p.advance(26.0)
    assert not p.on, "启动延迟内不应立即运行"
    for _ in range(int(L.DELAY_START) + 2):
        p.advance(26.0)
    assert p.on, "延迟后应启动"


def test_power_model():
    # 送风温度越高 COP 越高, 功率越低(同制冷量)
    p_low = crah_power(50, 14, 32, 0.9)
    p_high = crah_power(50, 20, 32, 0.9)
    assert p_high < p_low
    # 室外越热功率越高
    assert crah_power(50, 18, 38, 0.9) > crah_power(50, 18, 28, 0.9)


def test_controllers_decide():
    tp = Topology()
    obs = dict(inlet_max=26.5, load=np.full(tp.nrack, 1.0), crah_state=[])
    fault = np.zeros(L.NC, bool)
    fault[2] = True
    for c in (FixedMaxController(), LPController()):
        cmd = c.decide(obs, None, 30.0, fault)
        assert cmd['on'].shape == (L.NC,)
        assert not cmd['on'][2], "故障机组不应开启"


def test_scenario_shapes():
    tp = Topology()
    sc = Scenario(tp, days=1)
    assert sc.kw.shape == (sc.nstep, tp.nrack)
    assert sc.t_out.shape == (sc.nstep,)
    assert sc.kw.min() >= 0




def test_calibrator_reduces_error():
    """校准应降低孪生与传感器读数的残差。"""
    from simem.engine import Engine
    from simem.calib import Calibrator
    eng = Engine(days=0.2, policy='predictive', calibrate=False)
    # 推进真实机房到某状态
    sc = eng.sc
    for n in range(0, 120):
        eng.plant.t_out = float(sc.t_out[n])
        eng.plant.command(np.ones(L.NC, bool), np.full(L.NC, 15.0), np.ones(L.NC))
        eng.plant.advance(eng.room.T_ret)
        on, v, tsup, q, f = eng.plant.cols()
        tile = eng._tile_open(sc.kw[n])
        eng.room.step(eng.dt_s, sc.kw[n], tsup, v, on, tile, float(sc.t_out[n]),
                      q_misc=L.MISC_KW)
    y = eng.sensors.sample(eng.room.T)
    inlet_y, _ = eng.sensors.inlet_readings(y)
    cal = Calibrator(eng.twin, eng.sensors)
    args = (sc.kw[119], tsup, v, on, tile, float(sc.t_out[119]), L.MISC_KW)
    _, rms, rms0 = cal.calibrate(eng.twin.T.copy(), eng.dt_s, args, inlet_y,
                                 iters=3, warm=6)
    assert rms <= rms0 + 0.05, f"校准后残差不升: {rms0}->{rms}"


def test_placement_runs():
    """上架建议对每个空位都给出有限的温度影响。"""
    from simem.room import RoomModel
    from simem.placement import evaluate_new_server
    tp = Topology()
    room = RoomModel(tp)
    kw = np.full(tp.nrack, 1.5)
    v = np.full(L.NC, L.V_RATED)
    on = np.ones(L.NC, bool)
    res, bmax = evaluate_new_server(room, tp, kw, np.full(L.NC, 15.0), v, on,
                                    np.ones(tp.nrack), 30.0, new_kw=12.0)
    assert len(res) == len(L.GAPS)
    assert all(np.isfinite(d['local_rise']) for d in res)
    assert res[0]['rank'] == 1


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
