"""机房网格热-气流模型(仿真器核心)。

平面离散成 1m x 1m 网格,每个空气网格一个温度节点。机柜作为"取风-排风"元件:
正面从冷通道取风,出风温度 = 进风 + P/(rho*cp*V),从背面热通道排出。地板送风由
精密空调送入冷通道,按各机柜正面地板出风口的开度(压力驱动)分配;机柜排出的热风
由吊顶回风口回到空调。网格之间按风量交换热量 h_ij*(T_j - T_i),空机柜位(缺口)
用较大的交换系数模拟热风回流。

求解: 每一步对稳态能量方程做红黑 SOR;热惯性表现为每个网格与上一时刻自身的
交换(等效质量 m_i/dt)。以上一时刻温度热启动,每 1min 步 ~10-25 次双色扫描。
"""
from __future__ import annotations

import numpy as np

from . import layout as L


class Topology:
    """静态拓扑: 网格、邻接系数、机柜、送/回风权重。构建一次反复使用。"""

    def __init__(self):
        self._build_cells()
        self._build_neighbors()
        self._build_cabinets()
        self._build_supply_return()
        self._build_env()

    # ------------------------------------------------------------ 网格
    def _build_cells(self):
        self.gap_set = set(L.GAPS)
        self.row_names = L.ROW_NAMES
        cell_id = {}
        cx, cy, ct, cj, ci = [], [], [], [], []
        for j in range(1, L.NY - 1):
            row = L.row_of_j(j)
            for i in range(1, L.NX - 1):
                if row is not None and L.RACK_I0 <= i <= L.RACK_I1:
                    if (row, i) in self.gap_set:
                        t = L.GAP
                    else:
                        continue
                elif row is not None:
                    t = L.SIDE
                elif j in L.HOT_AISLE_J:
                    t = L.HOT
                elif j in L.COLD_AISLE_J:
                    t = L.COLD
                else:
                    t = L.SIDE
                cell_id[(i, j)] = len(cx)
                cx.append(i + 0.5)
                cy.append(j + 0.5)
                ct.append(t)
                cj.append(j)
                ci.append(i)
        self.cell_id = cell_id
        self.x = np.asarray(cx, float)
        self.y = np.asarray(cy, float)
        self.type = np.asarray(ct, np.int8)
        self.j = np.asarray(cj, np.int16)
        self.i = np.asarray(ci, np.int16)
        self.n = len(cx)
        self.is_cold = self.type == L.COLD
        self.is_hot = self.type == L.HOT
        self.is_gap = self.type == L.GAP
        self.is_side = self.type == L.SIDE

    def cell_at(self, i, j):
        return self.cell_id.get((i, j))

    # ------------------------------------------------------------ 邻接
    def _build_neighbors(self):
        pair = {}
        for (i, j), k in self.cell_id.items():
            for di, dj in ((1, 0), (0, 1)):
                q = self.cell_id.get((i + di, j + dj))
                if q is None:
                    continue
                ta, tb = self.type[k], self.type[q]
                if L.GAP in (ta, tb):
                    h = L.GAP_FLOW if (ta == L.GAP) != (tb == L.GAP) else 0.35
                elif ta == L.HOT and tb == L.HOT:
                    h = L.H_MIX_HOT
                elif ta == L.COLD and tb == L.COLD:
                    h = L.H_MIX
                elif L.SIDE in (ta, tb):
                    h = L.H_SIDE
                elif ta != tb:
                    h = 0.10
                else:
                    h = L.H_MIX
                pair[(k, q)] = h
        deg = np.zeros(self.n, int)
        for (k, q) in pair:
            deg[k] += 1
            deg[q] += 1
        maxd = deg.max()
        self.nbr = np.full((self.n, maxd), -1, np.int32)
        self.h0 = np.zeros((self.n, maxd))
        self.safe = np.zeros((self.n, maxd), np.int32)
        self.valid = np.zeros((self.n, maxd), bool)
        cnt = np.zeros(self.n, int)
        for (k, q), h in pair.items():
            for a, b in ((k, q), (q, k)):
                self.nbr[a, cnt[a]] = b
                self.h0[a, cnt[a]] = h
                cnt[a] += 1
        self.valid = self.nbr >= 0
        self.safe = np.where(self.valid, self.nbr, 0)
        par = (self.i + self.j) % 2
        self.mask_red = par == 0
        self.mask_black = par == 1
        self.parity = par.astype(np.int8)

    # ------------------------------------------------------------ 机柜
    def _build_cabinets(self):
        racks = []
        for name, j, d in L.ROWS:
            for i in range(L.RACK_I0, L.RACK_I1 + 1):
                if (name, i) in self.gap_set:
                    continue
                racks.append(dict(row=name, row_idx=self.row_names.index(name),
                                  i=i, j=j, d=d,
                                  front=self.cell_at(i, j + d),
                                  back=self.cell_at(i, j - d)))
        self.racks = racks
        self.nrack = len(racks)
        self.rack_ri = np.array([r['row_idx'] for r in racks], np.int16)
        self.rack_i = np.array([r['i'] for r in racks], np.int16)
        self.rack_j = np.array([r['j'] for r in racks], np.int16)
        self.rack_front = np.array([r['front'] for r in racks], np.int32)
        self.rack_back = np.array([r['back'] for r in racks], np.int32)
        self.rack_index = {(r['row'], r['i']): n for n, r in enumerate(racks)}
        self.rack_label = [(r['row'], r['i']) for r in racks]
        self.gap_cells = {g: self.cell_at(g[1], L.ROW_J[g[0]]) for g in L.GAPS}
        # 每排机柜的前/后网格, 用于热力图与"排进风温度"
        self.row_front_cells = []
        for r in self.row_names:
            cells = self.rack_front[self.rack_ri == self.row_names.index(r)]
            self.row_front_cells.append(cells)

    def rack_no(self, row, i):
        return self.rack_index.get((row, i))

    # ------------------------------------------------------------ 送/回风
    def _build_supply_return(self):
        """每台空调 -> 各机柜 / 各热区网格的静态权重。

        送风: 每台空调按距离把风量分给附近机柜的正面上方地板出风口,权重按机柜
        到空调的距离平滑衰减; 每台空调的列和为 1(压力驱动, 关小某处风口时其余
        风口风量自动增大)。
        回风: 每台空调从附近热通道网格按权重抽风,列和为 1。
        """
        crah_xy = np.array([[c['x'], c['y']] for c in L.CRAHS])
        rack_xy = np.stack([self.rack_i + 0.5, self.rack_j + 0.5], 1).astype(float)
        # 送风: 空调到机柜正面
        dist = np.sqrt(((rack_xy[:, None, :] - crah_xy[None, :, :]) ** 2).sum(-1))
        lam = 14.0
        w = np.exp(-dist / lam)
        self.rack_sup_w = w / w.sum(0, keepdims=True)        # (nrack, NC)
        # 回风: 网格到空调。热通道回风为主, 侧边通道也参与回风(端部气流),
        # 冷通道不抽。
        z = np.stack([self.x, self.y], 1)
        d = np.sqrt(((z[:, None, :] - crah_xy[None, :, :]) ** 2).sum(-1))
        wr = np.exp(-d / lam)
        wr[self.type == L.COLD] = 0.0
        wr[self.type == L.GAP] *= 0.6
        wr[self.type == L.SIDE] = 0.9 * np.exp(-d[self.type == L.SIDE] / lam)
        wr[self.type == L.HOT] *= 2.5
        wr = np.maximum(wr, 1e-9)
        self.w_ret = wr / wr.sum(0, keepdims=True)           # (n, NC)
        # 空调送风落到冷通道网格: 按"网格到最近空调的距离"分配, 与机柜功率无关
        # (地板出风口的开度决定单位面积冷风量)。这是冷通道被冷却的真实来源。
        cold = self.is_cold
        d_near = d.min(axis=1)
        wc = np.zeros(self.n)
        wc[cold] = np.exp(-d_near[cold] / lam)
        wc = np.maximum(wc, 0.0)
        self.cold_sup_raw = wc  # 供 step 按 tile_open 再归一
        # 空调覆盖的机柜子集(用于优化器局部灵敏度)
        self.crah_racks = [np.where(self.rack_sup_w[:, k] > 0.002)[0] for k in range(L.NC)]

    # ------------------------------------------------------------ 环境
    def _build_env(self):
        self.perim = ((self.i == 1) | (self.i == self.i.max()) |
                      (self.j == 1) | (self.j == self.j.max())).astype(float)
        m = np.full(self.n, 45.0)
        m[self.is_hot | self.is_cold] = 60.0
        self.mass = m
        # 每个机柜的"进风温度权重": 正面网格 + 同排相邻机柜少量(顶部进风混合)
        self.inlet_ring = np.zeros((self.nrack, 3), np.int32)
        self.inlet_ring[:, 0] = self.rack_front
        self.inlet_ring[:, 1] = self.rack_front
        self.inlet_ring[:, 2] = self.rack_front
        wgt = np.zeros((self.nrack, 3))
        wgt[:, 0] = 1.0
        for r in range(self.nrack):
            i = self.rack_i[r]
            for slot, (di, ww) in zip((1, 2), ((-1, 0.15), (1, 0.15))):
                q = self.cell_at(i + di, self.rack_j[r])
                if q is not None:
                    self.inlet_ring[r, slot] = q
                    wgt[r, slot] = ww
        self.inlet_w = wgt / wgt.sum(1, keepdims=True)


# ======================================================================
class RoomModel:
    """机房温度场推进器(参数可在线标定)。"""

    PARAM_KEYS = ('h_mix', 'h_mix_hot', 'gap_flow', 'h_side', 'h_wall',
                  'recirc', 'v_base', 'v_kw', 'tile_bypass')

    def __init__(self, topo=None, params=None):
        self.topo = topo or Topology()
        self.p = dict(h_mix=L.H_MIX, h_mix_hot=L.H_MIX_HOT, gap_flow=L.GAP_FLOW,
                      h_side=L.H_SIDE, h_wall=0.014, recirc=L.RECIRC,
                      v_base=L.V_CAB_BASE, v_kw=L.V_CAB_KW,
                      tile_bypass=L.TILE_BYPASS)
        if params:
            self.p.update(params)
        self.T = np.full(self.topo.n, 24.0)
        self.T_ret = np.full(L.NC, 26.0)
        self._rebuild_h(None)

    # ------------------------------------------------------------ 参数
    def apply_params(self, params):
        self._rebuild_h(params)

    def _rebuild_h(self, params):
        if params:
            self.p.update(params)
        P = self.p
        tp = self.topo
        ratio = np.ones(tp.n)
        ratio[tp.is_hot] = P['h_mix_hot'] / L.H_MIX_HOT
        ratio[tp.is_cold] = P['h_mix'] / L.H_MIX
        ratio[tp.is_side] = P['h_side'] / L.H_SIDE
        ratio[tp.is_gap] = P['gap_flow'] / L.GAP_FLOW
        s = 0.5 * (ratio[:, None] + ratio[tp.safe])
        self.hn = np.where(tp.valid, tp.h0 * s, 0.0)
        self.hsum = self.hn.sum(1)

    # ------------------------------------------------------------ 主推进
    def reset(self, t0=24.0):
        self.T = np.full(self.topo.n, t0)

    def cold_supply(self, v_crah, tile_open):
        """把每台空调的送风量按地板出风口开度铺到冷通道网格上。

        冷通道每个网格获得 sup_cell (m3/s), 与机柜功率无关——地板砖送风量由空调
        风量和风口开度决定。这是冷通道被冷却的来源, 机柜只是从冷通道"取风"。
        返回 (sup_cell[n], tsup_cell[n])。
        """
        tp = self.topo
        n = tp.n
        v_on = np.maximum(np.asarray(v_crah, float), 0.0)
        # 每个冷网格的风口开度: 取直接相邻机柜的开度均值(没有机柜则用 1.0)
        op_cell = np.ones(n)
        cnt = np.zeros(n)
        np.add.at(op_cell, tp.rack_front, 0.0)        # 初始化
        op_sum = np.zeros(n)
        np.add.at(op_sum, tp.rack_front, np.clip(tile_open, 0.05, 2.0))
        np.add.at(cnt, tp.rack_front, 1.0)
        op_cell = np.where(cnt > 0, op_sum / np.maximum(cnt, 1), 1.0)
        cold = tp.is_cold
        # 每台空调在冷区的权重 = 距离核 × 风口开度
        w = np.zeros((n, L.NC))
        for k in range(L.NC):
            if v_on[k] <= 1e-6:
                continue
            wk = tp.cold_sup_raw * op_cell * cold
            s = wk.sum()
            if s > 1e-9:
                w[:, k] = wk / s
        # 送风温度: 每台空调一个
        T_sup = np.asarray(self._tsup_cache, float)
        sup_cell = np.dot(w, v_on)
        num = np.dot(w, v_on * np.where(v_on > 1e-6, T_sup, 0.0))
        tsup_cell = np.where(sup_cell > 1e-9, num / np.maximum(sup_cell, 1e-9),
                             float(T_sup[v_on > 1e-6].mean()) if (v_on > 1e-6).any() else 25.0)
        return sup_cell, tsup_cell

    def step(self, dt_s, rack_kw, T_sup, v_crah, crah_on, tile_open, t_out,
             q_misc=0.0, iters=40, tol=5e-5):
        """推进一个时间步。

        质量/能量约定:
          - 机柜风扇需求 fan_v = v_base + v_kw*P; 空调总送风够则足量, 不够则缺风,
            缺风时进风温度向热通道温度靠近(热点的主要成因)。
          - 冷通道按 cab_v 供冷, 机柜以 cab_v 排出, 进风温升 = P/(rho*cp*cab_v)。
          - 空调送风大于机柜穿透风量的部分为旁通冷风, 直接混入回风。
        """
        tp = self.topo
        n = tp.n
        P = self.p
        T_old = self.T
        T = T_old.copy()
        T_sup = np.asarray(T_sup, float)
        v_on = np.maximum(np.asarray(v_crah, float), 0.0)
        tot_sup = float(v_on.sum())
        f, b = tp.rack_front, tp.rack_back

        # 机柜风扇需求 与 冷通道地板送风
        self._tsup_cache = T_sup
        fan_v = P['v_base'] + P['v_kw'] * np.maximum(rack_kw, 0.0)
        sup_cell, tsup_cell = self.cold_supply(v_crah, tile_open)
        # 每个机柜从其正面冷网格取风: 受该网格可供冷风的约束。
        # 同一冷网格若有多个机柜(共享), 按风扇需求比例分。
        sup_at_rack = sup_cell[f]
        # 同一网格上机柜的 fan 之和
        fan_sum = np.bincount(f, weights=fan_v, minlength=n)
        # 每个机柜能拿到的冷风 = 其 fan 占该网格 fan 和的比例 × 网格可供
        avail = np.where(fan_sum[f] > 1e-9,
                         sup_at_rack * fan_v / np.maximum(fan_sum[f], 1e-9), 0.0)
        cab_v = np.clip(np.minimum(fan_v, avail), 0.25 * fan_v, None)
        shortage = np.clip(fan_v - avail, 0.0, None) / np.maximum(fan_v, 1e-6)
        cab_dt = rack_kw / (1.206 * np.maximum(cab_v, 1e-3))
        # 缺风时进风向热通道温度靠近(顶部补风)
        t_front = T_old[f]
        t_hot_local = T_old[b]
        t_inlet_eff = t_front + shortage * (t_hot_local - t_front) * 0.9

        # 冷区: 地板送风注入网格, 机柜从网格抽走 cab_v。
        # 若送风量 > 抽走量, 多余冷风留在冷通道并最终经旁通回风; 若送风量 < 抽走量,
        # 机柜"抽"的空气里含一部分热通道补风(缺风), 由 shortage 表达。
        sup_use = np.bincount(f, weights=cab_v, minlength=n)      # 被机柜消耗的冷风
        sup_t = np.bincount(f, weights=cab_v * tsup_cell[f], minlength=n)
        # 网格送风总量 sup_cell 全部注入; 机柜抽取 cab_v 带走同等流量。
        m_sup_cell = sup_cell.copy()
        tsup_in = np.where(sup_cell > 1e-9, sup_cell * tsup_cell, 0.0)
        # 旁通: 冷通道中未被机柜抽走的冷风直接回风
        bypass_v = float(np.clip(sup_cell.sum() - float(cab_v.sum()), 0.0, None))
        # 热区回风抽出(质量 = 机柜排风 = cab_v)
        ret_flow = np.bincount(b, weights=cab_v, minlength=n)
        m_ret_cell = ret_flow.copy()
        T_hot_ret = np.dot(tp.w_ret.T, T_old)
        # 空调回风温度: 热区回风 + 旁通冷风
        v_byp_k = (v_on / max(tot_sup, 1e-9)) * bypass_v if tot_sup > 1e-9 else np.zeros(L.NC)
        t_ret_mix = np.where(v_on > 1e-6,
                             (T_hot_ret * np.maximum(v_on - v_byp_k, 0.0)
                              + np.asarray(T_sup, float) * v_byp_k) / np.maximum(v_on, 1e-9),
                             T_hot_ret)
        self.T_ret = t_ret_mix

        diag = self.hsum + m_sup_cell + m_ret_cell
        src = tsup_in
        # 墙体/渗透 + 环境锚。锚项保证矩阵严格对角占优(diag > Σ h),避免无送/回风
        # 的角落网格使线性系统奇异; 物理上对应机房门缝渗透与地板/吊顶的弱耦合。
        hw = P['h_wall']
        h_amb = 0.05
        diag = diag + tp.perim * hw + h_amb
        src = src + tp.perim * hw * t_out + h_amb * t_out
        # 机柜排风注入热区(温度 = 有效进风 + ΔT)
        src = src + np.bincount(b, weights=cab_v * (t_inlet_eff + cab_dt), minlength=n)
        # 顶部泄漏/缺口回流污染进风面
        rec = P['recirc'] * cab_v
        diag = diag + np.bincount(f, weights=rec, minlength=n)
        src = src + np.bincount(f, weights=rec * T_old[b], minlength=n)
        # 分散热负荷
        if q_misc > 0:
            src = src + q_misc * (1000.0 / L.RHO_CP) / n
        # 热惯性
        mdt = tp.mass / dt_s
        diag = diag + mdt
        src = src + mdt * T_old

        self._cab_v = cab_v
        self._cab_dt = cab_dt
        self._shortage = shortage
        self._bypass = bypass_v
        T = self._solve(T, diag, src, iters, tol)
        self.T = np.clip(T, 5.0, 90.0)
        return self.T

    # ------------------------------------------------------------
    def _solve(self, T, diag, src, iters, tol):
        """求解线性方程组 (diag - H) T = src。

        稀疏直接求解(优选, 精确); 无 scipy 时用红黑 SOR 迭代回退。
        """
        tp = self.topo
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import spsolve
        except Exception:
            return self._sor(T, diag, src, iters, tol)
        n = tp.n
        if getattr(self, '_sp_idx', None) is None:
            # 邻居项的 (行, 列) 索引; 展平顺序必须与 hn[tp.valid] 一致(行优先)。
            # np.nonzero 返回行优先顺序, 与布尔索引展平 hn 的顺序相同。
            r, c = np.nonzero(tp.valid)
            self._sp_rows = np.concatenate([np.arange(n), r])
            self._sp_cols = np.concatenate([np.arange(n), tp.nbr[r, c]])
            self._sp_idx = (r, c)
        r, c = self._sp_idx
        # A = diag*I - sum h_ij
        data = np.concatenate([diag, -self.hn[r, c]])
        A = csr_matrix((data, (self._sp_rows, self._sp_cols)), shape=(n, n))
        self.last_iter = 1
        self.last_res = 0.0
        return spsolve(A, src)

    def _sor(self, T, diag, src, iters, tol):
        tp = self.topo
        hn, safe, valid = self.hn, tp.safe, tp.valid
        omega = 1.6
        for it in range(iters):
            for mask in (tp.mask_red, tp.mask_black):
                S = (hn * T[safe]).sum(1)
                upd = (src + S) / diag
                T[mask] = (1 - omega) * T[mask] + omega * upd[mask]
            if it % 5 == 4 or it == iters - 1:
                res = np.abs(diag * T - src - (hn * T[safe]).sum(1)).max()
                if res < tol:
                    break
        self.last_res = float(res)
        self.last_iter = it + 1
        return T

    # ------------------------------------------------------------ 读数
    def rack_inlet(self, T=None):
        """每机柜进风温度(机柜正面 + 顶部进风修正)。"""
        tp = self.topo
        T = self.T if T is None else T
        inlet = (tp.inlet_w * T[tp.inlet_ring]).sum(1)
        return inlet + L.TOP_INLET_PENALTY

    def rack_exhaust(self, T=None):
        tp = self.topo
        T = self.T if T is None else T
        return T[tp.rack_back]

    def _predict_return(self, T):
        """返回最近一次 step 算出的各空调混合回风温度。

        混合回风包含热区回风与旁通冷风, 与 step 内 CRAH 看到的完全一致;
        预测器必须复用该值, 否则会把回风高估、制冷量判断失真。
        """
        return self.T_ret

    def hot_aisle_T(self, T=None):
        tp = self.topo
        T = self.T if T is None else T
        return T[tp.is_hot]

    def cold_aisle_T(self, T=None):
        tp = self.topo
        T = self.T if T is None else T
        return T[tp.is_cold]
