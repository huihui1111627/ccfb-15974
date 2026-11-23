"""机房温度预测与空调联动调节系统。

子模块:
    layout      几何/设备/物理常量
    room        网格热-气流模型
    crah        精密空调模型
    scenario    负载与气象场景
    calib       模型参数在线校准
    forecast    负载预测 / 温度预测 / 热点预警
    control     三种空调运行方式(含联动优化)
    engine      闭环仿真编排
    placement   新服务器上架位置评估
    acceptance  一个月验收
    app         网页服务
"""

__version__ = '1.0.0'
