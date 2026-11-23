#!/usr/bin/env bash
# 一键运行: 测试 -> 校准演示 -> 一个月验收 -> 上架建议 -> 启动网页
set -e
cd "$(dirname "$0")"

echo "== 核心测试 =="
python3 tests/test_core.py

echo "== 模型校准演示 =="
python3 -m simem.calib_demo

echo "== 新服务器上架建议 =="
python3 -m simem.placement_report

echo "== 一个月验收(约 1 小时, 三种策略 + 故障测试) =="
python3 -m simem.acceptance

echo "== 启动网页 http://127.0.0.1:8000 =="
python3 -m simem.app 8000
