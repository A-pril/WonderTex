#!/bin/bash

# 定义命令数组
commands=(
    # "python run_texture.py --config ./config/15.yaml"
    "python run_texture.py --config ./config/16.yaml"
    # "python run_texture.py --config ./config/17.yaml"
    # "python run_texture.py --config ./config/18.yaml"
    "python run_texture.py --config ./config/19.yaml"
    "python run_texture.py --config ./config/20.yaml"
    # "python run_texture.py --config ./config/21.yaml"
    # "python run_texture.py --config ./config/22.yaml"
    # "python run_texture.py --config ./config/23.yaml"
    # "python run_texture.py --config ./config/24.yaml"
    # "python run_texture.py --config ./config/25.yaml"
    # "python run_texture.py --config ./config/26.yaml"
    "python run_texture.py --config ./config/27.yaml"
    # "python run_texture.py --config ./config/28.yaml"
    # "python run_texture.py --config ./config/29.yaml"
    # "python run_texture.py --config ./config/30.yaml"
    # "python run_texture.py --config ./config/31.yaml"
    "python run_texture.py --config ./config/32.yaml"
)

# 使用 for 循环依次运行命令
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    $cmd  # 执行命令
    echo "Finished running: $cmd"
    echo "--------------------------------"
done