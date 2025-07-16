# 【核心】项目总控脚本，自动化执行所有任务

import yaml
import subprocess
import argparse
from pathlib import Path
import sys
import os

# 确保可以从src导入visualize
try:
    from src import visualize2 as visualize
except ImportError:
    print("WARNING: Could not import 'visualize2' from 'src'. Visualization step will be skipped.")
    visualize = None


def run_single_experiment(exp_params: dict):
    """构建并运行单次实验的命令，并实时流式传输输出。"""
    command = ['python', '-u', 'src/main.py']

    for key, value in exp_params.items():
        if isinstance(value, bool) and value:
            command.append(f'--{key}')
        elif value is not None:
            command.append(f'--{key}')
            command.append(str(value))

    print("\n" + "=" * 80)
    print(f"▶️  EXECUTING: {' '.join(command)}")
    print("=" * 80)

    # --- 核心修改：为子进程设置环境变量，强制其使用UTF-8输出 ---
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # -------------------------------------------------------------

    # 使用 Popen 进行更灵活的实时输出处理
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',  # 父进程也用UTF-8来解码
        errors='replace',
        bufsize=1,
        env=env  # 传递修改后的环境变量
    )

    # 逐行读取并打印子程序的输出
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()

    process.wait()  # 等待子程序结束

    if process.returncode != 0:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!!  ERROR: Experiment with tag '{exp_params.get('tag')}' failed.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print(f"\n✅  SUCCESS: Experiment with tag '{exp_params.get('tag')}' completed.")


def main():
    """主执行函数，读取yaml，分发实验任务"""
    parser = argparse.ArgumentParser(description="B-PINN 自动化实验与可视化总控脚本")
    parser.add_argument('experiment_groups', nargs='*',
                        help="要运行的实验组名称(来自experiments.yaml)。如果为空，则运行所有。")
    parser.add_argument('--skip_experiments', action='store_true', help='跳过所有实验，直接生成可视化图表。')
    args = parser.parse_args()

    with open('experiments.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    global_settings = config.get('global_settings', {})
    all_exp_groups = config.get('experiments', [])

    groups_to_run = [exp for exp in all_exp_groups if
                     exp.get('name') in args.experiment_groups] if args.experiment_groups else all_exp_groups

    if not args.skip_experiments:
        # --- 阶段一：预处理 ---
        print("\n" + "=" * 80)
        print("== [Stage 1/3] Checking and running data preprocessing...")
        print("=" * 80)
        # (此部分逻辑不变)
        all_datasets_needed = {exp['common_params']['dataset_name'] for exp in groups_to_run}
        for dataset_name in all_datasets_needed:
            raw_csv_path = Path('data/raw') / f"{dataset_name}.xes.csv"
            processed_dir = Path(global_settings.get('data_dir', 'data/processed/'))
            if not raw_csv_path.exists():
                print(f"  - ❌ WARNING: Raw data file for '{dataset_name}' not found. Skipping.")
                continue
            if not (processed_dir / f"{dataset_name}.pkl").exists():
                print(f"  - ⚠️ Processed data for '{dataset_name}' not found. Running automatically...")
                subprocess.run(['python', 'src/preprocessing.py', '--input', str(raw_csv_path), '--output_dir',
                                str(processed_dir)], check=True)
            else:
                print(f"  - ✅ Found existing processed data for '{dataset_name}'.")

        # --- 阶段二：执行所有实验 ---
        print("\n" + "=" * 80)
        print("== [Stage 2/3] Executing all specified experiments...")
        print("=" * 80)
        # (此部分逻辑不变)
        for exp in groups_to_run:
            group_name = exp.get('name', 'unnamed_group')
            print(f"\n#################### Starting Experiment Group: {group_name} ####################")
            common_params = exp.get('common_params', {})
            group_model_dir = Path('models') / group_name
            group_results_dir = Path('results') / group_name
            group_model_dir.mkdir(parents=True, exist_ok=True)
            group_results_dir.mkdir(parents=True, exist_ok=True)
            for run_config in exp.get('runs', []):
                full_params = {**global_settings, **common_params, **run_config}
                full_params['output_dir'] = str(group_model_dir)
                full_params['results_dir'] = str(group_results_dir)
                full_params['tag'] = f"{group_name}_{run_config['tag']}"
                full_params['overwrite'] = True
                run_single_experiment(full_params)
        print("\n✅ All experiments finished!")
    else:
        print("\nINFO: Skipping experiments as per --skip_experiments flag.")

    if visualize:
        print("\n" + "=" * 80)
        print("== [Stage 3/3] Generating all result visualizations...")
        print("=" * 80)
        visualize.main()


if __name__ == '__main__':
    main()