import numpy as np
import pandas as pd
import time
import sys
import os
import warnings
from PSFO import PSFO  # 导入算法类

# 忽略数值警告
warnings.filterwarnings('ignore')

def get_optimum(year, fid):
    """Return theoretical optimum for CEC functions."""
    if year == 2017: return fid * 100.0
    elif year == 2022:
        opts = {1: 300, 2: 400, 3: 600, 4: 800, 5: 900, 6: 1800,
                7: 2000, 8: 2200, 9: 2300, 10: 2400, 11: 2600, 12: 2700}
        return opts.get(fid, 0)
    return 0

def run_suite_comprehensive(year, dim, func_indices, repeats=30):
    POP_SIZE = 50
    MAX_ITER = 1000
    BOUNDS = [-100, 100]
    
    print(f"\n{'='*80}")
    print(f"PSFO - CEC {year} | Dim: {dim} | Runs: {repeats}")
    print(f"{'='*80}")
    
    summary_data = [] 
    detailed_data = [] 
    
    # 动态导入 opfunu
    try:
        if year == 2017: import opfunu.cec_based.cec2017 as cec_module
        elif year == 2022: import opfunu.cec_based.cec2022 as cec_module
    except ImportError:
        print(f"Error: 'opfunu' module not found. Please install it via pip.")
        return [], []

    for fid in func_indices:
        # 处理函数名称兼容性
        func_name = f"F{fid}{year}"
        if not hasattr(cec_module, func_name):
            if hasattr(cec_module, f"F{fid}"): func_name = f"F{fid}"
            else: continue

        func_class = getattr(cec_module, func_name)
        problem = func_class(ndim=dim)
        theoretical_opt = get_optimum(year, fid)
        
        errors = []
        
        sys.stdout.write(f"Running {func_name}: [")
        for run_id in range(repeats):
            # 实例化优化器
            optimizer = PSFO(problem.evaluate, dim, BOUNDS, POP_SIZE, MAX_ITER)
            best_val = optimizer.run()
            
            error = best_val - theoretical_opt
            if abs(error) < 1e-8: error = 0.0
            errors.append(error)
            
            if (run_id+1) % 5 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
        sys.stdout.write("] Done.\n")
        
        # 汇总数据
        summary_data.append({
            "Function": func_name,
            "Best": np.min(errors),
            "Worst": np.max(errors),
            "Mean": np.mean(errors),
            "Std": np.std(errors)
        })
        
        # 详细数据 (用于 Wilcoxon 测试)
        row_dict = {"Function": func_name}
        for i, err in enumerate(errors):
            row_dict[f"Run{i+1}"] = err
        detailed_data.append(row_dict)

    return summary_data, detailed_data

if __name__ == "__main__":
    start_time = time.time()
    REPEATS = 30
    
    # 创建结果目录
    if not os.path.exists("results"):
        os.makedirs("results")

    # --- 1. CEC 2017 (30维) ---
    # F2 在 CEC2017 中通常被排除 (不稳定)
    ids_2017 = [i for i in range(1, 31) if i != 2] 
    summ_17, raw_17 = run_suite_comprehensive(2017, dim=30, func_indices=ids_2017, repeats=REPEATS)
    
    if summ_17:
        pd.DataFrame(summ_17).to_csv("results/psfo2017_summary.csv", index=False)
        pd.DataFrame(raw_17).to_csv("results/psfo2017_detailed_runs.csv", index=False)

    # --- 2. CEC 2022 (20维) ---
    ids_2022 = list(range(1, 13))
    summ_22, raw_22 = run_suite_comprehensive(2022, dim=20, func_indices=ids_2022, repeats=REPEATS)
    
    if summ_22:
        pd.DataFrame(summ_22).to_csv("results/psfo2022_summary.csv", index=False)
        pd.DataFrame(raw_22).to_csv("results/psfo2022_detailed_runs.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\nAll Done! Total Time: {elapsed/60:.2f} minutes.")
    print("Results saved in 'results/' directory.")