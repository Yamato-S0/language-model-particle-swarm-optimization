import os
import sys
import json
import random
import pickle
import numpy as np
from pmlb import fetch_data
import sympy as sp
from sklearn.metrics import r2_score

sys.path.append("../src")

from optimizers.lmpso import LMPSO
from llms.languange_models import (
    LLAMA3_1_8B_Instruct,
)

from utils.setup import (
    create_unique_id,
    setup_directories,
    setup_logger,
)

from problems.symbolic_regression import SymbolicRegression

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python symbolic_regression.py <number of features> <idnex of dataset>"
        )
        sys.exit(1)

    # コマンドライン引数でデータセットのインデックスを取得
    num_features = int(sys.argv[1])
    dataset_index = int(sys.argv[2])

    # パラメータの設定
    NUM_PARTICLES = 80
    MAX_ITER = 50

    PROBLEM_TYPE = "symbolic_regression"

    # dataset dictをファイルから読み込む
    with open("../data/pmlb/dataset_dict.pkl", "rb") as f:
        dataset_dict = pickle.load(f)

    # インデックスの範囲を確認
    if dataset_index < 0 or dataset_index >= len(dataset_dict[num_features]):
        print(
            f"Invalid dataset index. Please choose a number between 0 and {len(dataset_dict[num_features]) - 1}."
        )
        sys.exit(1)

    dataset_name = dataset_dict[num_features][dataset_index]

    llm = LLAMA3_1_8B_Instruct()

    unique_id = create_unique_id()
    # ディレクトリの設定
    results_path, logs_path = setup_directories(
        PROBLEM_TYPE, "lmpso", unique_id, dataset_name, prefix=None
    )
    seed = random.randint(0, 1000000)

    logger = setup_logger(os.path.join(logs_path, f"seed_{seed}.log"))
    # 問題をfetch
    X, y = fetch_data(dataset_name, return_X_y=True, local_cache_dir="../data/pmlb")
    problem = SymbolicRegression(X, y, llm, logger)

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    max_new_tokens = 200
    inertia_length = 1

    config = {
        "NUM_PARTICLES": NUM_PARTICLES,
        "MAX_ITER": MAX_ITER,
        "unique_id": unique_id,
        "max_new_tokens": max_new_tokens,
        "inertia_length": inertia_length,
        "dataset_name": dataset_name,
        "dimensions": X.shape[1],
        "index_of_dataset": dataset_index,
    }

    with open(
        os.path.join(results_path, "config.json"), encoding="utf-8", mode="w"
    ) as f:
        json.dump(config, f)

    print(f"Starting optimization for seed {seed}")

    optimizer = LMPSO(
        problem,
        logger,
        llm,
        seed=seed,
        num_particles=NUM_PARTICLES,
        max_iter=MAX_ITER,
        max_new_tokens=max_new_tokens,
        inertia_length=inertia_length,
    )

    logger.info("Starting optimization for %s", PROBLEM_TYPE + "_" + dataset_name)
    logger.info("Config: %s", config)

    global_best_position, global_best_value = optimizer.optimize()

    logger.info(
        "Optimization completed. Global best position: %s", global_best_position
    )
    logger.info("Global best value: %s", global_best_value)
    optimizer.save_history(os.path.join(results_path, f"seed_{seed}.csv"))

    # position is a string representing a Python expression with the Sympy library
    expr = sp.sympify(global_best_position)
    # Create a lambda function from the expression
    f = sp.lambdify(problem.variables, expr, modules="numpy")
    # Evaluate the expression
    y_pred = f(*X.T)
    # Calculate the R2 score
    r2 = r2_score(y, y_pred)
    # Calculate the mean squared error
    mse = np.mean((y - y_pred) ** 2)

    logger.info("MSE: %s, R2: %s", mse, r2)
