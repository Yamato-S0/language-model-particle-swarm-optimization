import os
import random
import sys
import json
import argparse


sys.path.append("../src")

from optimizers.lmpso import LMPSO
from problems.tsp import TravelingSalesmanProblem as TSP
from utils.setup import (
    create_unique_id,
    setup_directories,
    setup_logger,
)
from llms.languange_models import LLAMA3_1_8B_Instruct

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Run TSP optimization using LMPSO.")
    parser.add_argument(
        "--num_of_cities", type=int, required=True, help="Number of cities for the TSP."
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed for the random number generator."
    )
    args = parser.parse_args()

    # パラメータの設定
    NUM_PARTICLES = 10
    MAX_ITERS = {5: 10, 10: 100, 20: 100, 30: 100}

    PROBLEM_TYPE = "tsp"
    BOUNDS = [0, 100]

    random_seeds = [random.randint(0, 100000) for _ in range(1)]

    num_of_city = args.num_of_cities
    seed = args.seed

    unique_id = create_unique_id()

    print(f"Starting optimization for {num_of_city} cities")

    max_new_tokens = num_of_city * 5

    pipe = LLAMA3_1_8B_Instruct()

    # for seed in SEEDS:
    random.seed(seed)

    # 問題の生成
    cities = set()
    while len(cities) < num_of_city:
        cities.add(
            (
                random.randint(BOUNDS[0], BOUNDS[1]),
                random.randint(BOUNDS[0], BOUNDS[1]),
            )
        )
    cities = list(cities)

    config = {
        "num_particles": NUM_PARTICLES,
        "max_iter": MAX_ITERS[num_of_city],
        "problem_type": PROBLEM_TYPE,
        "num_of_city": num_of_city,
        "bounds": BOUNDS,
        "cities": cities,
        "max_new_tokens": max_new_tokens,
        "llm": "LLAMA3.1.8B-Instruct",
    }

    print(f"Starting optimization for seed {seed}")
    # ディレクトリの設定
    results_path, logs_path = setup_directories(
        PROBLEM_TYPE,
        "lmpso",
        unique_id,
        seed,
        prefix=f"num_of_cities_{num_of_city}_seed",
    )
    with open(
        os.path.join(results_path, "config.json"),
        encoding="utf-8",
        mode="w",
    ) as f:
        json.dump(config, f)

    problem = TSP(cities=cities)

    for random_seed in random_seeds:
        logger = setup_logger(os.path.join(logs_path, f"seed_{random_seed}.log"))

        optimizer = LMPSO(
            problem,
            logger,
            pipe,
            seed=random_seed,
            num_particles=NUM_PARTICLES,
            max_iter=MAX_ITERS[num_of_city],
            max_new_tokens=max_new_tokens,
        )

        logger.info("Start optimization for %s", PROBLEM_TYPE)
        logger.info("Config: %s", config)

        global_best_position, global_best_value = optimizer.optimize()

        print(f"Optimization completed. Global best value: {global_best_value}")

        logger.info("Optimization completed. Global best value: %s", global_best_value)
        logger.info("Global best position: %s", global_best_position)

        # 結果の保存
        optimizer.save_history(os.path.join(results_path, f"seed_{random_seed}.csv"))
