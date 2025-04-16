import os
import sys
import json
import numpy as np
import random

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

from problems.heuristic_improvement_tsp import (
    TOUR_CONSTRUCTION_HEURISTICS,
    TravelingSalesmanHeuristic,
)

if __name__ == "__main__":

    def generate_tsp_problem(num_cities, bounds=(0, 100), seed=None):
        random.seed(seed)
        np.random.seed(seed)
        """ランダムな都市座標を生成"""
        cities = set()
        while len(cities) < num_cities:
            cities.add(
                (
                    random.randint(bounds[0], bounds[1]),
                    random.randint(bounds[0], bounds[1]),
                )
            )
        cities = list(cities)

        distances = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distances[i][j] = np.sqrt(
                    (cities[i][0] - cities[j][0]) ** 2
                    + (cities[i][1] - cities[j][1]) ** 2
                )
        return {"cities": cities, "distances": distances}

    seed = random.randint(0, 100000)

    NUM_PARTICLES = 25
    MAX_ITER = 40

    PROBLEM_TYPE = "tsp_heuristic"
    BOUNDS = (0, 100)
    DATASETS = {
        "0": generate_tsp_problem(100, BOUNDS, 0),
        "1": generate_tsp_problem(100, BOUNDS, 1),
        "2": generate_tsp_problem(100, BOUNDS, 2),
        "3": generate_tsp_problem(100, BOUNDS, 3),
        "4": generate_tsp_problem(100, BOUNDS, 4),
    }

    unique_id = "20241129_233000"
    results_path, logs_path = setup_directories(
        PROBLEM_TYPE, "lmpso", unique_id, "num_cities_100", prefix=None
    )
    max_new_tokens = 1000
    inertia_length = 1
    llm = LLAMA3_1_8B_Instruct()
    logger = setup_logger(os.path.join(logs_path, f"seed_{seed}.log"))
    problem = TravelingSalesmanHeuristic(datasets=DATASETS, logger=logger)
    config = {
        "num_particles": NUM_PARTICLES,
        "max_iter": MAX_ITER,
        "problem_type": PROBLEM_TYPE,
        "max_new_tokens": max_new_tokens,
        "inertia_length": inertia_length,
        "llm": "LLAMA3.1.8B-Instruct",
        "datasets": DATASETS,
        "initlization": TOUR_CONSTRUCTION_HEURISTICS,
    }

    # Custom function to handle NumPy arrays during JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(
        os.path.join(results_path, "config.json"), encoding="utf-8", mode="w"
    ) as f:
        json.dump(config, f, default=convert_to_serializable)

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

    logger.info("Starting optimization for %s", PROBLEM_TYPE)
    logger.info("Config: %s", config)

    global_best_position, global_best_value = optimizer.optimize()

    logger.info("Optimization completed. Global best value: %s", global_best_value)
    logger.info("Global best position: %s", global_best_position)

    optimizer.save_history(os.path.join(results_path, f"seed_{seed}.csv"))
    print(f"Optimization completed. Global best value: {global_best_value}")
