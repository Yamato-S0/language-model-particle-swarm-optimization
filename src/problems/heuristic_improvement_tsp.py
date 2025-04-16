import re
import ast
import numpy as np
import random
import inspect
from problems.optimization_problem import OptimizationProblem
from utils.code_manipulation import (
    add_code_wrapper,
    delete_code_wrapper,
    run_with_timeout,
)


def generate_tsp_problem(num_cities, bounds=(0, 100)):
    """ランダムな都市座標を生成"""
    cities = [
        (
            np.random.uniform(bounds[0], bounds[1]),
            np.random.uniform(bounds[0], bounds[1]),
        )
        for _ in range(num_cities)
    ]
    return cities


class HeuristicImprovementTSP(OptimizationProblem):
    def __init__(self, datasets: dict, llm=None, logger=None):
        self.datasets = datasets
        # datasets: {dataset_name: (cities, distances)}
        self.llm = llm
        self.logger = logger

    def evaluate(self, position):
        position = delete_code_wrapper(position)
        function_name_match = re.search(r"def\s+(\w+)\s*\(", position)
        if function_name_match:
            function_name = function_name_match.group(1)
        else:
            raise ValueError("Function name not found.")

        # Run the heuristic function with the given cities and distances
        globals_dict = {"np": np, "random": random}
        locals_dict = {}
        exec(position, globals_dict, locals_dict)
        random.seed(42)
        np.random.seed(42)
        heuristic = locals_dict[function_name]
        total_distance = 0
        for key in self.datasets:
            cities = self.datasets[key]["cities"]
            distances = self.datasets[key]["distances"]
            tour = heuristic(cities, distances)
            total_distance += (
                sum(distances[tour[i], tour[i + 1]] for i in range(len(cities) - 1))
                + distances[tour[-1], tour[0]]
            )
        return total_distance

    def initialize_position(self):
        if self.llm is None:
            position = random.choice(TOUR_CONSTRUCTION_HEURISTICS)
            return add_code_wrapper(position)
        else:
            messages = [
                {"role": "system", "content": self.generate_system_content()},
                {"role": "user", "content": self.initialize_velocity_lmpso()},
            ]
            self.logger.info("Initializing position")
            self.logger.info(f"messages: {messages}")

            position = None
            num_try = 0
            max_num_try = 5
            while position is None:
                if num_try >= max_num_try:
                    self.logger.error("Failed to initialize position")
                    # raise Exception("Failed to initialize position")
                    return random.choice(TOUR_CONSTRUCTION_HEURISTICS)
                num_try += 1
                self.logger.info(f"Initializing position (try {num_try})")
                output = self.llm.generate_output(messages, 500, 0.8)
                self.logger.info("Generated text: %s", output)

                try:
                    position = self.extract_position(output)
                    if not self.is_position_valid(position):
                        position = None
                except Exception as e:
                    pass

            return position

    def generate_system_content(self):
        content = "You are tasked with generating a heuristic function for the Traveling Salesman Problem (TSP). The function must take a list of city coordinates and a distance matrix as input and return a tour of the cities as a sequence of city indices. Provide the function inside a ```python``` code block, with no additional explanation or comments."

        content += "You can use the following template to start with:\n"

        content += "```python\n"
        content += """def heuristic(cities: list[tuple[float, float]], distances: np.ndarray) -> list[int]:
        \"\"\"Generate a tour of the cities using a heuristic algorithm.\"\"\"
    Args:
        cities (list[tuple[float, float]]): A list of city coordinates.
        distances (np.ndarray): A distance matrix between the cities.

    Returns:
        list[int]: A tour of the cities as a sequence of city indices. Each city should be visited exactly once. The length of the returned list should be equal to the number of cities.
        
        tour = [i for i in range(len(cities))]
        return tour
        """
        content += "```"

        return content

    def initialize_velocity_lmpso(self):
        return "Generate a heuristic function that generates a tour of the cities using a heuristic algorithm. Only provide the Python code within a ```python``` code block. Do not include comments or explanations."

    def extract_position(self, output):
        # 余計な空白などを削除
        output = output.strip()

        # ```python```で囲まれた部分を取得
        position_match = re.search(r"```python\s*([\s\S]+?)\s*```", output)
        if position_match:
            position_code = position_match.group(1)
            self.logger.info("Extracted Code Block:\n%s", position_code)
        else:
            raise ValueError("Position code not found in the provided code")

        # print("Extracted Code Block:\n", position_code)

        tree = ast.parse(position_code)

        # treeの中のFunctionDefノードを取得
        function = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        # functionのコードを取得
        function_code = ast.unparse(function[0])

        return add_code_wrapper(function_code)

    def is_position_valid(self, position):
        position = delete_code_wrapper(position)
        function_name_match = re.search(r"def\s+(\w+)\s*\(", position)
        if function_name_match:
            function_name = function_name_match.group(1)
        else:
            self.logger.info("Function name not found in the provided code")
            return False

        globals_dict = {"np": np, "random": random}
        locals_dict = {}
        exec(position, globals_dict, locals_dict)
        random.seed(42)
        np.random.seed(42)
        heuristic = locals_dict[function_name]

        for key in self.datasets:
            cities = self.datasets[key]["cities"]
            distances = self.datasets[key]["distances"]
            try:
                tour = run_with_timeout(heuristic, (cities, distances), timeout=10)
                _ = (
                    sum(distances[tour[i], tour[i + 1]] for i in range(len(cities) - 1))
                    + distances[tour[-1], tour[0]]
                )
                # Tour should be a list of integers which contains all the cities ranging from 0 to len(cities)-1
                if not isinstance(tour, list):
                    self.logger.info("Heuristic function did not return a list")
                    return False
                if not all(isinstance(city, int) for city in tour):
                    self.logger.info(
                        "Heuristic function did not return a list of integers"
                    )
                    return False
                if set(tour) != set(range(len(cities))):
                    self.logger.info("Heuristic function did not return a valid tour")
                    return False
                if len(tour) != len(cities):
                    self.logger.info(
                        "Heuristic function did not return a tour of the correct length"
                    )
                    return False
            except Exception as e:
                self.logger.info("Heuristic function raised an exception: %s", e)
                return False
        return True

    def generate_system_content_with_role(self, role):
        return self.generate_system_content()

    def generate_task_instruction(self, role):
        return "Based on the best heuristic functions, directly generate a new heuristic function that generates a tour of the cities using a heuristic algorithm. Focus on creating a unique and creative heuristic function that may outperform the existing heuristics for the Traveling Salesman Problem (TSP). Output only the heuristic function with the ```python``` code block. Do not provide additional information or explanation."

    def update_position(self, position, velocity):
        return 1

    def handle_constraints(self, position):
        return 1

    def initialize_velocity(self):
        return random.choice([0, 1, 2])


def nearest_neighbor(
    cities: list[tuple[float, float]], distances: np.ndarray
) -> list[int]:
    num_cities = len(cities)
    unvisited = set(range(num_cities))
    current_city = random.choice(list(unvisited))
    unvisited.remove(current_city)
    tour = [current_city]
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: distances[current_city, city])
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
    return tour


NEAREST_NEIGHBOR = inspect.getsource(nearest_neighbor).replace(
    "nearest_neighbor", "heuristic"
)


def nearest_insertion(
    cities: list[tuple[float, float]], distances: np.ndarray
) -> list[int]:
    n = len(cities)
    remaining = list(range(n))
    start = 0
    nearest = min(range(1, n), key=lambda x: distances[start][x])
    tour = [start, nearest]
    remaining.remove(start)
    remaining.remove(nearest)
    while remaining:
        nearest = min(
            remaining,
            key=lambda x: min(distances[x][tour[i]] for i in range(len(tour))),
        )
        best_insertion = None
        min_increase = float("inf")
        for i in range(len(tour)):
            increase = (
                distances[tour[i]][nearest]
                + distances[nearest][tour[(i + 1) % len(tour)]]
                - distances[tour[i]][tour[(i + 1) % len(tour)]]
            )
            if increase < min_increase:
                min_increase = increase
                best_insertion = i
        tour.insert(best_insertion + 1, nearest)
        remaining.remove(nearest)

    return tour


NEAREST_INSERTION = inspect.getsource(nearest_insertion).replace(
    "nearest_insertion", "heuristic"
)


def farthest_insertion(
    cities: list[tuple[float, float]], distances: np.ndarray
) -> list[int]:
    n = len(cities)
    remaining = list(range(n))
    start = 0
    farthest = max(range(1, n), key=lambda x: distances[start][x])
    tour = [start, farthest]
    remaining.remove(start)
    remaining.remove(farthest)
    while remaining:
        farthest = max(
            remaining,
            key=lambda x: min(distances[x][tour[i]] for i in range(len(tour))),
        )
        best_insertion = None
        min_increase = float("inf")
        for i in range(len(tour)):
            increase = (
                distances[tour[i]][farthest]
                + distances[farthest][tour[(i + 1) % len(tour)]]
                - distances[tour[i]][tour[(i + 1) % len(tour)]]
            )
            if increase < min_increase:
                min_increase = increase
                best_insertion = i
        tour.insert(best_insertion + 1, farthest)
        remaining.remove(farthest)

    return tour


FARTHEST_INSERTION = inspect.getsource(farthest_insertion).replace(
    "farthest_insertion", "heuristic"
)


def random_insertion(
    cities: list[tuple[float, float]], distances: np.ndarray
) -> list[int]:
    n = len(cities)
    remaining = list(range(n))
    start = 0
    random_city = random.choice(range(1, n))
    tour = [start, random_city]
    remaining.remove(start)
    remaining.remove(random_city)
    while remaining:
        random_city = random.choice(remaining)
        best_insertion = random.choice(range(len(tour)))
        tour.insert(best_insertion, random_city)
        remaining.remove(random_city)

    return tour


RANDOM_INSERTION = inspect.getsource(random_insertion).replace(
    "random_insertion", "heuristic"
)


def two_opt(cities: list[tuple[float, float]], distances: np.ndarray) -> list[int]:
    num_cities = len(cities)
    tour = list(range(num_cities))
    best_tour = tour
    best_distance = sum(distances[tour[i], tour[i + 1]] for i in range(num_cities - 1))
    improved = True
    while improved:
        improved = False
        for i in range(num_cities - 1):
            for j in range(i + 1, num_cities):
                new_tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                new_distance = sum(
                    distances[new_tour[i], new_tour[i + 1]]
                    for i in range(num_cities - 1)
                )
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
        tour = best_tour
    return tour


TWO_OPT = inspect.getsource(two_opt).replace("two_opt", "heuristic")


def genetic_algorithm(
    cities: list[tuple[float, float]], distances: np.ndarray
) -> list[int]:
    num_cities = len(cities)
    population_size = 100
    num_generations = 100
    mutation_rate = 0.01
    elite_size = 10

    def create_individual():
        return random.sample(range(num_cities), num_cities)

    def fitness(individual):
        return (
            sum(
                distances[individual[i], individual[i + 1]]
                for i in range(num_cities - 1)
            )
            + distances[individual[-1], individual[0]]
        )

    def crossover(parent1, parent2):
        child = [-1] * num_cities
        start = random.randint(0, num_cities - 1)
        end = random.randint(start + 1, num_cities)
        child[start:end] = parent1[start:end]
        remaining = [city for city in parent2 if city not in child]
        index = 0
        for i in range(num_cities):
            if child[i] == -1:
                child[i] = remaining[index]
                index += 1
        return child

    def mutate(individual):
        if random.random() < mutation_rate:
            i, j = random.sample(range(num_cities), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    population = [create_individual() for _ in range(population_size)]
    for _ in range(num_generations):
        population = sorted(population, key=lambda x: fitness(x))
        next_generation = population[:elite_size]
        while len(next_generation) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)
        population = next_generation

    return population[0]


GENETIC_ALGORITHM = inspect.getsource(genetic_algorithm).replace(
    "genetic_algorithm", "heuristic"
)

TOUR_CONSTRUCTION_HEURISTICS = [
    NEAREST_NEIGHBOR,
    NEAREST_INSERTION,
    FARTHEST_INSERTION,
    RANDOM_INSERTION,
]

TOUR_IMPROVEMENT_HEURISTICS = [TWO_OPT, GENETIC_ALGORITHM]
