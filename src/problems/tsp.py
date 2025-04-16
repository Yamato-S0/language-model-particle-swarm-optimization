import random
import re
import numpy as np

from problems.optimization_problem import OptimizationProblem


class TravelingSalesmanProblem(OptimizationProblem):
    def __init__(self, cities):
        self.cities = cities  # 都市の座標を格納するリスト [(x1, y1), (x2, y2), ...]
        self.num_cities = len(cities)

    def evaluate(self, position):
        """訪問順序に基づいて総移動距離を評価"""
        distance = 0
        for i in range(self.num_cities - 1):
            city_a = self.cities[position[i]]
            city_b = self.cities[position[i + 1]]
            distance += np.linalg.norm(np.array(city_a) - np.array(city_b))
        # 最後の都市から最初の都市に戻る距離を加算
        distance += np.linalg.norm(
            np.array(self.cities[position[-1]]) - np.array(self.cities[position[0]])
        )
        return np.round(distance, 2)

    def initialize_position(self):
        """初期の訪問順序をランダムに生成"""
        position = list(range(self.num_cities))
        random.shuffle(position)
        return position

    def initialize_velocity(self):
        """TSPでは速度の概念が訪問順序の変化として考えられる。ここではランダムな2-opt操作を返す"""
        i, j = np.random.choice(self.num_cities, size=2, replace=False)
        return (i, j)  # 2つの都市をスワップする操作

    def initialize_velocity_lmpso(self):
        return "Generate a sequence of city indices Randomly."

    def generate_system_content(self):
        """LLM用のmeta-promptのシステムコンテンツを生成"""
        list_of_cities = ", ".join(
            [f"({i}): {city}" for i, city in enumerate(self.cities)]
        )

        system_content = f"You are given a list of points with coordinates below: {list_of_cities}. Your task is to output a new trace that is different from the previous traces and has a length lower than any of the previous traces. The trace should traverse all points exactly once."

        return system_content

    def generate_system_content_with_role(self, role):
        return self.generate_system_content()

    def generate_task_instruction(self, role):
        return "Directly output a new trace that is different from the best traces and has a length lower than any of the previous traces by changing the current trace. The trace should traverse all points exactly once. Output only the new Python list of city indices, without any explanation."

    def update_position(self, position, velocity):
        """2つの都市をスワップして新しい訪問順序を生成"""
        i, j = velocity
        new_position = position.copy()
        new_position[i], new_position[j] = new_position[j], new_position[i]
        return new_position

    def is_position_valid(self, position):
        """訪問順序が有効であるか確認（全都市を一度ずつ訪問しているか）"""
        return (
            len(set(position)) == self.num_cities
            and len(position) == self.num_cities
            and min(position) == 0
            and max(position) == self.num_cities - 1
        )

    def handle_constraints(self, position):
        """制約違反の修正（基本的には不要だが、何らかの理由で無効な順序が出力された場合に対応）"""
        if not self.is_position_valid(position):
            # 重複を削除してランダムに再配置
            unique_cities = list(set(position))
            missing_cities = list(set(range(self.num_cities)) - set(unique_cities))
            random.shuffle(missing_cities)
            return unique_cities + missing_cities
        return position

    def extract_position(self, output):
        """LLMの出力から次の訪問順序を抽出"""
        pattern = r"\[([^\]]+)\]"

        matches = re.findall(pattern, output)
        if matches:
            match = matches[0]
            try:
                city_list = [int(x) for x in re.split(r"[,\s]+", match.strip()) if x]
                if len(city_list) == self.num_cities:
                    return city_list
                else:
                    print(
                        f"Dimension mismatch: expected {self.num_cities}, got {len(city_list)}"
                    )
                    return None
            except ValueError as e:
                print(f"Error during conversion to city list: {e}")
                return None
        else:
            print(f"No matches found in output: {output}")
        return None
