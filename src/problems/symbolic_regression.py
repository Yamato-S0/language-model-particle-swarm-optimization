import sympy as sp
import numpy as np
import re
import random
from sklearn.metrics import r2_score
import sys

sys.path.append("../src")

from problems.optimization_problem import OptimizationProblem


class SymbolicRegression(OptimizationProblem):
    def __init__(self, X, y, llm, logger):
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.m = X.shape[1]
        # 変数の定義
        self.variables = sp.symbols(" ".join([f"x{i}" for i in range(self.m)]))
        self.llm = llm
        self.logger = logger

    def evaluate(self, position):
        # position is a string representing a Python expression with the Sympy library
        # expr_str = self.delete_wrapper(position)
        # expr = sp.sympify(expr_str)
        # # Create a lambda function from the expression
        # f = sp.lambdify(self.variables, expr, modules="numpy")
        # # Evaluate the expression
        # y_pred = f(*self.X.T)
        # # Calculate the R2 score
        # r2 = r2_score(self.y, y_pred)
        # return -r2
        # evaluate by mean absolute error
        expr_str = self.delete_wrapper(position)
        expr = sp.sympify(expr_str)
        f = sp.lambdify(self.variables, expr, modules="numpy")
        y_pred = f(*self.X.T)
        mae = np.mean(np.abs(self.y - y_pred))
        return mae

    def initialize_position(self):
        messages = [
            {"role": "system", "content": self.generate_system_content()},
            {"role": "user", "content": self.initialize_velocity_lmpso()},
        ]
        self.logger.info("Initializing position")
        self.logger.info(f"messages: {messages}")
        print(messages)
        position = None
        num_try = 0
        max_num_try = 5
        while position is None:
            if num_try >= max_num_try:
                self.logger.error("Failed to initialize position")
                raise Exception("Failed to initialize position")

            num_try += 1
            self.logger.info(f"Initializing position (try {num_try})")

            output = self.llm.generate_output(messages, 300, 0.8)
            self.logger.info("Generated text: %s", output)
            print(output)

            try:
                position = self.extract_position(output)
                if not self.is_position_valid(position):
                    position = None

            except Exception as e:
                pass

        return position

    def initialize_velocity_lmpso(self):
        return "Briefly describe your thought process as you explore possible patterns in the data, and then present the symbolic expression within <eq>...</eq> tags. Keep your reasoning concise and to the point."

    def generate_system_content(self):
        # Xとyのインデックスをランダムにシャッフル
        if self.n >= 20:
            size = 20
        else:
            size = self.n
        indices = np.random.choice(self.n, size=size, replace=False)
        data_posints_str = display_points(self.X, self.y, indices)
        content = f"""You are a particle in a swarm of optimizers, aiming to find the best symbolic function that fits the given data. Follow the guidelines below to explore the solution space effectively.

        Guidelines:
        - Generate a short mathematical expressions in Python code using the Sympy library.
        - Use basic operations (addition, subtraction, multiplication, division) and elementary functions like sqrt, log, abs, neg, inv, sin, max, and min etc if necessary.
        - Focus on a creative, unique, and minimalistic approach to fully explore the solution space.
        - The expression should be short and simple, yet expressive and interpretable.
        - The expression should be tagged with <eq>...</eq> to indicate a mathematical expression. e.g., <eq>x0**2 + 2*x1 + 1</eq>

        Information Provided:
        - Variables: {', '.join([f"x{i}" for i in range(self.m)])}
        - Data Points: {data_posints_str}"""
        return content
        #         content = f"""You are a particle in a swarm of optimizers, aiming to find the best symbolic function that fits the given data.

        # Guidelines for your behavior as a particle:
        # 1. Your current "Personal Best" represents the best function you have found so far.
        # 2. "Global Best" represents the best function found by the swarm.
        # 3. To explore, occasionally:
        #    - Adjust your current solution slightly to refine it (local search).
        #    - Explore entirely new areas by introducing random variations (global search).
        # 4. Share insights with other particles by incorporating elements of the Global Best or Personal Best into your solution.

        # Task:
        # - Generate a symbolic function that fits the data in the form of a Python expression using the Sympy library.
        # - Strike a balance between exploiting your current knowledge (Personal Best) and exploring new possibilities.
        # - Tag your solution with <eq>...</eq>.

        # Information Provided:
        # - Variables: {', '.join([f"x{i}" for i in range(self.m)])}
        # - Data Points: {data_posints_str}"""
        # return content

    def generate_system_content_with_role(self, role):
        return self.generate_system_content()

    def is_position_valid(self, position):
        try:
            expr_str = self.delete_wrapper(position)
            expr = sp.sympify(expr_str)
            f = sp.lambdify(self.variables, expr, modules="numpy")
            y_pred = f(*self.X.T)
            mse = np.mean(np.abs(self.y - y_pred))
            # check if mse is float or int
            if not isinstance(mse, (int, float)):
                return False
            return True
        except Exception as e:
            return False

    def extract_position(self, output):
        # Extract all expressions from the output
        output_str = output.strip()
        expressions = re.findall(r"<eq>(.*?)</eq>", output_str, re.DOTALL)

        # Return the last expression if there are any matches, else None
        return expressions[-1] if expressions else None

    def generate_task_instruction(self, role):
        # return "Directly generate a new, simple symbolic function that may fit the data provided by making slight adjustments to your current function. Draw minimal inspiration from previous bests if necessary, but focus on creating a unique and creative expression that is concise and interpretable. Output only the expression, tagged with <eq>...</eq>, without additional explanation or information."
        # return "Generate a new symbolic function that may fit the data provided by making changes to your current function. Focus on creating a unique and creative expression that is concise and interpretable. The shorter the expression, the better. Briefly describe your thought process as you make adjustments to the function. The expression should be tagged with <eq>...</eq>. Remember to keep the expression concise and to the point."
        # return "Generate a new symbolic function that may fit the data provided by making slight changes to your current function. Focus on creating a unique and creative expression that is concise and interpretable to fully explore the solution space. The shorter the expression, the better. The expression should be tagged with <eq>...</eq>. Remember to keep the expression concise and to the point. Do not provide additional information or explanation."
        return """Instructions:
    - Balance between improving your current function (local search) and introducing significant variations (global search).
    - You may incorporate elements from the Global Best or other potential solutions to enhance your expression.
    - Focus on creating a unique and creative expression that is concise, interpretable, and diverse.
    - The shorter the expression, the better. Ensure that the expression is tagged with <eq>...</eq>.
    - Keep the response concise and to the point. Do not provide additional information or explanation."""

    # return "Generate a new symbolic function that may fit the data provided by making slight adjustments to your current function. Stick to your current function while exploring new possibilities. The expression should be tagged with <eq>...</eq>. Remember to keep the expression concise and to the point. Do not provide additional information or explanation."

    def add_wrapper(self, code):
        return f"<eq>{code}</eq>"

    def delete_wrapper(self, code):
        return code.replace("<eq>", "").replace("</eq>", "")

    def update_position(self, position, velocity):
        return 1

    def handle_constraints(self, position):
        return 1

    def initialize_velocity(self):
        return random.choice([0, 1, 2])


# xがn次元の時のデータポイントを適切に表示する関数
def display_points(X, y, indices):
    # ソート対象の列を指定してソート
    sorted_indices = sorted(
        indices, key=lambda i: tuple(X[i][j] for j in range(len(X[i])))
    )

    point_strs = []
    for i in sorted_indices:
        # Xの各値を条件に応じてフォーマット
        x_formatted = [
            (
                "0"
                if x == 0
                else f"{x:.2e}" if abs(x) < 0.01 else f"{x:.2f}".rstrip("0").rstrip(".")
            )
            for x in X[i]
        ]
        # yも同様にフォーマット
        y_formatted = (
            "0"
            if y[i] == 0
            else (
                f"{y[i]:.2e}"
                if abs(y[i]) < 0.01
                else f"{y[i]:.2f}".rstrip("0").rstrip(".")
            )
        )
        point_str = f"({', '.join(x_formatted)}) -> {y_formatted}"
        point_strs.append(point_str)
        print(point_str)
    return "\n".join(point_strs)
