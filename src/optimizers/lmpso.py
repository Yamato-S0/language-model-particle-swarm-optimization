import numpy as np
import pandas as pd
import random
import torch


from llms.languange_models import LargeLanguageModel


class Particle:
    def __init__(
        self, inertia_length, problem, logger, language_model: LargeLanguageModel
    ):
        self.logger = logger
        self.position = problem.initialize_position()
        self.velocity = problem.initialize_velocity_lmpso()
        self.previous_velocity = None
        self.inertia_length = inertia_length
        self.best_position = self.position
        self.best_value = float("inf")
        self.problem = problem
        self.llm = language_model
        self.inertia = []

    def update_velocity(self, global_best_position):
        self.previous_velocity = self.velocity
        if self.inertia_length > 0:
            self.inertia.append({"role": "user", "content": str(self.velocity)})
            if len(self.inertia) >= self.inertia_length * 2 - 1:
                self.inertia = self.inertia[-(self.inertia_length * 2 - 1) :]

        # パーソナルベストとグローバルベストの情報を元に速度を更新
        description = "Below are personal and global best. \n"
        # description = ""
        cognitive_velocity = generate_cognitive_velocity(self.best_position)
        social_velocity = generate_social_velocity(global_best_position)

        task_instruction = self.problem.generate_task_instruction(role="assistant")

        self.velocity = (
            description + cognitive_velocity + social_velocity + task_instruction
        )

        self.logger.info("Velocity updated: %s", self.velocity)

    def update_position(self, max_new_tokens, temperature, role="assistant"):
        messages = self.build_messages(role)
        self.logger.info("Messages: %s", messages)
        position = None
        num_try = 0
        while position is None:
            if num_try >= 5:
                self.logger.error("Failed to generate next position.")
                position = self.problem.initialize_position()
                break

            num_try += 1
            self.logger.info("Try %s to generate next position.", num_try)

            output = self.llm.generate_output(messages, max_new_tokens, temperature)
            self.logger.info("Generated text: %s", output)

            try:
                position = self.problem.extract_position(output)
                if not self.problem.is_position_valid(position):
                    position = None
            except:
                pass

        self.logger.info("Position Updated: %s", position)
        self.position = position
        self.inertia.append({"role": "assistant", "content": str(self.position)})

    def evaluate(self):
        self.logger.info("Evaluating Position: %s", self.position)
        value = self.problem.evaluate(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = np.copy(self.position)
        elif value == self.best_value:
            if random.random() < 0.5:
                self.best_value = value
                self.best_position = np.copy(self.position)
        self.logger.info("Value: %s", value)
        return value

    def build_messages(self, role="assistant"):
        system_content = self.problem.generate_system_content_with_role(role)
        messages = [{"role": "system", "content": system_content}]

        for inertia in self.inertia:
            if inertia is not None:
                messages.append(inertia)

        messages.append({"role": "assistant", "content": str(self.position)})
        messages.append({"role": "user", "content": str(self.velocity)})

        self.inertia.append({"role": "assistant", "content": str(self.position)})
        self.inertia.append({"role": "user", "content": str(self.velocity)})

        return messages


def generate_cognitive_velocity(best_position):
    # 最良の位置のみの情報を文字列に変換
    velocity = f"Personal Best: {best_position}\n"
    return velocity


def generate_social_velocity(global_best_position):
    # 最良の位置のみの情報を文字列に変換
    velocity = f"Global Best: {global_best_position}\n"
    return velocity


def generate_messages(system_content, inertia, position, velocity, task_instruction):
    if inertia is None:
        messages = [
            {"role": "system", "content": system_content},
            # positionを文字列に変換
            {"role": "assistant", "content": str(position)},
            # velocityも文字列に変換して結合
            {"role": "user", "content": str(velocity) + task_instruction},
        ]

    else:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(inertia)},
            # positionを文字列に変換
            {"role": "assistant", "content": str(position)},
            # velocityも文字列に変換して結合
            {
                "role": "user",
                "content": str(inertia) + str(velocity) + task_instruction,
            },
        ]

    return messages


class LMPSO:
    def __init__(
        self,
        problem,
        logger,
        language_model,
        seed,
        num_particles=10,
        max_iter=100,
        max_new_tokens=100,
        inertia_length=1,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.problem = problem
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.max_new_tokens = max_new_tokens
        self.particles = [
            Particle(inertia_length, problem, logger, language_model)
            for _ in range(num_particles)
        ]
        self.global_best_position = None
        self.global_best_value = float("inf")
        self.history = []
        self.logger = logger

    def optimize(self):
        iteration_without_improvement = 0
        for iteration in range(self.max_iter):
            self.logger.info("Iteration: %s", iteration)
            self.logger.info(
                "Iteration without improvement: %s", iteration_without_improvement
            )
            iteration_data = []

            self.logger.info("Evaluating particles")
            for i, particle in enumerate(self.particles):
                self.logger.info("Particle No.: %s", i)
                value = particle.evaluate()

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = np.copy(particle.position)
                    iteration_without_improvement = -1
                elif value == self.global_best_value:
                    if random.random() < 0.5:
                        self.global_best_value = value
                        self.global_best_position = np.copy(particle.position)

                iteration_data.append(
                    {
                        "position": np.copy(particle.position),
                        "velocity": np.copy(particle.velocity),
                        "score": value,
                    }
                )
            iteration_without_improvement += 1
            self.logger.info("Global best value: %s", self.global_best_value)
            self.logger.info("Global best position: %s", self.global_best_position)
            self.history.append(iteration_data)

            self.logger.info("Updating particles")
            temperature = 0.9
            for i, particle in enumerate(self.particles):
                self.logger.info("Particle No.: %s", i)
                particle.update_velocity(
                    self.global_best_position,
                )
                role = "assistant"
                particle.update_position(self.max_new_tokens, temperature, role)

        return self.global_best_position, self.global_best_value

    def save_history(self, filename):
        # 履歴をDataFrameとして保存
        flattened_history = []
        for iteration, particles_data in enumerate(self.history):
            for particle_id, data in enumerate(particles_data):
                flattened_history.append(
                    [
                        iteration,
                        particle_id,
                        data["position"],  # 配列のまま保存
                        data["velocity"],  # 配列のまま保存
                        data["score"],
                    ]
                )

        # データフレーム化（配列をそのまま保存）
        df = pd.DataFrame(
            flattened_history,
            columns=["Iteration", "Particle", "Position", "Velocity", "Score"],
        )

        # CSVファイルとして保存
        df.to_csv(filename, index=False)

        return df
