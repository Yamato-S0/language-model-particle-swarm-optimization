from abc import ABC, abstractmethod


class OptimizationProblem(ABC):
    @abstractmethod
    def evaluate(self, position):
        """位置に対する目的関数の評価"""
        pass

    @abstractmethod
    def initialize_position(self):
        """探索空間の初期位置を生成する"""
        pass

    @abstractmethod
    def initialize_velocity(self):
        """探索空間の初期速度を生成する（解空間に応じて速度の概念が異なる場合は適宜対応）"""
        pass

    @abstractmethod
    def generate_system_content(self):
        """LMPSOにおけるLLMのmeta-promptにおけるシステムコンテンツを生成する"""
        pass

    def generate_task_instruction(self, role):
        """LMPSOにおけるLLMのmeta-promptにおけるタスク指示を生成する"""
        pass

    def generate_system_content_with_role(self, role):
        """LMPSOにおけるLLMのmeta-promptにおけるシステムコンテンツを生成する（役割によって変化）"""
        pass

    @abstractmethod
    def update_position(self, position, velocity):
        """速度に基づいて位置を更新（非数値空間でも対応可能にする）"""
        pass

    @abstractmethod
    def is_position_valid(self, position):
        """新しい位置が問題の制約に合致するかを確認する（制約付き最適化）"""
        pass

    @abstractmethod
    def handle_constraints(self, position):
        """制約違反時に位置を修正するロジック"""
        pass

    @abstractmethod
    def extract_position(self, output):
        """LLMの出力から次の位置を抽出する"""
        pass
