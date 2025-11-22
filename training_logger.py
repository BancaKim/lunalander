"""
학습 데이터 로깅 유틸리티
시각화를 위한 학습 과정 데이터 저장
"""

import json
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """학습 과정 데이터를 기록하는 로거"""

    def __init__(self, algorithm_name):
        """
        Args:
            algorithm_name: 알고리즘 이름 (vanilla, double, dueling, d3qn)
        """
        self.algorithm = algorithm_name.lower().replace(' ', '_').replace('dqn', '').replace('_', '')
        if 'dueling' in self.algorithm and 'double' in self.algorithm:
            self.algorithm = 'd3qn'
        elif 'vanilla' not in self.algorithm:
            # vanilla, double, dueling, d3qn 중 하나로 매칭
            if 'double' in algorithm_name.lower():
                self.algorithm = 'double'
            elif 'dueling' in algorithm_name.lower():
                self.algorithm = 'dueling'

        self.data = {
            'algorithm': algorithm_name,
            'start_time': datetime.now().isoformat(),
            'episodes': [],
            'rewards': [],
            'avg_rewards': [],
            'best_rewards': [],
            'losses': [],
            'epsilons': [],
            'test_rewards': [],
            'test_episodes': [],
            'final_test_rewards': []
        }

        # 로그 디렉토리 생성
        self.log_dir = Path('training_logs')
        self.log_dir.mkdir(exist_ok=True)

    def log_episode(self, episode, reward, avg_reward, best_reward, loss, epsilon):
        """에피소드 데이터 기록"""
        self.data['episodes'].append(episode)
        self.data['rewards'].append(reward)
        self.data['avg_rewards'].append(avg_reward)
        self.data['best_rewards'].append(best_reward)
        self.data['losses'].append(loss if loss is not None else 0.0)
        self.data['epsilons'].append(epsilon)

    def log_test(self, episode, test_reward):
        """테스트 결과 기록"""
        self.data['test_episodes'].append(episode)
        self.data['test_rewards'].append(test_reward)

    def log_final_tests(self, test_rewards):
        """최종 테스트 결과 기록 (Best Model 3회)"""
        self.data['final_test_rewards'] = test_rewards

    def save(self):
        """데이터를 JSON 파일로 저장"""
        self.data['end_time'] = datetime.now().isoformat()

        filename = self.log_dir / f'{self.algorithm}_log.json'
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)

        print(f"📊 학습 데이터 저장: {filename}")

    def get_summary(self):
        """학습 요약 반환"""
        if not self.data['episodes']:
            return "No data logged yet"

        summary = f"\n{'='*60}\n"
        summary += f"학습 요약 - {self.data['algorithm']}\n"
        summary += f"{'='*60}\n"
        summary += f"에피소드: {len(self.data['episodes'])}\n"
        summary += f"최고 보상: {max(self.data['best_rewards']):.2f}\n"
        summary += f"최종 평균: {self.data['avg_rewards'][-1]:.2f}\n"
        summary += f"테스트 횟수: {len(self.data['test_rewards'])}\n"
        if self.data['final_test_rewards']:
            avg_final = sum(self.data['final_test_rewards']) / len(self.data['final_test_rewards'])
            summary += f"최종 테스트 평균: {avg_final:.2f}\n"
        summary += f"{'='*60}\n"

        return summary
