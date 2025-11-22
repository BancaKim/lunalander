"""
LunarLander-v3 DQN Algorithms - 시계열 시각화
4개 알고리즘의 학습 과정을 비교하는 그래프 생성
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 알고리즘 색상 정의
COLORS = {
    'vanilla': '#FF6B6B',      # Red
    'double': '#4ECDC4',       # Cyan
    'dueling': '#FFD93D',      # Yellow
    'd3qn': '#6BCB77'          # Green
}

LABELS = {
    'vanilla': 'Vanilla DQN',
    'double': 'Double DQN',
    'dueling': 'Dueling DQN',
    'd3qn': 'D3QN'
}


def load_training_data(algorithm):
    """알고리즘별 학습 데이터 로드"""
    data_file = Path(f'training_logs/{algorithm}_log.json')
    if data_file.exists():
        with open(data_file, 'r') as f:
            return json.load(f)
    return None


def plot_reward_comparison():
    """평균 보상 비교 그래프"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for algo in ['vanilla', 'double', 'dueling', 'd3qn']:
        data = load_training_data(algo)
        if data is None:
            continue

        episodes = data['episodes']
        avg_rewards = data['avg_rewards']

        # 이동 평균 계산 (50 에피소드)
        window = 50
        if len(avg_rewards) >= window:
            moving_avg = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
            moving_episodes = episodes[window-1:]
        else:
            moving_avg = avg_rewards
            moving_episodes = episodes

        # 원본 데이터 (투명)
        ax1.plot(episodes, avg_rewards, color=COLORS[algo], alpha=0.2, linewidth=0.5)
        # 이동 평균 (진하게)
        ax1.plot(moving_episodes, moving_avg, color=COLORS[algo],
                label=LABELS[algo], linewidth=2)

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Average Reward (10 episodes)', fontsize=12)
    ax1.set_title('학습 곡선 - 평균 보상 추이', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='목표 (200+)')

    # 최고 보상 비교
    for algo in ['vanilla', 'double', 'dueling', 'd3qn']:
        data = load_training_data(algo)
        if data is None:
            continue

        episodes = data['episodes']
        best_rewards = data['best_rewards']

        ax2.plot(episodes, best_rewards, color=COLORS[algo],
                label=LABELS[algo], linewidth=2)

    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Best Reward', fontsize=12)
    ax2.set_title('최고 보상 추이', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=300, color='red', linestyle='--', alpha=0.5, label='목표 (300+)')

    plt.tight_layout()
    plt.savefig('visualizations/reward_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 저장됨: visualizations/reward_comparison.png")
    plt.close()


def plot_test_performance():
    """테스트 성능 비교 (매 100 에피소드)"""
    fig, ax = plt.subplots(figsize=(14, 7))

    test_episodes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    for algo in ['vanilla', 'double', 'dueling', 'd3qn']:
        data = load_training_data(algo)
        if data is None:
            continue

        test_rewards = data.get('test_rewards', [])
        if len(test_rewards) > 0:
            ax.plot(test_episodes[:len(test_rewards)], test_rewards,
                   color=COLORS[algo], marker='o', markersize=8,
                   label=LABELS[algo], linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Test Reward', fontsize=12)
    ax.set_title('테스트 성능 비교 (매 100 에피소드)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=2, label='성공 기준 (200)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig('visualizations/test_performance.png', dpi=300, bbox_inches='tight')
    print("✅ 저장됨: visualizations/test_performance.png")
    plt.close()


def plot_loss_comparison():
    """Loss 추이 비교"""
    fig, ax = plt.subplots(figsize=(14, 7))

    for algo in ['vanilla', 'double', 'dueling', 'd3qn']:
        data = load_training_data(algo)
        if data is None:
            continue

        episodes = data['episodes']
        losses = data.get('losses', [])

        if len(losses) > 0:
            # 이동 평균으로 부드럽게
            window = 20
            if len(losses) >= window:
                moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
                moving_episodes = episodes[window-1:]
            else:
                moving_avg = losses
                moving_episodes = episodes

            ax.plot(episodes, losses, color=COLORS[algo], alpha=0.2, linewidth=0.5)
            ax.plot(moving_episodes, moving_avg, color=COLORS[algo],
                   label=LABELS[algo], linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss 추이 비교', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('visualizations/loss_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 저장됨: visualizations/loss_comparison.png")
    plt.close()


def plot_epsilon_comparison():
    """Epsilon 감소 추이"""
    fig, ax = plt.subplots(figsize=(14, 7))

    for algo in ['vanilla', 'double', 'dueling', 'd3qn']:
        data = load_training_data(algo)
        if data is None:
            continue

        episodes = data['episodes']
        epsilons = data.get('epsilons', [])

        if len(epsilons) > 0:
            ax.plot(episodes, epsilons, color=COLORS[algo],
                   label=LABELS[algo], linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax.set_title('Epsilon 감소 추이', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('visualizations/epsilon_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 저장됨: visualizations/epsilon_comparison.png")
    plt.close()


def plot_final_comparison():
    """최종 성능 비교 (막대 그래프)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    algorithms = []
    best_rewards = []
    final_avg_rewards = []
    final_test_rewards = []

    for algo in ['vanilla', 'double', 'dueling', 'd3qn']:
        data = load_training_data(algo)
        if data is None:
            continue

        algorithms.append(LABELS[algo])
        best_rewards.append(max(data['best_rewards']))
        final_avg_rewards.append(data['avg_rewards'][-1] if data['avg_rewards'] else 0)

        # 최종 3회 테스트 평균
        final_tests = data.get('final_test_rewards', [])
        final_test_rewards.append(np.mean(final_tests) if final_tests else 0)

    x_pos = np.arange(len(algorithms))
    colors = [COLORS[algo] for algo in ['vanilla', 'double', 'dueling', 'd3qn'][:len(algorithms)]]

    # 최고 보상
    bars1 = ax1.bar(x_pos, best_rewards, color=colors, alpha=0.8)
    ax1.set_ylabel('Best Reward', fontsize=11)
    ax1.set_title('최고 보상 비교', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms)
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(best_rewards):
        ax1.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # 최종 10-평균
    bars2 = ax2.bar(x_pos, final_avg_rewards, color=colors, alpha=0.8)
    ax2.set_ylabel('Final 10-Episode Average', fontsize=11)
    ax2.set_title('최종 10-에피소드 평균 비교', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algorithms)
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(final_avg_rewards):
        ax2.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # 최종 테스트 평균
    bars3 = ax3.bar(x_pos, final_test_rewards, color=colors, alpha=0.8)
    ax3.set_ylabel('Final Test Average', fontsize=11)
    ax3.set_title('최종 테스트 평균 (Best Model 3회)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=2)
    for i, v in enumerate(final_test_rewards):
        ax3.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # 종합 점수 (표준화)
    if len(algorithms) > 0:
        # 각 지표를 0-100으로 정규화
        norm_best = [(x / max(best_rewards)) * 100 for x in best_rewards]
        norm_avg = [(x / max(final_avg_rewards)) * 100 if max(final_avg_rewards) > 0 else 0 for x in final_avg_rewards]
        norm_test = [(x / max(final_test_rewards)) * 100 if max(final_test_rewards) > 0 else 0 for x in final_test_rewards]

        total_scores = [(a + b + c) / 3 for a, b, c in zip(norm_best, norm_avg, norm_test)]

        bars4 = ax4.bar(x_pos, total_scores, color=colors, alpha=0.8)
        ax4.set_ylabel('종합 점수 (0-100)', fontsize=11)
        ax4.set_title('종합 성능 비교', fontsize=13, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(algorithms)
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim(0, 105)
        for i, v in enumerate(total_scores):
            ax4.text(i, v + 2, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/final_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 저장됨: visualizations/final_comparison.png")
    plt.close()


def plot_success_rate():
    """성공률 분석 (200+ 보상 달성 비율)"""
    fig, ax = plt.subplots(figsize=(14, 7))

    for algo in ['vanilla', 'double', 'dueling', 'd3qn']:
        data = load_training_data(algo)
        if data is None:
            continue

        episodes = data['episodes']
        avg_rewards = data['avg_rewards']

        # 100 에피소드 단위로 성공률 계산
        window = 100
        success_rates = []
        success_episodes = []

        for i in range(0, len(avg_rewards), window):
            end_idx = min(i + window, len(avg_rewards))
            window_rewards = avg_rewards[i:end_idx]
            success_rate = sum(1 for r in window_rewards if r >= 200) / len(window_rewards) * 100
            success_rates.append(success_rate)
            success_episodes.append(episodes[end_idx - 1])

        ax.plot(success_episodes, success_rates, color=COLORS[algo],
               marker='o', markersize=6, label=LABELS[algo], linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('성공률 추이 (200+ 보상 달성 비율, 100 에피소드 단위)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, linewidth=2, label='목표 (80%)')

    plt.tight_layout()
    plt.savefig('visualizations/success_rate.png', dpi=300, bbox_inches='tight')
    print("✅ 저장됨: visualizations/success_rate.png")
    plt.close()


def create_all_visualizations():
    """모든 시각화 생성"""
    # 저장 디렉토리 생성
    Path('visualizations').mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("📊 시각화 생성 중...")
    print("="*60)

    try:
        plot_reward_comparison()
        plot_test_performance()
        plot_loss_comparison()
        plot_epsilon_comparison()
        plot_final_comparison()
        plot_success_rate()

        print("\n" + "="*60)
        print("✅ 모든 시각화 완료!")
        print("="*60)
        print("\n생성된 파일:")
        print("  - visualizations/reward_comparison.png")
        print("  - visualizations/test_performance.png")
        print("  - visualizations/loss_comparison.png")
        print("  - visualizations/epsilon_comparison.png")
        print("  - visualizations/final_comparison.png")
        print("  - visualizations/success_rate.png")
        print()

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    create_all_visualizations()
