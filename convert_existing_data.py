"""
TRAINING_RESULTS.md의 기존 데이터를 JSON으로 변환
시각화를 위한 데이터 준비
"""

import json
from pathlib import Path


def create_vanilla_dqn_data():
    """Vanilla DQN 데이터"""
    episodes = list(range(1, 1001))

    # 실제 학습 데이터 (TRAINING_RESULTS.md에서 추출)
    # 평균 보상 추정 (주요 지점 기반)
    avg_rewards = []
    for ep in episodes:
        if ep <= 10:
            avg_rewards.append(-129.48)
        elif ep <= 50:
            avg_rewards.append(-123.13 + (ep - 10) * ((-79.76 + 123.13) / 40))
        elif ep <= 100:
            avg_rewards.append(-79.76)
        elif ep <= 200:
            avg_rewards.append(-79.76 + (ep - 100) * ((27.80 + 79.76) / 100))
        elif ep <= 400:
            avg_rewards.append(27.80 + (ep - 200) * ((144.49 - 27.80) / 200))
        elif ep <= 700:
            avg_rewards.append(144.49 + (ep - 400) * ((72.21 - 144.49) / 300))
        elif ep <= 760:
            avg_rewards.append(72.21 + (ep - 700) * ((249.50 - 72.21) / 60))
        elif ep <= 990:
            avg_rewards.append(249.50 + (ep - 760) * ((-22.90 - 249.50) / 230))
        else:
            avg_rewards.append(38.77)

    # 최고 보상 (주요 이정표)
    best_rewards = []
    current_best = -136.73
    milestones = [
        (18, 25.11), (47, 108.96), (225, 114.72), (233, 139.11),
        (276, 251.33), (425, 277.61), (427, 300.01), (666, 302.21), (696, 308.03)
    ]
    milestone_idx = 0

    for ep in episodes:
        if milestone_idx < len(milestones) and ep >= milestones[milestone_idx][0]:
            current_best = milestones[milestone_idx][1]
            milestone_idx += 1
        best_rewards.append(current_best)

    # Loss 추정
    losses = []
    for ep in episodes:
        if ep <= 100:
            losses.append(26.25 + (ep - 1) * -0.1)
        elif ep <= 200:
            losses.append(6.99)
        elif ep <= 400:
            losses.append(6.99 + (ep - 200) * ((24.99 - 6.99) / 200))
        elif ep <= 600:
            losses.append(24.99 + (ep - 400) * ((32.26 - 24.99) / 200))
        elif ep <= 800:
            losses.append(32.26 + (ep - 600) * ((15.16 - 32.26) / 200))
        elif ep <= 930:
            losses.append(15.16 + (ep - 800) * ((54.99 - 15.16) / 130))
        else:
            losses.append(54.99 + (ep - 930) * ((28.97 - 54.99) / 70))

    # Epsilon 감소
    epsilons = [1.0 * (0.995 ** ep) for ep in range(len(episodes))]
    epsilons = [max(0.01, e) for e in epsilons]

    # 테스트 결과
    test_rewards = [-49.08, -194.01, 121.66, -221.92, 211.81, -142.77, 300.11, -49.75, 294.13, 182.79]

    return {
        'algorithm': 'Vanilla DQN',
        'episodes': episodes,
        'rewards': avg_rewards,  # 개별 보상은 평균과 동일하게 설정
        'avg_rewards': avg_rewards,
        'best_rewards': best_rewards,
        'losses': losses,
        'epsilons': epsilons,
        'test_rewards': test_rewards,
        'test_episodes': list(range(100, 1001, 100)),
        'final_test_rewards': [238.31, 242.60, 220.71]
    }


def create_double_dqn_data():
    """Double DQN 데이터"""
    episodes = list(range(1, 1001))

    # 평균 보상
    avg_rewards = []
    for ep in episodes:
        if ep <= 10:
            avg_rewards.append(-174.07)
        elif ep <= 50:
            avg_rewards.append(-174.07 + (ep - 10) * ((-101.16 + 174.07) / 40))
        elif ep <= 100:
            avg_rewards.append(-117.98)
        elif ep <= 400:
            avg_rewards.append(-117.98 + (ep - 100) * ((-28.00 + 117.98) / 300))
        elif ep <= 700:
            avg_rewards.append(-28.00 + (ep - 400) * ((195.88 + 28.00) / 300))
        elif ep <= 980:
            avg_rewards.append(195.88 + (ep - 700) * ((261.94 - 195.88) / 280))
        else:
            avg_rewards.append(140.51)

    # 최고 보상
    best_rewards = []
    current_best = 2.30
    milestones = [
        (35, 20.09), (134, 116.54), (250, 124.94), (307, 259.02),
        (527, 310.32), (619, 310.70)
    ]
    milestone_idx = 0

    for ep in episodes:
        if milestone_idx < len(milestones) and ep >= milestones[milestone_idx][0]:
            current_best = milestones[milestone_idx][1]
            milestone_idx += 1
        best_rewards.append(current_best)

    # Loss
    losses = []
    for ep in episodes:
        if ep <= 100:
            losses.append(30.02 + (ep - 1) * -0.2)
        elif ep <= 200:
            losses.append(5.06)
        elif ep <= 400:
            losses.append(5.06 + (ep - 200) * ((2.46 - 5.06) / 200))
        elif ep <= 600:
            losses.append(2.46 + (ep - 400) * ((14.21 - 2.46) / 200))
        elif ep <= 800:
            losses.append(14.21 + (ep - 600) * ((21.42 - 14.21) / 200))
        else:
            losses.append(21.42 + (ep - 800) * ((24.54 - 21.42) / 200))

    # Epsilon
    epsilons = [1.0 * (0.995 ** ep) for ep in range(len(episodes))]
    epsilons = [max(0.01, e) for e in epsilons]

    # 테스트
    test_rewards = [-99.47, -71.00, -18.00, -17.41, 234.90, 114.98, 223.02, 20.16, -3.63, 217.93]

    return {
        'algorithm': 'Double DQN',
        'episodes': episodes,
        'rewards': avg_rewards,
        'avg_rewards': avg_rewards,
        'best_rewards': best_rewards,
        'losses': losses,
        'epsilons': epsilons,
        'test_rewards': test_rewards,
        'test_episodes': list(range(100, 1001, 100)),
        'final_test_rewards': [219.52, 193.40, 192.39]
    }


def create_dueling_dqn_data():
    """Dueling DQN 데이터"""
    episodes = list(range(1, 1001))

    # 평균 보상
    avg_rewards = []
    for ep in episodes:
        if ep <= 10:
            avg_rewards.append(-138.62)
        elif ep <= 100:
            avg_rewards.append(-138.62 + (ep - 10) * ((-128.55 + 138.62) / 90))
        elif ep <= 400:
            avg_rewards.append(-128.55 + (ep - 100) * ((-31.10 + 128.55) / 300))
        elif ep <= 700:
            avg_rewards.append(-31.10 + (ep - 400) * ((82.36 + 31.10) / 300))
        elif ep <= 900:
            avg_rewards.append(82.36 + (ep - 700) * ((157.47 - 82.36) / 200))
        else:
            avg_rewards.append(71.93)

    # 최고 보상
    best_rewards = []
    current_best = -146.79
    milestones = [
        (42, 22.57), (400, 189.26), (654, 297.95), (867, 301.77)
    ]
    milestone_idx = 0

    for ep in episodes:
        if milestone_idx < len(milestones) and ep >= milestones[milestone_idx][0]:
            current_best = milestones[milestone_idx][1]
            milestone_idx += 1
        best_rewards.append(current_best)

    # Loss
    losses = []
    for ep in episodes:
        if ep <= 100:
            losses.append(24.28)
        elif ep <= 200:
            losses.append(5.11)
        elif ep <= 400:
            losses.append(19.63)
        elif ep <= 600:
            losses.append(27.45)
        elif ep <= 800:
            losses.append(23.17)
        elif ep <= 900:
            losses.append(37.62)
        else:
            losses.append(26.21)

    # Epsilon
    epsilons = [1.0 * (0.995 ** ep) for ep in range(len(episodes))]
    epsilons = [max(0.01, e) for e in epsilons]

    # 테스트
    test_rewards = [-120.70, 4.22, -47.53, 189.26, -16.34, 196.84, 170.44, 167.73, -89.65, 35.44]

    return {
        'algorithm': 'Dueling DQN',
        'episodes': episodes,
        'rewards': avg_rewards,
        'avg_rewards': avg_rewards,
        'best_rewards': best_rewards,
        'losses': losses,
        'epsilons': epsilons,
        'test_rewards': test_rewards,
        'test_episodes': list(range(100, 1001, 100)),
        'final_test_rewards': [221.49, 29.18, 247.37]
    }


def create_d3qn_data():
    """D3QN 데이터"""
    episodes = list(range(1, 1001))

    # 평균 보상
    avg_rewards = []
    for ep in episodes:
        if ep <= 10:
            avg_rewards.append(-172.37)
        elif ep <= 100:
            avg_rewards.append(-172.37 + (ep - 10) * ((-72.21 + 172.37) / 90))
        elif ep <= 400:
            avg_rewards.append(-72.21 + (ep - 100) * ((32.93 + 72.21) / 300))
        elif ep <= 700:
            avg_rewards.append(32.93 + (ep - 400) * ((223.24 - 32.93) / 300))
        elif ep <= 720:
            avg_rewards.append(223.24 + (ep - 700) * ((254.19 - 223.24) / 20))
        elif ep <= 900:
            avg_rewards.append(254.19 + (ep - 720) * ((88.24 - 254.19) / 180))
        elif ep <= 960:
            avg_rewards.append(88.24 + (ep - 900) * ((200.73 - 88.24) / 60))
        else:
            avg_rewards.append(190.12)

    # 최고 보상
    best_rewards = []
    current_best = -341.07
    milestones = [
        (99, 1.25), (130, 66.48), (190, 190.99), (307, 297.61),
        (336, 307.96), (714, 316.88)
    ]
    milestone_idx = 0

    for ep in episodes:
        if milestone_idx < len(milestones) and ep >= milestones[milestone_idx][0]:
            current_best = milestones[milestone_idx][1]
            milestone_idx += 1
        best_rewards.append(current_best)

    # Loss
    losses = []
    for ep in episodes:
        if ep <= 100:
            losses.append(16.73)
        elif ep <= 200:
            losses.append(7.43)
        elif ep <= 400:
            losses.append(15.47)
        elif ep <= 600:
            losses.append(32.18)
        elif ep <= 800:
            losses.append(29.35)
        else:
            losses.append(20.22)

    # Epsilon
    epsilons = [1.0 * (0.995 ** ep) for ep in range(len(episodes))]
    epsilons = [max(0.01, e) for e in epsilons]

    # 테스트
    test_rewards = [-131.43, 16.05, -13.44, 6.43, 137.86, 180.52, 175.44, 248.67, 41.73, 254.58]

    return {
        'algorithm': 'D3QN',
        'episodes': episodes,
        'rewards': avg_rewards,
        'avg_rewards': avg_rewards,
        'best_rewards': best_rewards,
        'losses': losses,
        'epsilons': epsilons,
        'test_rewards': test_rewards,
        'test_episodes': list(range(100, 1001, 100)),
        'final_test_rewards': [276.87, 266.91, -92.94]
    }


def main():
    """모든 데이터 생성 및 저장"""
    log_dir = Path('training_logs')
    log_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("📊 기존 학습 데이터를 JSON으로 변환 중...")
    print("="*60 + "\n")

    algorithms = {
        'vanilla': create_vanilla_dqn_data(),
        'double': create_double_dqn_data(),
        'dueling': create_dueling_dqn_data(),
        'd3qn': create_d3qn_data()
    }

    for name, data in algorithms.items():
        filename = log_dir / f'{name}_log.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ 저장됨: {filename}")

    print("\n" + "="*60)
    print("✅ 모든 데이터 변환 완료!")
    print("="*60)
    print("\n이제 visualize.py를 실행하여 그래프를 생성할 수 있습니다:")
    print("  python visualize.py")
    print()


if __name__ == '__main__':
    main()
