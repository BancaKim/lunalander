# DQN Algorithms Guide

LunarLander-v3에서 사용 가능한 모든 DQN 알고리즘 변형 가이드

---

## 🎯 사용 가능한 알고리즘

### 1. Vanilla DQN
**기본 Deep Q-Network**

```bash
python train.py 1000 dqn
```

**특징:**
- 가장 간단한 구현
- Q(s,a) ← r + γ × max_a' Q_target(s', a')
- 빠른 초기 학습
- ⚠️ Q-value 과대평가 문제

**권장 사용:**
- 학습 및 프로토타입
- 알고리즘 이해 목적

---

### 2. Double DQN (DDQN)
**Q-value 과대평가 해결**

```bash
python train.py 1000 ddqn
# 또는
python train.py 1000 double_dqn
```

**특징:**
- 행동 선택과 평가 분리
- Q(s,a) ← r + γ × Q_target(s', argmax_a' Q_online(s', a'))
- 안정적인 학습
- ✅ 실전 배포 권장

**권장 사용:**
- 실전 배포
- 안정성이 중요한 경우

**논문:** van Hasselt et al. (2016) "Deep RL with Double Q-learning"

---

### 3. Dueling DQN
**Value와 Advantage 분리**

```bash
python train.py 1000 dueling
# 또는
python train.py 1000 duel
python train.py 1000 dueling_dqn
```

**특징:**
- Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
- Value stream: 상태 가치 추정
- Advantage stream: 행동의 상대적 가치
- 더 나은 상태 가치 학습

**네트워크 구조:**
```
Input (8)
    ↓
Feature Extraction (128 → 128)
    ↓
    ├─→ Value Stream (64 → 1)
    └─→ Advantage Stream (64 → 4)
         ↓
    Combine: V(s) + (A(s,a) - mean(A))
```

**권장 사용:**
- 많은 행동이 비슷한 가치를 가질 때
- 상태 가치 추정이 중요한 경우

**논문:** Wang et al. (2016) "Dueling Network Architectures for Deep RL"

---

### 4. Dueling Double DQN (D3QN) ⭐ 최고
**두 기법의 결합**

```bash
python train.py 1000 d3qn
# 또는
python train.py 1000 dueling_ddqn
python train.py 1000 dueling_double_dqn
```

**특징:**
- Dueling 아키텍처 + Double DQN 학습
- Q-value 과대평가 방지 ✅
- 더 나은 상태 가치 추정 ✅
- 가장 안정적이고 성능 좋음 ✅

**권장 사용:**
- 최고 성능을 원할 때 (강력 권장!)
- 장기 학습이 가능한 경우
- 실전 배포 (최적)

**논문:**
- Wang et al. (2016) "Dueling Network Architectures"
- van Hasselt et al. (2016) "Double Q-learning"

---

## 📊 알고리즘 비교 요약

| 알고리즘 | Q-value 정확도 | 학습 안정성 | 수렴 속도 | 복잡도 | 추천도 |
|---------|--------------|-----------|----------|--------|--------|
| Vanilla DQN | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Double DQN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Dueling DQN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| D3QN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🔬 핵심 차이점

### Q-value 계산 방식

#### Vanilla DQN & Double DQN
```python
# 단일 출력 스트림
Q(s) = [Q(s,a₀), Q(s,a₁), Q(s,a₂), Q(s,a₃)]
```

#### Dueling DQN & D3QN
```python
# 두 개의 스트림 분리
V(s) = 상태 가치 (스칼라)
A(s,a) = [A(s,a₀), A(s,a₁), A(s,a₂), A(s,a₃)]

# 결합
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
```

### Target Q-value 계산

#### Vanilla DQN & Dueling DQN
```python
# Max 연산자 사용
target_q = r + γ × max_a' Q_target(s', a')
```

#### Double DQN & D3QN
```python
# 행동 선택과 평가 분리
a_best = argmax_a' Q_online(s', a')  # 온라인으로 선택
target_q = r + γ × Q_target(s', a_best)  # 타겟으로 평가
```

---

## 💡 실전 사용 가이드

### 시나리오 1: 빠른 프로토타입
**선택:** Vanilla DQN
```bash
python train.py 500 dqn
```
- 빠른 개발
- 간단한 구현
- 초기 결과 확인

### 시나리오 2: 실전 배포 (안정성 중요)
**선택:** Double DQN
```bash
python train.py 1000 ddqn
```
- 안정적인 학습
- 일관된 성능
- 검증된 알고리즘

### 시나리오 3: 최고 성능 (시간 충분)
**선택:** D3QN ⭐
```bash
python train.py 1000 d3qn
```
- 가장 안정적
- 최고 성능
- 장기 학습 가능

### 시나리오 4: 연구 / 비교
**선택:** 전체 비교
```bash
# 모든 알고리즘 테스트
python train.py 1000 dqn
python train.py 1000 ddqn
python train.py 1000 dueling
python train.py 1000 d3qn
```

---

## 🎓 학습 예시

### 기본 학습
```bash
# Double DQN으로 500 에피소드 (권장)
python train.py 500 ddqn

# D3QN으로 1000 에피소드 (최고)
python train.py 1000 d3qn
```

### GUI 표시하며 학습
```bash
# 학습 과정 실시간 관찰
python train.py 500 ddqn --show-gui
python train.py 1000 d3qn --show-gui
```

### 빠른 테스트
```bash
# 60 에피소드로 빠른 확인
python train.py 60 dqn
python train.py 60 ddqn
python train.py 60 dueling
python train.py 60 d3qn
```

---

## 📈 예상 성능

### 1000 에피소드 학습 후

| 알고리즘 | 예상 최고 보상 | 예상 성공률 | 학습 시간 |
|---------|--------------|-----------|----------|
| Vanilla DQN | 280-310 | 60-70% | ~7분 |
| Double DQN | 300-315 | 70-80% | ~7분 |
| Dueling DQN | 290-320 | 65-75% | ~8분 |
| D3QN | **310-330** | **75-85%** | ~8분 |

*Apple Silicon 기준

---

## 🔧 하이퍼파라미터

모든 알고리즘에 동일한 하이퍼파라미터 사용:

```python
# 네트워크
hidden_dim = 128          # Vanilla/Double: [8,128,128,4]
                         # Dueling/D3QN: [8,128,128] → V(64,1) + A(64,4)

# 학습
learning_rate = 1e-3
gamma = 0.99
batch_size = 64
buffer_capacity = 10000

# 탐색
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# 업데이트
target_update_freq = 10  # episodes
```

---

## 📚 참고 논문

### DQN (2015)
- **제목:** Playing Atari with Deep Reinforcement Learning
- **저자:** Mnih et al.
- **링크:** https://arxiv.org/abs/1312.5602

### Double DQN (2016)
- **제목:** Deep Reinforcement Learning with Double Q-learning
- **저자:** van Hasselt, Guez, Silver
- **링크:** https://arxiv.org/abs/1509.06461

### Dueling DQN (2016)
- **제목:** Dueling Network Architectures for Deep Reinforcement Learning
- **저자:** Wang et al.
- **링크:** https://arxiv.org/abs/1511.06581

---

## 🎯 권장 학습 순서

### 초보자
1. **Vanilla DQN** (500 에피소드) - 기본 이해
2. **Double DQN** (1000 에피소드) - 안정성 체감
3. **결과 비교** - 차이점 분석

### 중급자
1. **Double DQN** (1000 에피소드) - 베이스라인
2. **Dueling DQN** (1000 에피소드) - 아키텍처 비교
3. **D3QN** (1000 에피소드) - 최종 성능

### 고급자 / 연구
1. 모든 알고리즘 **2000 에피소드** 학습
2. 다양한 하이퍼파라미터 조합 실험
3. 통계적 유의성 검증 (여러 seed)

---

## 💻 코드 예시

### Python 스크립트에서 직접 사용

```python
from train import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, DuelingDoubleDQNAgent
import gymnasium as gym

# 환경 생성
env = gym.make("LunarLander-v3")
state_dim = 8
action_dim = 4

# 1. Vanilla DQN
agent = DQNAgent(state_dim, action_dim)

# 2. Double DQN
agent = DoubleDQNAgent(state_dim, action_dim)

# 3. Dueling DQN
agent = DuelingDQNAgent(state_dim, action_dim)

# 4. Dueling Double DQN (D3QN)
agent = DuelingDoubleDQNAgent(state_dim, action_dim)

# 학습 루프
for episode in range(1000):
    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        agent.train_step()

        obs = next_obs

    # 주기적으로 target network 업데이트
    if episode % 10 == 0:
        agent.update_target_network()
```

---

## 🏆 최종 권장 사항

### 일반 사용자
**선택:** Double DQN (ddqn)
- 안정적이고 검증됨
- 빠른 수렴
- 좋은 성능

```bash
python train.py 1000 ddqn
```

### 최고 성능 추구
**선택:** D3QN
- 가장 안정적
- 최고 성능
- 상태 가치 학습 우수

```bash
python train.py 1000 d3qn
```

### 학습 / 연구
**선택:** 모두 비교
- 알고리즘 특성 이해
- 성능 차이 분석
- 논문 작성용

```bash
for algo in dqn ddqn dueling d3qn; do
    python train.py 1000 $algo
done
```

---

**Updated:** 2025-11-21
**Available Algorithms:** 4 (DQN, Double DQN, Dueling DQN, D3QN)
**Recommendation:** D3QN for best results, DDQN for stability 🚀
