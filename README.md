# LunarLander-v3 DQN Algorithms Comparison

Gymnasium 공식 LunarLander-v3 환경에서 4개 DQN 알고리즘 비교 분석

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29.0+-green.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7+-red.svg)](https://matplotlib.org/)

---

## 🎯 프로젝트 개요

OpenAI Gymnasium의 LunarLander-v3 환경에서 4개의 DQN 알고리즘을 비교 분석합니다:
- **Vanilla DQN** - 기본 Deep Q-Network
- **Double DQN** - Q-value 과대평가 방지
- **Dueling DQN** - Value/Advantage 분리 아키텍처
- **D3QN** - Dueling + Double DQN 결합

### 주요 특징

- ✅ **4개 알고리즘 구현** - Vanilla, Double, Dueling, D3QN
- ✅ **공식 Gymnasium 환경** - 검증된 물리 엔진과 보상 시스템
- ✅ **시계열 시각화** - matplotlib 기반 학습 과정 비교 그래프
- ✅ **자동 영상 녹화** - 학습 과정 및 테스트 결과 mp4 저장
- ✅ **체크포인트 저장** - 주기적 모델 저장 및 최고 성능 모델 관리

### 학습 성과

| 알고리즘 | 최고 보상 | 최종 10-평균 | 최종 테스트 | 추천도 |
|---------|----------|-------------|------------|--------|
| **D3QN** | **316.88** 🏆 | **190.12** 🏆 | 150.28 | ⭐⭐⭐⭐⭐ |
| **Double DQN** | 310.70 | 140.51 | **201.77** 🥈 | ⭐⭐⭐⭐⭐ |
| **Vanilla DQN** | 308.03 | 38.77 | **233.87** 🥇 | ⭐⭐⭐ |
| **Dueling DQN** | 301.77 | 71.93 | 165.68 | ⭐⭐⭐ |

- ⏱️ **학습 시간**: 1000 에피소드 × 4 = 약 28분 (Apple Silicon)
- 📊 **시각화**: 6종류의 비교 그래프 제공

상세한 학습 결과는 [TRAINING_RESULTS.md](TRAINING_RESULTS.md)를 참고하세요.
시각화 가이드는 [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)를 참고하세요.

---

## 📦 설치

### 1. 저장소 클론 (또는 디렉토리 이동)

```bash
cd /Users/kimgun/Documents/lunalander
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

**requirements.txt 내용:**
```
gymnasium>=0.29.0
pygame>=2.5.0
numpy>=1.24.0
gymnasium[box2d]
moviepy>=1.0.3
opencv-python>=4.8.0
torch>=2.0.0
```

---

## 🚀 사용 방법

### 1. 빠른 테스트

공식 환경을 랜덤 에이전트로 빠르게 테스트:

```bash
python test.py
```

실시간 pygame 창에서 달 착륙선과 엔진 파티클 효과를 확인할 수 있습니다.

### 2. DQN 알고리즘 학습

#### 사용 가능한 알고리즘

```bash
# Double DQN (권장)
python train.py 1000 ddqn

# Dueling DQN
python train.py 1000 dueling

# Dueling Double DQN (최고 성능)
python train.py 1000 d3qn

# Vanilla DQN (비교용)
python train.py 1000 dqn
```

#### 빠른 테스트 (60 에피소드)

```bash
python train.py 60 ddqn
python train.py 60 dueling
python train.py 60 d3qn
```

#### 학습 과정 GUI로 관찰

```bash
# 주기적 테스트 시에도 pygame 창 표시
python train.py 500 ddqn --show-gui
python train.py 500 d3qn --show-gui
```

**알고리즘 상세 비교는 [ALGORITHMS.md](ALGORITHMS.md)를 참고하세요.**

---

## 📊 학습 결과 및 시각화

### 📈 시계열 비교 그래프

**6종류의 비교 시각화가 자동 생성됩니다:**

1. **학습 곡선** - 평균 보상 및 최고 보상 추이
2. **테스트 성능** - 매 100 에피소드 테스트 결과
3. **Loss 추이** - 학습 안정성 분석
4. **Epsilon 감소** - 탐험율 변화
5. **최종 비교** - 막대 그래프로 종합 성능 비교
6. **성공률** - 100 에피소드 단위 성공률 변화

**시각화 생성 방법:**
```bash
# 1. 기존 데이터를 JSON으로 변환
python convert_existing_data.py

# 2. 그래프 생성
python visualize.py
```

생성된 시각화는 `visualizations/` 디렉토리에 저장됩니다.
상세 가이드: [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)

---

### 최종 성과 (1000 에피소드)

**알고리즘별 비교:**

| 알고리즘 | 최고 보상 | 최종 10-평균 | 최종 테스트 | 학습 시간 |
|---------|----------|-------------|------------|----------|
| D3QN | 316.88 🏆 | 190.12 🏆 | 150.28 | ~7분 |
| Double DQN | 310.70 | 140.51 | 201.77 🥈 | ~7분 |
| Vanilla DQN | 308.03 | 38.77 | 233.87 🥇 | ~7분 |
| Dueling DQN | 301.77 | 71.93 | 165.68 | ~7분 |

**핵심 발견:**
- ⭐ **D3QN**: 가장 안정적인 학습 (190.12 최종 평균)
- ⭐ **Double DQN**: 가장 높은 최종 테스트 성공률 (100%)
- ⭐ **Vanilla DQN**: 최고 테스트 성능 (233.87) 하지만 불안정

상세한 분석은 [TRAINING_RESULTS.md](TRAINING_RESULTS.md)를 참고하세요.

---

## 📁 프로젝트 구조

```
lunalander/
├── train.py                    # 학습 스크립트 (메인)
├── test.py                     # 테스트 스크립트
├── requirements.txt            # 패키지 의존성
├── README.md                   # 이 파일
├── TRAINING_RESULTS.md         # 상세 학습 결과
├── models/                     # 저장된 모델
│   ├── best_model.pt          # 최고 성능 모델 (310.70)
│   ├── checkpoint_ep_50.pt
│   ├── checkpoint_ep_100.pt
│   └── ...
└── trained_videos/             # 학습 과정 영상
    ├── trained_ep_100_*.mp4
    ├── trained_ep_500_*.mp4
    ├── trained_ep_final_*.mp4
    └── ...
```

---

## 🎮 환경 상세

### LunarLander-v3

- **목표**: 달 착륙선을 안전하게 착륙 지점에 착륙시키기
- **관찰 공간**: 8차원 연속 벡터
  - x, y 좌표 (정규화)
  - x, y 속도
  - 각도, 각속도
  - 좌/우 다리 지면 접촉 여부 (boolean)
- **행동 공간**: 4개 이산 행동
  - 0: 아무것도 하지 않음
  - 1: 왼쪽 엔진 점화
  - 2: 메인 엔진 점화 (아래)
  - 3: 오른쪽 엔진 점화

### 보상 시스템

- **착륙 성공**: +100 ~ +140 (부드러운 착륙)
- **추락**: -100
- **엔진 사용**: 각 프레임마다 -0.3 (연료 소비)
- **거리/속도 패널티**: 착륙 지점에서 멀수록, 빠를수록 감점
- **목표**: 200+ 점수면 성공적인 착륙

---

## 🧠 알고리즘 상세

### Double DQN vs Vanilla DQN

| 특성 | Vanilla DQN | Double DQN |
|------|-------------|------------|
| Q-value 추정 | 과대평가 경향 | 더 정확한 추정 ✅ |
| 학습 안정성 | 보통 | 우수 ✅ |
| 수렴 속도 | 느림 | 빠름 ✅ |
| 권장 사용 | 학습용 | 실전 적용 ✅ |

### Double DQN 핵심 아이디어

**Vanilla DQN:**
```python
Q(s,a) ← r + γ × max_a' Q_target(s', a')
```

**Double DQN:**
```python
# 1. 온라인 네트워크로 행동 선택
a' = argmax_a' Q_online(s', a')

# 2. 타겟 네트워크로 해당 행동 평가
Q(s,a) ← r + γ × Q_target(s', a')
```

→ 행동 선택과 평가를 분리하여 과대평가 방지

---

## ⚙️ 하이퍼파라미터

학습에 사용된 주요 파라미터:

```python
Learning Rate (lr):         1e-3
Gamma (γ):                  0.99
Epsilon Start:              1.0
Epsilon End:                0.01
Epsilon Decay:              0.995
Replay Buffer Size:         10,000
Batch Size:                 64
Target Update Frequency:    10 episodes
Network Architecture:       [8, 128, 128, 4]
```

---

## 📈 학습 팁

### 권장 설정

1. **초보자**: 60 에피소드로 빠른 테스트
   ```bash
   python train.py 60 ddqn
   ```

2. **실전 사용**: 500 에피소드 (10-30분)
   ```bash
   python train.py 500 ddqn
   ```

3. **최고 성능**: 1000 에피소드 (20-60분)
   ```bash
   python train.py 1000 ddqn
   ```

### 시간 예상

| 환경 | 500 에피소드 | 1000 에피소드 |
|------|-------------|--------------|
| Apple Silicon (M1/M2/M3) | 5-15분 | 10-30분 |
| GPU (CUDA) | 5-15분 | 10-30분 |
| CPU (Intel/AMD) | 20-50분 | 40-90분 |

### 성능 향상 팁

- `--show-gui` 옵션 제거로 렌더링 오버헤드 감소
- GPU 사용 (PyTorch가 자동으로 CUDA/MPS 감지)
- 더 많은 에피소드 학습 (1000+)

---

## 🎬 영상 녹화

학습 및 테스트 과정이 자동으로 mp4 영상으로 저장됩니다:

- **주기적 테스트**: 100 에피소드마다 자동 녹화
- **최종 테스트**: 학습 완료 후 3회 테스트 녹화
- **저장 위치**: `trained_videos/`
- **프레임레이트**: 50 FPS

### 영상 파일 예시

```
trained_videos/trained_ep_100_20251121_143331.mp4
trained_videos/trained_ep_500_20251121_143523.mp4
trained_videos/trained_ep_final_0_20251121_143651.mp4
```

---

## 🔧 문제 해결

### Q: 학습이 너무 느려요
**A**: `--show-gui` 옵션을 제거하고, GPU를 사용하세요. Apple Silicon의 경우 MPS가 자동으로 활성화됩니다.

### Q: 에이전트가 학습을 못해요
**A**:
- 최소 500 에피소드 이상 학습하세요
- Double DQN(`ddqn`) 사용을 권장합니다
- 하이퍼파라미터를 조정해보세요

### Q: 파티클 효과가 안 보여요
**A**: pygame 창을 확인하세요. `--show-gui` 옵션을 추가하면 실시간으로 관찰할 수 있습니다.

### Q: 모델을 불러와서 테스트하고 싶어요
**A**: 학습 완료 후 자동으로 최종 테스트가 실행됩니다. 또는 train.py 코드에서 `agent.load("models/best_model.pt")`를 사용하세요.

---

## 📚 참고 자료

### 논문
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Double DQN Paper (van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)

### 문서
- [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 👤 작성자

**프로젝트 생성**: 2025-11-21
**학습 완료**: 2025-11-21
**최고 성능**: 310.70 (Episode 619)

---

## 🎉 다음 단계

학습이 완료되었다면:

1. ✅ [TRAINING_RESULTS.md](TRAINING_RESULTS.md)에서 상세 분석 확인
2. ✅ `trained_videos/` 폴더에서 학습 영상 시청
3. ✅ `models/best_model.pt`를 사용하여 실전 배포
4. 🚀 다른 Gymnasium 환경 도전 (CartPole, MountainCar 등)
5. 🚀 Prioritized Experience Replay, Dueling DQN 등 고급 기법 시도

Happy Landing! 🌙🚀
