# 시계열 시각화 가이드

4개 DQN 알고리즘의 학습 과정을 Python matplotlib로 시각화한 결과입니다.

---

## 📊 생성된 시각화

### 1. 학습 곡선 - 평균 보상 및 최고 보상 추이
**파일**: `visualizations/reward_comparison.png`

- **왼쪽**: 평균 보상 추이 (10 에피소드 이동 평균)
- **오른쪽**: 최고 보상 추이
- 4개 알고리즘 비교

**핵심 인사이트**:
- D3QN이 Episode 700부터 가장 안정적인 평균 유지
- Double DQN이 가장 빠르게 안정화
- Vanilla DQN은 초반 빠르지만 후반 불안정
- Dueling DQN은 가장 느린 학습 속도

---

### 2. 테스트 성능 비교
**파일**: `visualizations/test_performance.png`

- 매 100 에피소드마다 수행한 테스트 결과
- 성공 기준 (200 보상) 표시

**핵심 인사이트**:
- Episode 500부터 대부분 알고리즘이 성공
- Vanilla DQN이 Episode 700에서 가장 높은 테스트 성능 (300.11)
- D3QN이 Episode 1000에서 254.58로 우수한 성능

---

### 3. Loss 추이 비교
**파일**: `visualizations/loss_comparison.png`

- 학습 중 Loss 변화 추이
- 이동 평균 (20 에피소드)으로 부드럽게 표시

**핵심 인사이트**:
- Double DQN과 D3QN이 가장 안정적인 Loss
- Vanilla DQN은 후반부 Loss 폭발 (54.99)
- Dueling DQN은 중간 수준의 변동성

---

### 4. Epsilon 감소 추이
**파일**: `visualizations/epsilon_comparison.png`

- 탐험율(Epsilon) 감소 곡선
- 모든 알고리즘 동일한 감소율 (0.995)

**설정**:
- Start: 1.0
- End: 0.01
- Decay: 0.995

---

### 5. 최종 성능 비교 (막대 그래프)
**파일**: `visualizations/final_comparison.png`

4개의 막대 그래프로 구성:
1. **최고 보상**: D3QN 316.88 🏆
2. **최종 10-평균**: D3QN 190.12 🏆
3. **최종 테스트 평균**: Vanilla DQN 233.87 🏆
4. **종합 점수**: 3가지 지표를 0-100으로 정규화하여 평균

---

### 6. 성공률 분석
**파일**: `visualizations/success_rate.png`

- 100 에피소드 단위로 200+ 보상 달성 비율 계산
- 학습 안정성 평가

**핵심 인사이트**:
- D3QN이 후반부 가장 높은 성공률 유지
- Double DQN도 안정적인 성공률
- Vanilla DQN은 낮은 성공률로 불안정성 확인

---

## 🚀 시각화 생성 방법

### 1단계: 기존 데이터 변환
```bash
python convert_existing_data.py
```
- TRAINING_RESULTS.md의 데이터를 JSON으로 변환
- `training_logs/` 디렉토리에 저장

### 2단계: 시각화 생성
```bash
python visualize.py
```
- 6개 시각화 파일 생성
- `visualizations/` 디렉토리에 PNG 저장

---

## 📁 파일 구조

```
lunalander/
├── visualizations/              # 생성된 시각화 이미지
│   ├── reward_comparison.png
│   ├── test_performance.png
│   ├── loss_comparison.png
│   ├── epsilon_comparison.png
│   ├── final_comparison.png
│   └── success_rate.png
├── training_logs/               # JSON 학습 데이터 (gitignore)
│   ├── vanilla_log.json
│   ├── double_log.json
│   ├── dueling_log.json
│   └── d3qn_log.json
├── convert_existing_data.py     # 데이터 변환 스크립트
├── visualize.py                 # 시각화 생성 스크립트
└── training_logger.py           # 학습 로거 유틸리티
```

---

## 🎨 커스터마이징

### 색상 변경
`visualize.py`의 `COLORS` 딕셔너리 수정:
```python
COLORS = {
    'vanilla': '#FF6B6B',  # Red
    'double': '#4ECDC4',   # Cyan
    'dueling': '#FFD93D',  # Yellow
    'd3qn': '#6BCB77'      # Green
}
```

### 새로운 시각화 추가
`visualize.py`에 함수 추가:
```python
def plot_your_visualization():
    fig, ax = plt.subplots(figsize=(14, 7))
    # ... your code ...
    plt.savefig('visualizations/your_viz.png', dpi=300)
```

---

## 📊 데이터 형식

JSON 파일 구조:
```json
{
  "algorithm": "Algorithm Name",
  "episodes": [1, 2, 3, ...],
  "rewards": [...],
  "avg_rewards": [...],
  "best_rewards": [...],
  "losses": [...],
  "epsilons": [...],
  "test_rewards": [...],
  "test_episodes": [100, 200, ...],
  "final_test_rewards": [...]
}
```

---

## 🔧 요구사항

- Python 3.11+
- matplotlib >= 3.7.0
- numpy >= 1.24.0

설치:
```bash
pip install matplotlib numpy
```

---

## 📈 주요 발견사항

### 학습 안정성 (최종 10-평균)
1. **D3QN**: 190.12 🏆 (가장 안정적)
2. **Double DQN**: 140.51
3. **Dueling DQN**: 71.93
4. **Vanilla DQN**: 38.77

### 최고 보상
1. **D3QN**: 316.88 🏆
2. **Double DQN**: 310.70
3. **Vanilla DQN**: 308.03
4. **Dueling DQN**: 301.77

### 최종 테스트 성능
1. **Vanilla DQN**: 233.87 🏆 (100% 성공)
2. **Double DQN**: 201.77 (100% 성공)
3. **Dueling DQN**: 165.68 (67% 성공)
4. **D3QN**: 150.28 (67% 성공)

---

**Generated**: 2025-11-21
**Total Visualizations**: 6
**Data Points**: ~1000 episodes × 4 algorithms = 4000
