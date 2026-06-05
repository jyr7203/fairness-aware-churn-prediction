# Fairness-Aware Health Insurance Churn Prediction
### Orthogonalization + Reweighting

보험 해지(이탈) 예측 모델의 **그룹 공정성(group fairness)** 을 개선하기 위한 분석 프로젝트입니다.
민감변수(나이·성별 및 거주지/소득 대리변수)에 따라 발생하는 **거짓양성률 격차(False Positive Rate Gap, FPR Gap)** 를
두 가지 전처리(pre-processing) 기법으로 완화합니다.

---

## 핵심 아이디어

| 기법 | 목적 | 방법 |
|------|------|------|
| **Orthogonalization (직교화)** | 피처에 숨은 *간접차별* 경로 차단 | 각 피처를 민감변수 기저(age 더미 · gender · 교호작용 · age 연속형 · age²)로 Ridge 회귀한 뒤, 설명되는 성분을 제거하고 잔차만 사용 |
| **Reweighting (재가중)** | 집단 간 학습 불균형 보정 | `age_group × gender` 교차 그룹의 역빈도 표본 가중치 `N / (n_cells × cell_count)` 를 학습에 적용 |

- 직교화 임계값 `R²`는 **PR-AUC 하락 2% 이내** 제약을 만족하면서 **Avg FPR Gap이 최소**가 되는 값을 Pareto 탐색으로 선택합니다.
- 분류 임계값(threshold)은 Base 모델의 validation(2018년) F1 최댓값 지점을 **모든 민감변수에 공통 고정**하여, 공정성 개선이 임계값 조정이 아니라 학습 단계 효과에서 나오도록 통제합니다.
- 동일한 절차를 **XGBoost** 와 **GLM(로지스틱 회귀)** 두 모델에 적용합니다.

---

## 저장소 구성

| 파일 | 설명 |
|------|------|
| `ortho_rwh.ipynb` | 전체 분석 노트북 (XGBoost · GLM 직교화/재가중, 공정성 평가) |
| `requirements.txt` | 의존 패키지 목록 |
| `churn_model_xgb_final.pkl` | 학습된 XGBoost 모델 패키지 |
| `churn_model_glm_final.pkl` | 학습된 GLM 모델 패키지 |

> **데이터 미포함:** 원본 보험 계약 데이터(`*.pkl`)로 저장소에 포함하지 않습니다.

---

## 실행 방법

```bash
pip install -r requirements.txt
```

1. 노트북 상단 **`3. 파일 경로 설정`** 셀의 `BASE_PATH` **한 곳만** 본인 환경에 맞게 수정합니다.
   (모델·데이터·GLM 경로가 모두 `BASE_PATH`에서 파생됩니다.)
2. 위에서부터 순서대로 실행합니다. (Google Colab + GPU 환경 기준으로 작성)

---

## 주요 결과 (XGBoost, Base → Orth+RW)

테스트셋(2019년) 기준 민감변수별 FPR Gap이 크게 감소했습니다.

| 민감변수 | Base FPR Gap | Orth+RW FPR Gap | 개선율 |
|----------|:---:|:---:|:---:|
| age | 0.3083 | 0.0508 | **83.5%** |
| gender | 0.0348 | 0.0000 | **99.9%** |
| age × gender | 0.2784 | 0.0402 | **85.6%** |
| C_H (거주지) | 0.2344 | 0.1182 | 49.6% |
| C_GI (소득) | 0.1927 | 0.1243 | 35.5% |
| C_IE_T | 0.1668 | 0.0922 | 44.7% |

예측 성능(PR-AUC)은 Base 대비 2% 이내로 유지되어, **성능을 거의 희생하지 않고 공정성을 개선**합니다.

---

## 공정성 지표

- **FPR Gap**: 한 민감변수의 그룹들 사이 거짓양성률 최댓값 − 최솟값. (해지하지 않을 고객을 해지로 오분류하는 부담이 집단 간에 얼마나 불균등한지를 측정)
- 그룹별 FPR은 표본 ≥ 10이고 양·음성이 모두 존재하는 그룹에 대해서만 계산합니다.
