# 🏨 LG AIMers 해커톤 Phase 2 - Resort F&B 수요예측

본 리포지토리는 2025년 **LG AIMers AI 해커톤** Phase 2(온라인)에서 수행한  
리조트 식음업장 메뉴별 수요 예측 과제(Task)에 대한 문제 정의, 데이터 분석, 모델 설계, 결과 및 회고를 정리한 문서입니다.  

- **Phase 1**: 온라인 코딩 테스트 (개별)  
- **Phase 2**: 온라인 데이터 분석 & 모델 개발 (본 리포지토리)  
- **Phase 3**: 오프라인 본선 (주피터 환경, 코드 반출 불가 → 설명 위주)  

---

## 🧠 Overview

| 항목 | 내용 |
|------|------|
| 주최 | LG |
| 주제 | 리조트 식음업장 메뉴별 1주일 수요 예측 |
| 개발 환경 | Python / Jupyter Notebook / GPU |
| 팀 인원 | 2명 (전다함, 신준화) |
| 데이터 | 일별 메뉴별 판매량 + 업장/객실/날씨/단체예약 메타데이터 |
| 평가 방식 | SMAPE (메인), NMAE / NRMSE / R² (참고)<br>Public 50% + Private 100% |
| 성과 | 1500명 중 Phase 2 통과 → 100명 Phase 3 본선 진출 |

---

## 🧰 Tech Stack

- **Language & Framework**  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)  
  ![AutoGluon](https://img.shields.io/badge/AutoGluon-0099CC?style=for-the-badge&logo=amazonaws&logoColor=white)  
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

- **Data & Visualization**  
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)  
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)  
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)

---

## 🍽 Task: Resort F&B 메뉴별 수요 예측

### 📌 목표
28일간의 입력 데이터를 기반으로 **향후 7일간 메뉴별 수요(매출수량)**를 예측.  
업장별 가중치(특히 **담하, 미라시아**)가 적용된 **SMAPE**를 최종 평가 지표로 사용.

---

## 🗂 데이터 구성 (Phase 3 기준)

- **train.csv**: 2023-01-01 ~ 2024-06-15 메뉴별 일별 매출수량  
- **test/**: TEST_00~09 (각 28일 입력, 7일 예측 대상)  
- **meta/**: group(단체), hwadam(방문객), room(객실 판매), ski(내장객), weather(날씨)  
- **price.csv**: 메뉴별 평균 단가  
- **room_type.csv**: 객실 타입/기준 인원  
- **sample_submission.csv**: 제출 포맷

---

### 🔨 최종 사용 기법 (온라인 · Phase 2)

#### 1) 데이터 정리 & 출시 전(리드인) 0 제거
- `train.csv` → 표준 컬럼으로 정리: `date, store_menu, qty`
- `store_menu`를 `store`, `menu`로 분리
- **출시일(release_date)** = 해당 `store_menu`의 **최초 양수 판매일**  
  → 출시일 이전의 0 매출(리드인 구간)은 **학습에서 제외**해 cold-start 노이즈 제거

#### 2) 캘린더·계절성 파생
- 주차·월·요일 플래그: `dayofweek(0~6), month, is_weekend, is_fri/sat/sun`
- **연주기(365.25일)** 사인/코사인: `doy_sin, doy_cos`
- **고정 특일(01-01, 12-24, 12-25, 12-31)**  
  - 태그/플래그: `is_special_any`  
  - 앞/뒤 특일까지의 거리, 최소거리, ±2일 윈도우:  
    `days_to_special, days_from_special, min_days_to_special, is_special_window_k2`

#### 3) 메뉴 토큰화 & 매장별 Top-1 키워드 신호
- **토큰화 규칙**
  - 메뉴명 전체 + 괄호 내부 텍스트에서 한글/영문/숫자 토큰 추출 (`[0-9A-Za-z가-힣]+`)
  - 마지막 단어(강조) 우선 포함, 소문자 정규화
- **매장별 Top-1 토큰 추정(Keyword Signal)**
  - 매장 단위로 날짜×토큰 수요 행렬을 만들고, **Ridge(α=5.0)**로  
    “**다른 메뉴 합계**”를 타깃으로 회귀 → **|β|가 가장 큰 토큰 = Top-1**
  - 최소 조건: 매장 레코드 수≥50, 토큰 등장일≥3
  - 결과를 `store_top1_token_beta.csv`로 저장하여 이후 학습/추론에 사용

#### 4) 28→7 수퍼바이즈드 표본 생성 (학습용)
- 각 `store_menu`에 대해 **최근 28일(hist)**을 입력 윈도우로 사용,  
  그 다음 **7일(y1..y7)**을 타깃으로 생성
- **기본 통계/시계열 피처**
  - `last, mean7, mean28, std7, trend=(last+1)/(mean7+1)`
  - `lag7, lag14, rm3, rm14, rs14`
  - 0 패턴: `zero_streak_28, days_since_nz, zero_ratio28`
  - 스케일/분산: `mean7_over_mean28, vol14=rs14/(rm14+1e-6)`
- **히스토리 앵커 기반 수요 평균**
  - 28일 히스토리만 사용해 **요일/월 평균**을 계산  
    → `sm_dow_mean_next, sm_month_mean_next`
- **매장 집계 신호**
  - **Top-1 토큰 28일 합**: `store_top1tok_sum28`  
  - **매장 총수요 28일 합**: `store_total_sum28`  
  - 주말×토큰 상호작용: `wknd_x_top1`
- **날짜 파생(예측일 기준)**
  - `dow_next, is_wknd_next, is_fri_next/is_sat_next/is_sun_next, month_next, h(1~7)`
  - `doy_next, doy_sin_next, doy_cos_next`
  - 특일 플래그/거리(`…_next` 일체)

> ⚠️ **데이터 누수 방지**: 모든 피처는 **해당 28일 히스토리 내부**에서만 계산하고,  
> 테스트와 동일 규칙으로 생성되도록 고정.

#### 5) 검증 로직 (Last-window, 테스트와 1:1 매칭)
- `build_last_window_val()`을 **패치**하여,  
  검증 피처 생성이 테스트 로직과 **완전히 동일**하게 동작하도록 수정
  - 앵커 평균(요일/월), 특일 거리·플래그, Top-1토큰/총수요 28일 합 모두 **최근 28일**로 산출
- 목적: **학습-검증-테스트** 간 **특징 정의 불일치** 제거 → 신뢰도 높은 오프라인 성능 추정

#### 6) 가중치·타깃 보정
- **매장 가중치**: `담하`, `미라시아` 샘플의 `w=2.0`, 그 외 `w=1.0`
- 타깃 `y` 음수는 0으로 클리핑, `±∞`는 NaN→적절 처리

#### 7) 모델링 (AutoGluon Tabular · Fast 5 Ensemble + Stack 1)
- **학습 설정**
  - Label: `y`, Metric: `smape`, `sample_weight='w'`, `weight_evaluation=True`
  - `num_bag_folds=3`, `num_stack_levels=1`, `time_limit=1200s`
- **하이퍼파라미터 묶음**
  1) **LightGBM Quantile(α=0.80)**: 롱테일·하한제약 대응  
  2) **LightGBM GOSS**: 하드 샘플 가속 부스팅  
  3) **LightGBM Tweedie(v=1.3)**: 제로·롱테일 분포 대응  
  4) **XGBoost(tree_method=hist)**: 빠른 히스토그램 분할  
  5) **ExtraTrees**: 고분산 랜덤 포리스트형 베이스
- **단조 제약(Monotone Constraints)**
  - `last`, `mean7`, `store_total_sum28`는 **증가 시 예측도 증가(+1)**로 제한  
    → 과도한 역방향 학습 방지, 합리적 추세 보존

#### 8) 테스트 파이프라인 (TEST_00~09)
1. 파일별로 **각 메뉴의 최근 28일** 슬라이스 확보  
2. 28일 히스토리로 **동일 규칙**으로 피처 생성(위 4)과 동일)  
3. 별도 저장된 `store_top1_token_beta.csv`에서 매장별 Top-1 토큰 로드  
   → 해당 토큰이 포함된 메뉴들의 **최근 28일 합**을 매장 단위로 계산  
4. 예측(`pred.predict`) → `pred_long` 구성

#### 9) 제출 변환 & 후처리
- **제출 포맷 변환**:  
  `pred_long (test_prefix, date, store_menu, pred)` →  
  `sample_submission` 구조(`영업일자="TEST_xx+{1..7}일" × 열=메뉴`)에 **키 매칭** 삽입
- **반올림 버전** 추가 저장
- **0 → 1 치환(최솟값 보정)**:  
  제출 최종본에서 숫자 0 및 문자열 `"0"`을 모두 1로 치환  
  *(운영상 최소판매/재고 로직, 0-분모 이슈·SMAPE 안정화 및 언더포어캐스트 완화 목적)*

---

#### 사용 피처 총정리
- **기본 통계**: `last, mean7, mean28, std7, trend, lag7, lag14, rm3, rm14, rs14, diff_last_mean7`  
- **제로/분산**: `zero_streak_28, days_since_nz, zero_ratio28, mean7_over_mean28, vol14`  
- **날짜/주기**: `dow_next, is_wknd_next, is_fri_next, is_sat_next, is_sun_next, month_next, h, doy_next, doy_sin_next, doy_cos_next`  
- **특일**: `is_special_any_next, days_to_special_next, days_from_special_next, min_days_to_special_next, is_special_window_k2_next`  
- **앵커 평균**: `sm_dow_mean_next, sm_month_mean_next`  
- **매장 신호**: `store_top1tok_sum28, store_total_sum28, wknd_x_top1`  
- **가중치**: `w(담하/미라시아=2.0, 기타=1.0)`


---

## 📊 평가 전략

- **Phase 2**: Public 리더보드 기준 중상위권 유지  
- **Phase 3**: 오프라인 주피터 환경에서 추가 메타데이터 활용 및 코드 간소화  
  (보안상 코드 반출 불가 → 설명으로 대체)  
- 최종 Private 기준 **31팀 중 20위 기록**

---

## 🧭 회고

- 시계열 문제에서는 **도메인 기반 feature 설계**가 가장 효과적이었음  
- 메타데이터가 항상 성능 향상으로 이어지지 않음 → 노이즈 관리 필요  
- 단순 딥러닝보다 **AutoML + Feature Engineering** 조합이 안정적  
- 협업 시 GitHub/Colab 워크플로우 확립의 중요성 체감  

---

## 🧩 코드 공개 범위

- **온라인(Phase 2)**: 이 레포지토리에 공개 (모델링 및 전처리 코드)  
- **오프라인(Phase 3)**: 군사/보안 환경의 주피터 기반 개발 → 코드 반출 불가  
  → 핵심 로직 및 접근 방식만 문서화, 깃허브에 있는 코드는 'Phase 2 기준'
