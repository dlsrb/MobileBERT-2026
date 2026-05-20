# MobileBERT 기반 US Google Play MOBA 게임 사용자 반응 비교 분석

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyCharm](https://img.shields.io/badge/PyCharm-000000?style=flat&logo=pycharm&logoColor=white)](https://www.jetbrains.com/pycharm/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yellow?style=flat)](https://huggingface.co/)


---

### 1. 프로젝트 개요 설명

**[MOBA 장르의 특성과 라이브 서비스의 중요성]**
모바일 게임 시장에서 MOBA(Multiplayer Online Battle Arena) 장르는 실시간 5vs5 대전을 기반으로 하는 대표적인 경쟁 장르로 자리 잡고 있다. 
특히 MOBA 장르는 단순한 콘텐츠 소비형 게임과 달리 **영웅 밸런스, 네트워크 안정성(Ping), 매칭 시스템(MMR), 시즌 운영(LiveOps)** 등의 요소가 장기적인 흥행에 직접적인 영향을 미친다. 
실제로 동일한 장르의 게임이라도 서버 안정성이나 운영 정책에 따라 사용자 만족도와 게임 지속성이 크게 달라진다.

**[미국(US) 모바일 MOBA 시장의 경쟁 구도]**
현재 미국(US) 모바일 MOBA 시장에서는 **League of Legends: Wild Rift**와 **Mobile Legends: Bang Bang**이 대표적인 경쟁 구도를 형성하고 있다. 
두 게임은 동일한 장르임에도 불구하고 게임 템포, 진입 난이도, 과금 구조, 매칭 시스템 운영 방식 등에서 차이를 보이며, 이러한 차이는 실제 사용자 리뷰에도 고스란히 반영된다.

**[분석 모델 및 연구 목표]**
본 프로젝트에서는 미국 Google Play Store의 두 게임 리뷰 데이터를 직접 수집하여 사용자 반응을 비교 분석하고자 한다. 자연어 처리 모델인 **MobileBERT**를 활용하여 영문 리뷰의 감성을 긍정·부정으로 이진 분류하고, 주요 키워드를 추출하여 게임별 사용자 만족 요인과 불만 요인을 심층 탐색한다.

> **최종 목표:** 단순한 감성 분석에 그치지 않고, **QA(Quality Assurance)와 LiveOps 관점**에서 실제 사용자들이 어떤 문제를 가장 중요하게 인식하는지 확인한다. 이를 통해 게임 산업에서 서비스 운영 품질과 기술적 안정성이 사용자 경험에 어떠한 영향을 미치는지 데이터 기반으로 확인하는 것을 목표로 한다.

---

### 2. 데 이 터

#### 2.1 데이터 수집

본 프로젝트는 미국(US) Google Play Store의 MOBA 장르 리뷰 데이터를 직접 수집하여 구축하였다. 데이터 수집은 Python 기반 환경(PyCharm)에서 진행하며, Google Play 리뷰 수집 라이브러리를 활용하여 최근 리뷰 데이터를 크롤링한다.

**[수집 대상]**

| 게임명 (Game Title) | 개발사 (Developer) | 플랫폼 (Platform) |
|---|---|---|
| **League of Legends: Wild Rift** | Riot Games | Google Play Store |
| **Mobile Legends: Bang Bang** | MOONTON | Google Play Store |

### 2.1 데이터 수집 기준 및 항목

**[데이터 수집 개요]**

| 구분 | 세부 내용 |
|---|---|
| **수집 국가** | 미국 (US) |
| **수집 기간** | 2025년 11월 ~ 2026년 5월 (최근 6개월) |
| **수집 규모** | Wild Rift 약 25,000건, MLBB 약 25,000건 (총 50,000건 이상) |
| **수집 언어** | 영어 (English) |

<br>

**[수집 데이터 항목]**

| 항목 (Column) | 설명 (Description) |
|---|---|
| `reviewId` | 리뷰 고유 ID |
| `score` | 사용자 평점 (1~5점) |
| `content` | 리뷰 텍스트 원문 |
| `at` | 리뷰 작성 날짜 및 시간 |
| `thumbsUpCount` | 다른 사용자가 누른 '좋아요' 수 |
| `replyContent` | 개발사(운영진) 답변 텍스트 |
| `repliedAt` | 개발사 답변 날짜 및 시간 |

**[데이터 수집 방식 및 전처리]**

대규모 리뷰 수집 과정에서 누락을 방지하기 위해 Checkpoint 기반 분할 수집 방식을 적용한다. 또한 중복 리뷰 제거 및 비정상 데이터 제거를 위해 다음과 같은 1차 전처리를 수행한다.

* 텍스트 길이 10자 미만 제거
* 의미 없는 반복 문자열 제거
* 이모지 및 특수문자 위주 리뷰 제거
* 중복 리뷰 제거

<br>

**[데이터 수집 예시]**

| 날짜 (Date) | 평점 (Score) | 리뷰 텍스트 (Review Text) |
|---|---|---|
| 2026-03-15 | 5 | amazing gameplay and smooth graphics |
| 2026-02-11 | 1 | worst matchmaking ever |
| 2026-01-02 | 2 | high ping after update |

### 2.2 탐색적 데이터 분석 (EDA)

수집된 데이터를 기반으로 다음과 같은 탐색적 데이터 분석을 수행하여 데이터의 전반적인 특성을 파악한다.

**1. 게임별 리뷰 수 비교**
Wild Rift와 MLBB의 전체 리뷰 수량 및 기간별 리뷰 증가량(추이)을 시각화하여 비교한다.

**2. 평점 분포 분석**
1점~5점 평점 비율을 분석하여 데이터의 클래스 불균형 여부를 확인한다.
> **예상 특징:** 1점과 5점 리뷰 비율이 압도적으로 높게 나타나는 양극화 현상이 관찰되며, 중립(3점) 리뷰 비율은 상대적으로 낮을 것으로 예상된다.

**3. 리뷰 길이 분석**
리뷰별 단어 수(Word count) 및 문장 길이 분포를 분석한다.
> **분석 목적:** 짧은 감정 표현 위주의 리뷰 비율을 확인하고, 길이가 긴 장문 리뷰에서 구체적인 시스템/운영 불만 요소를 탐색하기 위함이다.

**4. 주요 단어 빈도 분석**
TF-IDF 및 N-gram 기법을 적용하여 주요 키워드를 추출하고 시각화한다.

| 게임명 (Game) | 예상 긍정 키워드 (Positive) | 예상 부정 키워드 (Negative) |
|---|---|---|
| **Wild Rift** | smooth graphics, champion design | high ping, fps drop |
| **MLBB** | fast matches, easy to play | toxic players, pay to win |

### 3. 학습 데이터 구축

전체 수집된 약 50,000건의 원시 데이터 중, MobileBERT 모델의 효과적인 파인튜닝(Fine-tuning)을 위한 고품질 학습 및 검증용 데이터셋을 별도로 구축한다.

#### 3.1 데이터 라벨링 및 품질 검증

**[라벨링 기준]**
대규모 데이터셋의 효율적인 구축을 위해 사용자 평점(Score)을 기반으로 다음과 같이 감성 라벨을 자동 부여(이진 분류)한다.
* 단, 긍정과 부정의 경계가 모호하여 모델의 학습 경계를 흐릴 수 있는 **중립 데이터(3점)는 데이터셋에서 제외**하여 분류의 명확성을 확보한다.

| 평점 (Score) | 라벨 (Label) | 비고 (Note) |
|---|---|---|
| **4 ~ 5점** | `1` (긍정, Positive) | 만족도 높은 유저 피드백 |
| **1 ~ 2점** | `0` (부정, Negative) | 결함 및 불만 유저 피드백 |
| **3점** | *제외 (Exclude)* | 데이터 분석 효율화 및 노이즈 방지 |

<br>

**[라벨 노이즈(Label Noise) 제거]**
실제 앱스토어 환경에서 빈번히 발생하는 '평점과 리뷰 본문 내용의 불일치' 문제를 해결하기 위해 **Rule-based 키워드 필터링** 거름망을 적용하여 데이터의 순도를 높인다.
* **불일치 예시 1:** 평점은 5점(긍정)이나 본문에 `worst`, `trash`, `bad`, `uninstall` 등의 극단적 불만이 포함된 경우
* **불일치 예시 2:** 평점은 1점(부정)이나 본문에 `best`, `love`, `masterpiece`, `perfect` 등 극찬 키워드가 포함된 밈(Meme)성 피드백인 경우

<br>

**[라벨 분포 시각화 계획]**
데이터 정제 후, 파이썬 시각화 라이브러리를 활용하여 다음 4가지 관점의 차트를 생성하고 데이터 특성을 검증한다.
> 1. 전체 데이터셋의 **긍정/부정 최종 비율** 시각화 (Pie Chart)
> 2. **Wild Rift vs MLBB** 두 게임 간의 라벨 분포 비교 (Grouped Bar Chart)
> 3. 정제 전/후 **평점별(1~5점) 데이터 수** 변화량 분석 (Bar Chart)

---

#### 3.2 학습 데이터셋 구성 및 분할

**[데이터 불균형 해소: 언더샘플링(Undersampling)]**
특정 라벨(예: 마켓 특성상 5점 만점 리뷰의 과도한 쏠림)로 인한 모델의 과적합(Overfitting)을 방지하고 예측 성능의 왜곡을 막기 위해, 긍정과 부정의 비율을 정확히 1:1 균형에 가깝게 맞추는 언더샘플링을 수행한다.

| 분류 (Label) | 최종 데이터 구축 규모 (수량) | 비고 (Note) |
|---|---|---|
| **긍정 (Positive)** | 약 15,000 ~ 20,000 건 | 두 게임 통합 긍정 데이터 풀 |
| **부정 (Negative)** | 약 15,000 ~ 20,000 건 | 두 게임 통합 부정 데이터 풀 |
| **최종 합계** | **약 30,000 ~ 40,000 건 규모** | **MobileBERT 학습용 균형 데이터셋** |

<br>

**[학습 및 검증 데이터 분리 (Data Split)]**
구축된 균형 데이터셋을 모델 학습용과 검증용으로 분할하되, 데이터 분할 시 발생할 수 있는 라벨 왜곡을 차단한다.
> * **분할 비율:** `Train : Validation = 8 : 2` (최종 균형 데이터셋 기준)
> * **분할 기법:** **계층적 분할(Stratified Split)** 적용
> * **목적:** 분할된 Train(약 24,000~32,000건)과 Validation(약 6,000~8,000건) 세트 내에서도 원본 데이터가 가진 긍정/부정 라벨 비율(50:50)이 정확하게 유지되도록 보장하여, 평가 지표의 신뢰성을 확보하기 위함이다.
<br>

**[학습 및 검증 데이터 분리 (Data Split)]**
구축된 균형 데이터셋을 모델 학습용과 검증용으로 분할하되, 데이터 분할 시 발생할 수 있는 라벨 왜곡을 차단한다.
> * **분할 비율:** `Train : Validation = 8 : 2`
> * **분할 기법:** **계층적 분할(Stratified Split)** 적용
> * **목적:** 분할된 Train과 Validation 세트 내에서도 원본 데이터가 가진 긍정/부정 라벨 비율(50:50)이 정확히 유지되도록 보장하여, 평가 지표의 신뢰성을 확보하기 위함이다.
### 4. MobileBERT 모델 학습 (Fine-tuning)

본 프로젝트에서는 모바일 및 엣지 디바이스 환경에 최적화된 경량 자연어 처리(NLP) 모델인 **MobileBERT**를 활용하여 고속의 감성 분류 파인튜닝을 수행한다.

* **사용 모델:** `google/mobilebert-uncased` (Hugging Face 라이브러리 연계)
* **학습 환경:** Google Colab GPU 환경 및 PyTorch 프레임워크 기반 학습 진행

#### 4.1 하이퍼파라미터 및 학습 설정 (Training Setup)
대규모 연산 오버헤드를 방지하고 학습의 안정성을 확보하기 위해 최적의 파라미터를 다음과 같이 고정하여 설정을 구성한다.

| 항목 (Parameter) | 설정값 (Value) | 비고 (Description) |
|---|---|---|
| **Epoch** | 2 ~ 3 | 과적합(Overfitting) 방지를 위한 타이트한 Epoch 설정 |
| **Batch Size** | 16 | Colab GPU 메모리(RAM) 타겟 최적화 배치 크기 |
| **Optimizer** | `AdamW` | 가중치 감쇠(Weight Decay)가 적용된 일반적인 NLP 옵티마이저 |
| **Loss Function** | `CrossEntropyLoss` | 다중/이진 분류를 위한 표준 크로스 엔트로피 손실 함수 |
| **Max Length** | 128 | 리뷰 텍스트 토큰화 시 최대 문장 길이 제한 (패딩 적용) |

<br>

#### 4.2 모델 성능 평가 지표 (Evaluation Metrics)
단순 정확도(Accuracy) 지표에만 의존하지 않고, 모델의 분류 타당성을 다각도로 검증하기 위해 오차행렬(Confusion Matrix) 기반의 4대 핵심 지표를 산출한다.

> * **Accuracy (정확도):** 전체 데이터 중 올바르게 분류한 비율
> * **Precision (정밀도):** 모델이 긍정/부정으로 예측한 것 중 실제 정답의 비율
> * **Recall (재현율):** 실제 긍정/부정 데이터 중 모델이 맞춘 비율
> * **F1-score:** 불균형 데이터셋 평가에 필수적인 정밀도와 재현율의 조화평균값

* **🎯 MOBA 특화 도메인 검증:** 특히 일반적인 영문 데이터셋과 달리, 본 장르 특유의 인게임 은어 및 줄임말(**`AFK`**(잠수), **`Feed`**(의도적 패배), **`Smurf`**(부캐학살), **`Diff`**(라인 격차) 등)이 포함된 유저의 특수 문맥을 MobileBERT가 얼마나 정확하게 포착하고 감성을 분류해내는지 중점적으로 교차 검증한다.

<br>

#### 4.3 학습 결과 시각화 계획 (Visualization)
학습이 진행됨에 따라 모델의 수렴 속도 및 과적합 여부를 모니터링하기 위해 `matplotlib`를 활용하여 다음 4가지 성능 추이 그래프를 생성하고 리포트에 첨부한다.
1. Epoch/Step 진행에 따른 **Training Loss(손실값)** 감소 추이 곡선
2. **Training Accuracy vs Validation Accuracy** 비교 학습 곡선 (Learning Curve)
3. Epoch별 **F1-score 및 Precision/Recall** 종합 성능 변동 그래프
5. 분석 결과 및 원인 분석

학습된 MobileBERT 모델을 전체 리뷰 데이터에 적용하여 감성 분석을 수행한다. 이후 TF-IDF 및 N-gram 분석을 통해 주요 불만 요인을 도출한다.

### 5. 분석 결과 및 원인 분석 (예상)

학습이 완료된 MobileBERT 모델을 정제된 전체 데이터셋에 적용하여 감성을 예측(Inference)하고, 긍정 및 부정으로 분류된 리뷰 풀(Pool)을 대상으로 N-gram 및 TF-IDF 기반의 단어 빈도 분석을 수행하여 다음과 같은 장르적 특성을 도출한다.

#### 5.1 League of Legends: Wild Rift 분석 결과 및 해석

**[사용자 감성 반응 요약]**

| 분류 | 세부 분석 키워드 (Keywords) | 비고 (Core Focus) |
|---|---|---|
| **긍정 반응** | `champion design`, `smooth graphics`, `competitive gameplay` | 뛰어난 원작 이식성 및 그래픽 품질 |
| **기술 QA 이슈** | `high ping`, `fps drop`, `crash after update` | 디바이스 최적화 및 네트워크 불안정 |
| **운영(LiveOps) 이슈** | `bad matchmaking`, `long match time`, `unfair ranking system` | 하드코어한 룰과 매칭 알고리즘 불만 |

* **💡 분석 및 해석 (Interpretation):** Wild Rift는 라이엇 게임즈의 PC 원작 IP 파워를 바탕으로 **높은 그래픽 품질과 원작 재현성(챔피언 디자인, 경쟁력 있는 게임 플레이)**에 대한 유저들의 만족도가 압도적으로 높을 것으로 예상된다. 반면, 하이엔드 그래픽 사양 요구와 잦은 네트워크 핑(Ping) 불안정, 업데이트 직후 발생하는 튕김(`crash`) 현상으로 인해 **기술 QA 영역에서의 불만 비율이 경쟁작 대비 상대적으로 높게 나타날 가능성**이 크다. 또한, 정교한 라인전 시스템으로 인해 판당 플레이 타임이 길어지고, MMR 매칭 시스템(`matchmaking`)에 대한 스트레스가 주요 부정적 토픽으로 연결된다.

---

#### 5.2 Mobile Legends: Bang Bang (MLBB) 분석 결과 및 해석

**[사용자 감성 반응 요약]**

| 분류 | 세부 분석 키워드 (Keywords) | 비고 (Core Focus) |
|---|---|---|
| **긍정 반응** | `fast matches`, `easy to learn`, `casual gameplay` | 캐주얼한 접근성 및 빠른 게임 템포 |
| **기술 QA 이슈** | `reconnect issue`, `lag problem` | 간헐적 네트워크 지연 및 재접속 불량 |
| **운영(LiveOps) 이슈** | `toxic players`, `AFK teammates`, `pay to win`, `unbalanced heroes` | 유저 매너 제재 실패 및 밸런스 불만 |

* **💡 분석 및 해석 (Interpretation):** MLBB는 모바일 플랫폼 환경에 맞춰 가볍게 기획된 시스템 특성상 **빠른 매칭 및 게임 진행(`fast matches`), 낮은 진입 장벽(`easy to learn`)**에 대한 긍정 피드백이 지배적일 것으로 보인다. 저사양 기기 최적화가 잘 되어 있어 클라이언트 크래시 같은 치명적인 기술 QA 이슈는 적으나, 대전 도중 발생하는 간헐적 렉(`lag`)이나 재접속 실패(`reconnect`)에 대한 불만이 존재한다. 가장 핵심적인 부정 요인은 운영(LiveOps) 파트에 집중되어 있는데, 트롤링이나 비매너 유저 제재 시스템에 대한 불만(`toxic players`, `AFK`), 그리고 특정 신규 캐릭터나 유료 스킨의 스탯 관여로 인한 밸런스 붕괴(`pay to win`, `unbalanced heroes`)가 상대적으로 큰 비중을 차지할 것으로 예상된다.

---

#### 5.3 주요 패치 전후 시계열 감성 분석 (Time-Series Analysis)

단순히 전체 기간의 평균치를 분석하는 것을 넘어, 두 게임의 핵심 분기점인 **'대규모 시즌/밸런스 업데이트 날짜'**를 기준으로 패치 전후 2주간의 사용자 감성 추이를 추적한다. 이를 통해 특정 QA 결함이나 운영 정책이 유저 여론에 미치는 영향을 실시간 지표로 증명한다.

* **League of Legends: Wild Rift:** 메이저 패치(예: 신규 시즌 시작 및 매칭 시스템 조정 빌드) 전후 2주
* **Mobile Legends: Bang Bang:** 대규모 밸런스 패치(예: 특정 신규 영웅 출시 및 과금 시스템 조정) 전후 2주

> **🎯 시계열 분석의 실무적 가치:** > 패치 당일 직후 3일간 부정 리뷰 비율이 급증하며 `crash`, `fps drop` 키워드가 동반 상승하는 구간을 시각화함으로써 **기술 QA의 공백**을 정량적으로 진단한다. 만약 패치 후 1주일이 지난 시점에도 긍정 감성이 회복되지 않고 `unfair matching` 등의 키워드가 유지될 경우, 이는 **시스템 기획 및 라이브 서비스(LiveOps)의 실패**로 해석하여 실무적인 개선 우선순위를 수치로 도출한다.
6. 느낀점 및 향후 개선 방향