import os
import pandas as pd
from sklearn.model_selection import train_test_split


def build_training_dataset(input_csv="moba_reviews_raw_50k.csv"):
    if not os.path.exists(input_csv):
        print(f"[오류] {input_csv} 파일이 없습니다. 수집을 먼저 완료하세요.")
        return

    print("=" * 50)
    print(" 3단계: 학습 데이터셋 구축 및 분할 파이프라인 가동")
    print("=" * 50)

    # 1. 원시 데이터 로드
    df = pd.read_csv(input_csv)
    initial_count = len(df)
    print(f"▶ 원시 데이터 로드 완료: 총 {initial_count}건")

    # 2. 잠재적 Data Leakage 방지를 위한 확실한 중복 제거
    df = df.drop_duplicates(subset=['reviewId']).copy()

    # 3. 중립(3점) 데이터 완전 제거 및 결측 방어
    df = df[df['sentiment_label'] != 'neutral'].copy()

    # [GPT 피드백 반영] 아주 드물게 섞일 수 있는 결측치(NaN) 원천 방어
    df = df.dropna(subset=['content', 'sentiment_label'])

    print(f"▶ 중복, 결측치, 중립(3점) 데이터 제거 후: 총 {len(df)}건 남음\n")

    # [GPT 피드백 반영] 언더샘플링 직전 데이터 편향성(분포) 로그 출력
    print("▶ 언더샘플링 직전 감성 라벨 분포:")
    print(df['sentiment_label'].value_counts())
    print("-" * 50)

    # 4. 클래스 불균형 해소를 위한 1:1 언더샘플링 (Undersampling)
    pos_df = df[df['sentiment_label'] == 'positive']
    neg_df = df[df['sentiment_label'] == 'negative']

    min_count = min(len(pos_df), len(neg_df))

    # 한쪽 클래스가 극단적으로 부족할 경우의 예외 방어
    if min_count == 0:
        print("[오류] 한쪽 클래스 데이터가 0건입니다. 데이터 수집 상태를 점검하세요.")
        return

    # 더 적은 수량에 맞춰 무작위 샘플링 (random_state 고정으로 재현성 확보)
    pos_sampled = pos_df.sample(n=min_count, random_state=42)
    neg_sampled = neg_df.sample(n=min_count, random_state=42)

    # 균형 데이터셋 병합 및 인덱스 셔플링
    balanced_df = pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"▶ 1:1 언더샘플링 적용 완료 (긍정 {min_count}건 / 부정 {min_count}건)")
    print(f"▶ 최종 구축된 균형 데이터셋: 총 {len(balanced_df)}건")

    # 5. 계층적 분할 (Stratified Split) - Train 8 : Validation 2
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=0.2,
        stratify=balanced_df['sentiment_label'],
        random_state=42
    )

    # 6. 최종 학습/검증 데이터셋 파일 저장
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_dataset.csv")
    val_path = os.path.join(output_dir, "val_dataset.csv")

    # 모델 학습용 필수 컬럼만 추출하여 저장
    train_df[['reviewId', 'content', 'sentiment_label']].to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df[['reviewId', 'content', 'sentiment_label']].to_csv(val_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print(" 학습 데이터셋 구축 완료 요약")
    print("=" * 50)
    print(f" - Train Dataset: {len(train_df)}건 저장 완료 ({train_path})")
    print(
        f"   ↳ 긍정 {len(train_df[train_df['sentiment_label'] == 'positive'])}건 / 부정 {len(train_df[train_df['sentiment_label'] == 'negative'])}건")
    print(f" - Validation Dataset: {len(val_df)}건 저장 완료 ({val_path})")
    print(
        f"   ↳ 긍정 {len(val_df[val_df['sentiment_label'] == 'positive'])}건 / 부정 {len(val_df[val_df['sentiment_label'] == 'negative'])}건")
    print("==================================================")


if __name__ == "__main__":
    build_training_dataset()