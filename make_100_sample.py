import os
import pandas as pd

def create_manual_inspection_file():
    print("▶ 100건 수동 검토용 데이터 추출 시작...")

    os.makedirs("dataset", exist_ok=True)

    df = pd.read_csv("moba_reviews_raw_50k.csv")

    if 'reviewId' in df.columns:
        df = df.drop_duplicates(subset=['reviewId'])

    df = df.dropna(subset=['content']).copy()
    df = df[df['score'] != 3].copy()

    df['sentiment_label'] = df['score'].apply(
        lambda x: 'positive' if x >= 4 else 'negative'
    )

    if len(df) < 100:
        raise ValueError(f"[오류] 표본 추출 가능한 데이터가 부족합니다. 현재 데이터 수: {len(df)}")

    sample_100 = df.sample(n=100, random_state=42).copy()

    # [수정됨] 실무 꿀팁 적용: 빈칸 대신 기계가 매긴 라벨을 기본값으로 복사해 둠
    sample_100['human_corrected_label'] = sample_100['sentiment_label']

    final_columns = [
        'reviewId',
        'score',
        'sentiment_label',
        'content',
        'human_corrected_label'
    ]

    if 'game_title' in sample_100.columns:
        final_columns.insert(1, 'game_title')

    sample_100[final_columns].to_csv(
        "dataset/manual_inspection_sample_100.csv",
        index=False,
        encoding='utf-8-sig'
    )

    print("▶ 완료! dataset 폴더 안에 'manual_inspection_sample_100.csv' 파일 생성 성공!")

if __name__ == "__main__":
    create_manual_inspection_file()