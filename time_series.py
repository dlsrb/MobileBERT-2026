import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# [클로드 피드백 반영] 실제 플레이스토어 날짜 컬럼명인 'at'으로 매핑 완료
DATE_COLUMN = 'at'


def run_time_series_analysis():
    print("=" * 60)
    print(" 5-3단계: 시간 흐름에 따른 감성 트렌드(시계열) 분석 시작")
    print("=" * 60)

    file_path = "dataset/predicted_reviews.csv"
    if not os.path.exists(file_path):
        print(f"[오류] {file_path} 파일이 없습니다. 5-1단계를 먼저 실행하세요.")
        return

    df = pd.read_csv(file_path)

    # 1. 날짜 컬럼 존재 여부 확인 및 데이터 타입 변환
    if DATE_COLUMN not in df.columns:
        print(f"[경고] 데이터에 '{DATE_COLUMN}' 컬럼이 없습니다.")
        print(f"현재 데이터셋 컬럼 목록: {list(df.columns)}")
        return

    # 날짜 형태로 강제 변환 (에러 데이터는 NaT 처리)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df = df.dropna(subset=[DATE_COLUMN]).copy()

    # 분석을 위해 연-월(YYYY-MM) 컬럼 생성
    df['year_month'] = df[DATE_COLUMN].dt.to_period('M')

    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)

    games = df['game_title'].dropna().astype(str).unique() if 'game_title' in df.columns else ["All_Games"]

    # 2. 게임별 시계열 집계 및 시각화
    for game in games:
        print(f"\n▶ [{game}] 시계열 데이터 트렌드 분석 중...")
        game_df = df[df['game_title'] == game] if 'game_title' in df.columns else df

        # 연월별, 감성별(positive/negative) 리뷰 건수 집계
        ts_df = game_df.groupby(['year_month', 'predicted_label']).size().unstack(fill_value=0)

        if ts_df.empty:
            continue

        # 가독성을 위해 연월 인덱스를 문자열로 변환
        ts_df.index = ts_df.index.astype(str)

        # [GPT 조언 연계] 보고서용 정량 시계열 표(CSV) 구조화 저장
        safe_game_name = str(game).replace(" ", "_").replace("/", "_").replace(":", "_")
        csv_path = f"{output_dir}/{safe_game_name}_sentiment_timeline.csv"
        ts_df.to_csv(csv_path, encoding='utf-8-sig')
        print(f"  - 시계열 통계 표(CSV) 저장 완료: {csv_path}")

        # ---- [차트] 월별 긍정/부정 선그래프 그리기 ----
        plt.figure(figsize=(14, 6))

        # 부정 트렌드 (붉은색 선 - QA 불만 요인)
        if 'negative' in ts_df.columns:
            plt.plot(ts_df.index, ts_df['negative'], marker='o', color='salmon', linewidth=2.5,
                     label='Negative (QA Issues)')
        # 긍정 트렌드 (푸른색 선 - 만족 포인트)
        if 'positive' in ts_df.columns:
            plt.plot(ts_df.index, ts_df['positive'], marker='s', color='skyblue', linewidth=2.5,
                     label='Positive (Satisfaction)')

        plt.title(f'[{game}] Monthly Sentiment and QA Issue Trend Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Timeline (Year-Month)', fontsize=12)
        plt.ylabel('Review Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()

        img_path = f"{output_dir}/{safe_game_name}_sentiment_timeline.png"
        plt.savefig(img_path, dpi=300)
        plt.close()
        print(f"  - 시계열 트렌드 차트(PNG) 저장 완료: {img_path}")

    print("\n" + "=" * 60)
    print(" 시계열 분석 완료! 모든 시간축 시각화 자료가 'visualization_results' 폴더에 보관되었습니다.")
    print("=" * 60)


if __name__ == "__main__":
    run_time_series_analysis()