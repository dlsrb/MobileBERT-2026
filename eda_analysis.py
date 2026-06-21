import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 깨짐 방지 설정 (리포트 출력용 표준 폰트 구성)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 환경 최적화
plt.rcParams['axes.unicode_minus'] = False


def run_moba_eda_pipeline(csv_path="moba_reviews_raw_50k.csv"):
    if not os.path.exists(csv_path):
        print(f"[오류] {csv_path} 파일이 존재하지 않습니다. 수집을 먼저 완료해주세요.")
        return

    print("=" * 50)
    print(" 2단계: 탐색적 데이터 분석(EDA) 및 리포트 시각화 파이프라인 가동")
    print("=" * 50)

    # 1. 데이터 로드 및 시계열 예외 방어 처리
    df = pd.read_csv(csv_path)

    # [GPT 반영] 타임존 유무에 상관없이 무조건 UTC aware로 통일 후 변환하여 런타임 에러 원천 차단
    df['at'] = pd.to_datetime(df['at'], utc=True)
    df['weekly'] = df['at'].dt.tz_convert(None).dt.to_period('W').dt.start_time

    # [클로드 반영] 리뷰 본문 '글자 수(Character Length)' 기술통계 산출 (보고서에 글자 수라고 명시 필요)
    df['content_len'] = df['content'].astype(str).str.len()

    # 3점 중립(neutral) 데이터 제외 규모 사전 모니터링 연산
    neutral_count = (df['sentiment_label'] == 'neutral').sum()
    neutral_ratio = (df['sentiment_label'] == 'neutral').mean() * 100

    # 콘솔창 기초 통계 정보 출력
    print(f"▶ 전체 데이터 볼륨: {df.shape[0]}행, {df.shape[1]}열")
    print(f"▶ 게임 타이틀별 데이터 수량:\n{df['game_title'].value_counts()}\n")
    print(f"▶ 사전 레이블링 감성 분포:\n{df['sentiment_label'].value_counts()}\n")
    print(f"▶ [핵심 정보] neutral(3점) 제외 대상 비율: {neutral_ratio:.1f}% ({neutral_count}건)")
    print(f"▶ 본문 글자 수 요약 통계:\n{df['content_len'].describe()}\n")
    print(f"▶ 500자 초과 롱(Long) 리뷰 비율: {(df['content_len'] > 500).mean() * 100:.1f}%\n")

    # 시각화 산출물 저장 폴더 생성
    output_dir = "eda_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 시각화 1: 통합 데이터셋 감성 라벨 비율 (Pie Chart)
    # ==========================================
    plt.figure(figsize=(8, 6))
    sentiment_counts = df['sentiment_label'].value_counts()

    color_map = {'negative': '#ff9999', 'positive': '#66b3ff', 'neutral': '#99ff99'}
    colors = [color_map.get(label, '#cccccc') for label in sentiment_counts.index]
    explode = [0.05] * len(sentiment_counts)

    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode
    )
    plt.title("통합 수집 데이터셋 감성(Sentiment) 라벨 비율", fontsize=14, weight='bold')
    pie_path = os.path.join(output_dir, "01_sentiment_pie_chart.png")
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ [차트 1 성공] 감성 비율 파이 차트 저장 완료: {pie_path}")

    # ==========================================
    # 시각화 2: 게임별 1~5점 평점 분포 비교 (Grouped Bar Chart)
    # ==========================================
    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x='score',
        hue='game_title',
        palette='Set2',
        order=[1, 2, 3, 4, 5]
    )

    plt.title("게임별(Wild Rift vs Mobile Legends) 스토어 평점 분포 비교", fontsize=14, weight='bold')
    plt.xlabel("구글 플레이스토어 별점 (Score)", fontsize=12)
    plt.ylabel("리뷰 수량 (건)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="게임 타이틀")

    bar_path = os.path.join(output_dir, "02_score_distribution_comparison.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ [차트 2 성공] 게임별 평점 분포 비교 바 차트 저장 완료: {bar_path}")

    # ==========================================
    # 시각화 3: 최근 6개월 사용자 리뷰 활성도 추이 (Time-Series Line Chart)
    # ==========================================
    plt.figure(figsize=(12, 6))

    # 주 단위 시계열 그룹핑 후 확실한 날짜 순 정렬 보장
    time_df = df.groupby(['weekly', 'game_title']).size().reset_index(name='count')
    time_df = time_df.sort_values('weekly')

    sns.lineplot(
        data=time_df,
        x='weekly',
        y='count',
        hue='game_title',
        marker='o',
        linewidth=2
    )

    plt.title("최근 6개월간 주차별(Weekly) 사용자 리뷰 활성도 추이", fontsize=14, weight='bold')
    plt.xlabel("리뷰 생성 날짜 (Time Series)", fontsize=12)
    plt.ylabel("유효 리뷰 수 (건)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.legend(title="게임 타이틀")

    line_path = os.path.join(output_dir, "03_weekly_collection_trend.png")
    plt.savefig(line_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ [차트 3 성공] 시계열 수집 추이 라인 차트 저장 완료: {line_path}")

    print("=" * 50)
    print(" 2단계 EDA 종료. 'eda_outputs' 폴더에 보고서용 고해상도 PNG 파일 3개가 안전하게 보관되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    run_moba_eda_pipeline()