# ==========================================
# [2회차] Step 2: 실제 수집 데이터 월별 리뷰 수량 추이 시각화
# (실전 구글 스토어 데이터 반영 버전)
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("=" * 50)
    print("[Step 2] 실전 코퍼스 기반 월별 데이터 볼륨 차트 가동")
    print("=" * 50)

    # 크롤러가 확보한 진짜 유저 데이터셋 로드
    path = "moba_reviews_cleaned_v2.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("❌ [오류] moba_reviews_cleaned_v2.csv 파일이 없습니다. Step 1을 먼저 실행하세요.")
        return

    # 월별/게임별 빈도수 통계 계산
    volume_stats = df.groupby(['game', 'date']).size().reset_index(name='count')

    # 논문 타임라인 순서 강제 고정
    target_months = ['2025-11', '2025-12', '2026-01', '2026-02', '2026-03', '2026-04', '2026-05']

    # 📊 [그림 1] 시각화 레이아웃 설정
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')

    colors = ['#1f77b4', '#ff7f0e']
    bottom_data = [0] * len(target_months)

    # 누적 막대그래프(Stacked Bar Chart) 빌드
    for idx, game_name in enumerate(['Wild_Rift', 'Mobile_Legends']):
        game_df = volume_stats[volume_stats['game'] == game_name].set_index('date').reindex(target_months,
                                                                                            fill_value=0).reset_index()
        counts = game_df['count'].values

        bars = ax.bar(target_months, counts, bottom=bottom_data, color=colors[idx], alpha=0.85, label=game_name,
                      width=0.5)

        # 각 막대 중앙에 수치 주석 바인딩 (zip 활용 고속 연산)
        for text_idx, (bar, b_val) in enumerate(zip(bars, bottom_data)):
            y_pos = b_val + bar.get_height() / 2
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{int(bar.get_height()):,}",
                        ha='center', va='center', color='white', fontsize=10, fontweight='bold')

        bottom_data = [b + c for b, c in zip(bottom_data, counts)]

    # 총량 표시 상단 주석 바인딩
    for text_idx, total_val in enumerate(bottom_data):
        if total_val > 0:
            ax.text(text_idx, total_val + (max(bottom_data) * 0.015), f"Total:\n{total_val:,}",
                    ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

    ax.set_title("Monthly Collected Review Volume Timeline (Real Google Play Dataset)", fontweight='bold', pad=20)
    ax.set_xlabel("Execution Timeline (Year-Month)", labelpad=12)
    ax.set_ylabel("Number of Cleaned Reviews (Count)", labelpad=12)
    ax.set_ylim(0, max(bottom_data) * 1.2)  # 상단 토탈 주석이 잘리지 않도록 마진 확보
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig("01_monthly_review_volume.png", dpi=300)
    plt.close()

    print("🎯 그림 1 저장 완료: 01_monthly_review_volume.png")
    print(f"📊 분석에 투입된 실전 데이터 총합: {len(df):,}건")
    print("=" * 50)


if __name__ == "__main__":
    main()