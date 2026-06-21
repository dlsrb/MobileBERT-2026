import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm


def main():
    # 폰트 및 테마 설정
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = 'Malgun Gothic' if 'Malgun Gothic' in available_fonts else 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font=selected_font)

    # 10만 건 마스터 데이터 로드 (라벨링 전 원본으로도 충분)
    file_name = "moba_reviews_predicted_final.csv"
    print(f"📂 데이터 로드 중: '{file_name}'...")
    df = pd.read_csv(file_name, encoding='utf-8-sig', on_bad_lines='skip')

    df = df.dropna(subset=['date', 'game'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    # 월별/게임별 리뷰 건수 집계
    monthly_counts = df.groupby(['year_month', 'game']).size().reset_index(name='Review Count')

    # 시각화
    plt.figure(figsize=(14, 6))
    ax = sns.lineplot(
        data=monthly_counts, x='year_month', y='Review Count',
        hue='game', marker='o', linewidth=2.5, palette=['#d62728', '#1f77b4']
    )

    plt.title('월별 모바일 MOBA 리뷰 작성 건수 추이 (Wild Rift vs MLBB)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('연도-월', fontsize=12)
    plt.ylabel('리뷰 건수', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)

    # X축 라벨 겹침 방지 (2개월 단위 출력)
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 2 != 0:
            label.set_visible(False)

    plt.tight_layout()
    output_file = 'readme 데이터용 사진파일/01_eda_monthly_distribution.png'
    plt.savefig(output_file, dpi=300)
    print(f"🎉 월별 분포 차트 저장 완료 -> '{output_file}'")


if __name__ == "__main__":
    main()