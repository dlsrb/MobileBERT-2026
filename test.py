import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def main():
    print("⏳ 데이터 불러오는 중...")
    df = pd.read_csv('moba_reviews_predicted_final.csv', encoding='utf-8-sig')

    date_col = 'date' if 'date' in df.columns else 'Date'
    df['year_month'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m')
    df['game_name'] = df['game'].apply(lambda x: 'MLBB' if 'Mobile_Legends' in x else 'Wild Rift')

    # 1. 월별 전체 리뷰 수 및 감성 라벨 카운트 (0: 부정, 1: 긍정)
    monthly_stats = df.groupby(['game_name', 'year_month', 'Sentiment']).size().unstack(fill_value=0)
    monthly_stats['Total'] = monthly_stats.sum(axis=1)

    # 정정된 라벨 반영: Sentiment가 0인 것이 '부정' 데이터입니다.
    monthly_stats['Neg_Ratio'] = (monthly_stats[0] / monthly_stats['Total']) * 100
    monthly_stats = monthly_stats.reset_index()

    # 시각화 데이터 정렬
    monthly_stats = monthly_stats.sort_values('year_month')

    # 차트 그리기
    games = ['MLBB', 'Wild Rift']
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for i, game in enumerate(games):
        game_data = monthly_stats[monthly_stats['game_name'] == game]
        ax1 = axes[i]

        # 주축 (왼쪽): 월별 전체 리뷰 수 (볼륨)
        sns.barplot(data=game_data, x='year_month', y='Total', color='lightgray', alpha=0.6, ax=ax1)
        ax1.set_ylabel(f'{game} 전체 리뷰 수', fontsize=11)

        # 보조축 (오른쪽): 부정 여론 비율 (선 그래프로 확 튀는 구간 표시)
        ax2 = ax1.twinx()
        sns.lineplot(data=game_data, x='year_month', y='Neg_Ratio', color='red', marker='o', linewidth=2, ax=ax2)
        ax2.set_ylabel('부정 여론 비율 (%)', color='red', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 100)

        # 특정 달에 확 튀는 스파이크 구간 강조
        if game == 'Wild Rift':
            spike_data = game_data[game_data['year_month'] == '2026-03']
            if not spike_data.empty:
                val = spike_data['Neg_Ratio'].values[0]
                ax2.annotate(f'Wild Rift Spike\n(부정: {val:.1f}%)',
                             xy=('2026-03', val), xytext=('2026-03', val - 15),
                             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                             color='red', fontweight='bold', ha='center')
        elif game == 'MLBB':
            spike_data = game_data[game_data['year_month'] == '2026-04']
            if not spike_data.empty:
                val = spike_data['Neg_Ratio'].values[0]
                ax2.annotate(f'MLBB Spike\n(부정: {val:.1f}%)',
                             xy=('2026-04', val), xytext=('2026-04', val + 15),
                             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                             color='red', fontweight='bold', ha='center')

        ax1.set_title(f'{game} 월별 리뷰 볼륨 및 부정 민심 추이 (Spike 구간 탐지)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('연월 (Year-Month)', fontsize=11)
        ax1.set_xticklabels(game_data['year_month'].unique(), rotation=45)

    plt.tight_layout()
    filename = 'readme 데이터용 사진파일/01_eda_monthly_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ '{filename}' 저장 완료!")


if __name__ == "__main__":
    main()