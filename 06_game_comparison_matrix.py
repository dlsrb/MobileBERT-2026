import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def main():
    print("⏳ 데이터 불러오는 중...")
    df = pd.read_csv('moba_reviews_predicted_final.csv', encoding='utf-8-sig')

    date_col = 'date' if 'date' in df.columns else 'Date'
    df['year_month'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m')
    df['game_name'] = df['game'].apply(lambda x: 'Mobile_Legends' if 'Mobile_Legends' in x else 'Wild_Rift')

    # 월별 부정(0) 비율 계산
    monthly_sentiment = df.groupby(['year_month', 'game_name', 'Sentiment']).size().unstack(fill_value=0)
    monthly_total = monthly_sentiment.sum(axis=1)

    # 🎯 라벨 0 기준 정정
    neg_ratio = (monthly_sentiment.get(0, 0) / monthly_total * 100).reset_index(name='Neg_Ratio')
    neg_ratio = neg_ratio.sort_values('year_month')

    fig, ax = plt.subplots(figsize=(15, 6))

    ml_data = neg_ratio[neg_ratio['game_name'] == 'Mobile_Legends']
    wr_data = neg_ratio[neg_ratio['game_name'] == 'Wild_Rift']

    ax.plot(ml_data['year_month'], ml_data['Neg_Ratio'], marker='.', markersize=8, color='#d62728', linewidth=2,
            label='Mobile_Legends (Negative Ratio)')
    ax.plot(wr_data['year_month'], wr_data['Neg_Ratio'], marker='.', markersize=8, color='#1f77b4', linewidth=2,
            label='Wild_Rift (Negative Ratio)')

    # 스파이크 표시
    wr_spike = wr_data[wr_data['year_month'] == '2026-03']
    if not wr_spike.empty:
        wr_val = wr_spike['Neg_Ratio'].values[0]
        ax.annotate(f'Wild_Rift Spike\n2026-03 ({wr_val:.1f}%)', xy=('2026-03', wr_val), xytext=('2026-03', wr_val - 4),
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5), color='#1f77b4', fontweight='bold',
                    ha='center', va='top')

    ml_spike = ml_data[ml_data['year_month'] == '2026-04']
    if not ml_spike.empty:
        ml_val = ml_spike['Neg_Ratio'].values[0]
        ax.annotate(f'Mobile_Legends Spike\n2026-04 ({ml_val:.1f}%)', xy=('2026-04', ml_val),
                    xytext=('2026-04', ml_val + 4),
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5), color='#d62728', fontweight='bold',
                    ha='center', va='bottom')

    ax.set_title('Negative Sentiment Proportion Comparison: Wild Rift vs MLBB', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Negative Proportion (%)', fontsize=11)
    ax.set_xlabel('Timeline (Year-Month)', fontsize=11)
    ax.legend(loc='upper right', frameon=True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = 'readme 데이터용 사진파일/06_game_comparison_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 그래프 생성 및 '{filename}' 저장 완료!")


if __name__ == "__main__":
    main()