import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm


def main():
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = 'Malgun Gothic' if 'Malgun Gothic' in available_fonts else 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="white", font=selected_font)

    raw_file_name = "moba_reviews_raw_50k.csv"
    predicted_file_name = "moba_reviews_predicted_final.csv"

    print(f"📂 데이터 병합 시작: '{raw_file_name}' + '{predicted_file_name}'...")

    try:
        df_raw = pd.read_csv(raw_file_name, encoding='utf-8-sig', usecols=['reviewId', 'score'])
        df_pred = pd.read_csv(predicted_file_name, encoding='utf-8-sig')
        df_merged = pd.merge(df_pred, df_raw, on='reviewId', how='inner')
        print(f"✅ 데이터 병합 완료! 총 {len(df_merged):,}건 매칭")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return

    df_merged = df_merged.dropna(subset=['score', 'Sentiment'])
    df_merged['score'] = df_merged['score'].astype(int)
    df_merged['Sentiment'] = df_merged['Sentiment'].astype(int)

    crosstab = pd.crosstab(df_merged['score'], df_merged['Sentiment'])
    crosstab.columns = ['부정(0)', '긍정(1)', '중립(2)', '관계없음(3)']
    crosstab.index.name = '유저 부여 별점(Score)'

    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt=',d', cmap='Purples', cbar=True,
                annot_kws={"size": 12, "weight": "bold"})

    plt.title('별점과 AI 텍스트 감성 예측의 불일치 분석 (3만 건 샘플 기준)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('유저가 누른 별점 (1~5점)', fontsize=13)
    plt.xlabel('AI가 예측한 텍스트 감성', fontsize=13)

    plt.tight_layout()
    output_file = '02_star_text_alignment_heatmap_30k.png'
    plt.savefig(output_file, dpi=300)
    print(f"🎉 히트맵 저장 완료 -> '{output_file}'")

    fake_positive = crosstab.loc[5, '부정(0)'] if 5 in crosstab.index and '부정(0)' in crosstab.columns else 0
    fake_negative = crosstab.loc[1, '긍정(1)'] if 1 in crosstab.index and '긍정(1)' in crosstab.columns else 0
    print("\n💡 [논문용 인사이트]")
    print(f"- 5점 만점인데 실제 욕설(부정) 리뷰: {fake_positive:,}건")
    print(f"- 1점 최하점인데 실제 칭찬(긍정) 리뷰: {fake_negative:,}건")


if __name__ == "__main__":
    main()