import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# 한글 폰트 설정 (깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def get_top_keywords(text_series, n=10):
    # 커스텀 불용어 (너무 뻔한 단어 제외)
    custom_stop_words = ['game', 'play', 'just', 'like', 'really', 'im', 'dont', 'wild', 'rift', 'mobile', 'legends',
                         'app', 'time', 'good', 'bad']
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

    vectorizer = CountVectorizer(stop_words=stop_words, max_features=1000, ngram_range=(1, 1))

    # 빈 데이터 처리
    if text_series.dropna().empty:
        return pd.DataFrame({'keyword': [], 'frequency': []})

    X = vectorizer.fit_transform(text_series.dropna())

    words = vectorizer.get_feature_names_out()
    freqs = X.sum(axis=0).A1

    df_freq = pd.DataFrame({'keyword': words, 'frequency': freqs})
    return df_freq.sort_values(by='frequency', ascending=False).head(n)


def plot_overall_keywords(df, sentiment_label, sentiment_name, colors, filename):
    # 감성 라벨 필터링
    df_filtered = df[df['Sentiment'] == sentiment_label]

    # 'game' 컬럼 소문자 유지
    wr_texts = df_filtered[df_filtered['game'].str.contains('Wild_Rift', case=False, na=False)]['content_lower']
    ml_texts = df_filtered[df_filtered['game'].str.contains('Mobile_Legends', case=False, na=False)]['content_lower']

    wr_top = get_top_keywords(wr_texts)
    ml_top = get_top_keywords(ml_texts)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'전체 기간(All-Time) 핵심 {sentiment_name} 키워드 비교 Top 10', fontsize=20, fontweight='bold')

    # MLBB 그래프
    if not ml_top.empty:
        sns.barplot(data=ml_top, x='frequency', y='keyword', ax=axes[0], palette=colors[0])
        for i, v in enumerate(ml_top['frequency']):
            axes[0].text(v + (max(ml_top['frequency']) * 0.01), i, f"{v:,}", va='center', fontweight='bold')

    axes[0].set_title('모바일 레전드 (Mobile Legends)', fontsize=16)
    axes[0].set_xlabel('언급 빈도수', fontsize=12)
    axes[0].set_ylabel('키워드', fontsize=12)

    # Wild Rift 그래프
    if not wr_top.empty:
        sns.barplot(data=wr_top, x='frequency', y='keyword', ax=axes[1], palette=colors[1])
        for i, v in enumerate(wr_top['frequency']):
            axes[1].text(v + (max(wr_top['frequency']) * 0.01), i, f"{v:,}", va='center', fontweight='bold')

    axes[1].set_title('와일드 리프트 (Wild Rift)', fontsize=16)
    axes[1].set_xlabel('언급 빈도수', fontsize=12)
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ '{filename}' 저장 완료!")


def main():
    print("⏳ 데이터 불러오는 중...")
    df = pd.read_csv('moba_reviews_predicted_final.csv', encoding='utf-8-sig')

    print("📊 전체 기간 부정(0) 키워드 시각화 생성 중...")
    plot_overall_keywords(df, 0, '부정', ['Reds_r', 'Blues_r'], '08_overall_keyword_negative_real.png')

    print("📊 전체 기간 긍정(1) 키워드 시각화 생성 중...")
    # 🔥 수정: 긍정 라벨을 다시 올바른 1로 복구!
    plot_overall_keywords(df, 1, '긍정', ['Oranges_r', 'Greens_r'], '08_overall_keyword_positive_real.png')


if __name__ == "__main__":
    main()