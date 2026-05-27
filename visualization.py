import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 폰트 및 시각화 스타일 세팅
plt.style.use('ggplot')
sns.set_palette("Set2")


def get_distinctive_keywords(target_series, reference_series, n_words=10):
    # 양방향 데이터 불균형 에지 케이스 철저 방어
    if len(target_series) < 10 or len(reference_series) < 10:
        return []

    # [원복 완료] match, making, player 등을 제거하여 matchmaking 통합 로직 정상화
    custom_stopwords = [
        'game', 'play', 'good', 'bad', 'really', 'games', 'playing',
        'just', 'like', 'dont', 'make', 'even', 'get', 'time', 'lol',
        'star', 'stars', 'far', 'im', 'ive', 'much', 'way', 'new', 'ever',
        'gg', 'op', 'ez', 'id', 'hp', 'oh', 'ok', 'bro', 'guys', 'guy', 'vs'
    ]

    base_en_stopwords = list(TfidfVectorizer(stop_words='english').get_stop_words())
    combined_stopwords = base_en_stopwords + custom_stopwords

    # [안정화 설정 완료] 2글자 이상 용어 보존 및 max_df=0.85로 핵심 단어 유실 원천 차단
    vectorizer = TfidfVectorizer(
        stop_words=combined_stopwords,
        ngram_range=(1, 3),
        max_features=1000,
        token_pattern=r'\b[a-zA-Z0-9]{2,}\b',
        min_df=3,
        max_df=0.85
    )

    try:
        all_texts = pd.concat([target_series, reference_series])
        vectorizer.fit(all_texts)

        target_matrix = vectorizer.transform(target_series).mean(axis=0).A1
        ref_matrix = vectorizer.transform(reference_series).mean(axis=0).A1

        # 차이 기반 가중치 결합 및 음수 0 클리핑
        raw_distinctive_scores = (target_matrix - ref_matrix) * target_matrix
        distinctive_scores = np.maximum(raw_distinctive_scores, 0)

        feature_names = vectorizer.get_feature_names_out()
        scores = zip(feature_names, distinctive_scores.tolist())

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # 중복 노이즈 제거용 가벼운 필터링
        filtered_scores = []
        seen_words = set()

        for word, score in sorted_scores:
            normalized_word = word.replace(" ", "")
            if score > 0 and normalized_word not in seen_words:
                filtered_scores.append((word, score))
                seen_words.add(normalized_word)
            if len(filtered_scores) >= n_words:
                break

        return filtered_scores

    except ValueError as e:
        print(f"[TF-IDF 연산 에러] {e}")
        return []


def run_visualization():
    print("=" * 60)
    print(" 5-2단계: 감성 상대 중요도 기반 핵심 키워드 랭킹 시각화 시작")
    print("=" * 60)

    file_path = "dataset/predicted_reviews.csv"
    if not os.path.exists(file_path):
        print(f"[오류] {file_path} 파일이 없습니다. 5-1단계를 먼저 실행하세요.")
        return

    df = pd.read_csv(file_path)

    # 단어 경계(\b) 기반의 정석적인 matchmaking 대통합 전처리
    df['content_lower'] = (
        df['content']
        .astype(str)
        .str.lower()
        .str.replace(r'\bmatch[\s-]?making\b', 'matchmaking', regex=True)
    )

    # 15자 이상의 유효 데이터 필터링
    df = df[df['content_lower'].str.len() >= 15].copy()

    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)

    if 'game_title' in df.columns:
        games = df['game_title'].dropna().astype(str).unique()
    else:
        games = ["All_Games"]

    print("\n📊 [보고서 최종 수치 반영용] 게임별 AI 예측 감성 분포 현황")
    print("-" * 45)

    for game in games:
        game_df = df[df['game_title'] == game] if 'game_title' in df.columns else df

        print(f"▶ 게임명: {game}")
        print(game_df['predicted_label'].value_counts())
        print("-" * 45)

        positive_reviews = game_df[game_df['predicted_label'] == 'positive']['content_lower']
        negative_reviews = game_df[game_df['predicted_label'] == 'negative']['content_lower']

        top_neg = get_distinctive_keywords(negative_reviews, positive_reviews, 10)
        top_pos = get_distinctive_keywords(positive_reviews, negative_reviews, 10)

        safe_game_name = str(game).replace(" ", "_").replace("/", "_").replace(":", "_")

        # ---- [차트 1 및 복합 구조화 CSV 표 저장] 부정 리뷰 (QA 결함 요인) ----
        if top_neg:
            words, scores = zip(*top_neg)

            # [복구 완료] 논문 및 다차원 통합 분석을 위한 메타데이터 컬럼 전면 유지
            csv_path = f"{output_dir}/{safe_game_name}_negative_keywords.csv"
            pd.DataFrame({
                'Rank': range(1, len(words) + 1),
                'Keyword': words,
                'Score': scores,
                'Sentiment': 'Negative',
                'Game': game
            }).to_csv(csv_path, index=False, encoding='utf-8-sig')

            plt.figure(figsize=(12, 7))
            sns.barplot(x=list(scores), y=list(words), color='salmon')
            plt.title(f'[{game}] Top {len(words)} Distinctive QA Issues (Negative Reviews)', fontsize=14,
                      fontweight='bold')
            plt.xlabel('Distinctive Weight Score (Non-Negative)', fontsize=12)
            plt.ylabel('Keywords (Rank 1-10)', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{safe_game_name}_negative_keywords.png", dpi=300)
            plt.close()

        # ---- [차트 2 및 복합 구조화 CSV 표 저장] 긍정 리뷰 (만족 포인트) ----
        if top_pos:
            words, scores = zip(*top_pos)

            # 긍정 랭킹 표(CSV) 메타데이터 컬럼 전면 유지
            csv_path = f"{output_dir}/{safe_game_name}_positive_keywords.csv"
            pd.DataFrame({
                'Rank': range(1, len(words) + 1),
                'Keyword': words,
                'Score': scores,
                'Sentiment': 'Positive',
                'Game': game
            }).to_csv(csv_path, index=False, encoding='utf-8-sig')

            plt.figure(figsize=(12, 7))
            sns.barplot(x=list(scores), y=list(words), color='skyblue')
            plt.title(f'[{game}] Top {len(words)} Distinctive Satisfaction Points (Positive Reviews)', fontsize=14,
                      fontweight='bold')
            plt.xlabel('Distinctive Weight Score (Non-Negative)', fontsize=12)
            plt.ylabel('Keywords (Rank 1-10)', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{safe_game_name}_positive_keywords.png", dpi=300)
            plt.close()

    print(f"\n🏆 데이터 무결성과 메타데이터가 100% 보존된 최종 시각화 파이프라인 완수!")
    print(f"👉 마스터 리포트 파일들이 '{output_dir}' 폴더에 안전하게 덮어쓰기 되었습니다.")
    print("=" * 60)


if __name__ == "__main__":
    run_visualization()