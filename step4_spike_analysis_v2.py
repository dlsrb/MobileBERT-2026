# ==========================================
# [2회차] Step 4: 이상치(Spike) 탐지 및 특정 월 TF-IDF 원인 규명 스크립트
# (시각화 루프 최적화 및 동적 Y축 상한선 교정 버전)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    print("=" * 50)
    print("[Step 4] 버그 교정 및 최적화 완료 - 이상치 포착 플롯 가동")
    print("=" * 50)

    # 3단계 예측 결과 파일 로드
    df = pd.read_csv("moba_reviews_predicted_v2.csv")

    # Label 0(부정/QA결함)인 데이터만 유효 불만 신호로 집계
    df['is_neg'] = df['pred_label'].apply(lambda x: 1 if x == 0 else 0)

    # 월별 부정 건수 및 전체 건수 집계 후 비율(%) 연산
    stats = df.groupby(['game', 'date']).agg(neg_cnt=('is_neg', 'sum'), total_cnt=('reviewId', 'count')).reset_index()
    stats['neg_ratio'] = (stats['neg_cnt'] / stats['total_cnt']) * 100

    # ---------------------------------------------------------
    # 📊 [그림 2] 월별 부정 리뷰 비율 추이선 시각화 빌드
    # ---------------------------------------------------------
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.figure(figsize=(12, 7))
    plt.grid(True, linestyle='--', alpha=0.5)

    target_months = ['2025-11', '2025-12', '2026-01', '2026-02', '2026-03', '2026-04', '2026-05']
    colors, markers = ['#1f77b4', '#ff7f0e'], ['o', 's']

    for idx, game_name in enumerate(['Wild_Rift', 'Mobile_Legends']):
        game_df = stats[stats['game'] == game_name].set_index('date').reindex(target_months, fill_value=0).reset_index()
        plt.plot(game_df['date'], game_df['neg_ratio'], marker=markers[idx], color=colors[idx], linewidth=2.5,
                 label=game_name)

        # 텍스트 주석 오프셋 조율 (Wild Rift는 상단, MLBB는 하단 배치)
        y_offset = 6 if game_name == 'Wild_Rift' else -14

        # 🔴 [버그 교정] iterrows() 탈피 및 고속 zip() 구조 완벽 적용
        # 판다스 행 순회 부하를 없애고 4개 리스트를 병렬 동기화 처리합니다.
        for x_val, neg_val, total_val, ratio_val in zip(
                game_df['date'], game_df['neg_cnt'], game_df['total_cnt'], game_df['neg_ratio']
        ):
            plt.text(x_val, ratio_val + y_offset, f"{ratio_val:.1f}%\n({int(neg_val)}/{int(total_val)})",
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor=colors[idx], alpha=0.8))

    plt.title("Monthly Negative Review Ratio Timeline (MobileBERT Base)", fontweight='bold', pad=15)
    plt.xlabel("Execution Timeline (Year-Month)", labelpad=10)
    plt.ylabel("Negative Feedback Ratio (%)", labelpad=10)

    # 🟡 [안전성 고도화] 실 데이터 이식 시 주석 유실을 막기 위해 최고 비율의 1.35배를 동적 맥시멈 지정
    max_ratio = stats['neg_ratio'].max()
    plt.ylim(0, max_ratio * 1.35)

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("02_score_distribution_comparison.png", dpi=300)
    plt.close()
    print("🎯 그림 2 저장 완료: 02_score_distribution_comparison.png")

    # ---------------------------------------------------------
    # 🎯 [그림 3] 이상치 폭등 스파이크 구간(2026-04) 핵심 불만 토픽 단어 추출
    # ---------------------------------------------------------
    # 4월달 데이터 중 '진짜 부정(pred_label == 0)' 데이터만 스나이핑 격리
    df_spike = df[(df['date'] == '2026-04') & (df['pred_label'] == 0)]

    # TF-IDF 벡터라이저 구동 (불용어 제거)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_spike['content_lower'])

    # 단어별 평균 가중치 연산 및 상위 15개 피처 스코어 추출
    importance = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    df_tfidf = pd.DataFrame({'word': tfidf.get_feature_names_out(), 'score': importance})
    df_top15 = df_tfidf.sort_values(by='score', ascending=False).head(15).sort_values(by='score', ascending=True)

    # 가로 바 차트 시각화 빌드
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.3, axis='x')
    bars = plt.barh(df_top15['word'], df_top15['score'], color='#2c3e50', height=0.6)

    # 바 우측 끝에 정확한 수치 바인딩 주석 처리
    for bar in bars:
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.4f}",
                 va='center', ha='left', fontsize=9, fontweight='bold')

    plt.title("Top 15 TF-IDF Feature Words in Target Spike Month (2026-04)", fontweight='bold', pad=15)
    plt.xlabel("TF-IDF Average Score", labelpad=10)
    plt.ylabel("Extracted Domain Keywords", labelpad=10)
    plt.xlim(0, df_top15['score'].max() * 1.15)  # 우측 수치 여백 확보
    plt.tight_layout()
    plt.savefig("03_spike_tfidf_top15.png", dpi=300)
    plt.close()
    print("🎯 그림 3 저장 완료: 03_spike_tfidf_top15.png")
    print("🎉 [성공] 전 공정 시뮬레이션 파이프라인 검증 가동 완수!")


if __name__ == "__main__":
    main()