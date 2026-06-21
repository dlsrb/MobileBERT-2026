# ==========================================
# [2회차] Step 1: 데이터 정제 및 라벨-텍스트 동기화 파이프라인
# (4-Class 스키마 및 문맥 정합성 완벽 반영 버전)
# ==========================================
import re
import numpy as np
import pandas as pd


def clean_text(text):
    if not isinstance(text, str): return None
    text_sub = text.replace('-', '')
    if not re.search(r"[a-zA-Z]{3,}", text_sub): return None
    return text


def main():
    print("=" * 50)
    print("[Step 1] 4-Class 문맥 정합성 동기화 빌더 가동")
    print("=" * 50)

    np.random.seed(2026)
    months = ['2025-11', '2025-12', '2026-01', '2026-02', '2026-03', '2026-04', '2026-05']
    review_ids = [f"rev_{i}" for i in range(125000)]
    review_ids[10], review_ids[500] = "rev_0", "rev_50"  # 중복 유입 시뮬레이션

    # 🔴 [기획 반영] 라벨별(0, 1, 2, 3) 독립 문장 컴포넌트 풀 정의
    corpus_pool = {
        'Wild_Rift': {
            0: {  # 부정 (QA 결함 신호)
                "ops": ["wild-rift is experiencing", "after the riot games update",
                        "every time i play ranked wild-rift"],
                "core": ["severe matchmaking issues and long queue times", "terrible lag spikes with high server ping",
                         "completely unbalanced rank draft matches"],
                "tails": ["matches me with bad teammates.", "making ranked unplayable.",
                          "completely trolls every game."]
            },
            1: {  # 중립 (사실 서술)
                "ops": ["wild-rift updated to", "riot games announced", "the developers changed"],
                "core": ["patch version 3.2 this week", "the champion balance scales",
                         "the gold distribution settings"],
                "tails": ["for the upcoming competitive season.", "in the latest patch notes.",
                          "on the official live servers."]
            },
            2: {  # 긍정 (만족, 칭찬)
                "ops": ["wild-rift matchmaking has", "the connection is absolutely", "i really love playing"],
                "core": ["improved a lot after the patch", "smooth and stable during team fights",
                         "wild-rift with my active guild members"],
                "tails": ["highly recommend this game!", "best mobile moba ever.", "fair ranked competitive play."]
            },
            3: {  # 관계없음 (노이즈, 일상)
                "ops": ["this game exists and", "i downloaded wild-rift because", "just checking out"],
                "core": ["i played it once on my tablet", "my friends told me to try it",
                         "the graphics interface layout"],
                "tails": ["nothing special honestly.", "will delete it soon.", "testing it out today."]
            }
        },
        'Mobile_Legends': {
            0: {  # 부정 (QA 결함 신호)
                "ops": ["mobile-legends mlbb has", "the latest moonton update caused",
                        "playing mobile-legends now means"],
                "core": ["network ping spikes and unbearable latency", "serious frame drops during 5v5 team fights",
                         "joystick delay and controls completely freezing"],
                "tails": ["solo queue is a nightmare.", "app crashes constantly on android.",
                          "losing stars because of bugs."]
            },
            1: {  # 중립 (사실 서술)
                "ops": ["mobile-legends added", "moonton released", "the roster system has"],
                "core": ["a new marksman hero today", "the classic 5v5 mode updates", "the latest skin line events"],
                "tails": ["according to the game store.", "available in the client menu.",
                          "for all global region players."]
            },
            2: {  # 긍정 (만족, 칭찬)
                "ops": ["mobile-legends runs", "the gameplay experience is", "moonton did a great job"],
                "core": ["incredibly smoothly on my device", "highly balanced and fun to play",
                         "optimizing the frame rate latency"],
                "tails": ["perfect 60fps performance.", "loving the new update!", "best moba game on mobile."]
            },
            3: {  # 관계없음 (노이즈, 일상)
                "ops": ["just typing random texts", "i am here only for", "mlbb is just"],
                "core": ["to see if this app works", "collecting daily check-in rewards",
                         "another casual arcade system"],
                "tails": ["not really my type of game.", "whatever i dont care.", "ok i guess."]
            }
        }
    }

    games, contents, dates, true_labels = [], [], [], []

    for i in range(125000):
        game_name = 'Wild_Rift' if i < 25000 else 'Mobile_Legends'
        date_str = np.random.choice(months)

        games.append(game_name)
        dates.append(date_str)

        # 🔴 [기획 반영] 월별 확률 분포 스케줄링에 따른 선제적 라벨 결정
        if date_str == '2026-04':
            # 4월 이상치 스파이크 구간 (부정 80% 폭등 기획 구현)
            label = np.random.choice([0, 1, 2, 3], p=[0.80, 0.10, 0.05, 0.05])
        else:
            # 평시 달 감성 비율 통제
            label = np.random.choice([0, 1, 2, 3], p=[0.60, 0.20, 0.10, 0.10])

        true_labels.append(label)

        # 파이프라인 정제 필터(단계 2, 3) 검증을 위한 의도적 원시 노이즈 주입 루프
        if i % 120 == 1:
            contents.append("!!!")
        elif i % 120 == 2:
            contents.append("12345a")
        elif i % 50 == 0:
            contents.append("bad")
        else:
            # 정상 문맥 매칭 생성: 결정된 라벨에 맞는 의미론적 문장 램덤 조합
            pool = corpus_pool[game_name][label]
            contents.append(
                f"{np.random.choice(pool['ops'])} {np.random.choice(pool['core'])} {np.random.choice(pool['tails'])}")

    df_raw = pd.DataFrame(
        {'reviewId': review_ids, 'game': games, 'date': dates, 'content': contents, 'Sentiment': true_labels})

    # 파이프라인 수행 (전처리 필터 통과 단계)
    df_pipe = df_raw.dropna(subset=['content']).drop_duplicates(subset=['reviewId'])
    df_pipe = df_pipe[df_pipe['content'].str.len() >= 10]
    df_pipe['content_cleaned'] = df_pipe['content'].apply(clean_text)
    df_pipe = df_pipe.dropna(subset=['content_cleaned'])
    df_pipe['content_lower'] = df_pipe['content_cleaned'].str.lower()

    df_final = df_pipe[['reviewId', 'game', 'date', 'content_lower', 'Sentiment']].reset_index(drop=True)
    df_final.to_csv("moba_reviews_cleaned_v2.csv", index=False)
    print(f"🎯 문맥 정합성 빌드 완료 및 파일 저장 성공 (총 건수: {len(df_final):,}건)")


if __name__ == "__main__":
    main()