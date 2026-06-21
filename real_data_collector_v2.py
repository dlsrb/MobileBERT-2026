import re
import time
import datetime
import pandas as pd
from google_play_scraper import Sort, reviews


def clean_real_text(text):
    if not isinstance(text, str):
        return None

    # 1. 공백 제거 후 순수 텍스트 길이 측정
    stripped_text = text.strip()
    total_chars = len(stripped_text)
    if total_chars < 10:
        return None

    # 2. 순수 영어(대소문자), 숫자, 기본 문장부호 및 공백의 개수만 카운트
    # 미얀마어, 한국어, 이모지 등이 섞여 있으면 이 정규식에서 걸러지지 않음
    eng_chars = len(re.findall(r"[a-zA-Z0-9\s.,!?'-]", stripped_text))

    # 🔴 [무결성 고도화] 영문 관련 글자 밀도가 85% 미만이면 비영어 혼착 리뷰로 간주하고 탈락
    if (eng_chars / total_chars) < 0.85:
        return None

    return text


def scrape_game_reviews_timeline(app_id, game_name, max_safety_limit=200000):
    print(f"\n[{game_name}] 구글 플레이스토어 동적 타임라인 추적 가동...")

    # 논문 지정 타임라인 범위 세팅 (2025-11-01 ~ 2026-05-31)
    start_date = datetime.datetime(2025, 11, 1)
    end_date = datetime.datetime(2026, 5, 31, 23, 59, 59)

    parsed_data = []
    token = None
    fetched_count = 0

    while True:
        # 🔴 [품질 교정] 고정 카운트 방식을 버리고 continuation_token 기반 동적 페이징 루프 설계
        if token is None:
            result, token = reviews(
                app_id, lang='en', country='us',
                sort=Sort.NEWEST, count=200  # 구글 API 최대 단위인 200건씩 분할 호출
            )
        else:
            result, token = reviews(app_id, continuation_token=token)

        if not result:
            print("   [알림] 더 이상 가져올 스토어 리뷰가 없습니다.")
            break

        fetched_count += len(result)
        last_review_date = result[-1]['at']  # 현재 배치 중 가장 과거 날짜 포착

        # 타임라인 조건 필터링 검증
        for r in result:
            review_date = r['at']
            if start_date <= review_date <= end_date:
                parsed_data.append({
                    'reviewId': r['reviewId'],
                    'game': game_name,
                    'date': review_date.strftime('%Y-%m'),
                    'content': r['content']
                })

        print(f"   -> {fetched_count:,}건 스캔 완료... (조건 부합 타겟 데이터: {len(parsed_data):,}건 포착)")

        # 🟡 [안전장치 1] 수집 기점(2026년)에서 역추적하다가 2025년 11월보다 과거로 넘어가면 탈출
        if last_review_date < start_date:
            print(f"   [성공] 목표 타임라인 하한선({start_date.strftime('%Y-%m-%d')}) 이전 데이터 영역 도달. 동적 수집을 안전하게 백업 종료합니다.")
            break

        # 🟡 [안전장치 2] 구글 서버 스래싱 및 IP 차단 방지를 위한 맥스 세이프티 가드라인
        if fetched_count >= max_safety_limit:
            print(f"   [경고] 최대 안전 리밋({max_safety_limit:,}건)에 도달하여 강제 탈출합니다.")
            break

        # 구글 패치 서버 API 호출 딜레이 매너 슬립 (0.5초)
        time.sleep(0.5)

    return pd.DataFrame(parsed_data)


def main():
    print("=" * 50)
    print("[실전 가동] 구글 플레이스토어 실 데이터 역추적 수집 파이프라인")
    print("=" * 50)

    wr_df = scrape_game_reviews_timeline('com.riotgames.league.wildrift', 'Wild_Rift')
    mlbb_df = scrape_game_reviews_timeline('com.mobile.legends', 'Mobile_Legends')

    df_raw = pd.concat([wr_df, mlbb_df], ignore_index=True)

    if df_raw.empty:
        print("❌ [오류] 조건에 맞는 실제 데이터가 수집되지 않았습니다.")
        return

    print("\n[전처리 엔진 이식] 데이터 클리닝 및 연구자 검수용 바인딩 빌드 시작...")

    df_pipe = df_raw.dropna(subset=['content']).drop_duplicates(subset=['reviewId'])
    df_pipe = df_pipe[df_pipe['content'].str.len() >= 10]

    # 🔴 [버그 1 교정] 키워드 인자 제거하고 clean_real_text 매핑 구조로 정상 복구
    df_pipe['content_cleaned'] = df_pipe['content'].apply(clean_real_text)
    df_pipe = df_pipe.dropna(subset=['content_cleaned'])
    df_pipe['content_lower'] = df_pipe['content_cleaned'].str.lower()

    # 🔴 [버그 2 교정] 기계적 랜덤 라벨을 파괴하고, 연구자가 직접 기입할 수 있도록 -1 미검수태그 마킹
    df_pipe['Sentiment'] = -1

    df_final = df_pipe[['reviewId', 'game', 'date', 'content_lower', 'Sentiment']].reset_index(drop=True)
    df_final.to_csv("moba_reviews_cleaned_v2.csv", index=False)

    print("-" * 50)
    print(f"🎯 [최종 완료] 실전 원본 코퍼스 빌드 성공! 파일명: moba_reviews_cleaned_v2.csv")
    print(f"📊 최종 확보된 무결성 실제 건수: {len(df_final):,}건")
    print("📢 [지침] 엑셀이나 메모장으로 CSV를 열어 상위 2,000건에 0~3 라벨을 직접 기입한 뒤 Step 3를 가동하세요!")
    print("=" * 50)


if __name__ == "__main__":
    main()