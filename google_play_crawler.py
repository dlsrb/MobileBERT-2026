import time
import re
import random
from datetime import datetime, timedelta, timezone
import pandas as pd
from google_play_scraper import Sort, reviews


def collect_moba_reviews_pipeline(app_id, game_name, target_count=25000):
    print(f"[{game_name}] 파이프라인 수집 시작 (목표 유효 데이터: {target_count}건)")

    # 1. 기준선은 무조건 UTC 타임존 명시 (Aware)
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
    print(f"[{game_name}] 유효 날짜 기준선 (최근 6개월): {six_months_ago.strftime('%Y-%m-%d')} 이후")

    valid_reviews = []
    processed_review_ids = set()  # 중복 체크용 해시셋
    continuation_token = None
    batch_size = 200

    # 제어 및 안정성 파라미터
    stop_collection = False
    retry_count = 0
    MAX_RETRIES = 5

    while len(valid_reviews) < target_count and not stop_collection:
        try:
            result, continuation_token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token
            )

            retry_count = 0

            if not result:
                print(f"[{game_name}] 더 이상 가져올 수 있는 스토어 리뷰가 없습니다.")
                break

            for rev in result:
                # 2. 평점 결측치 및 범위 검증
                score = rev.get('score')
                if score is None or not (1 <= score <= 5):
                    continue

                # 3. 날짜 데이터 결측치 방어
                review_date = rev.get('at')
                if review_date is None:
                    continue

                # 💡 [버그 원천 차단] 스크래퍼가 naive를 뱉든 aware를 뱉든 관계없이 UTC aware로 통일
                if review_date.tzinfo is None:
                    review_date = review_date.replace(tzinfo=timezone.utc)

                # 4. 이제 양쪽 다 확실한 UTC aware 상태이므로 에러 없이 대소 비교 가능
                if review_date < six_months_ago:
                    print(f"[{game_name}] 최근 6개월 범위를 벗어난 데이터 도달. 스캔을 종료합니다.")
                    stop_collection = True
                    break

                # 5. 중복 reviewId 차단
                r_id = rev.get('reviewId')
                if r_id in processed_review_ids:
                    continue

                # 6. 본문 결측치 및 타입 검증
                content = rev.get('content')
                if not content or not isinstance(content, str):
                    continue
                content_str = content.strip()

                # 7. URL 링크 제거
                content_str = re.sub(r'http\S+', '', content_str).strip()

                # 8. 연속 공백 정제
                content_str = " ".join(content_str.split())

                # 9. 문장 길이 필터링
                if len(content_str) < 10:
                    continue

                # 10. 이모지, 특수문자, 숫자 도배 리뷰 제거
                if not re.search(r'[a-zA-Z]', content_str):
                    continue

                # 11. 의미 없는 반복 문자열 제거
                unique_chars = len(set(content_str))
                if unique_chars <= 2:
                    continue
                if content_str.count(content_str[0]) / len(content_str) > 0.7:
                    continue

                # 12. 3점 중립 가이드라인 정립 및 사전 감성 라벨링
                if score in [1, 2]:
                    sentiment_label = 'negative'
                elif score in [4, 5]:
                    sentiment_label = 'positive'
                else:
                    sentiment_label = 'neutral'

                processed_review_ids.add(r_id)
                valid_reviews.append({
                    'reviewId': r_id,
                    'score': score,
                    'sentiment_label': sentiment_label,
                    'content': content_str,
                    'at': review_date,
                    'thumbsUpCount': rev.get('thumbsUpCount') or 0,
                    'replyContent': rev.get('replyContent') or "",
                    'repliedAt': rev.get('repliedAt') or ""
                })

                if len(valid_reviews) >= target_count:
                    break

            print(f"[{game_name}] 실시간 정제 완료 수량: {len(valid_reviews)} / {target_count}")
            time.sleep(1.5 + random.uniform(0.1, 0.5))

        except Exception as e:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                print(f"[{game_name}] 예외 혹은 차단이 연속 {MAX_RETRIES}회 발생하여 조기 종료합니다.")
                break

            wait_time = min(30 * (2 ** (retry_count - 1)), 300)
            print(f"[{game_name}] 에러 발생: {e} -> {wait_time}초 대기 후 재시도 ({retry_count}/{MAX_RETRIES})")
            time.sleep(wait_time)
            continue

    df = pd.DataFrame(valid_reviews)
    if not df.empty:
        df['game_title'] = game_name
    return df


if __name__ == "__main__":
    wild_rift_df = collect_moba_reviews_pipeline(
        app_id='com.riotgames.league.wildrift',
        game_name='Wild_Rift',
        target_count=25000
    )

    mlbb_df = collect_moba_reviews_pipeline(
        app_id='com.mobile.legends',
        game_name='Mobile_Legends',
        target_count=25000
    )

    print("\n" + "=" * 50)
    print("수집 및 1차 정제 파이프라인 최종 점검 로그")
    print("=" * 50)

    print(f"Wild Rift 수집 성공 수량: {len(wild_rift_df)}건")
    if not wild_rift_df.empty:
        print(f"↳ 수집 유효 기간: {wild_rift_df['at'].min()} ~ {wild_rift_df['at'].max()}")

    print(f"Mobile Legends 수집 성공 수량: {len(mlbb_df)}건")
    if not mlbb_df.empty:
        print(f"↳ 수집 유효 기간: {mlbb_df['at'].min()} ~ {mlbb_df['at'].max()}")

    total_moba_reviews = pd.concat([wild_rift_df, mlbb_df], ignore_index=True)

    if total_moba_reviews.empty:
        print("\n[위험] 수집된 유효 데이터가 존재하지 않습니다.")
    else:
        total_moba_reviews.reset_index(drop=True, inplace=True)
        output_filename = "moba_reviews_raw_50k.csv"
        total_moba_reviews.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n최종 통합본 파일 저장 완료: {output_filename}")
        print(f"최종 데이터 세트 차원(행, 열): {total_moba_reviews.shape}")