import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform


def set_korean_font():
    # 1. 스타일을 먼저 적용 (이후 폰트 설정이 덮어씌워지지 않도록 함)
    sns.set_style("whitegrid")

    # 2. 운영체제에 맞는 한글 폰트 자동 설정
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':  # Mac OS
        plt.rc('font', family='AppleGothic')
    else:  # Linux (Google Colab 등)
        plt.rc('font', family='NanumGothic')

    # 3. 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False


def main():
    print("⏳ 데이터 불러오는 중...")

    # 예외 처리 1: 파일 존재 여부 확인
    try:
        df = pd.read_csv('moba_reviews_predicted_final.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        print("❌ 에러: 'moba_reviews_predicted_final.csv' 파일을 찾을 수 없습니다.")
        return

    # 예외 처리 2: 컬럼 존재 여부 사전 검증 (KeyError 방지)
    date_col = 'date' if 'date' in df.columns else ('Date' if 'Date' in df.columns else None)
    game_col = 'game' if 'game' in df.columns else ('Game' if 'Game' in df.columns else None)

    if not date_col or not game_col:
        print(f"❌ 에러: 필수 컬럼 누락 (확인된 컬럼: {df.columns.tolist()})")
        return

    # 날짜 데이터 전처리 (결측치나 이상치 문자열은 NaT로 변환)
    df['year_month'] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m')

    # 예외 처리 3: isinstance(x, str) 추가로 NaN/Float 데이터로 인한 TypeError 방어
    df['game_name'] = df[game_col].apply(
        lambda x: 'Mobile Legends (MLBB)' if isinstance(x, str) and 'Mobile_Legends' in x else 'Wild Rift'
    )

    # 월별 리뷰 건수 집계 (날짜가 결측치인 행은 dropna로 안전하게 제거 후 집계)
    monthly_counts = df.dropna(subset=['year_month']).groupby(['year_month', 'game_name']).size().reset_index(
        name='Review_Count')

    # 폰트 및 스타일 적용 (반드시 figure 생성 전에 호출)
    set_korean_font()

    # 도화지 생성
    plt.figure(figsize=(14, 6))

    # 꺾은선 그래프 생성
    ax = sns.lineplot(data=monthly_counts, x='year_month', y='Review_Count', hue='game_name',
                      marker='o', markersize=8, linewidth=2.5, palette=['#d62728', '#1f77b4'])

    # 제목 및 축 레이블 설정
    plt.title('2025-11 ~ 2026-05 월별 리뷰 수 변화 추이', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('연월 (Year-Month)', fontsize=12)
    plt.ylabel('월간 리뷰 수 (건)', fontsize=12)

    # 범례 설정
    plt.legend(title='게임명', loc='upper left', fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 저장 및 메모리 누수 방지
    filename = 'readme 데이터용 사진파일/02_2_monthly_volume_trend.png'
    plt.savefig(filename, dpi=300)

    # 예외 처리 4: 반복 실행 시 메모리 누수(Memory Leak) 방지를 위한 명시적 종료
    plt.close()

    print(f"✅ '{filename}' 저장 완료!")


if __name__ == "__main__":
    main()