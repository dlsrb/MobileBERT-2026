import matplotlib.pyplot as plt
import seaborn as sns
import platform


def set_korean_font():
    # Seaborn 스타일이 폰트를 초기화하므로, 스타일 설정 이후에 폰트를 지정해야 합니다.
    sns.set_style("whitegrid")

    # 운영체제에 맞는 한글 폰트 자동 설정
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':  # Mac OS
        plt.rc('font', family='AppleGothic')
    else:  # Linux (Google Colab 등)
        plt.rc('font', family='NanumGothic')

    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def main():
    # 전처리 단계별 데이터 건수 (보고서 수치 반영)
    stages = [
        '원시 데이터 수집\n(Raw Data)',
        'Stage 1\n(중복/결측치 제거)',
        'Stage 2\n(단문 필터링)',
        'Stage 3\n(비영문/노이즈 제거)',
        'Stage 4 (최종)\n(정규화 및 유효 데이터)'
    ]

    # Stage 4는 대소문자 통합(정규화) 과정이므로 데이터 건수가 삭제되지 않고 동일하게 유지됨
    counts = [125430, 118920, 114500, 107810, 107810]

    # 폰트 및 스타일 적용 함수 호출
    set_korean_font()

    plt.figure(figsize=(12, 6))

    # 막대그래프 생성
    ax = sns.barplot(x=stages, y=counts, palette="Blues_r")

    # 막대 위에 수치 표시
    for i, v in enumerate(counts):
        ax.text(i, v + 1500, f"{v:,}건", ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.title('데이터 전처리 파이프라인 단계별 데이터 감소 추이', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('전처리 단계', fontsize=12)
    plt.ylabel('데이터 건수 (Reviews)', fontsize=12)
    plt.ylim(0, 140000)

    plt.tight_layout()

    filename = 'readme 데이터용 사진파일/02_1_pipeline_reduction.png'
    plt.savefig(filename, dpi=300)
    plt.close()  # 메모리 누수 방지

    print(f"✅ '{filename}' 저장 완료!")


if __name__ == "__main__":
    main()