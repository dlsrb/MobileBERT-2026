import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import sys

# 1. 한글 폰트 설정
os_name = platform.system()
if os_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif os_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif os_name == 'Linux':
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# 2. 실제 엑셀 파일 불러오기 (예외 처리 적용)
file_name = '수동라벨링 3000건.xlsx'

try:
    # CSV가 아닌 Excel 파일이므로 read_excel 사용
    df = pd.read_excel(file_name)
except FileNotFoundError:
    print(f"❌ 에러: '{file_name}' 파일을 찾을 수 없습니다.")
    print("스크립트와 동일한 폴더에 파일이 있는지, 파일명이 정확한지 확인해 주세요.")
    sys.exit() # 에러 발생 시 이후 코드 실행 중단

# 3. 실제 데이터 컬럼명 매핑 (수정 완료)
star_col = 'score'             # 실제 별점 컬럼명
manual_label_col = 'manual_label'  # 실제 수동 라벨링 컬럼명

# 안전장치: 데이터프레임에 해당 컬럼이 존재하는지 다시 한번 확인
if star_col not in df.columns or manual_label_col not in df.columns:
    print(f"❌ 에러: 데이터 내에 '{star_col}' 또는 '{manual_label_col}' 컬럼이 없습니다.")
    print(f"현재 파일에 있는 컬럼 목록: {list(df.columns)}")
    sys.exit()

# 4. Y축 기준 변경 (별점 그룹화)
def group_star(star_value):
    if star_value in [1, 2]:
        return '부정 (1~2점)'
    elif star_value == 3:
        return '중립 (3점)'
    elif star_value in [4, 5]:
        return '긍정 (4~5점)'
    else:
        return '기타'

df['star_group'] = df[star_col].apply(group_star)

# 5. 교차표 생성
heatmap_data = pd.crosstab(df['star_group'], df[manual_label_col])

# 6. 축 정렬 및 안전한 컬럼 매핑
y_order = ['부정 (1~2점)', '중립 (3점)', '긍정 (4~5점)']
if '기타' in heatmap_data.index:
    y_order.append('기타')
heatmap_data = heatmap_data.reindex(y_order, fill_value=0)

label_map = {0: '부정(0)', 1: '긍정(1)', 2: '중립(2)', 3: '관계없음(3)'}
heatmap_data = heatmap_data.rename(columns=label_map)

for col_name in label_map.values():
    if col_name not in heatmap_data.columns:
        heatmap_data[col_name] = 0

heatmap_data = heatmap_data[['부정(0)', '긍정(1)', '중립(2)', '관계없음(3)']]

# 7. 히트맵 시각화
plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=',.0f',
    cmap='Purples',
    cbar=True,
    annot_kws={"size": 12, "weight": "bold"}
)

# 8. 타이틀 및 라벨 설정
plt.title('별점 구간과 텍스트 감성의 교차 분석 (3,000건 수동 라벨링 기준)', fontsize=16, pad=20, weight='bold')
plt.ylabel('유저가 누른 별점 그룹', fontsize=13)
plt.xlabel('직접 분류한 텍스트 감성 라벨', fontsize=13)

# 9. 그래프 저장 및 출력
plt.tight_layout()

# [이 부분이 저장 코드입니다] 화면에 띄우기 전에 먼저 저장을 해야 합니다!
plt.savefig('readme 데이터용 사진파일/수동라벨링_히트맵_결과.png', dpi=300, bbox_inches='tight')

# 화면에 창 띄우기
plt.show()