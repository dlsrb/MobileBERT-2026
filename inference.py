import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# 1. 추론용 커스텀 데이터셋 클래스
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def run_inference():
    print("=" * 60)
    print(" 5-1단계: MobileBERT 대규모 데이터 감성 추론(Inference) 시작")
    print("=" * 60)

    # 1. 환경 및 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ 추론 디바이스: {device}")

    # 2. 원본 데이터 로드
    raw_data_path = "moba_reviews_raw_50k.csv"
    if not os.path.exists(raw_data_path):
        print(f"[오류] {raw_data_path} 파일이 없습니다.")
        return

    df = pd.read_csv(raw_data_path)

    # [GPT 조언 반영] 중복 리뷰 데이터 사전 차단
    if 'reviewId' in df.columns:
        df = df.drop_duplicates(subset=['reviewId']).copy()

    # 결측치(NaN) 및 빈 문자열 완벽 방어
    df = df.dropna(subset=['content']).copy()
    df = df[df['content'].astype(str).str.strip() != ""].copy()
    df['content'] = df['content'].astype(str)

    print(f"▶ 전처리 완료된 추론 대상 데이터: 총 {len(df)}건")

    # 기존 생성된 라벨을 정답지(true_label)로 일원화
    if 'sentiment_label' in df.columns:
        df = df.rename(columns={'sentiment_label': 'true_label'})
    else:
        df['true_label'] = df['score'].apply(
            lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
        )

    # 3. 학습된 모델 및 토크나이저 로드
    model_dir = "./mobilebert_finetuned"
    if not os.path.exists(model_dir):
        print(f"[오류] 학습된 모델 폴더({model_dir})가 없습니다. 4단계를 확인하세요.")
        return

    print("▶ 학습된 MobileBERT 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # 4. 데이터 로더 세팅 (Batch Size 32로 OOM 방지)
    batch_size = 32
    dataset = InferenceDataset(df['content'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predicted_labels = []
    confidences = []

    # 5. 추론 루프 시작 (tqdm 적용으로 진행률 표시)
    print(f"▶ AI 감성 예측 진행 중 (배치 크기: {batch_size})...")

    label_map = {0: 'negative', 1: 'positive'}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference Progress"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            predicted_labels.extend([label_map[p.item()] for p in preds])
            confidences.extend([round(prob.item(), 4) for prob in max_probs])

    # 6. 결과 병합
    df['predicted_label'] = predicted_labels
    df['confidence'] = confidences

    # 7. 컬럼 존재 여부 동적 체크 및 저장 로직
    columns_to_save = ['reviewId', 'score', 'true_label', 'predicted_label', 'confidence', 'content']

    if 'game_title' in df.columns:
        columns_to_save.insert(1, 'game_title')
    if 'at' in df.columns:
        columns_to_save.append('at')

    # 실제 존재하는 컬럼만 최종 추출
    final_columns = [col for col in columns_to_save if col in df.columns]
    final_df = df[final_columns]

    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predicted_reviews.csv")

    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f" 추론 완료! 결과 파일 저장 성공: {output_path}")
    print("=" * 60)

    print("\n▶ AI 모델 예측 결과 요약 (Predicted Labels):")
    print(final_df['predicted_label'].value_counts())
    print("\n▶ 100% 예측 완료. 이제 시각화(Visualization) 코드로 넘어갈 준비가 되었습니다.")


if __name__ == "__main__":
    run_inference()