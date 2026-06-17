import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import MobileBertTokenizerFast, MobileBertForSequenceClassification
from tqdm import tqdm
import logging


def main():
    # ----------------------------------------------------
    # 1. 환경 설정 및 장치 할당
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device for Inference: {device}")
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # ----------------------------------------------------
    # 2. 학습된 베스트 모델 및 토크나이저 로드
    # ----------------------------------------------------
    model_dir = "./best_mobilebert_model"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ '{model_dir}' 폴더가 없습니다. 모델 학습 완료 여부를 확인하세요.")

    print("🧠 Loading Trained MobileBERT Model...")
    tokenizer = MobileBertTokenizerFast.from_pretrained(model_dir)
    model = MobileBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()  # 추론 모드 전환

    # ----------------------------------------------------
    # 3. 10만 건 마스터 원본 데이터 로드 (속도 최적화)
    # ----------------------------------------------------
    master_file = "moba_reviews_cleaned_v3.csv"
    if not os.path.exists(master_file):
        master_file = "moba_reviews_cleaned_v2.csv"

    print(f"📂 Loading Master Data from '{master_file}'...")
    # engine='python' 제거로 C-engine 가동 (로드 속도 대폭 향상)
    df_master = pd.read_csv(master_file, encoding='utf-8-sig', on_bad_lines='skip')
    df_master = df_master.dropna(subset=['content_lower'])
    df_master['content_lower'] = df_master['content_lower'].astype(str)

    # ----------------------------------------------------
    # 4. 타겟팅 및 [버그 해결] 명시적 인덱스 추출
    # ----------------------------------------------------
    unlabeled_mask = df_master['Sentiment'] == -1
    target_indices = df_master[unlabeled_mask].index  # 나중에 정확한 위치에 넣기 위해 인덱스 저장
    texts_to_predict = df_master.loc[target_indices, 'content_lower'].tolist()

    print(f"🎯 Total unlabeled reviews to predict: {len(texts_to_predict):,} rows")

    if len(texts_to_predict) == 0:
        print("✅ 예측할 미검수(-1) 데이터가 없습니다.")
        return

    # ----------------------------------------------------
    # 5. [메모리 최적화] 실시간 토크나이징 커스텀 데이터셋
    # ----------------------------------------------------
    # 10만 건을 한 번에 텐서로 만들면 RAM이 터질 수 있으므로, 배치 단위로 그때그때 토크나이징
    class InferenceDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            # return_tensors="pt"가 만든 (1, max_len) 차원을 (max_len,)으로 축소 (DataLoader 호환용)
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    dataset = InferenceDataset(texts_to_predict, tokenizer)
    batch_size = 64
    inference_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    # ----------------------------------------------------
    # 6. 대규모 추론 (Inference) 시작 및 예외 방어
    # ----------------------------------------------------
    predictions = []

    print("🔥 Starting Memory-Safe Batch Inference...")
    inf_bar = tqdm(inference_dataloader, desc="Predicting")

    for batch in inf_bar:
        try:
            batch_ids, batch_mask = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
        except Exception as e:
            print(f"\n⚠️ Batch Error Skipped: {e}")
            # 배치에서 에러가 날 경우, 인덱스 길이 불일치 대참사를 막기 위해 기본값(-1)으로 채움
            fallback_len = len(batch[0])
            predictions.extend([-1] * fallback_len)

    # ----------------------------------------------------
    # 7. 예측 결과 병합 및 최종 저장 (데이터 타입 동기화)
    # ----------------------------------------------------
    # 추출해둔 명시적 인덱스(target_indices)에 예측값을 정확히 1:1 매핑
    df_master.loc[target_indices, 'Sentiment'] = predictions

    # float 변환 방지 및 소수점 없는 깔끔한 정수형 확정
    df_master['Sentiment'] = df_master['Sentiment'].astype(int)

    final_output_file = "moba_reviews_predicted_final.csv"
    df_master.to_csv(final_output_file, index=False, encoding='utf-8-sig')

    print("\n=== 🎉 Inference Completed! ===")
    print(f"✅ Predicted Data Saved to: '{final_output_file}'")
    print("📊 Final Sentiment Distribution:")
    print(df_master['Sentiment'].value_counts())


if __name__ == "__main__":
    main()