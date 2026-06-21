# ==========================================
# [2회차] Step 3: MobileBERT 4-Class 학습 및 고속 배치 추론 파이프라인
# (오차 폭발 완치 및 데이터 누수 방어 완전판)
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print("[Step 3] 문맥 동기화 기반 MobileBERT 엔진 가동")
    print("사용하는 장치 : ", device)
    print("=" * 50)

    logging.set_verbosity_error()

    # Step 1에서 정합성이 확보된 CSV 로드
    path = "moba_reviews_cleaned_v2.csv"
    df = pd.read_csv(path)

    # 데이터 누수 차단 추적 플래그 초기화
    df['is_train'] = 0

    # 3장 수동 검수 스펙 2,000건 무작위 추출 및 마킹
    df_train_sample = df.sample(n=2000, random_state=2026)
    df.loc[df_train_sample.index, 'is_train'] = 1

    df_train_sample = df_train_sample.reset_index(drop=True)
    text = list(df_train_sample["content_lower"].values)
    labels = df_train_sample["Sentiment"].values

    print("\n=== 정합성 검증 확인 (상위 2건 매핑 스냅샷) ===")
    print(" 문장: ", text[:2])
    print(" 라벨 : ", labels[:2])  # 이제 부정 텍스트에는 정확히 0번 라벨이 매칭됩니다.

    # 토큰화 진행
    tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
    inputs = tokenizer(text, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 🔴 단 한 줄로 마스크와 토큰 ID를 동시 셔플 (매핑 불일치 버그 원천 차단)
    tx, vx, tm, vm, ty, vy = train_test_split(
        input_ids, attention_mask, labels,
        test_size=0.2, random_state=2026
    )

    batch_size = 8

    train_data = TensorDataset(torch.tensor(tx), torch.tensor(tm), torch.tensor(ty))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(torch.tensor(vx), torch.tensor(vm), torch.tensor(vy))
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    # 모델 선언 및 최적화 설정
    model = MobileBertForSequenceClassification.from_pretrained("mobilebert-uncased", num_labels=4)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epoch = 4

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_dataloader),
        num_training_steps=len(train_dataloader) * epoch
    )

    # 학습 및 평가 루프
    epoch_results = []
    for e in range(epoch):
        model.train()
        total_train_loss = 0.0
        process_bar = tqdm(train_dataloader, desc=f"Training epoch {e + 1}", leave=False)

        for batch in process_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch
            model.zero_grad()

            output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
            loss = output.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            process_bar.set_postfix({'loss': loss.item()})
        avg_train_loss = total_train_loss / len(train_dataloader)

        # 학습 정확도 평가
        model.eval()
        train_preds, train_true = [], []
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch
            with torch.no_grad():
                output = model(batch_ids, attention_mask=batch_mask)
            preds = torch.argmax(output.logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())
        train_acc = np.sum(np.array(train_preds) == np.array(train_true)) / len(train_preds)

        # 검증 정확도 평가
        valid_preds, valid_true = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch
            with torch.no_grad():
                output = model(batch_ids, attention_mask=batch_mask)
            preds = torch.argmax(output.logits, dim=1)
            valid_preds.extend(preds.cpu().numpy())
            valid_true.extend(batch_labels.cpu().numpy())
        valid_acc = np.sum(np.array(valid_preds) == np.array(valid_true)) / len(valid_preds)

        epoch_results.append([avg_train_loss, train_acc, valid_acc])

    print("\n=== 학습 및 검증 결과 (오차 폭발 해결 완료) ===")
    for idx, (loss, tacc, vacc) in enumerate(epoch_results, start=1):
        print(f"Epoch {idx}: 학습오차 - {loss:.4f}, 학습정확도 - {tacc:.4f}, 검증정확도 - {vacc:.4f}")

    model.save_pretrained("mobilebert_mo_v2")
    print("\n=== 모델 가중치 저장 완료 (mobilebert_mo_v2) ===")

    # [4장 기획] 고속 배치 추론 및 데이터 누수 격리부
    print("\n=== [4장 기획] 전체 데이터 추론 시작 (학습 샘플 격리 + 고속 배치 적용) ===")
    model.eval()

    df_infer = df[df['is_train'] == 0].reset_index(drop=True)
    infer_texts = list(df_infer["content_lower"].values)
    infer_preds = []

    inference_batch_size = 256
    for start_idx in tqdm(range(0, len(infer_texts), inference_batch_size), desc="Inferencing"):
        batch_texts = infer_texts[start_idx:start_idx + inference_batch_size]
        tokens = tokenizer(batch_texts, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
        b_ids = tokens["input_ids"].to(device)
        b_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(b_ids, attention_mask=b_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        infer_preds.extend(preds.cpu().numpy())

    df.loc[df['is_train'] == 0, 'pred_label'] = infer_preds
    df.loc[df['is_train'] == 1, 'pred_label'] = df.loc[df['is_train'] == 1, 'Sentiment']
    df['pred_label'] = df['pred_label'].astype(int)

    df.to_csv("moba_reviews_predicted_v2.csv", index=False)
    print(f"💾 4단계 입력용 데이터 바인딩 완료: moba_reviews_predicted_v2.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()