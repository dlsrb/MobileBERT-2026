import pandas as pd
import numpy as np
from huggingface_hub.utils.tqdm import progress_bar_states
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

def main():
    # 0. GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용하는 장치 : ", device)


    # 1. 학습 시 경고 메세지 제거
    logging.set_verbosity_error()

    #2. 데이터 확인
    path = "imdb_reviews_sample.csv"
    df = pd.read_csv(path, encoding="cp949")
    text = list(df["Text"].values)
    labels = df["Sentiment"].values

    print("\n=== 데이터 확인 ===")
    print(" 문장: ", text[:5])
    print(" 라벨 : ", labels[:5])

    # 3. 텍스트 데이터 토큰화
    tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
    inputs = tokenizer(text, truncation= True, max_length=256, add_special_tokens=True, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    num_to_print = 3
    print("\n=== 토큰화 샘플 ===")
    for j in range(num_to_print):
        print(f"\n{j+1}번째 데이터")
        print("토큰: ", input_ids[j])
        print("어텐션 마스크: ", attention_mask[j])

    # 4. 데이터 분리
    tx, vx, ty, vy = train_test_split(input_ids, labels, test_size=0.2, random_state=2026)
    tm, vm, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2026)

    # 5. torch에 학습 시키기 위한 데이터 설정
    batch_size = 8

    train_inputs = torch.tensor(tx)
    train_labels = torch.tensor(ty)
    train_mask = torch.tensor(tm)
    train_data = TensorDataset(train_inputs, train_mask, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_inputs = torch.tensor(vx)
    valid_labels = torch.tensor(vy)
    valid_mask = torch.tensor(vm)
    valid_data = TensorDataset(valid_inputs, valid_mask, valid_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    # 6. 사전학습 언어모델 설정
    model = MobileBertForSequenceClassification.from_pretrained("mobilebert-uncased", num_labels=2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epoch =4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epoch)

    # 7. 학습 및 검증
    epoch_results = []
    for e in range(epoch):
        # --- 학습 단계 ---
        model.train()
        total_train_loss = 0.0

        process_bar = tqdm(train_dataloader, desc=f"Training epoch {e+1}", leave=False)

        for batch in process_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch
            #학습 할 때 그래디언트(모수 업데이트 방향) 초기화
            model.zero_grad()

            # Forward Pass: 데이터를 넣고 라벨과 손실까지 계산한다
            output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
            loss = output.loss
            total_train_loss += loss.item()

            # Backward Pass: 손실 값을 바탕으로 모델의 모수를 업데이트 한다
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # 배치별로 loss를 tqdm 진행 상태에 업데이트 한다
            process_bar.set_postfix({'loss':loss.item()})
        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- 학습 정확도 평가 ---
        # 이 단계는 매우 무겁기 때문에 통상적으로 생략하고 loss를 많이 본다
        model.eval()
        train_preds, train_true = [], []

        progress_bar_t = tqdm(train_dataloader, desc=f"Validation epoch {e+1}", leave=False)
        for batch in progress_bar_t:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch
            with torch.no_grad():
                output = model(batch_ids, attention_mask=batch_mask)
            logits = output.logits
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())
        train_acc = np.sum(np.array(train_preds) == np.array(train_true))/len(train_preds)

        # ---검증 정확도 평가 ---
        # 이 단계는 학습하지 않은 데이터에 대한 정확도 이므로 반드시 해야한다.
        valid_preds, valid_true = [], []

        progress_bar_v = tqdm(valid_dataloader, desc=f"Validation epoch {e+1}", leave=False)
        for batch in progress_bar_v:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch
            with torch.no_grad():
                output = model(batch_ids, attention_mask=batch_mask)
            logits = output.logits
            preds = torch.argmax(logits, dim=1)
            valid_preds.extend(preds.cpu().numpy())
            valid_true.extend(batch_labels.cpu().numpy())
        valid_acc = np.sum(np.array(valid_preds) == np.array(valid_true))/len(valid_preds)

        epoch_results.append([avg_train_loss, train_acc, valid_acc])

    #  8. 결과 저장
    print("\n=== 학습 및 검증 결과 ===")
    for idx, (loss, tacc, vacc) in enumerate(epoch_results, start=1):
        print(f"Epoch {idx}: 학습오차 - {loss:.4f}, 학습정확도 - {tacc:.4f}, 검증정확도 - {vacc:.4f}")

    # 9. 모델 저장
    print("\n=== 모델 저장 ===")
    model.save_pretrained("mobilebert_imdb.pt")
    print(" 모델 저장 완료")


if __name__ == "__main__":
    main()