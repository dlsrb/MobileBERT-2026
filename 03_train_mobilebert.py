import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import MobileBertTokenizerFast, MobileBertForSequenceClassification, get_linear_schedule_with_warmup, \
    logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm


def main():
    # ----------------------------------------------------
    # 1. 환경 및 폰트 설정 (GPU 가속 확인 및 경고 제거)
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 사용하는 장치 : ", device)
    logging.set_verbosity_error()

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = 'Arial' if 'Arial' in available_fonts else 'DejaVu Sans'

    # ----------------------------------------------------
    # 2. 클로드 정제 완료 파일 로드
    # ----------------------------------------------------
    file_name = "샘플_2000건_cleaned.csv"
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"❌ '{file_name}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

    df = pd.read_csv(file_name, encoding='utf-8-sig')
    df['Sentiment'] = df['Sentiment'].astype(int)
    df['content_lower'] = df['content_lower'].astype(str)

    text = df['content_lower'].tolist()
    labels = df['Sentiment'].tolist()

    print("\n=== 데이터 분포 확인 ===")
    print(f"총 샘플 수: {len(df)}건")
    print("클래스별 분포:\n", df['Sentiment'].value_counts())

    # ----------------------------------------------------
    # 3. 텍스트 데이터 토큰화 (MobileBERT 규격)
    # ----------------------------------------------------
    tokenizer = MobileBertTokenizerFast.from_pretrained('google/mobilebert-uncased')
    inputs = tokenizer(text, truncation=True, max_length=128, add_special_tokens=True, padding="max_length")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # ----------------------------------------------------
    # 4. 데이터 분리 (계층적 분할로 클래스 비율 유지)
    # ----------------------------------------------------
    tx, vx, ty, vy = train_test_split(input_ids, labels, test_size=0.2, random_state=42, stratify=labels)
    tm, vm, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=42, stratify=labels)

    # ----------------------------------------------------
    # 5. [버그 방어] 클래스 가중치 동적 계산 (0 나누기 방지)
    # ----------------------------------------------------
    class_counts = np.bincount(ty, minlength=4)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # 0건 방어
    computed_weights = len(ty) / (4.0 * class_counts)
    class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float).to(device)
    print(f"⚖️ 계산된 클래스 가중치 (Class Weights): {computed_weights}")

    # ----------------------------------------------------
    # 6. 파이토치 DataLoader 빌드
    # ----------------------------------------------------
    batch_size = 16

    train_data = TensorDataset(torch.tensor(tx), torch.tensor(tm), torch.tensor(ty))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(torch.tensor(vx), torch.tensor(vm), torch.tensor(vy))
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    print(f"📦 Train Batches: {len(train_dataloader)} | Validation Batches: {len(valid_dataloader)}")

    # ----------------------------------------------------
    # 7. 사전학습 언어모델 및 최적화 설정 (4진 분류: num_labels=4)
    # ----------------------------------------------------
    model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * epochs
    )

    # 불균형 방어용 가중치가 주입된 로우레벨 손실함수 정의
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 조기 종료(Early Stopping) 제어 변수
    best_val_f1 = 0.0
    patience = 2
    patience_counter = 0
    save_dir = "./best_mobilebert_model"

    # ----------------------------------------------------
    # 8. 정석 파이토치 학습 및 검증 루프
    # ----------------------------------------------------
    print("\n🔥 모델 학습 및 다중 지표 검증을 시작합니다...")

    for e in range(epochs):
        # --- [1] Train Phase ---
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_dataloader, desc=f"Training Epoch {e + 1}/{epochs}", leave=False)

        for batch in train_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch

            model.zero_grad()

            # Forward Pass (출력 도출)
            outputs = model(batch_ids, attention_mask=batch_mask)
            logits = outputs.logits

            # 가중치가 반영된 손실 연산
            loss = criterion(logits, batch_labels)
            total_train_loss += loss.item()

            # Backward Pass (역전파 및 최적화)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그래디언트 클리핑
            optimizer.step()
            scheduler.step()

            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- [2] Validation Phase (논문용 Macro 지표 계산) ---
        model.eval()
        valid_preds, valid_true = [], []
        val_bar = tqdm(valid_dataloader, desc=f"Evaluating Epoch {e + 1}/{epochs}", leave=False)

        for batch in val_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch

            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_mask)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            valid_preds.extend(preds.cpu().numpy())
            valid_true.extend(batch_labels.cpu().numpy())

        # Sklearn 기반 논문 디펜스용 다중 지표 연산
        val_acc = accuracy_score(valid_true, valid_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            valid_true, valid_preds, average='macro', zero_division=0
        )

        print(
            f"📢 [Epoch {e + 1}] Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        # --- [3] Best Model Save & Early Stopping Check ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0  # 카운터 초기화

            # 베스트 모델 및 토크나이저 명시적 저장
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"   🌟 검증 Macro F1 개선됨! 베스트 모델 저장 완료 -> {save_dir}")
        else:
            patience_counter += 1
            print(f"   ⚠️ 성능 개선 없음 (Patience: {patience_counter}/{patience})")

        if patience_counter >= patience:
            print("🛑 Early Stopping 조기 종료 조건 만족. 학습을 중단합니다.")
            break

    # ----------------------------------------------------
    # 9. 최종 검증 결과 바탕으로 논문용 혼동 행렬 시각화
    # ----------------------------------------------------
    print("\n📊 최종 베스트 모델 기준 혼동 행렬(Confusion Matrix)을 생성합니다...")

    # 평가를 위해 저장했던 베스트 가중치 다시 로드
    best_model = MobileBertForSequenceClassification.from_pretrained(save_dir).to(device)
    best_model.eval()

    final_preds, final_true = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        batch_ids, batch_mask, batch_labels = batch
        with torch.no_grad():
            outputs = best_model(batch_ids, attention_mask=batch_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        final_preds.extend(preds.cpu().numpy())
        final_true.extend(batch_labels.cpu().numpy())

    # 혼동 행렬 도출
    cm = confusion_matrix(final_true, final_preds)

    # 영문 시각화 차트 그리기
    sns.set_theme(style="white", font=selected_font)
    plt.figure(figsize=(8, 7))

    class_names = ['0: Neg', '1: Pos', '2: Neu', '3: Irr']
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, annot_kws={"size": 14, "weight": "bold"}
    )

    plt.title('Confusion Matrix: Loss-Weighted MobileBERT (PyTorch)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xticks(fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    output_matrix_file = 'readme 데이터용 사진파일/04_model_confusion_matrix.png'
    plt.savefig(output_matrix_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 모든 작업이 끝났습니다! 혼동 행렬 차트 저장 완료 -> '{output_matrix_file}'")


if __name__ == "__main__":
    main()