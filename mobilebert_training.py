import os
import random
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score


# 1. 전역 시드 고정 함수 (재현성 완벽 보장)
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 2. 파이토치 커스텀 데이터셋 클래스 정의
class MobaReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# 3. 평가 지표 계산 함수
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    return {'accuracy': acc, 'f1': f1}


def run_mobilebert_training():
    print("=" * 60)
    print(" 4단계: MobileBERT 모델 파인튜닝 시작")
    print("=" * 60)

    # 전역 시드 적용
    set_global_seed(42)

    # 1. Trainer가 텐서 디바이스를 자동 관리하므로 직접 할당에는 미사용 (로깅 전용)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ 학습 디바이스 할당 확인: {device}")

    # 2. 데이터 로드
    train_path = "dataset/train_dataset.csv"
    val_path = "dataset/val_dataset.csv"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[오류] 학습 데이터셋 파일이 존재하지 않습니다. 3단계를 먼저 수행하세요.")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print("\n▶ 학습 데이터(Train) 라벨 분포:")
    print(train_df['sentiment_label'].value_counts())
    print("\n▶ 검증 데이터(Validation) 라벨 분포:")
    print(val_df['sentiment_label'].value_counts())
    print("-" * 60)

    # 3. 라벨 인코딩 및 엄격한 결측치 방어망 구축
    label_map = {'negative': 0, 'positive': 1}

    train_mapped = train_df['sentiment_label'].map(label_map)
    val_mapped = val_df['sentiment_label'].map(label_map)

    if not train_mapped.notna().all():
        raise ValueError("[오류] Train 세트에 매핑 실패한 라벨이 존재합니다.")
    if not val_mapped.notna().all():
        raise ValueError("[오류] Validation 세트에 매핑 실패한 라벨이 존재합니다.")

    train_labels = train_mapped.tolist()
    val_labels = val_mapped.tolist()

    train_texts = train_df['content'].astype(str).tolist()
    val_texts = val_df['content'].astype(str).tolist()

    # 4. 모델 및 토크나이저 동적 로드 (경고 차단)
    model_name = "google/mobilebert-uncased"
    print(f"▶ 사전 학습된 모델 로드 중: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # 5. 텍스트 토큰화 (max_length 고정 패딩)
    print("▶ 텍스트 데이터 토큰화 진행 중 (max_length=128)...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding='max_length',
        max_length=128
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding='max_length',
        max_length=128
    )

    train_dataset = MobaReviewDataset(train_encodings, train_labels)
    val_dataset = MobaReviewDataset(val_encodings, val_labels)

    # fp16 하드웨어 지원 여부를 더욱 엄격하게 검증하여 NaN Loss 방지
    use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7

    # 6. 최적화된 하이퍼파라미터 셋업
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,  # [추가] AI가 차분하게 학습하도록 진도율 조절
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        fp16=False,  # [수정] GPU 가속 충돌(NaN 에러) 원천 차단
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 7. 모델 학습 실행
    print("▶ 파인튜닝 학습 루프 시작...")
    trainer.train()

    # 8. 최종 최고 성능 모델 및 토크나이저 안전 저장
    output_model_dir = "./mobilebert_finetuned"
    os.makedirs(output_model_dir, exist_ok=True)

    # load_best_model_at_end 옵션으로 인해 trainer 안에 최고 성능 모델이 있으므로 trainer를 통해 저장
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print("\n" + "=" * 60)
    print(f" 학습 완료 및 최종 모델 저장 성공: {output_model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_mobilebert_training()