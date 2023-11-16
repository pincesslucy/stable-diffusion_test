from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class TrainingConfig:
    image_size = 128  # 생성되는 이미지 해상도
    train_batch_size = 16
    eval_batch_size = 16  # 평가 동안에 샘플링할 이미지 수
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no`는 float32, 자동 혼합 정밀도를 위한 `fp16`
    output_dir = "ddpm-butterflies-128"  # 로컬 및 HF Hub에 저장되는 모델명

    push_to_hub = True  # 저장된 모델을 HF Hub에 업로드할지 여부
    hub_private_repo = False
    overwrite_output_dir = True  # 노트북을 다시 실행할 때 이전 모델에 덮어씌울지
    seed = 0


config = TrainingConfig()