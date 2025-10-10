# MNIST 분류 모델 97% 이상 달성

# Trainer 클래스와 콜백을 활용하여, MNIST 손글씨 숫자 데이터셋을 분류하는 MLP 모델 구축.
# 1. torchvision.dataset.MINIST를 활용하여 데이터셋과 데이터로더를 준비.
# 2. Trainer 객체 생성.
#    CheckpointCallback, EarlyStoppingCallback, LoggingCallback 사용
#    CheckpointCallback : 모델 체크포인트 저장
#    EarlyStoppingCallback : 검증 손실이 일정 epoch 동안 향상되지 않으면 학습 조기 종료
#    LoggingCallback : 학습 로그 출력 (epoch, loss)
# 3. 분류 97% 이상 달성. (lr, parmateter 조정)
# 4. 모든 랜덤 시드 (random, numpy, torch)를 고정했을 때, 항상 동일한 결과가 나와야 함. (재현성)
#    테스트 정확도 +-0.2%
# 5. 실행 후, 학습 로그가 정상적으로 출력되고 가장 좋은 성능의 모델의 체크포인프 파일(.pth)가 실제로 생성되어야 함.


import torch
import time
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

# MNIST 데이터셋 준비
from torchvision import datasets, transforms

# 모델의 기본 구조 정의
@dataclass
class TrainingConfig:
    epochs : int = 100
    learning_rate : float = 1e-4
    batch_size = 64
    hidden_layers : List[int] = field(default_factory = lambda : [512, 256])
    use_mixed_precision: bool = True

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")

# 1. 
# MNIST 데이터를 담는 Dataset 클래스
class MNISTDataset(Dataset) :
    def __init__(self, data : torch.Tensor, targets : torch.Tensor):
        self.data = data
        self.targets = targets

    def __len__(self) -> int :
        return len(self.data)
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], self.targets[idx]

# normalized 된 MNIST 데이터셋을 Tensor 형태로 변환하여 불러옴.
data = datasets.MNIST('dataset/', train = True, download = True,
                      transform = transforms.Compose([
                          transforms.ToTensor(), 
                          transforms.Normalize(mean = (0.5,), std = (0.5,))]))
# 불러온 MNIST 데이터셋을 내가 정의한 Dataset 클래스에 담음.
mnist_data = MNISTDataset(data.data, data.targets)

model_config = TrainingConfig()
# 데이터 로더에 데이터셋을 담아서 정의
mnist_dataloader = DataLoader(
    dataset = mnist_data,
    batch_size = model_config.batch_size,
    shuffle = True
)

# 2.
# Callback을 활용하여 Trainer 객체를 정의.
# Callback 기본 구조 정의
class BaseCallback:
    def on_train_begin(self, trainer): pass
    def on_epoch_begin(self, trainer): pass
    def on_batch_end(self, trainer): pass
    def on_epoch_end(self, trainer): pass
    def on_train_end(self, trainer): pass
    def checkpoint(self, trainer): pass # on_epoch_end 시점에 이전보다 성능이 좋으면 .pth 파일로 모델 저장.
    def earlystoppint(self, trainer): pass # 일정 횟수 이상 성능 향상이 없으면 학습 조기 종료.


# Trainer 클래스 정의
class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, 
                 # optimlr_scheduler._LRScheduler -> learning rate 스케줄러. 학습 도중에 학습률을 동적으로 조정하는 데 사용됨.
                 scheduler: Optional[torch.optimlr_scheduler._LRScheduler] = None,
                 callbacks: Optional[List[BaseCallback]] = None,
                 device: str = "cuda"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.callbacks = callbacks if callbacks else []
        self.device = device
        # state 변수에는 학습의 상태를 지속적으로 업데이트 함.
        self.state = {}

    # 콜백 메서드를 실행하는 함수
    def _run_callbacks(self, event_name : str):
        for callback in self.callbacks:
            # getattr 함수 : 객체의 속성, 메서드를 동적으로 가져올 때 사용함.
            # 여기에서는 콜백의 각 이벤트 메서드를 호출하는 용도로 사용됨.
            # 즉, 콜백에서만 사용되는 함수는 아님.
            getattr(callback, event_name)(self)

    # 학습을 수행하는 파트
    def fit(self, num_epochs: int):
        self._run_callbacks("on_train_begin")
        # 모델 학습 시작
        for i in range(num_epochs):
            self._run_callbacks("on_epoch_begin")
            # 학습/검증 루프 
            for batch in self.train_loader:
                # 배치 학습 루프
                self._run_callbacks("on_batch_end")

            self._run_callbacks("on_epoch_end")
        # TODO: 이 부분에서 만약 state 가 stop 이라면 학습을 멈추도록 예외 처리를 추가해야 함.
        self._run_callbacks("on_train_end")


class LoggingCallBack(BaseCallback):        
    def on_train_begin(self, trainer : Trainer) :
        print("=== Training started ===")
        trainer.state.get('epoch', 0) # epoch를 담을 state
        trainer.state.get('batch', 0) # batch 순번을 담을 state
        trainer.state.get('stop', False) # 학습 종료 여부
        trainer.state.get('best_metric', None) # 가장 좋은 성능을 기록
        trainer.state.get('cur_model_state', None) # 현재 모델의 파라미터
        trainer.state.get('best_model_state', None) # 가장 성능 좋은 모델의 파라미터
        trainer.state.get('no_improve_epochs', 0) # 성능 향상이 없는 epoch 수

    def on_epoch_begin(self, trainer) : 
        next_epoch = trainer.state['epoch'] + 1 # 다음 Epoch 계산
        print(f"=== Epoch {next_epoch} started ===")
        trainer.state['epoch'] = next_epoch # epoch 상태 업데이트

    def on_batch_end(self, trainer) : 
        next_batch = trainer.state['batch'] + 1 # 다음 Batch 계산
        print(f"=== Batch {next_batch} processed ===")
        trainer.state['batch'] = next_batch # batch 상태 업데이트

    def on_epoch_end(self, trainer): 
        epoch = trainer.state['epoch']
        print(f"=== Epoch {epoch} ended ===")

    def on_train_end(self, trainer):
        print("=== Training ended ===")

    # on_epoch_end 시점에 이전보다 성능이 좋으면 .pth 파일로 모델 저장.
    def checkpoint(self, trainer) : 
        cur_model_state = trainer.state['cur_model_state']
        prev_model_state = trainer.state['best_model_state']
        
        if prev_model_state is None or cur_model_state['accuracy'] > prev_model_state['accuracy'] :
            trainer.state['best_model_state'] = cur_model_state # 가장 성능 좋은 모델의 파라미터를 교체
            print("New best model found, saving checkpoint...")
            daytime = time.strftime('%Y%m%d_%H%M%S')
            torch.save(cur_model_state.state_dict(), f'model_weights_{daytime}.pth')

        
    def earlystoppint(self, trainer): # 일정 횟수 이상 성능 향상이 없으면 학습 조기 종료.
        no_impove = trainer.state.get('no_imporve_epochs', 0)
        if no_impove >= 100 :
            trainer.state['stop'] = True
