# Gradient Check 구현 (수치 미분 vs Autograd 결과 비교)

# 1. scala loss 함수를 1개 만든다. (output이 scala로 나오도록)
# 2. 현재 parameter 값에서 auto grad로 구한 값과 중앙차분법으로 계산한 값을 추출한다.
# 3. 이 둘을 비교한다.
# 4. 오차범위가 특정 수치 아래로 나오면 넘어간다.
# 5. 오차범위가 특정 수치 이상이면 h값ㅇ르 조정하며 loss값을 확인한다.
# =======================================
import numpy as np
from dataclasses import dataclass 



# =======================================
@dataclass
class data_Config:
    learning_rate : float = 1e-3
    epoch : int = 1000 # epoch 수
    pos_val : float = 1e-5 # 오차 범위의 기준값을 설정하는 변수
    h : float = 1e-5 # 수치 미분에서 사용할 아주 작은 값
    print_each : bool = True         # 로그 출력 여부



# =======================================
# scala loss 함수 ============================
def loss_func(y_true, y_pred, eps = 1e-9):
    # cross entropy loss

    # ERROR 250927
    # cross entropy 는 분류 문제에서 주로 사용되는 loss.
    # 따라서, 해당 문제에는 적합하지 않은 계산식.

    # y_true 와 y_pred 는 shape이 같은 행렬.
    # y_pred에는 log를 씌워서 true 에서 멀어질수록 더 큰 값을 가지도록 한다.
    # 여기에서 eps는 0이 되지 않게 하기 위한 방어책으로 아주 작은 값을 사용한다.
    y_pred = np.log(y_pred + eps)
    # true와 pred를 곱하여 true에 가까울수록 loss가 0에 가까워지도록 한다.
    val = y_true * y_pred
    # 그리고 모든 원소의 합을 구해서 총 loss를 구한다.
    val = np.sum(val)
    # log 함수에서는 0~1사이의 값은 음의 값을 가지므로 -1을 곱하여 양의 값을 가지도록 한다.
    return -val

def mse_loss(y_true, y_pred):
    # mean squared error loss
    # 평균 제곱 오차
    # 평균 차의 제곱이 작아지도록 하는 loss 함수
    return np.mean((y_true - y_pred) ** 2)

# 수치 미분 (중앙차분법) ===========================
def numerical_func(func, x, h=1e-5):
    # 양쪽의 기울기를 구하여 중앙 기울기 값을 구하는 함수.
    # 여기에서 h는 아주 작은 값으로, x값에서 양 옆의 기울기를 구할 때 사용한다.
    # 여기에서 func는 loss 함수를 의미한다.
    # 먼저 x 에서 h를 뺐을 때의 값이 input으로 들어오면 어떤 기울기가 나올지 구한다.
    prev = func(x - h)
    # 다음으로 x에서 h를 더했을 때의 값이 input으로 들어오면 어떤 기울기가 나올지 구한다.
    next = func(x + h)
    # 그리고 이 둘의 차이를 구한다.
    val = next - prev
    # 이 차이를 2h로 나누어 중앙 기울기를 구한다.
    # 2*h로 나누는 이유는 양 옆의 기울기를 구할 때 각각 h만큼 이동했기 때문이다.
    val = val / (2 * h)
    return val

# 수치 미분 다변수 함수의 수치적 그래디언트 
def numerical_gradient(func, x, h=1e-5):
    grad = np.zeros_like(x) # x와 같은 shape의 0으로 채워진 행렬을 만든다.
    # x의 모든 원소에 대해서 수치 미분을 수행한다.
    for idx in range(x.size):
        tmp = x[idx] # 원소 하나를 꺼냄
        x[idx] = tmp + h # h만큼 더함
        fxh1 = func(x) # f(x + h) 계산
        x[idx] = tmp - h # h만큼 뺌
        fxh2 = func(x) # f(x - h) 계산
        grad[idx] = (fxh1 - fxh2) / (2 * h) # 중앙차분법
        x[idx] = tmp # 원래 값으로 복원
    return grad

# Auto grad =========================
# auto gradient는 자동으로 미분을 해주는 장치로, pytorch 에서는 value라는 클래스로 구현되어 있다.
# 이를 구현하여 auto gradient를 재현한다.
class AutoGrad:
    def __init__(self, data, _children = (), _op = ''):
        # data : 실제 input 값
        # _children : 이전 노드의 포인터를 가리킨다
        # 이전 노드를 기록하는 이유는 backward를 할 때 이전 노드 어디로 가야하는지 알아야 하기 때문이다.
        # _op : 어떤 연산이 사용되었는지를 기록한다 (연산은 +, * 등등...)
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    def __add__(self, other):
        # 더하기 연산
        # isinstance(확인하고자 하는 데이터 값, 확인하고자 하는 데이터 타입)
        # isinstance의 return 은 bool 로 반환한다.
        # 만약 other 의 값이 AutoGrad일 경우, other의 값을 그대로 사용한다.
        if isinstance(other, AutoGrad):
            pass
        else:
            # 만약 other의 값이 AutoGrad가 아닐 경우, AutoGrad로 감싸준다.
            other = AutoGrad(other)

        # input data와 other로 받은 data를 더하여 output data를 만든다.
        out = AutoGrad(self.data + other.data, (self, other), '+')

        # add function에 대한 backward function을 정의한다.
        # 여기에 backward를 정의하는 이유는 연산에 따라 backward가 달라지기 때문이다.
        def _backward():
            # ERROR 250927
            # self.grad 는 other.grad 와 out.grad 의 곱셈을 업데이트 하고,
            # other.grad 는 self.grad 와 out.grad 의 곱셈을 업데이트 한다.
            self.grad = other.grad + out.grad
            other.grad = self.grad + out.grad

        # backward function을 out에 연결하여 backward가 가능하도록 한다.
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        # 곱하기 연산
        if isinstance(other, AutoGrad):
            pass
        else:
            other = AutoGrad(other)

        out = AutoGrad(self.data * other.data, (self, other), '*')

        def _backward():
            # ERROR 250927
            # self.grad 는 other.grad 와 out.grad 의 곱셈을 업데이트 하고,
            # other.grad 는 self.grad 와 out.grad 의 곱셈을 업데이트 한다.
            self.grad = other.grad * out.grad
            other.grad = self.grad * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # 역전파 함수를 정의한다.
        # 값을 저장할 임시 리스트를 만든다.
        topo = []
        # 방문한 노드를 기록할 집합을 만든다.
        visited = set()
        # 위상정렬을 수행하는 함수를 만든다.
        def build_topo(v):
            # 만약 v가 이미 방문한 노드가 아니라면 for문을 돌면서 노드를 방문한다. 
            if v not in visited:
                # 방문한 노드를 visited 집합에 추가한다.
                visited.add(v)
                # 이전 노드가 가지고 있는 노드들을 방문한다.
                for child in v._prev:
                    # 재귀적으로 방문한다.
                    build_topo(child)
                # 방문이 끝난 노드를 topo 리스트에 추가한다.
                topo.append(v)
        build_topo(self)
        # 최종 노드의 기울기를 1로 설정한다.
        self.grad = 1.0
        # 역전파를 수행한다.
        for node in reversed(topo):
            node._backward()


# =======================================
def test_func(x, w, b, y_true, config : data_Config):
    # loss 의 기준 함수
    h = config.h
    ep = 0

    # AutoGrad loss 계산 =========================
    auto_x, auto_w, auto_b = AutoGrad(x), AutoGrad(w), AutoGrad(b)
    # 순전파
    y_pred_auto = auto_x * auto_w + auto_b
    # ERROR 250927
    # loss 를 한 뒤에 역전파를 해야하는데 왜 순전파를 하고 바로 역전파를 했을까??
    # loss 계산
    loss_auto = mse_loss(y_true, y_pred_auto.data)
    # 역전파
    AutoGrad(loss_auto).backward()

    # # 기울기를 업데이트 함
    # auto_x.data = auto_x.data - config.learning_rate * auto_x.grad
    # auto_w.data = auto_w.data - config.learning_rate * auto_w.grad
    # auto_b.data = auto_b.data - config.learning_rate * auto_b.grad

    auto_grad = np.array([auto_x.grad, auto_w.grad, auto_b.grad])

    # 수치 미분 loss 계산 =========================
    # 수치 미분 함수 생성
    # ERROR 250927
    # yhat 계산이 들어가야 하는데 안 넣고 람다를 사용했었음.
    # 그래서 내부 함수를 이용해서 yhat를 계산하도록 함.
    # num_func = lambda v: mse_loss(y_true, v)
    def f_num(theta):
        xv, wv, bv = theta
        yhat = xv * wv + bv
        return mse_loss(y_true, yhat)
        
    theat0 = np.array([x, w, b], dtype=np.float64)
    # 수치 미분 계산
    numerical_grad = numerical_gradient(f_num, theat0, h)

    # ERROR 250927
    # loss 가 아니라 grad 를 비교해야 하는 것임.
    # 그래서 auto grad 값을 vector화 하여 numerical grad 값과 비교함.
    # 오차범위 계산 / 상대 오차 계산 (백터 normalization 기준)
    # linalg : 선형대수 함수 라이브러리
    # norm : 벡터의 크기를 계산하는 함수
    grad_length = np.linalg.norm(auto_grad - numerical_grad)
    # auto grad 와 numerical grad 의 크기를 더한 값이 0이 되는 것을 방지하기 위해 아주 작은 값을 더해준다.
    den = np.linalg.norm(auto_grad) + np.linalg.norm(numerical_grad) + 1e-12
    dif = grad_length / den

    if config.print_each:
        print("=== Gradient Check (Linear + MSE) ===")
        print(f"x0,w0,b0,y_true = {x}, {w}, {b}, {y_true}")
        print("autograd grad [dL/dx, dL/dw, dL/db]:", auto_grad)
        print("numeric  grad [dL/dx, dL/dw, dL/db]:", numerical_grad)
        print("relative error:", dif)
        print(f"threshold(pos_val): {config.pos_val}")


    # 튜닝 루프 =========================
    ep = 0
    while dif > config.pos_val and ep < config.epoch:
        ep += 1
        print(f"\nepoch : {ep} ==============================")

        # 기울기를 업데이트하기 위한 수식
        # 만약 오차범위가 1e-3보다 크면 0.5배, 작으면 0.8배의 h값을 곱한 값을 min의 매개변수로 가짐.
        # 1e-12과 min 사이의 값으로 h값을 조정함.
        h = max(1e-12, min(1e-1, h * (0.5 if dif > 1e-3 else 0.8)))
        numerical_grad = numerical_gradient(f_num, theat0, h)
        grad_length = np.linalg.norm(auto_grad - numerical_grad)
        den = np.linalg.norm(auto_grad) + np.linalg.norm(numerical_grad) + 1e-12
        dif = grad_length / den

        if config.print_each:
            print(f"h={h:.2e}\trel_err={dif:.3e}")
            
    passed = dif <= config.pos_val
    if config.print_each:
        print("PASS" if passed else "FAIL")
                
    return passed, dif, auto_grad, numerical_grad, h

    # while True:

        # numerical_grad = numerical_gradient(num_func, np.array([x, w, b]), h)


                

        # dif = np.abs(auto_grad - numerical_grad)
        # print(f"오차 범위 : {dif}")

        # # 오차범위가 특정 수치 이상일 경우, h의 값을 조정하여 loss 값을 확인한다.
        # if dif > config.pos_val:
        #     if dif > 0:
        #         h = h - config.learning_rate
        #     else:
        #         h = h + config.learning_rate
        #     print(f"[ NOTE ] 오차 범위가 {config.pos_val} 이상이므로 h 값을 {h:.4f}로 조정합니다.")
        # else:
        #     print(f"[ NOTE ] 오차 범위가 {config.pos_val} 이하이므로 학습을 종료합니다.")
        #     break

        # if ep > config.epoch:
        #     print(f"[ NOTE ] 최대 epoch {ep - 1}에 도달하여 학습을 종료합니다.")
        #     break
        # # break




# =======================================
x = 2.0
w = 3.0
b = 1.0
y_t = 7.0

print(x, w, b, y_t)

config = data_Config()

passed, rel_err, g_auto, g_num, h_used = test_func(x, w, b, y_t, config)
assert passed, f"Gradient check failed: rel_err={rel_err}, h={h_used}"

# print("x : \n", x, "\ny_true : \n", y_t)