import numpy as np
import turtle
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =========================================================
# 1. データ準備 (Iris データセット & 2Dマップ設定)
# =========================================================
np.random.seed(42)
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
NUM_CLASSES = len(class_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================================
# 2. パルスネットワーク・セルの定義
# =========================================================
NUM_CELLS = 70

class PulseCell:
    def __init__(self, cell_id):
        self.id = cell_id
        self.x = np.random.uniform(-300, 300)
        self.y = np.random.uniform(-200, 200)
        
        self.a = np.random.uniform(-1.5, 1.5, 4)
        self.class_tendency = np.zeros(NUM_CLASSES)
        
        self.links = {}  # {next_id: 結合ウェイト}
        self.pulse_energy = 0.0
        self.pulse_type = "none" # "periodic" or "isolated"
        self.visits = 0

memory_space = [PulseCell(i) for i in range(NUM_CELLS)]

# リンク（道路網）の自動構築：ユークリッド距離が近いノード間を結合
for i, cell_a in enumerate(memory_space):
    for j, cell_b in enumerate(memory_space):
        if i != j:
            dist = np.hypot(cell_a.x - cell_b.x, cell_a.y - cell_b.y)
            if dist < 110:
                cell_a.links[j] = 1.0 / (dist + 1.0)

# =========================================================
# 3. Turtle グラフィックスの設定と描画関数
# =========================================================
screen = turtle.Screen()
screen.setup(width=900, height=700)
screen.bgcolor("#0B0B16")
screen.title("Dual-Mode: Auto-Stream & Interactive Pulse Map (Iris)")
screen.tracer(0)

t_drawer = turtle.Turtle()
t_drawer.hideturtle()
t_drawer.speed(0)

# 状態管理変数
auto_stream_active = True
current_sample_idx = 0
current_epoch = 0
perm = np.random.permutation(len(X_train))

def draw_pulse_map(status_info="Running Dual Mode (Auto-Stream + Manual Clicks)"):
    t_drawer.clear()
    
    # マップ外枠
    t_drawer.penup()
    t_drawer.goto(-420, -310)
    t_drawer.pendown()
    t_drawer.pencolor("#333355")
    t_drawer.pensize(3)
    for _ in range(2):
        t_drawer.forward(840)
        t_drawer.left(90)
        t_drawer.forward(620)
        t_drawer.left(90)
        
    # 操作ガイドとステータス表示
    t_drawer.penup()
    t_drawer.goto(-400, 270)
    t_drawer.pencolor("#FFEE00")
    t_drawer.write("CONTROLS: [SPACE] Toggle Auto-Stream | [Left-Click] Isolated | [Right-Click] Periodic", font=("Courier", 9, "bold"))
    
    t_drawer.penup()
    t_drawer.goto(-400, 245)
    status_color = "#66FF66" if auto_stream_active else "#FF6666"
    t_drawer.pencolor(status_color)
    mode_str = "AUTO-STREAM: ON" if auto_stream_active else "AUTO-STREAM: PAUSED"
    t_drawer.write(f"{mode_str} | {status_info}", font=("Courier", 10, "normal"))

    # リンク（道路網）の描画
    for cell in memory_space:
        for n_id, weight in cell.links.items():
            target = memory_space[n_id]
            t_drawer.penup()
            t_drawer.goto(cell.x, cell.y)
            t_drawer.pendown()
            t_drawer.pencolor("#222244")
            t_drawer.pensize(max(1, weight * 2.5))
            t_drawer.goto(target.x, target.y)

    # セル（ノード）の描画
    for cell in memory_space:
        t_drawer.penup()
        t_drawer.goto(cell.x, cell.y - (5 + cell.pulse_energy * 18))
        t_drawer.pendown()
        
        if cell.pulse_energy > 0.15:
            if cell.pulse_type == "periodic":
                t_drawer.fillcolor("#FFEE00") # 黄色：周期パルス
            else:
                t_drawer.fillcolor("#FF00FF") # マゼンタ：孤立パルス
        else:
            t_drawer.fillcolor("#444466") # 通常時
            
        t_drawer.begin_fill()
        t_drawer.circle(5 + cell.pulse_energy * 15)
        t_drawer.end_fill()

    screen.update()

# =========================================================
# 4. 手動クリックによるパルス注入処理
# =========================================================
def inject_manual_pulse(x_click, y_click, p_type):
    dists = [np.hypot(c.x - x_click, c.y - y_click) for c in memory_space]
    nearest_id = np.argmin(dists)
    
    p_strength = 1.0 if p_type == "isolated" else 0.85
    memory_space[nearest_id].pulse_energy = p_strength
    memory_space[nearest_id].pulse_type = p_type
    memory_space[nearest_id].visits += 1
    
    draw_pulse_map(f"Manual {p_type.capitalize()} Pulse -> Node #{nearest_id}")

# =========================================================
# 5. 非同期自動ストリーミング・ループ (ontimer駆動)
# =========================================================
def auto_stream_step():
    global current_sample_idx, current_epoch, perm
    
    # 全セルのパルスエネルギーを自然減衰
    for cell in memory_space:
        cell.pulse_energy *= 0.65
        if cell.pulse_energy < 0.05:
            cell.pulse_type = "none"
            
    if auto_stream_active:
        # Iris訓練データからサンプルをストリーミング
        idx = perm[current_sample_idx]
        x = X_train[idx]
        true_y = y_train[idx]
        
        dists = [np.linalg.norm(c.a - x) for c in memory_space]
        start_id = np.argmin(dists)
        
        is_periodic = (current_sample_idx % 4 == 0)
        p_type = "periodic" if is_periodic else "isolated"
        p_strength = 0.85 if is_periodic else 1.0
        
        memory_space[start_id].pulse_energy = p_strength
        memory_space[start_id].pulse_type = p_type
        memory_space[start_id].visits += 1
        
        # リンクを通じた波及効果
        for cid, cell in enumerate(memory_space):
            if cell.pulse_energy > 0.3:
                for n_id in cell.links:
                    memory_space[n_id].pulse_energy = min(1.0, memory_space[n_id].pulse_energy + cell.pulse_energy * 0.3)
        
        info = f"Epoch {current_epoch+1} | Stream Step {current_sample_idx+1}/{len(X_train)} | Iris Class: {class_names[true_y]}"
        draw_pulse_map(info)
        
        current_sample_idx += 1
        if current_sample_idx >= len(X_train):
            current_sample_idx = 0
            current_epoch += 1
            perm = np.random.permutation(len(X_train))
    else:
        draw_pulse_map("Stream Paused. Click freely or press SPACE to resume.")

    # 180msごとに次のフレームを非同期で実行（マウス操作を妨げない）
    screen.ontimer(auto_stream_step, 180)

# =========================================================
# 6. キー・マウスイベントのバインド
# =========================================================
def toggle_stream():
    global auto_stream_active
    auto_stream_active = not auto_stream_active

screen.listen()
screen.onkey(toggle_stream, "space")

screen.onclick(lambda x, y: inject_manual_pulse(x, y, "isolated"), btn=1)
screen.onclick(lambda x, y: inject_manual_pulse(x, y, "periodic"), btn=3)

# 初期化とループ開始
draw_pulse_map("System Initialized. Dual-mode active.")
screen.ontimer(auto_stream_step, 500)

turtle.done()
