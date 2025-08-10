import torch
import time

def prepare_transition_model(P, n_states, actions, device='cuda'):
    """
    将 P 转换为适合 GPU 并行的张量形式：
    next_states[s, a] = 下一个状态编号
    rewards[s, a] = 奖励
    """
    next_states = torch.zeros((n_states, len(actions)), dtype=torch.long, device=device)
    rewards = torch.zeros((n_states, len(actions)), dtype=torch.float32, device=device)

    for s in range(n_states):
        for a in actions:
            p, s2, r = P[s][a][0]  # 这里假设确定性转移 p=1
            next_states[s, a] = s2
            rewards[s, a] = r

    return next_states, rewards


def update_value_iteration_gpu(n_states, V, actions, gamma, next_states, rewards):
    """
    GPU 加速的 Value Iteration 更新
    V: torch.tensor [n_states]  (GPU)
    next_states: torch.long [n_states, n_actions]
    rewards: torch.float32 [n_states, n_actions]
    """
    start_time = time.time()

    # 获取下一个状态的 V 值
    V_next = V[next_states]  # shape: [n_states, n_actions]

    # 计算每个动作的 Q 值
    action_values = rewards + gamma * V_next

    # 更新 V
    new_V, _ = torch.max(action_values, dim=1)

    # 计算 delta
    delta = torch.max(torch.abs(new_V - V)).item()

    V.copy_(new_V)  # 原地更新

    elapsed = time.time() - start_time
    return delta, elapsed


import numpy as np
import matplotlib
matplotlib.use('Agg')  # 👈 必须在 import pyplot 之前！
import matplotlib.pyplot as plt

# --- 参数 ---
N = 100  # Grid 大小 N×N
gamma = 0.999
theta = 1.0
actions = [0, 1, 2, 3]  # up, right, down, left
action_names = {0: '^', 1: '>', 2: 'v', 3: '<'}
n_states = N * N

def generate_z_maze(N):
    maze = np.ones((N, N), dtype=int)  # 1代表墙，0代表通道
    
    for r in range(N):
        if r % 2 == 0:
            # 偶数行全部通道
            maze[r, :] = 0
        else:
            # 奇数行全部墙，但在左边或右边留一个通道口形成连接
            if ((r - 1) // 2) % 2 == 0:
                maze[r, -1] = 0  # 右侧开口
            else:
                maze[r, 0] = 0   # 左侧开口
    
    # 起点和终点必须是通道
    maze[0, 0] = 0
    maze[N-1, N-1] = 0
    return maze
maze = generate_z_maze(N)

# --- 状态转移 P ---
P = {}
for s in range(n_states):
    P[s] = {}
    row, col = divmod(s, N)
    for a in actions:
        if s in [n_states - 1]:  # 终点（吸收状态）
            # probability, next state, reward
            P[s][a] = [(1.0, s, 100)]
        else:
            next_r, next_c = row, col
            if a == 0:  # up
                next_r = max(0, row - 1)
            elif a == 1:  # right
                next_c = min(N - 1, col + 1)
            elif a == 2:  # down
                next_r = min(N - 1, row + 1)
            elif a == 3:  # left
                next_c = max(0, col - 1)

            # 撞墙检测
            if maze[next_r, next_c] == 1:
                next_r, next_c = row, col

            next_s = next_r * N + next_c
            P[s][a] = [(1.0, next_s, -1)]

            
device = 'cuda'
next_states, rewards = prepare_transition_model(P, n_states, actions, device=device)


import time

# --- Value Iteration ---
# --- 测试：从起点 S 出发，使用最优策略是否能到达终点 T ---
def test_policy_rollout(policy, maze, N, max_steps=None):
    if max_steps is None:
        max_steps = N * N  # 最大步数限制

    action_names = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
    
    # 起点
    r, c = 0, 0
    state = r * N + c
    goal = N * N - 1  # 终点状态索引
    path = [(r, c)]
    

    for step in range(1, max_steps + 1):
        if state == goal:
            print(f"✅ 成功！在第 {step-1} 步到达终点 T({N-1},{N-1})")
            print(f"🏆 路径长度: {len(path)} 步")
            return True

        # 获取当前状态下的最优动作
        action_probs = policy[state]
        action = np.argmax(action_probs)  # 确定性策略


        # 执行动作
        next_r, next_c = r, c
        if action == 0:   # up
            next_r = max(0, r - 1)
        elif action == 1: # right
            next_c = min(N - 1, c + 1)
        elif action == 2: # down
            next_r = min(N - 1, r + 1)
        elif action == 3: # left
            next_c = max(0, c - 1)

        # 检查是否撞墙
        if maze[next_r, next_c] != 1:
            r, c = next_r, next_c
            state = r * N + c
            path.append((r, c))

    # 超出最大步数仍未到达
    if state != goal:
        return False

def update_value_iteration(n_states, V, actions, gamma):
    start_time = time.time()
    delta = 0
    for s in range(n_states):
        v = V[s]
        action_values = [sum(p * (r + gamma * V[s2]) for p, s2, r in P[s][a]) for a in actions]
        V[s] = max(action_values)
        delta = max(delta, abs(v - V[s]))
    elapsed = time.time() - start_time
    return delta, elapsed

def value_iteration():
    global theta
    V = torch.zeros(n_states, dtype=torch.float32, device=device)
    deltas = []
    iteration = 0
    totalT = 0

    while True:
        while True:
            delta, elapsed = update_value_iteration_gpu(
                n_states, V, actions, gamma, next_states, rewards
            )
            totalT += elapsed
            deltas.append(delta)

            if iteration % 10 == 0:
                print(f"Iteration: {len(deltas)}, Max Delta: {delta:.4f}")

            iteration += 1
            if delta < theta:
                break

        # 在计算 policy 前，把 V 转回 CPU numpy
        V_cpu = V.cpu().numpy()

        policy = np.zeros((n_states, len(actions)))
        for s in range(n_states):
            action_values = [
                sum(p * (r + gamma * V_cpu[s2]) for p, s2, r in P[s][a])
                for a in actions
            ]
            best_action = np.argmax(action_values)
            policy[s] = np.eye(len(actions))[best_action]

        if test_policy_rollout(policy, maze, N):
            print(f"✅ Value Iteration 完成！总迭代次数: {len(deltas)}, 总耗时: {totalT:.2f} 秒")
            return policy, V_cpu, deltas
        else:
            print("theta too large, retrying...", theta)
            theta /= 2.0

policy, V, deltas = value_iteration()
print("end = value iteration")

# --- 绘制 Value Iteration 收敛曲线并保存 ---
plt.figure(figsize=(10, 6))
plt.plot(deltas, marker='o', linestyle='-', color='b', markersize=4, linewidth=1.5)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Max Delta (Δ)', fontsize=12)
plt.title('Value Iteration Convergence Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 可选：使用对数坐标观察收敛速度
plt.tight_layout()  # 自动调整布局

# 保存图像
plt.savefig('value_iteration_convergence.png', dpi=200, bbox_inches='tight')
plt.close()  # 关闭图形，释放内存

print("✅ 收敛曲线已保存为: value_iteration_convergence.png")

# 调用打印
print("\n=== 迷宫结构和最优策略 ===")
# 假设N, policy, maze已经定义好
def plot_maze_with_policy(policy, maze, N, save_path='maze_with_policy.png'):
    # 定义动作符号
    action_symbols = {0: '^', 1: '>', 2: 'v', 3: '<'}
    
    fig, ax = plt.subplots(figsize=(N, N))
    ax.set_aspect('equal')

    # 绘制网格背景
    for r in range(N):
        for c in range(N):
            if maze[r, c] == 1:
                facecolor = 'gray'
                text = " "
            elif r * N + c == 0:
                facecolor = 'green'
                text = "S"
            elif r * N + c == n_states - 1:
                facecolor = 'red'
                text = "T"
            else:
                facecolor = 'white'
                a = np.argmax(policy[r * N + c])
                text = action_symbols[a]
            
            rect = plt.Rectangle((c, N-r-1), 1, 1, facecolor=facecolor, edgecolor='black')
            ax.add_patch(rect)
            ax.text(c+0.5, N-r-1+0.5, text, ha='center', va='center', fontsize=16)

    plt.xlim(0, N)
    plt.ylim(0, N)
    ax.axis('off')  # 关闭坐标轴
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 迷宫和策略图已保存为: {save_path}")

# 调用函数
plot_maze_with_policy(policy, maze, N)  # 请确保这里的N, policy, maze是之前定义好的变量