# TactileTTT 项目交接文档

用途：将本文件交给新的 Codex 对话，使其直接继续当前工作。  
更新时间：2026-09-03
本地仓库：/Users/babyna/TactileTTT  
服务器仓库：/workspace/mnt/sqzhang26/TactileTTT

## 1. 研究目标

我们准备研究“灵巧手触觉 + Test-Time Training”。核心问题是：反应式触觉策略通常只使用当前或很短的触觉，而长程、接触丰富的任务需要记住过去发生过什么，例如有效接触、无效接触、滑动、重抓以及物体接触动力学。

当前设想：

- 短期触觉直接输入策略，负责即时反馈；
- 长期触觉通过 TTT fast weights 累积为跨 action chunk 的 memory；
- 部署期间不需要质量、摩擦、软硬度等物理属性标签；
- 后续版本再加入 VisualTTT，与 TactileTTT 异步更新，验证视觉和触觉长期记忆的互补性。

研究定位：

- RoboTTT 说明长历史可以通过 TTT fast weights 注入机器人策略；
- FM-VLA 等工作说明触觉长历史有价值，但固定长度压缩面对大量接触事件存在容量问题；
- ViTacPhys 等方法依赖质量、摩擦、软硬度标签或预训练属性预测器；
- 我们希望用部署过程中的触觉序列在线更新 memory，不显式预测物理属性。

## 2. 当前验证任务与数据

第一阶段任务：

1. 完成 3 次有效按钮按压；
2. 中间随机插入 1～2 次无效按压；
3. 完成有效次数后拿起物体并放进盒子。

目的不是立即证明复杂物理适应，而是先验证：策略能否利用长期触觉区分有效/无效接触，并记住已经完成多少次有效按压。

数据目录：

    /Users/babyna/TactileTTT/data/press_button_4_times

数据量：

    press_button_0: 10 episodes, 4,072 frames
    press_button_1: 62 episodes, 29,832 frames
    press_button_2: 29 episodes, 13,493 frames
    总计: 101 episodes, 47,397 frames

重要数据风险：press_button_1 约 5.87% 的帧至少包含一个绝对值接近 255 的 raw taxel 饱和值。需要确认这是合法量程上限还是异常值，并避免它主导触觉归一化统计。

## 3. 当前模型设计

当前配置名：

    pi05_tactile_ttt_v1

两阶段训练配置：

    pi05_tactile_ttt_v1_warmup  # 只训练路径中包含tactile_ttt的参数
    pi05_tactile_ttt_v1         # 加载warm-up params后联合训练

结构：

    RGB + language + robot state → π0.5

    当前帧 raw tactile [1,5,120,3]
        → TPE → 5个手指token → π0.5 Action Expert

    Action Expert的每个Transformer block：
        Attention → layer-specific TTT-MLP → FFN

    每层fast state：W1,b1,W2,b2
    K/V binding inner update：MLP(K; fast state)拟合V
    Q读取更新后的fast state，并通过逐通道residual gate注入block

已经确定：

- 保留 Tactile Patch Encoder（TPE）；
- TTT只插入Action Expert，不修改VLM分支；
- 每个Action Expert Transformer block在attention后各有一个独立TTT-MLP；
- fast state为两层256→256→256 MLP，并按layer、episode分别携带；
- fast weight 每个 action chunk 更新一次；
- action horizon H=16，训练与部署更新频率一致；
- 一条 episode 内持续携带 fast weight，episode 开始时重置；
- sequence training 中，一次slow-weight optimization对整条episode全部有效chunk loss求平均；
- 不使用future tactile prediction；
- 不使用双AE、teacher AE、future-flow和tactile refiner；
- 当前只验证TactileTTT，后续版本再加入VisualTTT。

关键配置：

    action_horizon = 16
    force_input_frames = 16
    tactile_history_offsets = (-15, ..., 0)
    tactile_ttt_memory_dim = 256
    tactile_ttt_mlp_dim = 256
    tactile_ttt_inner_lr = 0.1
    tactile_ttt_residual_gate_init = 0.001
    disable_future_tactile = True

## 4. Fast-weight读写

代码：src/openpi/models/tactile_ttt.py

每个Action Expert block写入：

    K = key_proj(LN(write_tokens))
    V = value_proj(LN(write_tokens))
    prediction = MLP(K; W1,b1,W2,b2)
    inner loss = mean(||prediction - V||²)
    fast_state_t = fast_state_{t-1} - inner_lr × contact_gate × grad(inner loss)

读取：

    Q = query_proj(LN(current_tokens))
    memory = output_proj(MLP(Q; fast_state_t))
    enhanced = current_tokens + tanh(residual_gate[channel]) × memory

residual_gate 是每层可训练的slow vector parameter，初始化为0.001：

- tanh(residual_gate)接近0：策略几乎不读取TTT；
- 非零：memory才会影响Action Expert。

## 5. 三个实验配置

### 纯视觉baseline

    pi05_xhand_full_finetune_h16

输入视觉、语言和robot state，不使用TPE/raw tactile。当前结果有效，不受触觉归一化问题影响。

### 短期触觉baseline

    pi05_tactile_direct16

过去16帧触觉经过TPE后直接输入策略，没有fast-weight memory。原训练受到错误触觉quantile normalization影响，需要修复后重跑。

### TactileTTT

    pi05_tactile_ttt_v1

使用episode sequence training，fast weight跨chunk累积。原训练同时受到触觉归一化和接触门控失效影响，需要修复后重跑。

## 6. 已有训练结果

纯视觉π0.5 baseline：

    30,000 steps
    final train loss ≈ 0.00576
    final eval loss  ≈ 0.00807

Direct16：

    30,000 steps
    final train loss ≈ 0.00738
    final eval loss  ≈ 0.00859

Direct16触觉预处理错误，因此不能由此得出“触觉没有帮助”。

已废弃的TactileTTT v0诊断结果：

    step 250: eval loss = 0.08161824
    step 500: eval loss = 0.07069663
    step 750: eval loss = 0.07085311
    step 744附近 train loss ≈ 0.02086505

Step 750：

    eval/student_action_arm  = 0.06548920
    eval/student_action_hand = 0.14166917
    eval/tactile_ttt/contact_gate = 1.00000000
    eval/tactile_ttt/fast_weight_norm = 0.88156766
    eval/tactile_ttt/reconstruction = 0.13886708

结论：旧TTT v0明显差于视觉baseline，而且500到750基本不再改善。v0代码与配置入口已经删除，旧checkpoint不能用于v1续训。

旧checkpoint仅用于诊断：

    /workspace/mnt/sqzhang26/TactileTTT/checkpoints/
    pi05_tactile_ttt_v0/pi05_tactile_ttt_v0_press_0829/750

## 7. 已发现并修复：触觉归一化

当前openpi让π0.5对输入使用quantile normalization：

    x' = 2 × (x - q01) / (q99 - q01) - 1

它适合state/action，但不适合稀疏raw tactile。当前effort统计：

    mean = [-0.0042728, -0.0049407, 0.9401863]
    std  = [ 0.2330233,  0.2617692, 15.1095915]
    q01  = [-0.0568, -0.008, 0.0]
    q99  = [-0.0260, -0.008, 1.989]

触觉y轴出现q01=q99=-0.008，quantile分母接近零。零taxel原来会变成：

    quantile: [2.69, 15999, -1]

正确z-score为：

    z-score: [0.018, 0.019, -0.062]

TPE预训练配置属于PI0，原本使用z-score；接入π0.5后被切换为quantile，产生输入分布不一致。

### 已完成的本地修复

现在支持按字段归一化：

    state/action → π0.5 quantile normalization
    effort/raw tactile → z-score normalization

已对以下配置启用：

    pi05_tactile_current
    pi05_tactile_direct16
    pi05_tactile_ttt_v1

训练和部署policy使用相同规则；视觉baseline不受影响。

相关修改：

    src/openpi/transforms.py
    src/openpi/training/config.py
    src/openpi/training/data_loader.py
    src/openpi/policies/policy_config.py
    src/openpi/models/tactile_tokenizer_test.py

验证状态：

- py_compile通过；
- git diff --check通过；
- 新增混合归一化单元测试；
- Mac无法运行完整pytest，因为项目依赖强制安装CUDA版JAX；应在服务器运行目标测试。

## 8. 已完成：未归一化calc_force接触门控

当前门控使用归一化raw taxel：

    magnitude = ||normalized_raw_taxel||
    taxel_gate = sigmoid((magnitude - 1.0) / 0.5)

原quantile数值爆炸导致门控恒为1。即使改成z-score，threshold=1.0也没有物理意义，而且raw taxel存在饱和值。

### 数据驱动物理阈值

每帧定义：

    s_t = max over five fingers ||calc_force[t,finger]||₂

对全部47,397帧的log(1+s_t)做两类聚类：

    低力簇log中心 = 0.156376，对应0.169 N
    高力簇log中心 = 3.194516，对应23.398 N
    两个log中心的中点还原 = 4.341 N

分组统计：

    无接触平均 = 0.339 N
    无接触中位数 = 0 N
    无接触p95 = 3.162 N

    接触平均 = 33.900 N
    接触中位数 = 24.083 N
    接触p05 = 5.831 N

建议门控：

    score_t = max over past 16 frames and five fingers ||calc_force||₂
    gate_t = sigmoid((score_t - 4.3 N) / 0.5 N)

当前已把未归一化calc_force [T,5,3]单独传入模型，仅供contact gate使用：

- TPE继续读取z-score后的raw taxel；
- contact gate读取未归一化calc force；
- 训练与部署使用完全相同门控；
- 不要通过减小当前raw-taxel threshold解决。

## 9. 已加入TTT诊断指标

本地代码已加入：

    tactile_ttt/residual_gate_raw
    tactile_ttt/residual_gate
    tactile_ttt/fast_weight_update_norm
    tactile_ttt/memory_contribution_norm
    tactile_ttt/memory_to_current_ratio

含义：

- fast_weight_norm = ||W_t||_F，只说明fast weight非零；
- fast_weight_update_norm = ||W_t-W_{t-1}||_F，说明当前chunk是否写入；
- residual_gate = tanh(raw gate)，说明策略是否允许memory进入Action Expert；
- memory_contribution_norm说明memory实际注入量；
- memory_to_current_ratio说明memory贡献相对当前触觉token的比例。

相关文件：

    src/openpi/models/tactile_ttt.py
    src/openpi/models/pi0_latent_flow.py

## 10. 如何证明TTT存储了正确记忆

必须做推理消融：

1. Full TTT：正常读写并跨chunk携带memory；
2. Reset：每个chunk将W_t重置为W_0；
3. No-read：将residual gate强制设为0；
4. Shuffled-memory：交换不同episode的fast weights。

还应从W_t或memory tokens中用线性probe解码：

- 已完成有效按压次数0/1/2/3；
- 最近一次按压是否有效；
- 当前任务阶段：按压/拿取/放置。

只有Full优于Reset和No-read、交换memory导致性能下降，并能解码有效按压次数，才能证明TTT保存并使用了任务相关长期记忆。

## 11. 公平比较与数据设计

原训练方式不同：

- baseline随机采样action chunk，训练30,000 slow-weight steps；
- TTT每个batch读取完整episode sequence，一个slow step对序列全部有效chunk loss求平均；
- TTT序列长度最高约40个chunk。

正式实验需要增加公平baseline：

    相同episode-sequence loader
    相同序列loss平均
    相同slow optimization steps
    但不使用fast-weight TTT

当前eval只有2个batch。正式实验应固定随机种子、覆盖全部validation episodes、报告chunk-weighted action loss，并重点报告真机闭环成功率。

数据还必须避免模型通过时间、手臂位置或视觉阶段猜测进度。应随机化有效/无效顺序和持续时间，每次尝试后回到相近观察/机器人状态，并让有效接触间隔超过16帧。

最终指标：

    恰好完成3次有效按压
    能够忽略或恢复无效按压
    随后成功拿取并放入盒子

### 两阶段训练约定

当前采用两个独立训练run：

1. `pi05_tactile_ttt_v1_warmup`：冻结VLM、原Action Expert、TPE和action head，只用action outer loss训练TTT slow parameters与fast-state初始化参数；
2. `pi05_tactile_ttt_v1`：从warm-up checkpoint的`params`初始化，重新创建optimizer，然后联合训练。

阶段二禁止对阶段一目录使用`--resume`。warm-up optimizer只包含TTT参数，而joint optimizer需要覆盖全部解冻参数，两者opt-state树不兼容。阶段二应新建exp，并通过：

    --weight-loader.pi0-params-path <warmup-checkpoint>/params

加载模型参数。episode runtime fast state不会跨run保存；checkpoint保存并传递的是学习到的每层fast-state初始化参数。

单阶段和两阶段使用完全相同的student-only TactileTTT模型参数树。TTT配置不再构造teacher action head、teacher tactile tokenizer、蒸馏projector、future query/flow、flow VAE投影、patch auxiliary head以及pi05不读取的连续state projector；warm-up与joint仅由冻结规则区分。该结构变更后不能对旧TTT train-state直接使用`--resume`，应从基础pi05参数或旧checkpoint的`params`新建run。

## 12. 下一步执行顺序

### P0：服务器验证v1代码

1. 同步本地v1修改到服务器；
2. 运行目标pytest和data-loader batch shape检查；
3. 初始化模型并确认基础Action Expert权重加载、新增TTT参数随机初始化；
4. 检查单个训练step能编译且无NaN/OOM；
5. 不从旧v0 checkpoint续训。

### P1：小规模诊断训练

1. 从π0.5基础权重重新开始，不从旧750 checkpoint继续正式训练；
2. 运行TTT-only warm-up 250 sequence optimizer steps；
3. 从warm-up params新开joint run，运行750 sequence optimizer steps；
4. 重跑pi05_tactile_direct16；
5. 检查contact gate是否有0～1变化；
6. 检查residual gate是否离开0.001；
7. 检查memory contribution是否非零；
8. 确认无NaN和异常fast-weight增长后再完整训练。

### P2：正式验证

1. 视觉π0.5；
2. Direct16；
3. sequence baseline但无TTT；
4. Full TactileTTT；
5. Reset、No-read与Shuffled-memory；
6. 真机闭环有效按压计数与拿取放置成功率；
7. memory语义probe。

## 13. 当前本地代码状态

当前存在未提交修改，用户明确要求不要自动commit或push。

修改文件：

    TACTILE_TTT_HANDOFF.md
    readme_forceWAM.md
    src/openpi/models/gemma.py
    src/openpi/models/pi0_config.py
    src/openpi/models/pi0_latent_flow.py
    src/openpi/models/tactile_ttt.py
    src/openpi/models/tactile_ttt_test.py
    src/openpi/training/config.py
    src/openpi/training/config_tactile_ttt_test.py

其中：

- gemma.py：在每个Action Expert Transformer block的attention后插入layer-specific TTT-MLP；
- tactile_ttt.py：实现两层fast MLP、K/V binding、Q读取、可学习inner LR和vector residual gate；
- pi0_latent_flow.py：维护每层episode fast state，训练与部署按chunk携带，flow denoising只提交一次更新；TTT配置只构造student与TTT所需参数，不构造teacher/蒸馏/future-flow辅助模块；
- pi0_config.py、training/config.py：删除v0入口并注册pi05_tactile_ttt_v1；
- tactile_ttt_test.py：覆盖状态写入、禁写mask、门控、residual gate与Gemma逐层集成；
- training/config.py：另外注册TTT-only warm-up冻结规则和pi05_tactile_ttt_v1_warmup；
- config_tactile_ttt_test.py：验证warm-up只训练TTT，joint配置解除warm-up冻结；
- readme_forceWAM.md：训练命令切换为warm-up 250 steps + joint 750 steps。

## 14. 服务器信息

    服务器Python:
    /workspace/mnt/sqzhang26/FactileLDM/env/.venv/bin/python

    π0.5基础权重:
    /workspace/mnt/sqzhang26/.cache/openpi-assets/checkpoints/pi05_base/params

    服务器仓库:
    /workspace/mnt/sqzhang26/TactileTTT

本地触觉归一化统计：

    /Users/babyna/TactileTTT/assets/pi05_tactile_current/
    press_button_4_times/norm_stats.json

## 15. 给新对话的直接指令

请先完整阅读本文件，然后继续：

> 在不提交代码的前提下，验证pi05_tactile_ttt_v1逐层TTT-MLP实现。门控使用过去16帧五指未归一化calc_force最大合力，threshold=4.3 N，temperature=0.5 N；TPE仍使用z-score raw tactile。训练与部署每个action chunk只提交一次fast-state更新，不从旧v0 checkpoint续训。

## 16. TactileTTT-v2（2026-09-03）

新增配置：

    pi05_tactile_ttt_v2_warmup
    pi05_tactile_ttt_v2

v1中每个Action Expert block的TTT对完整suffix做K/V/Q投影，因此每个投影均有21个token：

    5个当前帧手指触觉token + 16个带噪action token

v2改为：

- 普通Action Expert suffix仍为5个当前触觉token+16个action token，共21个Q；
- 16帧触觉经TPE得到80个token，构造相邻帧同手指预测对；
- K使用前15帧，共15×5=75个token；V使用后15帧，共75个token；
- 只在Action Expert第2/5/8/11/14/17层启用TTT，其余层作为Transformer anchor；
- TTT残差从block输入并行产生，再与attention残差相加；
- contact gate仍只控制fast-weight写入，读取不受门控；
- residual gate初始化从0.001提高到0.01；
- v1配置、参数结构与已有checkpoint入口继续保留。

本地已通过Python 3.14语法编译和`git diff --check`。Mac环境不能安装CUDA JAX，仍需在服务器运行：

    pytest -q src/openpi/models/tactile_ttt_test.py src/openpi/training/config_tactile_ttt_test.py
