# KMT scratch 准入竞态：根因分析与修复报告

- 仓库：`/home/zhaosiying/zcode-gem5-hybrid2`（分支 `zcode/gem5-hybrid-cta2`，HEAD `62d197403`，仅工作区改动，未提交）
- 取证材料：`/home/zhaosiying/zcode-lane/artifacts/blog-perf-2026-09/results/L0-attempt2-stall-forensics/`
- 复现配置（L0 arm）：`timing_fidelity=cycle-accurate functional_fast=0`，`SAGR_LANE_FASTCOPY_MODE=legacy`，`GEMSIM_IDLE_PARK_S=0`
- 修复文件：`src/dev/amdgpu/host_gpu_kmt_native_dispatch.cc`（1 处改动，+29/−20 行）
- 验证状态：`g++ -fsyntax-only`（含正确 include 根）通过，exit 0；按任务要求未运行 scons/构建、未 commit

---

## 1. 确切根因

### 1.1 握手协议的真实形态（宿主侧证据）

宿主 ROCr（`projects/rocm-systems` fork，`libhsa-runtime64.so.1` 直驱）在创建 AQL 队列时：

1. 创建 `queue_inactive_signal`（本环境为 `DefaultSignal`，`event_mailbox_ptr == 0` —— 证据：若为 InterruptSignal，gem5 的报错会是 "interruptible queue inactive signal is unsupported"，而实际报错是 "KMT queue scratch request failed"）；
2. 注册异步处理器：`hsa_amd_signal_async_handler(queue_inactive_signal, HSA_SIGNAL_CONDITION_NE, 0, DynamicQueueEventsHandler, queue)`（`amd_aql_queue.cpp:326`）；
3. 运行时 async-events 线程对该信号**忙轮询**：任何信号缺 EopEvent 时 `wait_hint` 被强制为 `HSA_WAIT_STATE_ACTIVE`（`signal.cpp:213-220`），随后 `continue` 无睡眠自旋（`signal.cpp:315-317`），观测延迟为微秒级；
4. 处理器路径 `DynamicQueueEventsHandler` → `HandleInsufficientScratch`（`amd_aql_queue.cpp:1310`）：从队列创建时 `ReserveScratch()` 预留的池中**纯宿主本地** carve 出 scratch（池有容量时无需经 gem5 transport），`InitScratchSRD()` 写 MQD 的 `scratch_backing_memory_location`/`scratch_workitem_byte_size`/tmpring 等字段（`amd_aql_queue.cpp:2011+`），最后 `hsa_signal_store_screlease(queue_inactive_signal, 0)` "Restart the queue"（`amd_aql_queue.cpp:1257`）。

即：**信号值一旦非零，其后续所有取值（0=重启、512=回收中、-1ull=handler 终结于 `amd_aql_queue.cpp:1384`、0x8000…=析构终结于 `:361`）都由宿主状态机独占所有**。gem5 写入的 `1` 是合法 CP 错误码（bit0 命中 "insufficient scratch"）。

### 1.2 gem5 侧缺陷（旧代码，`host_gpu_kmt_native_dispatch.cc`）

`prepare()` 在 `Admission::ScratchRequired`（首次，信号值必为 0）时：写 `1` → 回读 → **要求回读值 == 1**；`Admission::QueueInactive`（重访，信号值必非 0）时：**要求值 == 1 才继续等**。任一不满足 → `fail("KMT queue scratch request failed")` → `host_gpu_bridge.cc:3695` warn → 调度 `closeClient` 杀掉客户端。

### 1.3 竞态时序（对取证日志的解释）

gem5 的 `memory.write`/`memory.read` 是两次独立的访存操作（`host_gpu_kmt_memory.cc:267-288`：共享 backing memcpy / `process_vm_writev`，写完成即对宿主进程可见，回读是另一趟 syscall + gem5 事件队列往返）：

1. gem5 写 `1` 落入宿主内存；
2. ROCr 忙轮询线程在**微秒级**观察到 1，随即（池容量足够时）完成 carve scratch → 写 MQD → **回写 0 重启队列**——全部宿主本地，全程无需 gem5 参与；
3. cycle-accurate + legacy copy 下，gem5 两次访存之间隔着事件队列工作与 CPU 争用（gem5 自身 ~70% CPU），墙上时间被拉长到毫秒级，**宿主在 gem5 回读执行之前就把 1 消费成了 0**；
4. 回读 `0 != 1` → `waiting = false` → 唯一一条 warn（`gem5-live-at-stall.log:455193`）→ `closeClient`；
5. 宿主引擎侧队列被从脚下抽走，而它正在忙等该 dispatch 的完成信号（`HSA_WAIT_STATE_ACTIVE` 自旋）；gem5 侧 `GEMSIM_IDLE_PARK_S=0` 同样忙自旋 → `walltime.jsonl`：1193 条 dispatch 后零 retire、gem5 60-72% CPU 的"stall"。

**为何只有慢配置触发**：fast 配置下写+回读在同一次 bridge 服务趟里背靠背执行（墙上间隔亚微秒），忙轮询线程几乎不可能插入；慢配置把窗口拉大到宿主必然赢。这正解释了缺陷的时序敏感性。

对任务给出的三个假设的裁定：

- "宿主 shim 根本没收到通知？" —— 否。接收方不是 shim，是 ROCr 的 async signal handler（DefaultSignal 忙轮询），通知通路完好；
- "inactive signal 写回丢失？" —— 否。写成功落地；错误在于 gem5 把宿主的合法重启（0）解释成投递失败；
- "ScratchRetryDelay 耗尽走 Failed？" —— 否。`ScratchRetryDelay = 1e9 ticks`（1ms 仿真时间）重试本身无上限、无超时判定；Failed 来自 `prepare()` 内部的取值断言，与重试次数无关。

### 1.4 同根的次级缺陷（一并消除）

重访路径 `waiting = signalValue == 1` 对其它 ROCr 合法值（512 回收中、-1ull 终结、0x8000… 析构）一律硬失败。本次 stall 由 1.3 的回读竞态主导（宿主实际是健康的——scratch 已装好、信号已清零——所以引擎是"自旋"而非报错退出），但两者同属"gem5 越权解释宿主拥有的 mailbox 取值"这一设计缺陷，修复以同一不变量一并覆盖。

---

## 2. 改动清单与机制

### `src/dev/amdgpu/host_gpu_kmt_native_dispatch.cc`（`HostKmtNativeDispatch::prepare()` 的 scratch 准入分支，唯一改动）

**改动内容**：

1. 删除"回读必须等于 1"断言：回读仅保留为**可达性探测**（`memory.write`/`memory.read` 操作本身失败才失败——那是真实的 transport/pin 级错误，应 fail-closed）；
2. 删除按 admission 区分的 `waiting = signalValue == 1` 两个分支：只要 mailbox 存在（handle 非零、非 interrupt-event 信号、无地址溢出）且本 dispatch 的 scratch 需求未满足，**一律置 `scratchPending` 延期**；
3. 发布逻辑保留且幂等：仅当观测值为 0（mailbox 空闲）才写 `1`；完成回读即视为投递成功，不再解读取值；
4. 代码注释按仓库规范陈述不变量：inactive signal 是由 ROCr `DynamicQueueEventsHandler` 服务的 CP→runtime 错误 mailbox，发布后取值语义归宿主；队列拆除经 KMT destroy 协议终结延期，而非在此处失败 dispatch。

**不变量（修复后）**：

- 准入只判三件事：(a) mailbox 存在且可达 → 延期重试；(b) mailbox 内存不可达（write/read 操作失败）→ fail-closed；(c) interrupt-event 型信号 → 维持原有显式不支持失败（非竞态，报错响亮，不在本缺陷范围）；
- `prepare()` 返回延期 ⟺ `scratchPending=true` ⟺ bridge 得到 `DeferredForScratch` ⟺ `kmtScratchRetryEvent` 以 `curTick()+1e9` 无限重臂（`host_gpu_bridge.cc:3685-3691`，仅在 `shutdown()` 反调度；dispatch 退休路径 `finishKmtDispatch` 末尾会再调 `serviceKmtDispatches()`（`:3781`），重试循环无活性漏斗）。

### 为什么这系统性消除了竞态

gem5 不再要求对 mailbox 取值的任何**独占观察窗口**。宿主 handler 与 gem5 准入的每一种交错——写在宿主观测前/后、回读在宿主重启前/后、重试落在宿主处理中间（值=1）或之后（MQD 已就绪、值=0）——都单调地映射为"延期"，成功仅由**宿主发布的稳定终态**（MQD scratch 足够 且 信号=0）在后某次轮询中判定。宿主侧终结态（-1ull 等）也不再从握手内部杀客户端，而是等宿主自己的错误回调/队列销毁经协议到达，两侧清理路径收敛。原 bug 的触发面（慢配置下"宿主太快消费请求"被误判为失败）在构造上不复存在。

### 对既有测试的兼容性（静态核对，未运行）

`host_gpu_kmt_native_dispatch.test.cc` 两个相关用例语义不变：

- `RequestsRuntimeScratchThenPinsTheMappedAllocation`：值 0 → 发布 1（内存中随后读到 1，断言仍成立）→ pending；MQD 更新+信号清零后 prepare 成功；
- `TreatsPublishedScratchRequestAsPending`：值 1 → 不重写（值保持 1 的断言成立）→ pending，failure 字符串仍为 "KMT queue scratch allocation pending"。

无任何测试断言被删除的"按取值失败"行为。

---

## 3. 后续建议（未实施，超出最小修复范围）

- 若未来 ROCr 以 `g_use_interrupt_wait` 模式创建该信号（`event_mailbox_ptr != 0`），需为 shim 实现 KFD event 语义后再放开 interruptible 分支；
- `ScratchRetryDelay`（1e9 ticks）在 cycle-accurate 下每次重试的墙上代价可观，可考虑宿主侧完成分配后主动 doorbell 重投以缩短唤醒延迟——但仅是性能优化，正确性已由本修复保证。
