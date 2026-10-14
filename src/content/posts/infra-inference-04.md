---
title: 推理系统-04-CUDA Graph
published: 2026-10-13
description: ''
image: ''
tags: ["推理加速", "cuda", "AI Infra"]
category: 'AI Infra'
draft: false
lang: ''
slug: infra-inference-04
---

# CUDA Graphs

# 1. 背景

Kernel Fused的优化有限，一次decode step要经过28层，即使每层都fused，整个step仍然存在大量的kernel launch、embedding、Linear、norm等调度动作。当batch size较小的时候，每步之处理1个token，CPU的launch overhead会占据很明显的比例。

CUDA Graphs解决的不是数学计算量，而是重复执行同一形状decode step时候的CPU调度成本。GPU的调度很小的时候，那CPU的调度就需要花费大量的时间，如果没有cuda Graph，那么会产生下面的情况：

!image.png

# Kernel Launch是什么？

Kernel Launch通过把ABCDE的拓扑关系，组成一张静态图`Launch Graph`，只需要调用一次`Launch Graph`以后，就可以直接调度ABCDE的GPU资源而不需要每次Launch结束以后才进行Launch

瓶颈有的时候不出现在GPU计算本身，一次Python算子调用要经过Python Dispatcjer、PyTorch C++、CUDA runtime/driver，最后才把kernel提交给GPU stream。如果kernel很短，host可能来不及工作，GPU上就会出现绿色之间的空闲，我们叫做`launch gap`

CUDA Graph把kernel、memcpy、memset等GPU操作表示成节点，把stream顺序或event表达成依赖边。它不会融合kernel，也不会减少模型FLOPs，它优化的是整组操作的提交方式：先记录完整工作流，然后通过`cudaGraphLaunch`提交可执行图。解决了kernel很快的时候，launch花费的时间很显眼的问题。

# 怎么使用Cuda Graph

1. Definition：创建节点和依赖，得到`cudaGraph_t`。可以逐个调用Graph API，也可以用stream capture记录现有CUDA代码
    
    ```cpp
    // stream 捕获
    
    cudaGraph_t graph;
    
    cudaStreamBeginCapture(stream);
    
    kernel_A<<< ..., stream >>>(...);
    kernel_B<<< ..., stream >>>(...);
    libraryCall(stream);
    kernel_C<<< ..., stream >>>(...);
    
    cudaStreamEndCapture(stream, &graph);
    ```
    
2. instantiaion：校验拓扑关系，并提前完成大量launch初始化，得到可重复执行的`cudaGraphExec_t`。这一步比较的耗时，但是只要做过一次就可以重复利用
    
    ```cpp
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, 0);
    ```
    
3. Execution：把executable graph launch到一个CUDA stream。同一个graphExec可以反复执行，不需要再次实例
    
    ```cpp
    cudaGraphLaunch(graphExec, stream);
    ```
    

完整代码如下：

```cpp
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, 0);
cudaGraphLaunch(graphExec, stream);
```

Pytorch中使用了`torch.cuda.CUDAGraph`和`torch.cuda.graph(…)` 封装了底层对象

```cpp
g = torch.cuda.CUDAGraph()

with torch.cuda.graph(g, pool=shared_pool):
		static_logits.copy_(model.forward()) # capture
		
static_input.copy_(new_input)
g.replay()
```

## warmup

使用捕获成功的条件成功需要warmup，一次forward经常包含真正推理阶段不会重复出现的工作：

- FlashInfer/Triton JIT编译和模块加载
- cuBLAS、allocator、library handle的懒加载
- workspace和临时tensor分配
- CUDA Graph第一次launch所需要的device upload

## 静态地址

下面的写法不能replay

```python
batch.input_ids = new_input
g.replay()
```

正确的做法是长期持有capture时使用的静态tensor，只更新内容

```python
static_input.copy_(new_input)
# 只改变值但是不改变地址
g.replay
output = static_logits[:real_batch_size]
```

## 为什么CUDA Graph要求内存地址相对稳定

因为默认情况下，Graph Capture 会记录 Kernel 参数，包括它访问的 GPU Buffer 指针。如果Capture时候记录的Buffer地址可以随时发生改变，那新的Tensor地址是可能改变的，Graph Replay是很专一死板的，仍然会使用之前Capture的指针，这样会导致结果不正确甚至访问已释放的内存。因此需要使用`Static Buffer`。

## 动态Shape的处理策略

graph replay的限制很多，kernel的拓扑和shape都必须和capture兼容，但是连续批处理真实的batch size会改变：

```python
step 0: 4 reqs
step 1: 4 reqs
step 2: 3 reqs
step 3: 2 reqs
```

不可能给每一种batch size都捕获一张图，而是准备少量的bucket（使用典型的shape数据）

```python
graph_bs = [1, 2, 4]

real bs = 3 -> padding bs = 4
real bs = 2 -> padding bs = 2
real bs = 1 -> padding bs = 1
```

准备一些预备的，1、2、4，然后选择适合的graph运行即可

padding使用dummy request（推理中人为构造的，不对应真实用户的请求，用于CUDA Graph预热、捕获以及运行时Batch Padding）

dummy request有着专门的page和kv cache，都是dummy的

总结如下：

- Bucketing：捕获几个代表性的 Shape；
- Padding：把运行时输入填充到匹配的 Shape；
- Eager Fallback：不适合的 Shape 采用普通执行；
- Graph Update/Recapture：在允许的情况下修改或重新捕获。

## 为什么只捕获decode，不管prefill

因为prefill和prompt长度相关，动态范围太大了，强行padding token会浪费大量的算力。

Decode阶段每次请求之处理一个新token，单次的Kernel工作量可能比较小，但是Transformer的每层都有超级多kernel需要执行，因此需要Graph来降低Kernel Launch的时间。

## Attention metadata必须graph-aware

模型静态并不代表attention metadata静态，每个decode step的序列长度和page table内容仍然在变化。FlashInfer提供了`CUDAGraphBatchDecodeWithPagedKVCacheWrapper` （太长了）

## CUDA Graph一定能带来性能提升吗

- 不一定，对于大GEMM和GPU Kernel运行本身占主要执行时间的时候，减少的开销占比很少；
- Capture、实例化、静态Buffer等也需要花销
- 只有Host Launch明显瓶颈的时候，CUDA Graph才会带来显著收益

## CUDA Graph的几个阶段

Capture时候Kernel真的执行了吗

```python
Warmup（真实执行）
       ↓
Capture（记录工作流）
       ↓
Instantiate（准备可执行 Graph）
       ↓
Replay（真实执行）
```

warmup的时候会执行，Capture一般只是记录到Graph图，等后续Graph Launch才会执行捕获以后的Kernel工作

使用graph的replay替代原本的`model.forward()`操作