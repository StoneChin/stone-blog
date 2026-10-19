---
title: 【CUDA】Warp Shuffle
published: 2025-05-23
description: ''
image: ''
tags: ['cuda']
category: 'cuda'
draft: false
lang: ''
slug: cuda-warp-shuffle
---

# Warp Shuffle


在学习过程中，shuffle函数的理解和我一开始的理解出现了一部分的偏差，因此将学习过程记录下来。

参考资料：

https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions

https://www.cnblogs.com/bandaoyu/p/19216933

# 基本概念

warp是CUDA的重要级别，也是CUDA开展高性能工作的一个核心概念。warp的出现往往伴随着element-wise类型的reduce方法：**`shuffle`**洗牌


```cpp
T __shfl_sync     (unsigned mask, T value, int      srcLane,  int width=warpSize);
T __shfl_up_sync  (unsigned mask, T value, unsigned delta,    int width=warpSize);
T __shfl_down_sync(unsigned mask, T value, unsigned delta,    int width=warpSize);
T __shfl_xor_sync (unsigned mask, T value, int      laneMask, int width=warpSize);
```

参数说明：

- mask: 一个32位掩码，每个位代表warp中的一个线程（从0开始编号）。只有掩码中设置为1的对应线程才会参与操作。通常使用全1掩码（0xffffffff）表示整个warp参与。
- var: 要交换的变量。
- srcLane: 指定从哪个线程读取数据。**分组内的相对索引**（0 ~ width-1）。 (__shfl_sync)
- delta: 向上或向下移动的偏移量。 (__shfl_up_sync 和 __shfl_down_sync)
- laneMask : 用于按位异或的掩码，与当前线程ID异或后得到目标线程ID。(__shfl_xor_sync)
- return：目标线程的var值。**非mask指定的线程返回值未定义**

其中width参数（可选），允许把wrap划分成更小的组（必须是2的幂，同时小于等于wrapsize）

# **1. __shfl_sync: 从指定线程复制数据**

一句话总结：`srcLane`里的数据`value`，给`mask`覆盖下的所有线程。是一种广播机制

下面的代码是从lane=0中取出数据广播给统一warp下的所有线程

```cpp
示例：线程1获取线程0的val值
__global__ void kernel() {
    int val = threadIdx.x;  // 每个线程的ID即其值
    int target = __shfl_sync(0xFFFFFFFF, val, 0); // 所有线程都读取线程0的值
    printf("Thread %d: target=%d\n", threadIdx.x, target);
}
/* 输出：
   Thread 0: target=0
   Thread 1: target=0
   ... (所有线程输出0) */
```

下面的图片是从2中取出数据进行洗牌广播，上面是直接广播，下面是每16为一组然后去取lane=2的进行广播

!40075b1a10c719f1cb6d2d0946442eec.png

# **2. __shfl_up_sync 序号向前**

__shfl_up_sync是序号向前的偏移

```cpp
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=32)
```

所有参与的线程`tid=n`从线程`tid = n - delta`中获得数据，如果出现`tid = n - delta < 0`，那么当前线程保持不变。

例如warp大小是4，线程0、1、2、3，如果线程2调用__shfl_up_sync(0xf, x, 1)，那么线程2调用的时候获得线程1的值，线程0调用因为-1<0，因此值仍然是原来的本身。

下面代码的例子中，warp大小是4，delta是2，那么就是，线程3拿到线程1的值，线程2拿到线程0的值，而线程1和线程0因为偏移以后<0，因此保持原本的值

```cpp
示例：每个线程从相对本身靠前2个位置的线程获取值
__global__ void kernel() {
    int val = threadIdx.x;
    int result = __shfl_up_sync(0xFFFFFFFF, val, 2);
    printf("Thread %d: result=%d\n", threadIdx.x, result);
}
/* 输出：
   Thread 0: result=0  // 0-2<0 返回自身
   Thread 1: result=1  // 1-2<0 返回自身
   Thread 2: result=0  // 2-2=0
   Thread 3: result=1  // 3-2=1
   ... */
```

下图表示： mask=0xffffffff 覆盖所有线程，delta=2，所以所有线程参与，从ID=本身 ID -`delta` 的线程获取数据，前 2 个因为 ID -`delta`<0,所以返回的是自身的数据：

!333daa0933321df82e107bd7ff302e58.png

# **3. __shfl_down_sync 序号向后**

和2类似的道理，只不过和2相反

```cpp
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=32);
```

例如`__shfl_down_sync(0xf, x, 2)`，warp大小是4，那么就是，线程0拿到线程2的值，线程1拿到线程3的值，最后的线程2和线程3保持原来。

下图： mask=0xffffffff 覆盖所有线程，delta=3，所以所有线程参与，从ID=本身 ID +3 的线程获取数据，后 3 个因为 ID +`delta`> width-1,所以返回的是自身的数据。

!909575cb304b24429b6ebc4f94da280d.png

## warp shuffle的使用

!1f99f3c9f1f71a3ccaa9033948c15714.png

再warp shuffle中，经常会使用`__shfl_down_sync(0xFFFFFFFF, x, offset)` 的形式，本质上就是lane小的位置在warp中递归的获取lane+offse位置的值进行规约。

# **4. __shfl_xor_sync: 按线程ID的异或操作交换数据**

用的没有那么多，这里给出示例自行理解

```cpp
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=32);
```

所有参与的线程（tids=n)，各自从 tidt=(tids **XOR laneMask)**的线程取值

例如：warp大小为4（线程0,1,2,3），laneMask=1。则：

线程0：0^1=1 -> 从线程1获取

线程1：1^1=0 -> 从线程0获取

线程2：2^1=3 -> 从线程3获取

线程3：3^1=2 -> 从线程2获取

这实际上实现了两两交换（0<->1, 2<->3）。如果laneMask=3，则：

线程0：0^3=3 -> 从线程3获取

线程1：1^3=2 -> 从线程2获取

线程2：2^3=1 -> 从线程1获取

线程3：3^3=0 -> 从线程0获取

这实现了4个线程的循环交换（0->3, 3->0; 1->2, 2->1），也就是相邻两个线程对交换。

注意：这些函数要求warp内所有活动线程（由mask指定）必须执行相同的洗牌操作，否则结果未定义。

下面给出一个简单的代码示例（概念性代码，实际使用时需注意线程同步和掩码设置）：

```cpp
示例：相邻线程交换数据（laneMask=1）
__global__ void kernel() {
    int val = threadIdx.x;
    int result = __shfl_xor_sync(0xFFFFFFFF, val, 1);
    printf("Thread %d: result=%d\n", threadIdx.x, result);
}
/* 输出：
   Thread 0: result=1  // 0^1=1
   Thread 1: result=0  // 1^1=0
   Thread 2: result=3  // 2^1=3
   Thread 3: result=2  // 3^1=2
   ... */
```

下图：

![](https://i-blog.csdnimg.cn/img_convert/0cf6cb806fa94d7fc667875567685c2a.png)

# 常用模板

在reduce中，常常使用warp shuffle进行warp内的规约操作，并将规约后的值存入到shared memory以进行下一步规约。一般的warp的大小是32，以reduce为求和函数为例子，所用的warp shuffle模板如下：

```cpp
__device__ __forceinline__
float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(
            0xffffffff, value, offset
        );
    }
    return value;
}
```

用的次数超级多，经过五轮的shuffle，把32个lane的值逐步向加，最后lane 0 的v就是warp部分和的总和，可以使用if判断lane的下标是不是0，并存入shared memory进行下一步操作