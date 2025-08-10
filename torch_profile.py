import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity



def torchInfo_print(model,inputs,labels):
    with profile(
            activities=[
                ProfilerActivity.CPU,  # 分析CPU
                ProfilerActivity.CUDA  # 分析GPU
            ],
            schedule=torch.profiler.schedule(
                wait=1,  # 跳过前1次迭代
                warmup=1,  # 预热1次迭代
                active=3,  # 分析3次迭代
                repeat=1  # 只运行1轮
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # 保存TensorBoard日志
            record_shapes=True,  # 记录输入形状
            profile_memory=True,  # 分析内存
            with_stack=True  # 记录调用栈
    ) as prof:
        for step in range(5):  # 总共运行5次迭代
            # 前向传播
            with record_function("forward_pass"):
                outputs = model(inputs)

            # 计算损失
            with record_function("compute_loss"):
                loss = criterion(outputs, labels)

            # 反向传播
            with record_function("backward_pass"):
                loss.backward()

            # 参数更新
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

            # 让Profiler记录当前步骤
            prof.step()

    # 4. 打印关键指标
    print(prof.key_averages().table(
        sort_by="cuda_time_total",  # 按GPU耗时排序
        row_limit=10  # 只显示前10行
    ))

    # 5. 启动TensorBoard查看可视化结果
    # 在终端运行: tensorboard --logdir=./log
    # 然后在浏览器打开 http://localhost:6006/



if __name__ == "__main__":
    # 1. 加载或创建PyTorch模型
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cpu")
    model.eval()  # 设置为评估模式

    # 2. 创建虚拟输入数据
    inputs = torch.randn(32, 3, 518, 518).cuda()
    labels = torch.randint(0, 10, (32,)).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)