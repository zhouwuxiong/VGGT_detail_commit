from vggt.models.vggt import VGGT


###################     torchsummary  ###############
def torchsummary_print(model):
    from torchsummary import summary

    summary(model.to('cuda'), input_size=(3, 518, 518))

def dict_print(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

def torchInfo_print(model):
    from torchinfo import summary
    summary(model, input_size=(1, 3, 518, 518), depth=10)

def torchviz_print(model):
    import torch
    from torchviz import make_dot

    # 1. 初始化模型和输入
    dummy_input = torch.randn(1, 3, 518, 518).to('cpu')  # 根据你的输入尺寸调整

    # 2. 生成计算图
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))

    # 3. 保存为图片（需要安装Graphviz）
    dot.render("results/model_architecture", format="png", cleanup=True)

def hiddenlayer_print(model):
    import torch
    import hiddenlayer as hl

    # 1. 构建转换器
    hl_graph = hl.build_graph(model.to('cpu'), torch.zeros([1, 3, 518, 518]).to('cpu'))

    # 2. 保存为图片
    hl_graph.save("results/model_architecture.png", format="png")

if __name__ == "__main__":
    # 1. 加载或创建PyTorch模型
    print("loading model ...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cpu")
    model.eval()  # 设置为评估模式
    # 方法 1： print 直接打应
    print(model)
    # 方法 2 ： dict 打印
    print("dict_print")
    dict_print(model)
    # 方法 3 ： torchsummary
    # print("torchsummary_print(model)")
    # torchsummary_print(model)
    # 方法 4 : torch_info
    print("torchInfo_print(model)")
    torchInfo_print(model)

    # 方法 5 :
    # torchviz_print(model)

    # 方法 6
    print("hiddenlayer_print(model)")
    hiddenlayer_print(model)
