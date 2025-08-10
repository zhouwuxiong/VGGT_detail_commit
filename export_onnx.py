import torch
import torchvision
from vggt.models.vggt import VGGT

def torch_onnx_export(model):
    torch.save(model.to('cpu'), "vggt_1B.pth")  # PyTorch模型
    # 或者导出为ONNX
    torch.onnx.export(
        model,  # 要导出的模型
        torch.randn(1, 3, 518, 518).to('cpu'),  # 模型输入
        "vggt_1B.onnx",  # 输出文件名
        input_names=["input"],  # 输入节点名称
        output_names=["output"],  # 输出节点名称
        dynamic_axes={
            'input': {0: 'batch_size'},  # 动态维度
            'output': {0: 'batch_size'}
        },
        opset_version=14  # ONNX算子集版本
    )



if __name__ == "__main__":
    # 1. 加载或创建PyTorch模型
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cpu")
    model.eval()  # 设置为评估模式

    torch_onnx_export(model)
    print("PyTorch模型已成功导出为ONNX格式")