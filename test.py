import torch
import numpy as np
# 根据你跑的任务导入对应的模型（这里以你日志里的 LDL 任务 MLPModel 为例）
from models.mlp_model import MLPModel 

def run_experiment():
    # 1. 载入你已经训练好的模型权重 (请替换为你的真实 ckpt 路径)
    ckpt_path = "../Logs/SFNO/MAFW/formal/run_0/last_step=100000.ckpt"  
    
    # 注意：你需要填入与你训练时一致的初始化参数
    # 以下参数参考自你的 configs/ldl.yaml
    model = MLPModel.load_from_checkpoint(
        ckpt_path,
        data_dim=1024,       # <--- 改成 1024
        hidden_dim=512,      # <--- 改成 512
        latent_dim=1024,     # <--- 改成 1024
        output_dim=11,       # <--- 改成 11
        in_proj='linear',
        out_proj='linear',
        f_add_blocks=0,
        h_add_blocks=4,
        g_add_blocks=0,
        strict=False 
    )
    
    # 【关键】将模型设为评估模式！这一步会关闭 BatchNorm 等带来的训练期波动
    model.eval()
    model.cuda() # 放到 GPU 上

    print("模型加载完毕。开始实验...")

    # 2. 准备一条固定的测试数据
    fixed_X = torch.randn(1, 1024).cuda()

    # 3. 连续进行 100 次推理
    outputs = []
    
    with torch.inference_mode(): # 使用你代码 base.py 里的推荐方式
        for i in range(100):
            # 注意：我们这里【没有】设置任何随机种子！
            # 内部调用的是 base.py 里的 inference 方法
            pred = model.inference(fixed_X, method='dopri5')
            outputs.append(pred)

    # 4. 验证结果是否一致
    first_output = outputs[0]
    is_deterministic = True

    for i in range(1, 100):
        # 计算当前输出与第一次输出之间的绝对误差
        diff = torch.abs(first_output - outputs[i])
        max_diff = torch.max(diff).item()
        
        # 考虑到 GPU 浮点数极度微小的舍入误差，设置一个合理的阈值 (1e-6)
        if max_diff > 1e-7:
            is_deterministic = False
            print(f"❌ 发现不同！第 {i+1} 次输出与第 1 次不一致。最大差值: {max_diff}")
            print(f"第 1 次输出: {first_output.cpu().numpy()}")
            print(f"第 {i+1} 次输出: {outputs[i].cpu().numpy()}")
            break

    if is_deterministic:
        print("✅ 验证成功：连续 100 次推理，结果百分百完全一致！")
        print("这证明 SFNode 在推理时不包含任何生成式采样（未加噪），是一个完全确定性的模型。")
        print(f"最终的固定输出值: {first_output.cpu().numpy()}")

if __name__ == "__main__":
    run_experiment()