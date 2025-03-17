import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA可用，您可以使用GPU进行计算。")
        print(f"可用的GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA不可用，您将使用CPU进行计算。")

if __name__ == "__main__":
    check_cuda()
    print(torch.cuda.is_available())  # 应该返回True
    print(torch.cuda.device_count())   # 应该返回可用的GPU数量
    print(torch.__version__)
    # print(torch.cuda.get_device_name(0))  # 显示第一个GPU的名称
