import torch
import torch.nn as nn


# 定义一个名为 ParallelBranches 的模型，将两个分支并行地组合起来
class ParallelBranches(nn.Module):
    def __init__(self):
        super(ParallelBranches, self).__init__()

        # 分支1：定义第一个分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            # 其他层的定义...
        )

        # 分支2：定义第二个分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            # 其他层的定义...
        )

    def forward(self, x):
        # 将输入 x 同时传递给两个分支
        x_branch1 = self.branch1(x)
        x_branch2 = self.branch2(x)

        # 在需要时合并两个分支的输出
        x_combined = torch.add(x_branch1, x_branch2)  # 在通道维度上拼接

        return x_combined

class ParallelBranches_cut(nn.Module):
    def __init__(self):
        super(ParallelBranches_cut, self).__init__()

        # 分支1：定义第一个分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            # 其他层的定义...
        )


    def forward(self, x):
        # 将输入 x 同时传递给两个分支
        x_branch1 = self.branch1(x)
        x_combined = torch.add(x_branch1, 0)  # 在通道维度上拼接
        return x_combined


# 创建模型实例
model = ParallelBranches()

state_dict = model.state_dict()
torch.save(state_dict,'./models/test.pth')

print(state_dict.keys())

new_state_dict = state_dict.copy()


del new_state_dict['branch2.0.weight']
del new_state_dict['branch2.0.bias']

# 打印模型结构，查看两个分支是否被并行地组合起来
print(model)

torch.save(new_state_dict,'./models/test_cut.pth')


testmodel = torch.load('./models/test_cut.pth')

module_cut = ParallelBranches_cut()


module_cut.load_state_dict(testmodel, strict=False)
print(module_cut)
