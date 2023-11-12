from pathlib import Path
import sastvd as svd
import sastvd.linevd as lvd
from dgl.nn.pytorch import GNNExplainer
import torch


# 检查是否有可用的 CUDA 设备，如果有，则使用第一个设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_new")
chkpt = svd.processed_dir() / "clean_codebert/202307141026_1fca96a_update_get_data.sh/epoch=9-step=30.ckpt"
# 加载检查点文件
checkpoint = torch.load(chkpt)
# 提取 state_dict
# 如果检查点是由 PyTorch Lightning 保存的，则 state_dict 会被存储在 'state_dict' 键下
model_state_dict = checkpoint['state_dict']

data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=1024,
    nsampling_hops=2,
    gtype="pdg+raw",
    splits="default",
    feat="codebert",
)
# 使用 DataLoader 获取测试数据
test_loader = data.test_dataloader()


class LitGNNWrapper(lvd.LitGNN):
    def forward(self, graph, feat, **kwargs):
        # 调整 forward 调用以匹配原始 LitGNN forward 方法
        # 这里忽略了 GNNExplainer 可能传递的其他 kwargs
        output = super().forward(g=graph, test=False, e_weights=[], feat_override='')
        return output[0]

# 创建包装类的实例，而不是原始 LitGNN
model_wrapper = LitGNNWrapper()

print("model_wrapper:", model_wrapper.state_dict().keys())
print("ckpt:", model_state_dict.keys())

model_wrapper.load_state_dict(model_state_dict)

# 现在，使用这个包装模型与 GNNExplainer
explainer = GNNExplainer(model_wrapper, num_epochs=200, num_hops=2, lr=0.1)


# 解释预测结果
input_nodes = [0, 1, 2]  # 选择需要解释的节点
for g in test_loader:
    # 将图移动到适当的设备
    g = g.to(device)
    node_feat_mask = explainer.explain_graph(g, g.ndata['_CODEBERT'], target_nodes=input_nodes)
    print("节点重要性分数:")
    print(node_feat_mask)
