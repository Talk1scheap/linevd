from pathlib import Path
import sastvd as svd
import sastvd.linevd as lvd
from dgl.nn.pytorch import GNNExplainer
import torch
import matplotlib.pyplot as plt

# 检查是否有可用的 CUDA 设备，并使用可用的第一个设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_new")
chkpt = svd.processed_dir() / "clean_codebert/202307141026_1fca96a_update_get_data.sh/epoch=9-step=30.ckpt"

# 加载检查点，并提取模型的状态字典
checkpoint = torch.load(chkpt, map_location=device)
model_state_dict = checkpoint['state_dict']

# 设置数据加载器
data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=1,  # 假设我们一次处理一个图，以便于解释
    nsampling_hops=2,
    gtype="pdg+raw",
    splits="default",
    feat="codebert",
)
test_loader = data.test_dataloader()


# 定义模型包装类
class LitGNNWrapper(lvd.LitGNN):
    def forward(self, graph, feat, **kwargs):
        output = super().forward(g=graph, test=False, e_weights=[], feat_override='')
        return output[0]


# 创建包装类实例
model_wrapper = LitGNNWrapper().to(device)
model_wrapper.load_state_dict(model_state_dict)

# 实例化 GNNExplainer
explainer = GNNExplainer(model=model_wrapper, num_hops=2, lr=0.01)

# 选择需要解释的一个节点
input_node = 0  # 或任何你想解释的特定节点

# 解释预测结果
for g in test_loader:
    g = g.to(device)
    # 计算解释
    node_feat_mask, edge_mask = explainer.explain_graph(g, g.ndata['_CODEBERT'], target_nodes=[input_node])

    # 可视化特征重要性
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(node_feat_mask)), node_feat_mask.cpu().numpy())
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title(f'Top Feature Importance for Node {input_node}')
    plt.savefig(f'feature_importance_node_{input_node}.png')
    plt.show()

    # 可视化子图（此处为伪代码，根据实际需求实现可视化)
    # visualize_subgraph(g, edge_mask, save_path=f'subgraph_node_{input_node}.pdf')

    break  # 如果我们只解释一个批处理中的一个图，就可以在这里停止