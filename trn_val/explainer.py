import sastvd as svd
import sastvd.linevd as lvd
import torch
from sastvd.linevd.gnnexplainer import get_node_importances
import matplotlib.pyplot as plt
import networkx as nx

# 检查是否有可用的 CUDA 设备，如果有，则使用第一个设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_new")
chkpt = svd.processed_dir() / "clean_codebert/202307141026_1fca96a_update_get_data.sh/epoch=9-step=30.ckpt"

# Load model
model = lvd.LitGNN()
model = lvd.LitGNN.load_from_checkpoint(chkpt, strict=False)

data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=1024,
    nsampling_hops=2,
    gtype="pdg+raw",
    splits="default",
    feat="codebert",
)
# 使用 DataLoader 获取测试数据
test_loader = data.test_dataloader()

# 解释预测结果
for g in test_loader:
    g = g.to(device)
    node_importances = get_node_importances(model, g)

    print("节点重要性分数的长度:", len(node_importances))
    print("图g的节点数:", g.number_of_nodes())

    # 确保 node_importances 的长度与 g 的节点数量一致
    assert len(node_importances) == g.number_of_nodes(), "长度不匹配"

    G = nx.DiGraph()  # 或者 nx.Graph() 取决于图的类型

    # 添加节点和边
    for node in range(g.number_of_nodes()):
        if hasattr(node_importances[node], 'item'):
            importance = node_importances[node].item()
        else:
            importance = node_importances[node]
        G.add_node(node, importance=importance)

    for edge in g.edges():
        G.add_edge(edge[0], edge[1])

    # 检查和设置节点颜色
    node_colors = []
    for node in G.nodes():
        if 'importance' not in G.nodes[node]:
            print(f"Importance not set for node {node}")
        node_colors.append(G.nodes[node].get('importance', 0))  # 如果 importance 不存在，使用默认值 0

    # 绘制图形
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Blues)
    plt.show()


    print("节点重要性分数:")
    print(node_importances)