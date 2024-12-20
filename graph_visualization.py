import json
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt

graph_path = 'filtered_results/hetero_graph_small_1_0.pt'
data = torch.load(graph_path)

metadata_csv_path = 'datasets/filtered_metadata.csv'
metadata = pd.read_csv(metadata_csv_path)
item_asins = metadata['parent_asin'].tolist()
item_titles = metadata['title'].tolist()

item_index_to_info = {idx: (asin, title) for idx, (asin, title) in enumerate(zip(item_asins, item_titles))}

user_id = "AGGGJOYI5L5LXQKIEF6WIJYQIE7Q"

selected_asins = [
    'B07HGT7JC8',  # Final Fantasy X & X-2 HD Remaster - Xbox One
    'B08XD54VJY'   # The Legend of Zelda: Skyward Sword HD - Nintendo Switch
]

selected_item_indices = [item_asins.index(asin) for asin in selected_asins]

G = nx.DiGraph()

G.add_node(user_id, color='lightblue', size=700)

for item_index in selected_item_indices:
    asin, title = item_index_to_info.get(item_index, ('Unknown ASIN', 'Unknown Title'))
    G.add_node(f'{asin} ({title})', color='lightgreen', size=500)
    G.add_edge(user_id, f'{asin} ({title})', label='Purchased')

item_item_edge_index = data['item', 'similar', 'item'].edge_index
item_item_edge_attr = data['item', 'similar', 'item'].edge_attr

normalized_similarities = item_item_edge_attr / 2.0

for item_index in selected_item_indices:
    asin, title = item_index_to_info.get(item_index, ('Unknown ASIN', 'Unknown Title'))
    similar_items_mask = item_item_edge_index[0] == item_index
    similar_items = item_item_edge_index[1][similar_items_mask]
    similarity_scores = normalized_similarities[similar_items_mask].flatten().tolist()

    top_similar_items = similar_items[:3]
    top_similarity_scores = similarity_scores[:3]

    for sim_item_index, sim_score in zip(top_similar_items.tolist(), top_similarity_scores):
        sim_asin, sim_title = item_index_to_info.get(sim_item_index, ('Unknown ASIN', 'Unknown Title'))
        G.add_node(f'{sim_asin} ({sim_title})', color='orange', size=400)
        G.add_edge(f'{asin} ({title})', f'{sim_asin} ({sim_title})', label=f'Sim: {sim_score:.2f}')

plt.figure(figsize=(12, 10))

node_colors = [G.nodes[n]['color'] for n in G.nodes]
node_sizes = [G.nodes[n]['size'] for n in G.nodes]

pos = nx.spring_layout(G, seed=41, k=0.5, iterations=100)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

nx.draw_networkx_edges(G, pos, edge_color='gray')

nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

plt.title(f'User {user_id} Interactions and Similar Items')
plt.show()