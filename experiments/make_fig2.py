"""
Fig 2: Behavior graph comparison — reference style
"""
import os; os.environ['PYTHONIOENCODING'] = 'utf-8'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
                      'font.size': 9, 'axes.unicode_minus': False})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
fig.patch.set_facecolor('white')

# Normal graph
G1 = nx.DiGraph()
G1.add_edge('get_balance', 'read_email', weight=3.0)
pos1 = nx.shell_layout(G1)
nx.draw_networkx_nodes(G1, pos1, ax=ax1, node_color='#C8E6C9', edgecolors='#2E7D32',
                        node_size=1800, linewidths=1.5)
nx.draw_networkx_labels(G1, pos1, ax=ax1, font_size=8, font_weight='bold', font_color='#1B5E20')
nx.draw_networkx_edges(G1, pos1, ax=ax1, node_size=1800, width=2.5, edge_color='#2E7D32',
                        arrows=True, arrowsize=15, arrowstyle='-|>',
                        connectionstyle='arc3,rad=0.2')
nx.draw_networkx_edge_labels(G1, pos1, ax=ax1,
    edge_labels={('get_balance','read_email'): 'w=3.0'},
    font_size=8, label_pos=0.5)
ax1.set_title('Normal Session', fontsize=10, fontweight='bold', color='#2E7D32', pad=8)
ax1.axis('off')

# Attack graph
G2 = nx.DiGraph()
edges = [('list_contacts','read_email'), ('read_email','send_email'),
         ('send_email','delete_record'), ('store_memory','send_email'),
         ('list_contacts','store_memory')]
for u,v in edges:
    G2.add_edge(u, v, weight=1.0)
pos2 = nx.shell_layout(G2)
nx.draw_networkx_nodes(G2, pos2, ax=ax2, node_color='#FFCDD2', edgecolors='#C62828',
                        node_size=1800, linewidths=1.5)
nx.draw_networkx_labels(G2, pos2, ax=ax2, font_size=7.5, font_weight='bold', font_color='#B71C1C')
nx.draw_networkx_edges(G2, pos2, ax=ax2, node_size=1800, width=2.0, edge_color='#C62828',
                        arrows=True, arrowsize=15, arrowstyle='-|>',
                        connectionstyle='arc3,rad=0.15')
el2 = {e: 'w=1.0' for e in edges}
nx.draw_networkx_edge_labels(G2, pos2, ax=ax2, edge_labels=el2,
    font_size=7.5, label_pos=0.5)
ax2.set_title('Attack Session', fontsize=10, fontweight='bold', color='#C62828', pad=8)
ax2.axis('off')

plt.tight_layout()
plt.savefig('figures/behavior_graphs.png', dpi=350, bbox_inches='tight', facecolor='white', pad_inches=0.15)
plt.close()
print('Fig2 done - reference style')
