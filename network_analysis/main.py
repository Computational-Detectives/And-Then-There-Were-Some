import networkx as nx
from na import *

G=full_analysis(file="data/edge_list_owen.csv",
                division=deaths,
                only_main_characters=True,
                visualization=True,
                analysis=False)

#nx.write_gexf(G, 'graph.gexf')         #FILE FOR GEPHI