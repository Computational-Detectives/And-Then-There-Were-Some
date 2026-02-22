import networkx as nx
from na import *

G=full_analysis(file="data/edge_list_owen.csv",     #if you want wargrave and owen as the same node, you can imput the file edge_list.csv
                division=deaths,                    #here you can also put division=blocks or division=chapters as parameters
                only_main_characters=True,          #here you choose if the nodes are only the ten protagonists or not
                visualization=True,                 #this creates the images in a folder entitled 'networks'
                analysis=False)                     #this prints the measures results

#nx.write_gexf(G, 'graph.gexf')  #If you activate this operation, you get a file you can put in GEPHI software and there you can perform further analysis
