import networkx as nx
from na import *

story={'story':[(0,60964),()]}



graphs_list=full_analysis(file="network_analysis/data/processed_triples.csv",     #if you want wargrave and owen as the same node, you can imput the file edge_list.csv
                            division=new_deaths,                    #here you can also put division=blocks or division=chapters as parameters
                            only_main_characters=False,          #here you choose if the nodes are only the ten protagonists or not
                            visualization=True,                 #this creates the images in a folder entitled 'networks'
                            analysis=False,                     #this prints the measures results
                            sentiment=True)                   #special parameter, only for returning graphs with sentiment 


calculate_sentiment_evolution(graphs_list)

print('\n\nDONE! 🥳🎉')
#nx.write_gexf(G, 'graph.gexf')  #If you activate this operation, you get a file you can put in GEPHI software and there you can perform further analysis