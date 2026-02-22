import pandas as pd
import ast
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyvis.network import Network
import os

blocks={'entire_book':[(3,66980),(1,5021)],    #the whole book, from the very beginning to the very end
        'entire_story':[(3,59218),(1,4544)],   #the whole story, except the epilogues
        'first':[(3,36525),(1,2773)],          #from chapter 1 to 9 (from the beginning until trial included)
        'second':[(36529,49430),(2774,3747)],     #from chapter 10 to 13 (after the trial until the fake death included)
        'third':[(49434,59218),(3749,4544)],      #from chapter 14 to 15/16 (the conclusion)
        'epilogue':[(59220,62734),(4545,4770)],   #Police in Scotland Yard analyzing the case
        'manuscript':[(62753,66980),(4771,5021)]} #the final chapter, containing Wargrave's confession
    #'block_name':[(tokens_ids_interval),(sentences_ids_interval)]

chapters={'1':[(3,3883),(1,269)],
          '2':[(),(271,655)],
          '3':[(),(657,1008)],
          '4':[(),(1009,1235)],
          '5':[(),(1237,1519)],
          '6':[(),(1520,1839)],
          '7':[(),(1840,2050)],
          '8':[(),(2051,2350)],
          '9':[(),(2352,2773)],
          '10':[(),(2774,2988)],
          '11':[(),(2989,3256)],
          '12':[(),(3257,3481)],
          '13':[(),(3483,3747)],
          '14':[(),(3749,4097)],
          '15':[(53509,57476),(4098,4393)],
          '16':[(),(4394,4544)],
          '17':[(),(4545,4770)],
          '18':[(),(4771,5021)]
          }

deaths={'1_marston':[(),(1,1292)],      
        '2_ethel_rogers':[(),(1293,1609)],
        '3_macarthur':[(),(1610,2486)],
        '4_thomas_rogers':[(),(2487,3057)],
        '5_brent':[(),(3058,3344)],
        '6_wargrave':[(),(3345,3749)],
        '7_blore':[(),(3750,4314)],
        '8_armstrong':[(),(4315,4393)],
        '9_lombard':[(),(4394,4455)],
        '10_claythorne':[(),(4456,4544)]
        }

main_characters=["Philip Lombard",
                 "Vera Elizabeth Claythorne",
                 "William Henry Blore",
                 "Edward George Armstrong",
                 "Emily Brent",
                 "Lawrence John Wargrave",
                 "Ethel Rogers",
                 "Anthony James Marston",
                 "John Gordon Macarthur",
                 "Thomas Rogers",
                 "Owen"]

def estrai_grafo(path, section, only_main=False):

    if 'avp_triples' in path:
        block=blocks[section][0]
        dataf=pd.read_csv(path, sep='\t', dtype={'index':int})
        dataf=dataf[(dataf['role_left'] == 'agent')&(dataf['role_right'] == 'patient')&(block[0] <= dataf['index'])&(dataf['index'] <= block[1])]
        dataf=dataf[["name_left","lemma","name_right", "negated"]]
        triplestore = list(dataf.itertuples(index=False, name=None))
        G = nx.MultiDiGraph()
        G.add_edges_from((s, o, {'label': p}) for s, p, o, n in triplestore)
        print('Graph Created!!!')

        if only_main==True:
            G=G.subgraph(main_characters)
            print("Subgraph with only the ten main characters created!!!")

        net = Network(notebook=True, height="1000px", width="100%", directed=False)

        for node, attrs in G.nodes(data=True):
            net.add_node(
                node,
                label=str(attrs.get("name_left", node)),
                title=str(attrs)
            )

        for u, v, attrs in G.edges(data=True):
            net.add_edge(
                u,
                v,
                label=attrs.get("label"),
            )

        # net.show(out_dir)
        # Create output directory if it does not yet exist
        if not os.path.isdir('graphs'):
            os.makedirs('graphs')

        net.save_graph(str(f'graphs/{section}.html'))


        return G
    
    elif 'edge_list' in path:
        
        dataf=pd.read_csv(path, sep=',')[["source_name","sentence_ids","target_name"]]
        dataf['sentence_ids'] = dataf['sentence_ids'].apply(ast.literal_eval)

        triplestore = list(dataf.itertuples(index=False, name=None))
        
        real_edges=[]
        for e in triplestore:
            for c in range(len(e[1])):
                real_edges.append((e[0],e[2],e[1][c]))   
        block=blocks[section][1]

        filtered_edges=[]
        for st in real_edges:
            if block[0] <= int(st[2]) <= block[1] :
                filtered_edges.append(st)

        G = nx.MultiGraph()
        G.add_edges_from((x, y) for x,y,s in filtered_edges)
        if only_main==True:
            G=G.subgraph(main_characters)
            print("Subgraph with only the ten main characters created!!!")

        #G.add_edges_from((x, y, {'penwidth': p}) for x, p, y in triplestore)
        print('Graph Created!!!')

        '''
        net = Network(notebook=True, height="1000px", width="100%", directed=True)

        net.set_options("""
        {
            "physics": {
                "enabled": false
            }
        }
        """)

        for node, attrs in G.nodes(data=True):
            net.add_node(
                node,
                label=str(attrs.get("name_left", node)),
                title=str(attrs)
            )

        for u, v, attrs in G.edges(data=True):
            net.add_edge(
                u,
                v,
                label=attrs.get("label"),
            )

        # net.show(out_dir)
        # Create output directory if it does not yet exist
        if not os.path.isdir('graphs'):
            os.makedirs('graphs')

        net.save_graph(str(f'graphs/{section}.html'))
        '''
        
        return G

def display_results(d):
    ordinato_desc = sorted(d.items(), key=lambda item: item[1], reverse=True)
    for node, value in ordinato_desc:
        print(f'{node};{value}')
    return ordinato_desc

def calculate_results(d):
    ordinato_desc = sorted(d.items(), key=lambda item: item[1], reverse=True)
    return ordinato_desc

def network_analysis(G):

    print('\n\nDEGREE CENTRALITY')
    display_results(nx.degree_centrality(G))

    print('\n\nBETWEENNESS CENTRALITY')
    display_results(nx.betweenness_centrality(G))

    print('\n\nCLOSENESS CENTRALITY')
    display_results(nx.closeness_centrality(G))

    if type(G)==nx.MultiDiGraph:
        G = nx.DiGraph(G)   #we convert the multidigraph to a simple digraph to perform the following measures
        print('\n\nMultiDiGraph converted to DiGraph')
    
    elif type(G)==nx.MultiGraph:
        G = nx.Graph(G)   #we convert the multigraph to a simple graph to perform the following measures
        print('\n\nMultiGraph converted to Graph')


    print('\n\nEIGENVECTOR CENTRALITY (changing to simple digraph)')
    display_results(nx.eigenvector_centrality(G))

    print('\n\nKATZ CENTRALITY (changing to simple digraph)')
    display_results(nx.katz_centrality(G))

def visualize(G):
    # Converti multigraph a grafo semplice se necessario
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        G_simple = nx.Graph(G)
    else:
        G_simple = G

    pos = nx.spring_layout(G_simple, k=2, iterations=50, seed=42)

    # Calcola centralità
    degree_cent = nx.degree_centrality(G_simple)
    node_sizes = [v * 5000 for v in degree_cent.values()]

    between_cent = nx.betweenness_centrality(G_simple)
    node_colors = list(between_cent.values())

    # CORREZIONE: crea figura e axes esplicitamente
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw network
    nodes = nx.draw_networkx_nodes(G_simple, pos,
                                    node_size=node_sizes,
                                    node_color=node_colors,
                                    cmap=plt.cm.plasma,
                                    ax=ax)

    nx.draw_networkx_labels(G_simple, pos, 
                            font_size=9, 
                            font_weight='bold',
                            ax=ax)

    nx.draw_networkx_edges(G_simple, pos,
                        edge_color='gray',
                        width=0.5,
                        ax=ax)

    # CORREZIONE: aggiungi colorbar usando l'oggetto nodes
    plt.colorbar(nodes, ax=ax, label='Betweenness Centrality')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_degree_and_betwenness(G, title):
    # Converti multigraph a grafo semplice se necessario
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        G_simple = nx.Graph(G)
    else:
        G_simple = G

    pos = nx.spring_layout(G_simple, k=2, iterations=50, seed=42)

    # Calcola centralità
    degree_cent = nx.degree_centrality(G_simple)   
    node_sizes = [v * 5000 for v in degree_cent.values()]

    between_cent = nx.betweenness_centrality(G_simple)
    node_colors = list(between_cent.values())

    # Crea figura e axes
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw network
    nodes = nx.draw_networkx_nodes(G_simple, pos, 
                                    node_size=node_sizes,
                                    node_color=node_colors,
                                    cmap=plt.cm.plasma,
                                    ax=ax)

    nx.draw_networkx_labels(G_simple, pos, 
                            font_size=9, 
                            font_weight='bold',
                            ax=ax)

    nx.draw_networkx_edges(G_simple, pos,
                        edge_color='gray',
                        width=0.5,
                        ax=ax)

    # Colorbar per betweenness
    plt.colorbar(nodes, ax=ax, label='Betweenness Centrality')

    # LEGGENDA per degree centrality (dimensione nodi)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='High Degree',
            markerfacecolor='gray', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Medium Degree',
            markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low Degree',
            markerfacecolor='gray', markersize=5)
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Degree Centrality')

    ax.axis('off')
    plt.tight_layout()

    if not os.path.isdir('networks'):
        os.makedirs('networks')

    plt.savefig(f'networks/{title}.png', dpi=300, bbox_inches='tight')  #otherwise we can choose .pdf or .svg format

def visualize_closeness(G, title):
    # Converti multigraph a grafo pesato semplice
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        G_simple = nx.Graph()
        for u, v in G.edges():
            if G_simple.has_edge(u, v):
                G_simple[u][v]['weight'] += 1
            else:
                G_simple.add_edge(u, v, weight=1)
    else:
        G_simple = G

    # Aggiungi distanza invertita per closeness corretta
    for u, v, d in G_simple.edges(data=True):
        d['distance'] = 1 / d['weight']

    pos = nx.spring_layout(G_simple, k=2, iterations=50, seed=42)

    # --- NODI: closeness centrality → scala di rossi ---
    closeness_cent = nx.closeness_centrality(G_simple, distance='distance')
    node_colors = list(closeness_cent.values())
    node_sizes = [v * 3000 + 300 for v in node_colors]

    # --- EDGES: peso → intensità del nero ---
    edge_weights = [G_simple[u][v]['weight'] for u, v in G_simple.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    edge_colors = [(1 - w/max_w * 0.85) for w in edge_weights]
    edge_colors_rgb = [(c, c, c) for c in edge_colors]
    edge_widths = [0.5 + (w / max_w) * 3 for w in edge_weights]

    # --- FIGURA ---
    fig, ax = plt.subplots(figsize=(14, 10))

    nodes = nx.draw_networkx_nodes(G_simple, pos,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap=plt.cm.Reds,
                                   ax=ax)

    nx.draw_networkx_labels(G_simple, pos,
                            font_size=9,
                            font_weight='bold',
                            ax=ax)

    nx.draw_networkx_edges(G_simple, pos,
                           edge_color=edge_colors_rgb,
                           width=edge_widths,
                           ax=ax)

    plt.colorbar(nodes, ax=ax, label='Closeness Centrality')

    '''
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=3, label='High co-occurrence'),
        Line2D([0], [0], color='gray', linewidth=1.5, label='Medium co-occurrence'),
        Line2D([0], [0], color='lightgray', linewidth=0.5, label='Low co-occurrence'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Edge Weight')
    '''

    ax.axis('off')
    plt.tight_layout()

    if not os.path.isdir('networks'):
        os.makedirs('networks')

    plt.savefig(f'networks/{title}.png', dpi=300, bbox_inches='tight')


def full_analysis(file, division, only_main_characters, visualization, analysis):
    
    global blocks
    blocks=division.copy()    

    sections=list(blocks.keys())         
    for se in sections:

        print('\n\n\n***', se.upper(), '***\n\n\n')

        G=estrai_grafo(file, section=se, only_main=only_main_characters)  #only_main means if you want to take into account only the ten protagonists
        #G.remove_node('Lawrence John Wargrave')            #if we are interested in removing some characters to see what happens
        
        if visualization==True:
            visualize_closeness(G, se)

        if analysis==True:
            network_analysis(G)
    
    return G


#EXAMPLE OF USAGE
#full_analysis(file="data/edge_list_owen.csv",
#              division=deaths,
#              only_main_characters=False,
#              visualization=True,
#              analysis=False)


def killer_hypothesis(G):

    display_results(nx.closeness_centrality(G))

    for c in main_characters:
        print('\nHYPOTHESIS: OWEN IS', c.upper())
        newG=nx.contracted_nodes(G,c,"Owen")
        d=calculate_results(nx.closeness_centrality(newG))
        if d[0][0]==c:
            print('Possible: Highest closeness centrality\n'.upper())
        else:
            print('Impossible: Not highest closeness centrality\n'.upper())