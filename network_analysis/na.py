import pandas as pd
import numpy as np
import ast
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyvis.network import Network
import os
import nltk
nltk.download('vader_lexicon')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

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

def estrai_grafo(path, section, only_main=False, sentiment=False):

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

        if sentiment==True:

            tokens=pd.read_csv('network_analysis\data\preproc_attwn.tokens', sep='\t')[["sentence_ID","word"]]
            tokens['word'] = tokens['word'].fillna('')
            sia=SentimentIntensityAnalyzer()
            G = nx.Graph()
            for x,y,s in filtered_edges:
                filtered_tokens=tokens[tokens['sentence_ID']==int(s)]
                filtered_tokens_list=list(filtered_tokens['word'])
                sentence=' '.join(filtered_tokens_list)
                sent=sia.polarity_scores(sentence)['compound']
                if (x,y) in G.edges:
                    G.edges[x, y]['sentiment']+=sent
                else:
                    G.add_edge(x,y, sentiment=sent)

            if only_main==True:
                G=G.subgraph(main_characters)
                print("Subgraph with only the ten main characters created!!!")

            return G

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
    try:
        # Increased max_iter to 1000 to improve convergence chances
        eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
        display_results(eig_cent)
    except nx.PowerIterationFailedConvergence:
        print('Analysis Failed: Eigenvector centrality failed to converge.')

    print('\n\nKATZ CENTRALITY (changing to simple digraph)')
    try:
        # Katz is sensitive to the alpha parameter
        katz_cent = nx.katz_centrality(G, max_iter=1000)
        display_results(katz_cent)
    except nx.PowerIterationFailedConvergence:
        print('Analysis Failed: Katz centrality failed to converge.')
    except Exception as e:
        # Catches other potential mathematical errors (e.g., alpha too large)
        print(f'Analysis Failed: An error occurred during Katz calculation ({e})')

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

    plt.savefig(f'network_analysis/networks/{title}.png', dpi=300, bbox_inches='tight')


def full_analysis(file, division, only_main_characters, visualization, analysis, sentiment):
    
    global blocks
    blocks=division.copy()    

    graphs_list=[]

    sections=list(blocks.keys())         
    for se in sections:

        print('\n\n\n***', se.upper(), '***\n\n\n')

        G=estrai_grafo(file, section=se, only_main=only_main_characters, sentiment=sentiment)  #only_main means if you want to take into account only the ten protagonists
        #G.remove_node('Lawrence John Wargrave')            #if we are interested in removing some characters to see what happens
        
        if visualization==True:

            if sentiment==True:
                visualize_sentiment_graph(G, se)
            else:
                visualize_closeness(G, se)

        if analysis==True:
            network_analysis(G)
    
        graphs_list.append(G)

    return graphs_list


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


def simplify_multigraph(mg):
    if len(mg.nodes)==1:
        return nx.Graph(mg)
    # Crea un grafo semplice (Graph) partendo dai nodi del multigrafo
    g = nx.Graph()
    # Calcola la molteplicità di ogni coppia (u, v) e imposta il peso come 1/molteplicità
    g.add_edges_from((u, v, {'weight': 1/mg.number_of_edges(u, v)}) for u, v in mg.edges())
    return g

def print_graph_summary(g):
    print(f"Resoconto Grafo: {g.number_of_nodes()} nodi, {g.number_of_edges()} archi")
    print("-" * 30)
    # Itera sugli archi chiedendo anche i dati (data=True) per accedere al peso
    for u, v, data in g.edges(data=True):
        peso = data.get('weight', 'N/A')
        print(f"Arco ({u} <-> {v}) | Peso (1/n): {peso:.4f}")

def plot_centrality_evolution(labels, characters_data):
    """
    Plots closeness centrality evolution for multiple characters.
    
    :param labels: List of strings (phases/deaths)
    :param characters_data: Dictionary {character_name: [centrality_scores]}
    """
    x_ticks = list(range(len(labels)))
    x_points = [x - 0.5 for x in x_ticks]
    
    plt.figure(figsize=(12, 7))

    for name, scores in characters_data.items():
        # Ensure scores match the number of labels
        plt.plot(x_points, scores, marker='o', label=name, linewidth=1.5)

    # Label mapping and axis formatting
    plt.xticks(x_ticks, labels, rotation=45)
    plt.xlim(-1, len(labels) - 1)
    plt.xlabel("Timeline (Deaths)")
    plt.ylabel("Closeness Centrality")
    plt.title("Character Centrality Evolution - And Then There Were None")
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legend outside the plot
    plt.tight_layout()
    
    plt.show()


def calculate_closeness_evolution(graphs_list):    # the graph_list must be based on deaths
    # this list is useful to calculate closeness based on distance:
    weighted_graphs_list=[]
    for mg in graphs_list:
        weighted_graphs_list.append(simplify_multigraph(mg))

    characters_closeness_centralities={}
    for ch in main_characters:
        characters_closeness_centralities[ch]=[]
        for step in weighted_graphs_list:
            if ch in step.nodes:
                cl=nx.closeness_centrality(step, distance='weight')[ch]    
            else:
                cl=0
            characters_closeness_centralities[ch].append(cl)

    labels=list(deaths.keys())
    plot_centrality_evolution(labels, characters_closeness_centralities)




GREEN     = "#2ca02c"
RED       = "#d62728"
GRAY      = "#999999"
WIDTH_MIN = 0.5
WIDTH_MAX = 6.0


def visualize_sentiment_graph(G: nx.Graph, title: str = "sentiment_graph"):
    """
    Visualize a graph coloring edges by their 'sentiment' attribute:
      - Green (thicker = stronger positive) if sentiment > 0
      - Gray  (thin)                        if sentiment == 0
      - Red   (thicker = stronger negative) if sentiment < 0

    The image is saved to 'networks/{title}.png'.

    Args:
        G:     NetworkX graph with a 'sentiment' attribute on every edge.
        title: Output filename (no extension).
    """
    # Collapse multigraphs by averaging sentiment on parallel edges
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G_simple = nx.Graph()
        for u, v, data in G.edges(data=True):
            s = data.get("sentiment", 0)
            if G_simple.has_edge(u, v):
                G_simple[u][v]["sentiment"] += s
                G_simple[u][v]["count"] += 1
            else:
                G_simple.add_edge(u, v, sentiment=s, count=1)
        for u, v, d in G_simple.edges(data=True):
            d["sentiment"] = d["sentiment"] / d["count"]
    else:
        G_simple = G

    pos = nx.spring_layout(G_simple, k=2, iterations=50, seed=42)

    # --- Edges: fixed color per sign, width scaled by intensity ---
    sentiments = np.array(
        [d.get("sentiment", 0) for _, _, d in G_simple.edges(data=True)],
        dtype=float,
    )

    if sentiments.size == 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_nodes(G_simple, pos, ax=ax, node_size=800, node_color="#D9D9D9")
        nx.draw_networkx_labels(G_simple, pos, ax=ax, font_size=9, font_weight="bold")
        ax.set_title('Before death: '+title, fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        os.makedirs("networks", exist_ok=True)
        plt.savefig(os.path.join("networks", f"{title}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Image saved to: networks/{title}.png (single node, no edges)")
        return

    pos_max = float(sentiments.max()) if sentiments.max() > 0 else 1.0
    neg_min = float(sentiments.min()) if sentiments.min() < 0 else -1.0

    edge_colors, edge_widths = [], []
    for s in sentiments:
        if s > 0:
            edge_colors.append(GREEN)
            edge_widths.append(WIDTH_MIN + (s / pos_max) * (WIDTH_MAX - WIDTH_MIN))
        elif s < 0:
            edge_colors.append(RED)
            edge_widths.append(WIDTH_MIN + (s / neg_min) * (WIDTH_MAX - WIDTH_MIN))
        else:
            edge_colors.append(GRAY)
            edge_widths.append(WIDTH_MIN)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(14, 10))

    nx.draw_networkx_nodes(G_simple, pos, ax=ax, node_size=800, node_color="#D9D9D9")
    nx.draw_networkx_labels(G_simple, pos, ax=ax, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G_simple, pos, ax=ax, edge_color=edge_colors, width=edge_widths)

    legend_elements = [
        Line2D([0], [0], color=GREEN, linewidth=WIDTH_MIN, label="Positive sentiment (low)"),
        Line2D([0], [0], color=GREEN, linewidth=WIDTH_MAX,  label="Positive sentiment (high)"),
        Line2D([0], [0], color=GRAY,  linewidth=WIDTH_MIN, label="Neutral sentiment (0)"),
        Line2D([0], [0], color=RED,   linewidth=WIDTH_MIN, label="Negative sentiment (low)"),
        Line2D([0], [0], color=RED,   linewidth=WIDTH_MAX,  label="Negative sentiment (high)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Edge Sentiment")
    ax.set_title('Before death: '+title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    os.makedirs("networks", exist_ok=True)
    out_path = os.path.join("networks", f"{title}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved to: {out_path}")