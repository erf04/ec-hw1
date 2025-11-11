import networkx as nx

def build_seller_network(all_sellers: dict[str, list]):

    G = nx.Graph()

    # Add nodes
    for product, sellers in all_sellers.items():
        for s in sellers:
            G.add_node(s.name, product=product, influence_score=getattr(s, "influence_score", 0))

    # Add edges between sellers of same product
    for product, sellers in all_sellers.items():
        seller_names = [s.name for s in sellers]
        for i, s1 in enumerate(seller_names):
            for s2 in seller_names[i + 1:]:
                G.add_edge(s1, s2)

    return G


def compute_network_influence(G):

    pagerank_scores = nx.pagerank(G, alpha=0.85)
    nx.set_node_attributes(G, pagerank_scores, "influence_score")
    return pagerank_scores


def update_sellers_with_network_influence(all_sellers: dict[str, list], influence_scores: dict[str, float]):

    for product, sellers in all_sellers.items():
        for s in sellers:
            s.influence_score = influence_scores.get(s.name, 0)
