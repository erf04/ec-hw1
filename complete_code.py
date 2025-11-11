# -*- coding: utf-8 -*-
"""
Project: Simulation of Price and Advertising Competition in an Online Market
Following the tasks from EC-hw1.pdf

This script is structured into 5 parts, matching the tasks:
1. Data Preparation
2. Model Sellers, Demand, and Profit
3. Game Simulation & Nash Equilibrium
4. Adding Social Influence
5. Visualization
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import io  # Used to simulate the data file

# --- Constants for the Model ---

# We define model parameters here so they can be easily changed.
# These are from the Task II demand function: D_i = base + (alpha * m_i) + (beta * (p_j - p_i)) + (gamma * influence)
COST = 5.0             # (Task II) Assumed constant cost per product (e.g., 5 units)
BASE_DEMAND = 1000     # (Task II) Base demand for the product
ALPHA = 5.0            # (Task II) Sensitivity to advertising (how much 1 unit of ad budget increases demand)
BETA = 50.0            # (Task II) Sensitivity to price difference (how much demand changes per unit of price difference)
GAMMA = 15.0           # (Task II) Sensitivity to social influence

# Simulation parameters for Task III
# We need to define the "search space" for sellers. What prices/ad budgets can they choose?
# We'll check prices from COST+1 up to 20
PRICE_RANGE = np.arange(COST + 1, 20.0, 0.5)
# We'll check ad budgets from 0 up to 1000
AD_BUDGET_RANGE = np.arange(0, 1001, 50)


# ==============================================================================
# --- Task I: Data Preparation ---
# ==============================================================================

def load_and_clean_data():
    """
    Simulates loading and cleaning the dataset.
    
    Since we can't access the provided link, we create a sample CSV string
    to mimic the file. You can replace this part with:
    df = pd.read_csv('your_file_path.csv')
    """
    print("--- Task I: Starting Data Preparation ---")
    
    # 1. Simulate the data file
    # This data string mimics the structure described in the PDF

    # Read the simulated CSV data into a pandas DataFrame
    # In your real code, replace this line with:
    # df = pd.read_csv('your_dataset_url_or_path.csv', encoding='utf-8')
    try:
        df = pd.read_excel("data.xlsx")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    print(f"Original data shape: {df.shape}")

    # 2. Perform Preprocessing (as described in the PDF)
    
    # Remove rows with null (missing) values
    df.dropna(inplace=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Remove rows with negative Quantity (which are likely returns)
    df = df[df['Quantity'] > 0]
    
    # Remove rows with zero or negative Price
    df = df[df['Price'] > 0]

    print(f"Cleaned data shape: {df.shape}")

    # 3. Perform a simple analysis (as requested in the PDF)
    print("\n--- Simple Data Analysis ---")
    
    # Group by product (using 'Description') and calculate mean price and total quantity
    product_analysis = df.groupby('Description').agg(
        MeanPrice=('Price', 'mean'),
        TotalQuantity=('Quantity', 'sum')
    ).reset_index()

    print("Top 5 Products by Quantity Sold:")
    print(product_analysis.sort_values(by='TotalQuantity', ascending=False).head())
    
    print("\nTop 5 Most Expensive Products (Average):")
    print(product_analysis.sort_values(by='MeanPrice', ascending=False).head())
    
    print("--- Task I: Data Preparation Complete ---\n")
    return df

# ==============================================================================
# --- Task II: Model Sellers, Demand, and Profit ---
# ==============================================================================

"""
Here we define the core logic of the market.
We create functions for demand and profit as specified in the PDF.
We are modeling a "duopoly" (two-seller market) for a single product.
Seller 'i' is our seller, and seller 'j' is the competitor.
"""

print("--- Task II: Defining Demand and Profit Functions ---")

def calculate_demand(p_i, m_i, p_j, m_j, influence_score=0):
    """
    Calculates the demand for seller 'i' based on the formula from Task II.
    D_i = base_demand + (alpha * m_i) + (beta * (p_j - p_i)) + (gamma * influence_score)
    
    Args:
        p_i (float): Price of seller i
        m_i (float): Advertising budget of seller i
        p_j (float): Price of seller j (competitor)
        m_j (float): Advertising budget of seller j (competitor)
        influence_score (float): Social influence score (from Task IV)
    
    Returns:
        float: Calculated demand for seller i.
    """
    # Price difference effect: (beta * (p_j - p_i))
    # If our price (p_i) is lower than p_j, this term is positive, increasing demand.
    price_effect = BETA * (p_j - p_i)
    
    # Advertising effect: (alpha * m_i)
    # Our advertising budget (m_i) increases our demand.
    # Note: The model in the PDF doesn't include the competitor's ads (m_j)
    # in *our* demand, but a real model might make it a ratio, e.g., (m_i / (m_i + m_j)).
    # We will stick to the formula provided.
    ad_effect = ALPHA * m_i
    
    # Social influence effect: (gamma * influence_score)
    influence_effect = GAMMA * influence_score
    
    # Calculate total demand
    demand = BASE_DEMAND + price_effect + ad_effect + influence_effect
    
    # Demand cannot be negative, so we set a floor of 0.
    return max(0, demand)

def calculate_profit(p_i, m_i, p_j, m_j, influence_score=0):
    """
    Calculates the profit for seller 'i' based on the formula from Task II.
    Profit_i = (p_i - cost) * D_i - m_i
    
    Args:
        (Same as calculate_demand)
    
    Returns:
        float: Calculated profit for seller i.
    """
    # First, get the demand for this scenario
    demand_i = calculate_demand(p_i, m_i, p_j, m_j, influence_score)
    
    # Calculate profit
    # (Price per unit - Cost per unit) * Number of units sold - Money spent on ads
    profit = (p_i - COST) * demand_i - m_i
    
    return profit

print("Functions 'calculate_demand' and 'calculate_profit' are defined.")
print("--- Task II: Complete ---\n")


# ==============================================================================
# --- Task III: Game Simulation & Nash Equilibrium ---
# ==============================================================================

def find_best_response(p_j, m_j, influence_score=0):
    """
    Finds the best strategy (price and ad budget) for seller 'i',
    given the strategy of seller 'j'.
    
    This function iterates through all possible strategies for seller 'i'
    (defined in PRICE_RANGE and AD_BUDGET_RANGE) and finds the combination
    that maximizes profit.
    """
    best_profit = -np.inf  # Initialize with a very low number
    best_strategy = (None, None) # (best_p_i, best_m_i)
    
    # Iterate through all possible prices we can set
    for p_i in PRICE_RANGE:
        # Iterate through all possible ad budgets we can set
        for m_i in AD_BUDGET_RANGE:
            # Calculate our profit for this combination
            profit = calculate_profit(p_i, m_i, p_j, m_j, influence_score)
            
            # If this profit is the best we've seen, save it
            if profit > best_profit:
                best_profit = profit
                best_strategy = (p_i, m_i)
                
    return best_strategy, best_profit

def run_simulation(influence_score=0):
    """
    Simulates the game by having sellers iteratively find their best response
    to each other until their strategies stabilize (Nash Equilibrium).
    """
    print(f"--- Task III: Running Simulation (Influence Score: {influence_score}) ---")
    
    # 1. Initialize strategies for Seller A and Seller B
    # We'll start them at a random point.
    # (p_a, m_a) = (price_seller_A, ad_budget_seller_A)
    p_a = np.random.choice(PRICE_RANGE)
    m_a = np.random.choice(AD_BUDGET_RANGE)
    
    # (p_b, m_b) = (price_seller_B, ad_budget_seller_B)
    p_b = np.random.choice(PRICE_RANGE)
    m_b = np.random.choice(AD_BUDGET_RANGE)

    print(f"Initial: P_A: {p_a}, M_A: {m_a} | P_B: {p_b}, M_B: {m_b}")

    # 2. Iterate to find equilibrium
    # We'll loop for a fixed number of iterations (e.g., 20)
    # The game should converge much faster.
    max_iterations = 20
    for i in range(max_iterations):
        # Store the "old" strategies to check for convergence
        old_p_a, old_m_a = p_a, m_a
        old_p_b, old_m_b = p_b, m_b
        
        # --- Seller A's Turn ---
        # Seller A finds their best response to Seller B's *last* move (p_b, m_b)
        (p_a, m_a), profit_a = find_best_response(p_b, m_b, influence_score)
        
        # --- Seller B's Turn ---
        # Seller B finds their best response to Seller A's *new* move (p_a, m_a)
        (p_b, m_b), profit_b = find_best_response(p_a, m_a, influence_score)
        
        print(f"Iter {i+1}: P_A: {p_a:.2f}, M_A: {m_a} | P_B: {p_b:.2f}, M_B: {m_b} | Profit_A: {profit_a:.0f}, Profit_B: {profit_b:.0f}")
        
        # --- Check for Convergence (Equilibrium) ---
        # If both sellers' strategies didn't change this round, we've found the equilibrium.
        if (p_a == old_p_a and m_a == old_m_a) and (p_b == old_p_b and m_b == old_m_b):
            print(f"\nNash Equilibrium found after {i+1} iterations.")
            break
            
    if i == max_iterations - 1:
        print("\nSimulation reached max iterations without perfect convergence.")

    print("--- Task III: Simulation Complete ---")
    
    # Return the final state
    equilibrium_strategies = {
        'seller_A': {'price': p_a, 'ad_budget': m_a, 'profit': profit_a},
        'seller_B': {'price': p_b, 'ad_budget': m_b, 'profit': profit_b}
    }
    return equilibrium_strategies


# ==============================================================================
# --- Task IV: Adding Social Influence ---
# ==============================================================================

def create_social_network(customer_ids):
    """
    Creates a social network using NetworkX based on the customer IDs
    from the dataset.
    
    This is a simplified model. A real one might be built from
    co-purchase data ("customers who bought X also bought Y").
    
    Here, we will:
    1. Create a graph where each customer is a node.
    2. Create random connections (edges) to simulate a social network.
    3. Designate some high-connectivity nodes as "influencers".
    4. Calculate a global "influence score".
    """
    print(f"\n--- Task IV: Creating Social Network ---")
    
    if customer_ids.empty:
        print("No customer IDs to build network.")
        return 0
        
    G = nx.Graph()
    
    # 1. Add customers as nodes
    unique_customers = customer_ids.unique()
    G.add_nodes_from(unique_customers)
    
    # 2. Create random connections (edges)
    # We'll create a "small-world" network (Barabási-Albert graph)
    # which is common in social networks. It has "hubs".
    n_nodes = len(unique_customers)
    if n_nodes < 5:
        print("Not enough unique customers to build a meaningful network. Returning 0 influence.")
        return 0
        
    # Create a scale-free graph with n_nodes, where each new node connects to 2 existing nodes
    G = nx.barabasi_albert_graph(n_nodes, 2)
    # Relabel nodes to match customer IDs (for realism)
    mapping = dict(zip(G.nodes(), unique_customers))
    G = nx.relabel_nodes(G, mapping)

    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 3. Designate influencers
    # We'll say the top 5% of nodes with the highest "degree" (most connections)
    # are influencers.
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    n_influencers = max(1, int(n_nodes * 0.05)) # At least 1, or 5%
    influencers = [node for node, degree in degrees[:n_influencers]]
    
    print(f"Designated {len(influencers)} influencers (e.g., {influencers[0]}).")
    
    # 4. Calculate a global influence score
    # This is a simple metric. We'll say the score is:
    # (Total connections of influencers) + (Total connections of non-influencers / 10)
    # This just gives a single number to plug into our model.
    influencer_connections = sum(degree for node, degree in degrees[:n_influencers])
    normal_connections = sum(degree for node, degree in degrees[n_influencers:])
    
    # A simple weighted score
    influence_score = influencer_connections + (normal_connections * 0.1)
    # We normalize it by dividing by the number of nodes to keep it stable
    influence_score = influence_score / n_nodes
    
    print(f"Calculated global influence score: {influence_score:.2f}")
    print("--- Task IV: Complete ---\n")
    
    return influence_score


# ==============================================================================
# --- Task V: Visualization ---
# ==============================================================================

def plot_profit_landscape(seller_strategy, competitor_strategy, influence_score=0):
    """
    Creates a 2D heatmap showing a seller's profit for various
    price/ad combinations, given a *fixed* strategy for the competitor.
    
    This helps visualize *why* the equilibrium is the best choice.
    """
    print("--- Task V: Generating Profit Landscape Plot ---")
    
    # We will plot Seller A's profit landscape
    p_j = competitor_strategy['price']
    m_j = competitor_strategy['ad_budget']
    
    # Create a matrix to hold profit values
    profit_matrix = np.zeros((len(PRICE_RANGE), len(AD_BUDGET_RANGE)))
    
    # Calculate profit for every combination
    for i, p_i in enumerate(PRICE_RANGE):
        for j, m_i in enumerate(AD_BUDGET_RANGE):
            profit_matrix[i, j] = calculate_profit(p_i, m_i, p_j, m_j, influence_score)
            
    # Find the location of the max profit (our best response)
    max_profit_loc = np.unravel_index(np.argmax(profit_matrix), profit_matrix.shape)
    best_p = PRICE_RANGE[max_profit_loc[0]]
    best_m = AD_BUDGET_RANGE[max_profit_loc[1]]
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    # We flip the profit matrix for intuitive plotting
    sns.heatmap(np.flipud(profit_matrix), 
                xticklabels=AD_BUDGET_RANGE, 
                yticklabels=np.flip(PRICE_RANGE), 
                cmap="viridis")
    
    # Adjust ticks to be less crowded
    plt.xticks(ticks=np.arange(0, len(AD_BUDGET_RANGE), 2), labels=AD_BUDGET_RANGE[::2])
    plt.yticks(ticks=np.arange(0, len(PRICE_RANGE), 2), labels=np.flip(PRICE_RANGE)[::2])
    
    plt.title(f"Seller A Profit Landscape (vs. Seller B at P={p_j}, M={m_j})")
    plt.xlabel("Seller A: Advertising Budget (m_i)")
    plt.ylabel("Seller A: Price (p_i)")
    
    # Mark the equilibrium point
    plt.axvline(x=max_profit_loc[1], color='red', linestyle='--', label=f'Best Ad Budget: {best_m}')
    plt.axhline(y=len(PRICE_RANGE) - 1 - max_profit_loc[0], color='red', linestyle='--', label=f'Best Price: {best_p}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("task_v_profit_landscape.png")
    print("Saved 'task_v_profit_landscape.png'")


def plot_comparison(results_no_net, results_with_net):
    """
    Creates a simple bar chart to compare profits with and
    without the social network.
    """
    print("--- Task V: Generating Comparison Plot ---")
    
    # Data for plotting
    labels = ['Seller A', 'Seller B']
    profit_no_net = [results_no_net['seller_A']['profit'], results_no_net['seller_B']['profit']]
    profit_with_net = [results_with_net['seller_A']['profit'], results_with_net['seller_B']['profit']]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, profit_no_net, width, label='Profit (No Network)')
    rects2 = ax.bar(x + width/2, profit_with_net, width, label='Profit (With Network)')

    # Add some text for labels, title and axes
    ax.set_ylabel('Equilibrium Profit')
    ax.set_title('Profit Comparison: With vs. Without Social Network Influence')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.0f')
    ax.bar_label(rects2, padding=3, fmt='%.0f')

    fig.tight_layout()
    plt.savefig("task_v_profit_comparison.png")
    print("Saved 'task_v_profit_comparison.png'")
    print("--- Task V: Complete ---")

# ==============================================================================
# --- Main Execution ---
# ==============================================================================

if __name__ == "__main__":
    
    # --- Task I ---
    # Load and clean the (simulated) data
    cleaned_df = load_and_clean_data()
    
    # --- Task III (Part 1) ---
    # Run the simulation *without* social influence
    # We pass influence_score=0
    results_no_network = run_simulation(influence_score=0)
    
    print("\n--- FINAL EQUILIBRIUM (NO NETWORK) ---")
    print(f"Seller A: {results_no_network['seller_A']}")
    print(f"Seller B: {results_no_network['seller_B']}")
    
    # --- Task IV ---
    # Create the social network and get the influence score
    if cleaned_df is not None and not cleaned_df.empty:
        global_influence_score = create_social_network(cleaned_df['Customer ID'])
    else:
        print("Skipping network creation due to data loading error.")
        global_influence_score = 0
        
    # --- Task III (Part 2) ---
    # Re-run the simulation *with* the social influence score
    results_with_network = run_simulation(influence_score=global_influence_score)
    
    print("\n--- FINAL EQUILIBRIUM (WITH NETWORK) ---")
    print(f"Seller A: {results_with_network['seller_A']}")
    print(f"Seller B: {results_with_network['seller_B']}")
    
    # --- Task V ---
    # Plot the results
    
    # 1. Plot the profit landscape for a seller
    # We show Seller A's options, assuming Seller B is at the *final*
    # equilibrium strategy (from the "with network" simulation)
    plot_profit_landscape(
        results_with_network['seller_A'],
        results_with_network['seller_B'],
        global_influence_score
    )
    
    # 2. Plot the comparison bar chart
    plot_comparison(results_no_network, results_with_network)
    
    print("\n--- All Tasks Complete. Check for 'task_v_profit_landscape.png' and 'task_v_profit_comparison.png' ---")