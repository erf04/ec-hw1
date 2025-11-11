from random import randint
from modeling.seller import Seller
import pandas as pd

def simulate_nash(sellers: list[Seller], max_iterations=1000, epsilon=0.1):
    """
    Simulate competition until Nash equilibrium is approximately reached.
    Returns final seller states.
    """
    # If sellers already have network-based influence, use it; otherwise randomize
    influence_scores = {
        s.name: getattr(s, "influence_score", randint(3, 8)) for s in sellers
    }

    history = []

    # Initial computation
    for s in sellers:
        s.compute_demand(sellers, influence_scores[s.name])
        s.compute_profit()

    for iteration in range(max_iterations):
        max_change = 0
        for s in sellers:
            old_price, old_ads, old_profit = s.price, s.ad_budget, s.profit

            s.update_strategy(sellers, influence_scores[s.name])

            # Compute the change magnitude
            delta = max(
                abs(s.price - old_price),
                abs(s.ad_budget - old_ads),
                abs(s.profit - old_profit),
            )
            max_change = max(max_change, delta)

        # Recompute all profits after this iteration
        for s in sellers:
            s.compute_demand(sellers, influence_scores[s.name])
            s.compute_profit()
            # Record history
            history.append({
                "Iteration": iteration + 1,
                "Seller": s.name,
                "Price": s.price,
                "AdBudget": s.ad_budget,
                "Profit": s.profit
            })

        if max_change < epsilon:
            print(f"Nash equilibrium reached at iteration {iteration + 1}")
            break

    history_df = pd.DataFrame(history)

    return sellers,history_df
