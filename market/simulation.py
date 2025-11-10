from random import randint
from modeling.seller import Seller

def simulate_nash(sellers:list[Seller], max_iterations=1000, epsilon=0.01):
    """
    Simulate competition until strategies stabilize within epsilon or max iterations reached.
    Returns final seller states.
    """
    influence_scores = {s.name: randint(3,8) for s in sellers}
    
    # Initial computation
    for s in sellers:
        s.compute_demand(sellers, influence_scores[s.name])
        s.compute_profit()
    
    for it in range(max_iterations):
        max_change = 0
        for s in sellers:
            old_price, old_ads, old_profit = s.price, s.ad_budget, s.profit
            s.update_strategy(sellers, influence_scores[s.name])
            
            # Track largest absolute change in this iteration
            max_change = max(max_change,
                             abs(s.price - old_price),
                             abs(s.ad_budget - old_ads),
                             abs(s.profit - old_profit))
        
        if max_change < epsilon:
            print(f"Nash equilibrium reached at iteration {it+1}")
            break
    
    return sellers

