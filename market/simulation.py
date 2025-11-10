from modeling.seller import Seller

def simulate_market(sellers: list[Seller], influence_scores: dict, max_iter=100, tolerance=1e-3, learning_rate=0.1):
    """
    Simulates iterative strategy adjustments for multiple sellers until reaching equilibrium.

    Parameters:
        sellers (list[Seller]): List of Seller objects.
        influence_scores (dict): Influence score for each seller.
        max_iter (int): Maximum number of iterations to simulate.
        tolerance (float): Stop if average change in profit is below this.
        learning_rate (float): How aggressively sellers adjust strategies.

    Returns:
        list[Seller]: Sellers with equilibrium prices and ad budgets.
    """
    history = []  # To track total market profit evolution

    for iteration in range(max_iter):
        total_profit_change = 0

        for s in sellers:
            # 1️⃣ Compute current demand and profit
            old_profit = s.compute_profit()

            # 2️⃣ Try small adjustments to find better direction
            old_price, old_ads = s.price, s.ad_budget

            # Test slightly higher and lower prices
            for delta_price in [-1, 1]:
                test_price = old_price + delta_price
                s.price = test_price
                s.compute_demand(sellers, influence_scores[s.name])
                profit_new = s.compute_profit()
                if profit_new > old_profit:
                    old_profit = profit_new
                    old_price = test_price

            # Test slightly higher and lower ad budgets
            for delta_ads in [-1, 1]:
                test_ads = old_ads + delta_ads
                s.ad_budget = test_ads
                s.compute_demand(sellers, influence_scores[s.name])
                profit_new = s.compute_profit()
                if profit_new > old_profit:
                    old_profit = profit_new
                    old_ads = test_ads

            # 3️⃣ Update price & ads gradually (learning rate)
            s.price += learning_rate * (old_price - s.price)
            s.ad_budget += learning_rate * (old_ads - s.ad_budget)

            # 4️⃣ Recompute final profit after update
            s.compute_demand(sellers, influence_scores[s.name])
            new_profit = s.compute_profit()

            total_profit_change += abs(new_profit - old_profit)

        # Record iteration info
        history.append(sum(s.profit for s in sellers))

        # 5️⃣ Check convergence (Nash condition)
        if total_profit_change / len(sellers) < tolerance:
            print(f"✅ Equilibrium reached after {iteration+1} iterations.")
            break

    return sellers, history
