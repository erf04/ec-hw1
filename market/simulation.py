from modeling.seller import Seller
import numpy as np

def simulate_market(sellers:list[Seller], influence_scores, max_iter=200, tolerance=1e-5, learning_rate=0.3):
    """
    Simulates iterative strategy adjustments for multiple sellers until reaching equilibrium.
    Tweaked for smoother convergence and more iterations.
    """
    history = []

    for iteration in range(max_iter):
        total_profit_change = 0

        for s in sellers:
            # 1️⃣ Compute current demand and profit
            s.compute_demand(sellers, influence_scores[s.name])
            old_profit = s.compute_profit()

            # Keep track of old values
            old_price, old_ads = s.price, s.ad_budget
            best_price, best_ads = old_price, old_ads
            best_profit = old_profit

            # 2️⃣ Search a local neighborhood for better strategies
            price_candidates = [old_price - 2, old_price - 1, old_price, old_price + 1, old_price + 2]
            ad_candidates = [old_ads - 2, old_ads - 1, old_ads, old_ads + 1, old_ads + 2]

            for p in price_candidates:
                for m in ad_candidates:
                    if p <= 0 or m < 0:
                        continue  # invalid
                    s.price, s.ad_budget = p, m
                    s.compute_demand(sellers, influence_scores[s.name])
                    profit = s.compute_profit()
                    if profit > best_profit:
                        best_profit = profit
                        best_price, best_ads = p, m

            # 3️⃣ Move gradually toward the best found strategy
            s.price += learning_rate * (best_price - old_price)
            s.ad_budget += learning_rate * (best_ads - old_ads)

            # 4️⃣ Recalculate final profit and measure change
            s.compute_demand(sellers, influence_scores[s.name])
            new_profit = s.compute_profit()
            total_profit_change += abs(new_profit - old_profit)

        history.append(sum(s.profit for s in sellers))

        # 🧩 Print progress occasionally
        if iteration % 10 == 0 or iteration == max_iter - 1:
            avg_price = np.mean([s.price for s in sellers])
            print(f"Iteration {iteration:3d} → Total Profit={sum(s.profit for s in sellers):.2f}, Avg Price={avg_price:.2f}")

        # 5️⃣ Check convergence
        if total_profit_change / len(sellers) < tolerance:
            print(f"\n✅ Equilibrium reached after {iteration+1} iterations.")
            break

    return sellers, history
