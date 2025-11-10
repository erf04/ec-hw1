import numpy as np

class Seller:
    def __init__(self, name, price, ad_budget, cost, alpha=0.5, beta=-2.0, gamma=1.0, base_demand=50):
        """
        Represents a single seller (player) in the market.

        Parameters:
            name (str): Name or ID of the seller
            price (float): Current price per unit
            ad_budget (float): Current advertising budget
            cost (float): Production cost per unit
            alpha, beta, gamma (float): Model coefficients
            base_demand (float): Baseline demand without ads or influence
        """
        self.name = name
        self.price = price
        self.ad_budget = ad_budget
        self.cost = cost
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.base_demand = base_demand
        
        # These will be computed later
        self.demand = 0
        self.profit = 0

    def compute_demand(self, competitors, influence_score):
        """
        Computes the demand for this seller based on its own price and
        advertising level, plus competitors' prices and social influence.
        """
        avg_competitor_price = np.mean([s.price for s in competitors if s.name != self.name])
        self.demand = (
            self.base_demand
            + self.alpha * self.ad_budget
            + self.beta * (self.price - avg_competitor_price)
            + self.gamma * influence_score
        )
        self.demand = max(0, self.demand)  # Demand can't be negative
        return self.demand

    def compute_profit(self):
        """
        Computes profit = (price - cost) * demand - ad_budget
        """
        self.profit = (self.price - self.cost) * self.demand - self.ad_budget
        return self.profit

    def __repr__(self):
        return f"Seller({self.name}: Price={self.price}, Ads={self.ad_budget}, Profit={self.profit:.2f}, Demand={self.demand:.2f})"
