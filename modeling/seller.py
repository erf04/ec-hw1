import numpy as np
from random import randint
import pandas as pd
from config.constants import COST,ALPHA,BETA,GAMMA,BASE_DEMAND

class Seller:
    def __init__(self, name, price, ad_budget, cost, alpha=ALPHA, beta=BETA, gamma=GAMMA, base_demand=BASE_DEMAND):
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
        Computes demand using all competitors individually.
        D_i = base_demand + alpha*ad_budget + beta*sum(price_i - price_j) + gamma*influence
        """
        price_diff_sum = sum(self.price - s.price for s in competitors if s.name != self.name)
        
        self.demand = (
            self.base_demand
            + self.alpha * self.ad_budget
            + self.beta * price_diff_sum
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

    def info(self):
        return {
            "Seller": self.name,
            "Price": self.price,
            "Ads": self.ad_budget,
            "Demand": round(self.demand, 2),
            "Profit": round(self.profit, 2)
        }
    
    def update_strategy(self, rivals, influence, step_price=0.5, step_ads=1):
        """
        Simple greedy update: try small changes in price and ads, pick the best profit.
        Returns True if strategy changed.
        """
        best_profit = self.profit
        best_price = self.price
        best_ads = self.ad_budget
        
        # try increasing/decreasing price
        for delta_p in [-step_price, 0, step_price]:
            for delta_a in [-step_ads, 0, step_ads]:
                delta_p = min(max(delta_p, -0.1), 0.1)
                delta_a = min(max(delta_a, -0.5), 0.5)
                self.price += delta_p
                self.ad_budget += delta_a
                self.compute_demand(rivals, influence)
                self.compute_profit()
                
                if self.profit > best_profit:
                    best_profit = self.profit
                    best_price = self.price
                    best_ads = self.ad_budget
                
                # revert temporarily
                self.price -= delta_p
                self.ad_budget -= delta_a

            step_price *= 0.99
            step_ads *= 0.99
        
        # apply best found
        changed = (self.price != best_price) or (self.ad_budget != best_ads)
        self.price = best_price
        self.ad_budget = best_ads
        self.profit = best_profit
        return changed
    
    def __repr__(self):
        return f"Seller({self.name}: Price={self.price}, Ads={self.ad_budget}, Profit={self.profit:.2f}, Demand={self.demand:.2f})"





def create_sellers_for_product(df: pd.DataFrame, description: str, max_sellers: int = 2) -> list[Seller]:
    """
    Given a Description, generate sellers based on unique prices.
    Returns a list of Seller objects.
    """
    product_sales = df[df["Description"] == description]
    unique_prices = sorted(product_sales["Price"].unique())[:max_sellers]  # limit sellers

    sellers = [
        Seller(
            name=f"{description}_Seller_{i+1}",
            cost=min(unique_prices)*0.4,          # cost as 40% of lowest price
            price=p,
            ad_budget=randint(5, 15)              # random ad budget
        )
        for i, p in enumerate(unique_prices)
    ]
    return sellers



def get_multi_price_products(df:pd.DataFrame,n=5):

    # Step 1: Count unique prices per product
    price_variability = df.groupby("Description")["Price"].nunique().reset_index(name="unique_prices")

    # Filter products with multiple price levels
    multi_price_products = price_variability[price_variability["unique_prices"] > 1]

    # Pick top n products with most sales variability (or you can choose more)
    top_products = multi_price_products.head(n)["Description"].tolist()
    return top_products