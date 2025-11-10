from preparation.preparation import *


# part 1 
df = load_data("data.xlsx")
df_clean = clean_data(df)
summary = summarize_data(df_clean)
summary.head()
print(summary)


# part 2 with example 

from modeling.seller import Seller
from config.constants import *

# Create sellers
s1 = Seller("Seller A", price=30, ad_budget=10, cost=COST, alpha=ALPHA, beta=BETA, gamma=GAMMA, base_demand=BASE_DEMAND)
s2 = Seller("Seller B", price=32, ad_budget=12, cost=COST, alpha=ALPHA, beta=BETA, gamma=GAMMA, base_demand=BASE_DEMAND)
s3 = Seller("Seller C", price=29, ad_budget=8, cost=COST, alpha=ALPHA, beta=BETA, gamma=GAMMA, base_demand=BASE_DEMAND)

sellers = [s1, s2, s3]

# Suppose we have influence scores from a social network later
influence_scores = {"Seller A": 5, "Seller B": 3, "Seller C": 2}

# Compute demand & profit for each seller
for s in sellers:
    s.compute_demand(sellers, influence_scores[s.name])
    s.compute_profit()

# Print results
for s in sellers:
    print(s)