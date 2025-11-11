## 📘 Final Report: Competitive Market Simulation Project Considering Social Network Influence

**Link**: https://github.com/erf04/ec-hw1.git

### Introduction

In this project, the goal was to design and simulate a **simple competitive market** among several sellers offering similar products. Each seller must set their price and advertising budget in a way that maximizes profit in the presence of competitors and under the influence of social relationships.

**Game Theory** was used to analyze seller behavior, and the market equilibrium point of **Nash Equilibrium** was obtained numerically. Additionally, the effects of social network influence on decision-making and seller profits were examined.

---

## ⚙️ Project Implementation Stages

### 🔹 Task I – Data Preparation

In this stage, initial sales data was extracted from a file containing information about products, prices, sales quantities, countries, and customer IDs.

The data cleaning process included removing duplicate records, missing values, and negative values. Then, for each product, the average price and demand were calculated to serve as the basis for defining the model.

---

### 🔹 Task II – Defining Seller Model, Demand Function, and Profit

To simulate seller behavior, several hypothetical sellers were defined, each with the following characteristics:

| Feature | Description |
|---------|-------------|
| `price` | Selling price of the product |
| `marketing` | Advertising budget |
| `cost` | Fixed production cost |
| `influence_score` | Level of social influence in the network |
| `profit` | Seller's final profit function |

The demand function for each seller (i) was defined as:

\[
D_i = base\_demand + (\alpha \times m_i) + (\beta \times (p_i - p_j)) + (\gamma \times influence\_score)
\]

Where:

* \( base\_demand \): Base demand without advertising or network effects
* \( \alpha \): Advertising impact coefficient
* \( \beta \): Demand sensitivity to price differences with competitors
* \( \gamma \): Social network influence on demand

The profit function was modeled as:

\[
Profit_i = (p_i - cost) \times D_i - m_i
\]

---

### 🔹 Task III – Game Simulation and Finding Nash Equilibrium

To reach **Nash Equilibrium**, an iterative algorithm was used where each seller updates their price and advertising budget based on competitor behavior in each step.

When no seller had an incentive to change their strategy, the stable state of the system was considered as **Nash Equilibrium**.

This simulation was performed using the `simulate_nash` function, which checks system convergence through multiple iterations (up to 10,000 iterations).

---

### 🔹 Task IV – Adding Social Network Effects

In this section, the **NetworkX** library was used to examine the role of social networks.

1. First, a seller network was built using the `build_seller_network` function, connecting sellers of similar products.
2. Then, using the `compute_network_influence` function, the influence level of each node (seller) was calculated using the **PageRank** algorithm.
3. Finally, with the `update_sellers_with_network_influence` function, these values were added to the seller model to be incorporated into the demand function.

The game was then re-run considering social network influence to examine the impact of social influence on price, profit, and demand.

---

### 🔹 Task V – Results Visualization

**Matplotlib** and **Seaborn** libraries were used to analyze the final results. Two main types of charts were created:

1. **Profit change charts relative to price and advertising** at Nash Equilibrium (2D or 3D):
   These charts show how sellers can reach the maximum profit area by changing advertising and pricing strategies.

2. **Comparison charts of social network impact on sales and profit**:
   In this section, differences between values before and after applying social network effects were calculated and plotted:
   \[
   ΔPrice = Price_{After} - Price_{Before}
   \]
   \[
   ΔProfit = Profit_{After} - Profit_{Before}
   \]
   \[
   ΔDemand = Demand_{After} - Demand_{Before}
   \]
   Results showed that increasing the social network influence coefficient (\( \gamma \)) led to a significant increase in demand and average seller profits, especially for sellers positioned at the center of the network (influencers).

---

## 📊 Results Analysis

* In the no-network scenario, sellers tended to compete by lowering prices to increase demand.
* With the addition of social networks, more influential sellers were able to achieve higher sales without reducing prices.
* The overall system's average profit increased in the presence of social network effects.
* Charts showed that social relationships lead to faster market stability and more favorable equilibrium.

---

## 🧩 Summary and Conclusion

In this project, a competitive market simulation was implemented using game theory, including the following elements:

* Dynamic determination of price and advertising budget
* Analysis of Nash Equilibrium effects among sellers
* Social network modeling and its impact on market behavior

Results demonstrate that **social network influence can play a key role in improving profitability and market stability**. Sellers with central positions in the network (influencers) can achieve higher profits with less advertising and higher prices.

---

## 📁 Project Outputs

* Cleaned data files
* Nash simulation script (`simulate_nash.py`)
* Network analysis file (`network.py`)
* Results comparison file (`network_vs_no_network.xlsx`)

---