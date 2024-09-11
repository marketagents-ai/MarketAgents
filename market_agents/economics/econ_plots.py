import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from market_agents.economics.utility import (
    create_utility_function, create_cost_function,
    StepwiseUtility, CobbDouglasUtility, InducedUtility,
    StepwiseCost, QuadraticCost
)
from market_agents.economics.econ_models import Basket, Good

def plot_utility_function(utility_function, cost_function, goods: List[str], ranges: Dict[str, tuple], cash_range: tuple, title: str):
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(title)

    cash = np.linspace(cash_range[0], cash_range[1], 101)  # 101 points to include both endpoints
    cash_mean = np.mean(cash)
    x = np.linspace(0, 100, 101)  # 101 points from 0 to 100

    # 1. Good alone
    y_goods = np.array([utility_function.goods_utility.evaluate(Basket(cash=0, goods=[Good(name=goods[0], quantity=q)])) for q in x])
    axs[0, 0].plot(x, y_goods)
    axs[0, 0].set_title(f"Goods Utility ({goods[0]})")
    axs[0, 0].set_xlabel(goods[0])
    axs[0, 0].set_ylabel("Utility")

    # 2. Cost function (total cost)
    y_cost = np.array([cost_function.evaluate(Basket(cash=0, goods=[Good(name=goods[0], quantity=q)])) for q in x])
    axs[0, 1].plot(x, y_cost)
    axs[0, 1].set_title(f"Total Cost Function ({goods[0]})")
    axs[0, 1].set_xlabel(goods[0])
    axs[0, 1].set_ylabel("Cost")

    # 3. Combined 1 good, cash, and cost
    y_combined = np.array([utility_function.evaluate(Basket(cash=c, goods=[Good(name=goods[0], quantity=q)])) 
                           for q, c in zip(x, cash)])
    axs[1, 0].plot(x, y_combined, label="Combined Utility")
    axs[1, 0].plot(x, y_goods, label="Goods Utility")
    axs[1, 0].plot(x, utility_function.cash_weight * cash, label="Cash Utility")
    axs[1, 0].plot(x, y_cost, label="Total Cost", linestyle='--')
    axs[1, 0].set_title(f"Combined Utility and Cost ({goods[0]} and Cash)")
    axs[1, 0].set_xlabel(f"{goods[0]} / Cash")
    axs[1, 0].set_ylabel("Utility / Cost")
    axs[1, 0].legend()

    # 4. 2 goods vs utility 3D plot
    if len(goods) > 1:
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        X, Y = np.meshgrid(x, x)
        
        Z = np.array([[utility_function.evaluate(Basket(cash=float(cash_mean), goods=[Good(name=goods[0], quantity=float(i)), Good(name=goods[1], quantity=float(j))])) for i in X[0]] for j in Y[:, 0]])
        surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis')
        ax_3d.set_title(f"2D Utility ({goods[0]}, {goods[1]})")
        ax_3d.set_xlabel(goods[0])
        ax_3d.set_ylabel(goods[1])
        ax_3d.set_zlabel("Utility")
        ax_3d.view_init(elev=20, azim=-45)
    else:
        axs[1, 1].text(0.5, 0.5, "2D plot not available for 1 good", ha='center', va='center')

    plt.tight_layout()
    plt.show()

def plot_cost_function(cost_function, goods: List[str], ranges: Dict[str, tuple], title: str):
    if len(goods) == 1:
        plot_1d_cost(cost_function, goods[0], ranges[goods[0]], title)
    elif len(goods) == 2:
        plot_2d_cost(cost_function, goods, ranges, title)
    else:
        raise ValueError("Can only plot 1 or 2 goods")

def plot_1d_cost(cost_function, good: str, good_range: tuple, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title)

    x = np.linspace(good_range[0], good_range[1], 100)
    y = [cost_function.evaluate(Basket(cash=0, goods=[Good(name=good, quantity=q)])) for q in x]

    ax.plot(x, y)
    ax.set_title("Cost Function")
    ax.set_xlabel(good)
    ax.set_ylabel("Cost")

    plt.tight_layout()
    plt.show()

def plot_2d_cost(cost_function, goods: List[str], ranges: Dict[str, tuple], title: str):
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
    fig.suptitle(title)

    x = np.linspace(ranges[goods[0]][0], ranges[goods[0]][1], 20)
    y = np.linspace(ranges[goods[1]][0], ranges[goods[1]][1], 20)
    X, Y = np.meshgrid(x, y)

    Z = np.array([[cost_function.evaluate(Basket(cash=0, goods=[Good(name=goods[0], quantity=float(i)), Good(name=goods[1], quantity=float(j))])) for i in x] for j in y])

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title("Cost Function")
    ax.set_xlabel(goods[0])
    ax.set_ylabel(goods[1])
    ax.set_zlabel("Cost")

    plt.tight_layout()
    plt.show()

def plot_marginal_utility_function(utility_function, cost_function, goods: List[str], ranges: Dict[str, tuple], cash_range: tuple, title: str):
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(title + " (Marginal)")

    cash = np.linspace(cash_range[0], cash_range[1], 101)  # 101 points to include both endpoints
    cash_mean = np.mean(cash)
    x = np.linspace(0, 100, 101)  # 101 points from 0 to 100

    # 1. Marginal Good Utility
    y_marginal_goods = np.array([utility_function.marginal_utility(Basket(cash=0, goods=[Good(name=goods[0], quantity=q)]), goods[0]) for q in x])
    axs[0, 0].plot(x, y_marginal_goods)
    axs[0, 0].set_title(f"Marginal Goods Utility ({goods[0]})")
    axs[0, 0].set_xlabel(goods[0])
    axs[0, 0].set_ylabel("Marginal Utility")

    # 2. Marginal Cost
    y_marginal_cost = np.array([cost_function.marginal_cost(Basket(cash=0, goods=[Good(name=goods[0], quantity=q)]), goods[0]) for q in x])
    axs[0, 1].plot(x, y_marginal_cost)
    axs[0, 1].set_title(f"Marginal Cost Function ({goods[0]})")
    axs[0, 1].set_xlabel(goods[0])
    axs[0, 1].set_ylabel("Marginal Cost")

    # 3. Combined Marginal Utility and Cost
    axs[1, 0].plot(x, y_marginal_goods, label="Marginal Goods Utility")
    axs[1, 0].plot(x, y_marginal_cost, label="Marginal Cost", linestyle='--')
    axs[1, 0].plot([0, 100], [utility_function.cash_weight, utility_function.cash_weight], label="Marginal Cash Utility", linestyle=':')
    axs[1, 0].set_title(f"Combined Marginal Utility and Cost ({goods[0]})")
    axs[1, 0].set_xlabel(goods[0])
    axs[1, 0].set_ylabel("Marginal Utility / Cost")
    axs[1, 0].legend()

    # 4. 2 goods vs marginal utility 3D plot
    if len(goods) > 1:
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        X, Y = np.meshgrid(x, x)
        
        Z = np.array([[utility_function.marginal_utility(Basket(cash=float(cash_mean), goods=[Good(name=goods[0], quantity=float(i)), Good(name=goods[1], quantity=float(j))]), goods[0]) for i in X[0]] for j in Y[:, 0]])
        surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis')
        ax_3d.set_title(f"2D Marginal Utility ({goods[0]})")
        ax_3d.set_xlabel(goods[0])
        ax_3d.set_ylabel(goods[1])
        ax_3d.set_zlabel("Marginal Utility")
        ax_3d.view_init(elev=20, azim=-45)
    else:
        axs[1, 1].text(0.5, 0.5, "2D plot not available for 1 good", ha='center', va='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    goods_1 = ["apple"]
    goods_2 = ["apple", "banana"]
    base_values = {"apple": 10.0, "banana": 8.0}
    ranges = {"apple": (0, 100), "banana": (0, 100)}  # Updated ranges
    cash_range = (0, 100)

    # Utility functions
    utility_types = ["stepwise", "cobb-douglas"]
    for utility_type in utility_types:
        # 2-good case
        utility_function_2 = create_utility_function(utility_type, goods_2, base_values, num_units=100, noise_factor=0.1, cash_weight=1.0, cobb_douglas_scale=10)
        
        # Create corresponding cost function
        cost_type = "stepwise" if utility_type == "stepwise" else "quadratic"
        cost_function_2 = create_cost_function(cost_type, goods_2, base_values, num_units=100, noise_factor=0.1, cost_scale=1)  # Adjust cost_scale as needed
        
        # Plot total utility and cost
        plot_utility_function(utility_function_2, cost_function_2, goods_2, ranges, cash_range, f"{utility_type.capitalize()} Utility and {cost_type.capitalize()} Cost (2 Goods)")
        
        # Plot marginal utility and cost
        plot_marginal_utility_function(utility_function_2, cost_function_2, goods_2, ranges, cash_range, f"{utility_type.capitalize()} Utility and {cost_type.capitalize()} Cost (2 Goods)")