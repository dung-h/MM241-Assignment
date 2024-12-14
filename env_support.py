import sys
from PySide6.QtWidgets import QApplication
import gymnasium as gym
import gym_cutting_stock
from core_ga import Stock, Product, ParallelCuttingStockGA
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from gui import ResultsWindow

class EnvironmentOptimizer:
    """Handles optimization of cutting stock problems from gym environment"""
    
    @staticmethod
    def convert_environment_data(observation: Dict) -> Tuple[List[Stock], List[Product]]:
        """Convert gym environment observation to Stock and Product lists"""
        stocks = []
        products = []
        
        # Convert stocks
        for idx, stock_data in enumerate(observation['stocks']):
            # Get actual stock dimensions
            width = np.sum(np.any(stock_data != -2, axis=1))
            height = np.sum(np.any(stock_data != -2, axis=0))
            
            stock = Stock(
                id=idx,
                length=float(width),
                width=float(height)
            )
            stocks.append(stock)
        
        # Convert products
        for idx, product_data in enumerate(observation['products']):
            size = product_data['size']
            product = Product(
                id=idx,
                length=float(size[0]),  # width in environment
                width=float(size[1]),   # height in environment
                quantity=int(product_data['quantity'])
            )
            products.append(product)
        
        return stocks, products

    @staticmethod
    def run_optimization(observation: Dict) -> Tuple[Dict, float, Dict]:
        """Run optimization and return results"""
        # Convert data
        stocks, products = EnvironmentOptimizer.convert_environment_data(observation)
        
        # Initialize and run GA
        ga = ParallelCuttingStockGA(
            num_islands=6,
            island_population=50,
            generations=60,
            crossover_rate=0.85,
            mutation_rate=0.25
        )
        
        # Run optimization
        solution, fitness, stats = ga.optimize(stocks, products)
        return solution, fitness, stats, stocks, products

def main():
    # Initialize Qt Application
    app = QApplication(sys.argv)
    
    # Create and reset environment
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode=None  # Disable gym rendering as we'll use our own
    )
    observation, info = env.reset(seed=42)
    
    print("Starting optimization...")
    print(f"Number of stocks: {len(observation['stocks'])}")
    print(f"Number of product types: {len(observation['products'])}")
    
    # Run optimization
    solution, fitness, stats, stocks, products = EnvironmentOptimizer.run_optimization(observation)
    
    if solution:
        print("\nOptimization completed successfully!")
        print(f"Execution time: {stats['execution_time']:.2f} seconds")
        print(f"Best fitness: {stats['best_fitness']:.2f}")
        print(f"Average utilization: {stats['average_utilization']:.2f}%")
        print(f"Fulfillment ratio: {stats['overall_fulfillment_ratio']:.2f}")
        
        # Create and show results window
        results_window = ResultsWindow()
        results_window.initialize_display(solution, stocks, products)
        results_window.show()
        
        # Start Qt event loop
        sys.exit(app.exec())
    else:
        print("Optimization failed to find a valid solution")
        env.close()
        sys.exit(1)

if __name__ == "__main__":
    main()