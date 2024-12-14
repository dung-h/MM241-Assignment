# test_pga.py
import logging
from core_ga import ParallelCuttingStockGA, Stock, Product

def test_pga():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create test data
    stocks = [
        Stock(id=0, length=20.0, width=10.0),
        Stock(id=1, length=30.0, width=20.0)
    ]
    
    products = [
        Product(id=0, length=2.0, width=1.0, quantity=20),
        Product(id=1, length=3.0, width=2.0, quantity=2),
        Product(id=2, length=4.0, width=3.0, quantity=2),
    ]

    # Initialize and run PGA
    pga = ParallelCuttingStockGA(
        num_islands=2,
        island_population=50,
        generations=100
    )
    
    solution, fitness, stats = pga.optimize(stocks, products)
    
    if solution:
        print("Optimization succeeded!")
        print(f"Best fitness: {fitness}")
        print(f"Stats: {stats}")
    else:
        print("No solution found!")

if __name__ == '__main__':
    test_pga()
