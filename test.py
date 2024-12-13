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

# from gui import MainWindow
# from PySide6.QtWidgets import QApplication
# import sys
# from core_ga import Stock, Product, Placement
# import copy

# def test_display():
#     # Initialize app and main window
#     app = QApplication(sys.argv)
#     window = MainWindow()
    
#     # Create test data
#     # 1. Create stocks
#     window.stocks = [
#         Stock(id=0, length=20.0, width=10.0),
#         Stock(id=1, length=30.0, width=20.0)
#     ]
    
#     # 2. Create products
#     window.products = [
#         Product(id=0, length=2.0, width=1.0, quantity=2),
#         Product(id=1, length=3.0, width=2.0, quantity=2)
#     ]
    
#     # 3. Create a mock solution
#     mock_solution = {
#         0: [  # First stock
#             Placement(product_id=0, x=1.0, y=1.0, rotated=False),
#             Placement(product_id=0, x=4.0, y=1.0, rotated=False),
#             Placement(product_id=1, x=7.0, y=1.0, rotated=True)
#         ],
#         1: [  # Second stock
#             Placement(product_id=1, x=2.0, y=2.0, rotated=False)
#         ]
#     }
    
#     # Add product references to placements
#     for stock_id, placements in mock_solution.items():
#         for placement in placements:
#             product = next(p for p in window.products if p.id == placement.product_id)
#             placement.product = copy.deepcopy(product)
    
#     # 4. Create mock statistics
#     mock_stats = {
#         'execution_time': 2.5,
#         'generations': 100,
#         'islands': 4,
#         'best_fitness': 850.5,
#         'total_area_used': 14.0,
#         'average_utilization': 35.0
#     }
    
#     # Store current solution and stats
#     window.current_solution = mock_solution
#     window.current_stats = mock_stats
    
#     # Set up ID mappings (optional, for testing display names)
#     window.current_id_mappings = {
#         'stock': {0: 'A', 1: 'B'},
#         'demand': {0: 'Small', 1: 'Medium'}
#     }
    
#     # Display results
#     window.display_results(mock_solution, mock_stats)
    
#     # Show window
#     window.show()
    
#     # Enable export button
#     window.export_button.setEnabled(True)
    
#     return app, window

# if __name__ == "__main__":
#     app, window = test_display()
#     sys.exit(app.exec())