import subprocess
import time
import statistics
import os
import random
from typing import List, Dict, Tuple
import logging
import csv
import json
from core_ga import ParallelCuttingStockGA, Stock, Product

def ensure_dataset_folder():
    """Create dataset folder if it doesn't exist"""
    if not os.path.exists('Datasets'):
        os.makedirs('Datasets')

class AutomatedMetric:
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.format_input_path = os.path.join(base_dir, "Format_input.txt")
        self.dataset_path = os.path.join(base_dir, "Dataset.txt")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Ensure base directory exists
        os.makedirs(base_dir, exist_ok=True)

    def modify_input_params(self) -> dict:
        """Randomly modify Format_input.txt while keeping problem type 11"""
        params = {
            'seed': random.randint(1, 2147483646),
            'dimensions': (random.randint(100, 300), random.randint(100, 300)),  # min/max stock size
            'item_dims': (random.randint(10, 50), random.randint(50, 100)),     # min/max item size
            'n_large_obj': (random.randint(5, 10), random.randint(10, 15)),     # min/max number of large objects
            'n_items': (random.randint(5, 10), random.randint(10, 20))          # min/max number of items
        }
        
        # Read the template file
        with open(self.format_input_path, 'r') as f:
            lines = f.readlines()
        
        # Find and update values
        for i, line in enumerate(lines):
            if "Enter an integer seed" in line:
                lines[i + 1] = f"{params['seed']}\n"
            elif "insert integer values for the minimum and maximum size dimension of the large object" in line:
                lines[i + 1] = f"{params['dimensions'][0]} {params['dimensions'][1]}\n"
            elif "insert integer values for the minimum and maximum size dimension of the items" in line:
                lines[i + 1] = f"{params['item_dims'][0]} {params['item_dims'][1]}\n"
            elif "insert the minimum and maximum number of different large objects" in line:
                lines[i + 1] = f"{params['n_large_obj'][0]} {params['n_large_obj'][1]}\n"
            elif "insert the minimum and maximum number of different item types" in line:
                lines[i + 1] = f"{params['n_items'][0]} {params['n_items'][1]}\n"
        
        # Write back to file
        with open(self.format_input_path, 'w') as f:
            f.writelines(lines)
        
        return params

    def run_2dcpackgen(self) -> bool:
        """Run 2DCPackGen with automated input"""
        try:
            # Create command file that provides input automatically
            with open('run_commands.txt', 'w') as f:
                f.write("1\n")  # Select existing input parameter file
                f.write(f"{self.format_input_path}\n")  # Provide format input path
                f.write(f"{self.dataset_path}\n")  # Provide dataset output path
                f.write("0\n")  # Exit command

            # Run 2DCPackGen with input redirection
            with open('run_commands.txt', 'r') as commands:
                process = subprocess.Popen(
                    ['./2DCPackGen'],
                    stdin=commands,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()

            if process.returncode != 0:
                self.logger.error(f"2DCPackGen failed: {stderr}")
                return False

            # Cleanup
            if os.path.exists('run_commands.txt'):
                os.remove('run_commands.txt')

            return True

        except Exception as e:
            self.logger.error(f"Failed to run 2DCPackGen: {str(e)}")
            return False
        
    def save_dataset(self, problem_no: int, params: dict, parsed_data: List[Dict]) -> str:
        """Save dataset to Datasets folder after parsing, in a json format"""
        dataset_path = f'Datasets/problem_{problem_no}.json'
        
        data_to_save = {
            'params': params,
            'instances': parsed_data
        }

        with open(dataset_path, 'w') as dest:
            json.dump(data_to_save, dest, indent=4)
        return dataset_path
      
    

    def prepare_data_for_pga(self, dataset: Dict) -> Tuple[List[Stock], List[Product]]:
        """Convert parsed data into PGA input format"""
        stocks = [Stock(id=i, length=s['length'], width=s['width']) 
                for i, s in enumerate(dataset['stocks'])]
        
        products = [Product(id=i, length=p['length'], width=p['width'], quantity=p['demand'])
                   for i, p in enumerate(dataset['items'])]
        
        return stocks, products

    def calculate_demand_fulfilled(self, solution: Dict, products: List[Product]) -> float:
        """Calculate percentage of product demand fulfilled"""
        if not solution:
            return 0.0
            
        placed_quantities = {p.id: 0 for p in products}
        
        # Count placed items
        for placements in solution.values():
            for placement in placements:
                placed_quantities[placement.product_id] += 1
        
        # Calculate fulfillment ratio
        fulfillment_ratios = []
        for p in products:
            ratio = placed_quantities[p.id] / p.quantity
            fulfillment_ratios.append(ratio)
        
        return sum(fulfillment_ratios) / len(fulfillment_ratios)
    
    def evaluate_instance(self, instance: Dict, runs: int = 10) -> List[Dict]:
        """Evaluate a single instance multiple times and return list of individual results"""
        try:
            stocks, products = self.prepare_data_for_pga(instance)
            ga = ParallelCuttingStockGA()
            
            results = []
            for run in range(runs):
                self.logger.info(f"Running evaluation {run + 1}/{runs}")
                start_time = time.time()
                solution, fitness, stats = ga.optimize(stocks, products)
                end_time = time.time()
                
                if solution:  # Only append valid solutions
                    results.append({
                        'used_stocks': len(solution),
                        'fitness': float(fitness),  # Ensure numeric type
                        'demand_fulfilled': self.calculate_demand_fulfilled(solution, products),
                        'time': end_time - start_time,
                        'stats': stats
                    })
                    
                self.logger.info(f"Run {run + 1} completed: fitness = {fitness}")

            return results  # Return list of individual results

        except Exception as e:
            self.logger.error(f"Failed to evaluate instance: {str(e)}")
            return []

    def run_performance_evaluation(self, n_problems: int = 10) -> List[Dict]:
        """Run complete performance evaluation with enhanced automation"""
        try:
            all_results = []
            ensure_dataset_folder()
            
            for i in range(n_problems):
                self.logger.info(f"\nProblem {i+1}/{n_problems}:")
                
                # Generate new dataset
                self.logger.info(f"  - Modifying input parameters...")
                params = self.modify_input_params()
                self.logger.info(f"    Generated parameters: {params}")

                # Run 2DCPackGen
                if not self.run_2dcpackgen():
                    self.logger.error(f"Failed to generate dataset for problem {i+1}")
                    continue

                # Parse and save the dataset
                self.logger.info("Parsing generated dataset...")
                instances = self.parse_dataset(self.dataset_path)
                if not instances:
                    self.logger.error("No valid instances found in dataset")
                    continue

                dataset_path = self.save_dataset(i+1, params, instances)
                self.logger.info(f"  - Dataset saved to: {dataset_path}")
                
                self.logger.info(f"Running evaluation (10 runs)")
                instance_results = self.evaluate_instance(instances[0])  # Get list of results
                
                if instance_results:  # Only process if we have valid results
                    # Calculate averages and store results
                    avg_results = self.aggregate_results(instance_results)
                    avg_results['problem_no'] = i + 1
                    avg_results['params'] = params
                    all_results.append(avg_results)
                    
                    self.logger.info(f"\nResults for Problem {i+1}:")
                    self.display_problem_results(avg_results)
                
            return all_results

        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {str(e)}")
            return []

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple runs"""
        if not results:
            return {
                'best_result': 0,
                'best_fitness': 0,
                'avg_result': 0,
                'avg_fitness': 0,
                'avg_time': 0,
                'avg_trials': 0,
                'avg_demand_fulfilled': 0
            }
        
        try:
            # Calculate averages safely
            return {
                'best_result': min(r['used_stocks'] for r in results),
                'best_fitness': max(r['fitness'] for r in results),
                'avg_result': statistics.mean(r['used_stocks'] for r in results),
                'avg_fitness': statistics.mean(r['fitness'] for r in results),
                'avg_time': statistics.mean(r['time'] for r in results),
                'avg_trials': statistics.mean(r['stats'].get('generations', 0) for r in results),
                'avg_demand_fulfilled': statistics.mean(r['demand_fulfilled'] for r in results)
            }
        except Exception as e:
            self.logger.error(f"Error in aggregating results: {str(e)}")
            return {
                'best_result': 0,
                'best_fitness': 0,
                'avg_result': 0,
                'avg_fitness': 0,
                'avg_time': 0,
                'avg_trials': 0,
                'avg_demand_fulfilled': 0
            }

    def parse_dataset(self, filename: str) -> List[Dict]:
        """Parse the dataset file into a workable format"""
        instances = []
        with open(filename, 'r') as f:
            # Skip until we find first asterisk line
            while True:
                line = f.readline().strip()
                if line.startswith('*****'):
                    break
            
            # Skip lines until we find second asterisk line
            while True:
                line = f.readline().strip()
                if line.startswith('*****'):
                    # Now the next line will be our number
                    num_instances = int(f.readline().strip())
                    break
            
            # Parse each instance
            for inst in range(num_instances):
                instance = {}
                
                # Get number of large objects
                n_stocks = int(f.readline().strip())
                
                # Parse stock data
                stocks = []
                for _ in range(n_stocks):
                    data = f.readline().strip().split('\t')
                    length, width, available, value = map(int, data)
                    stocks.extend([{'length': length, 'width': width} for _ in range(available)])
                instance['stocks'] = stocks
                
                # Get number of items
                n_items = int(f.readline().strip())
                
                # Parse item data
                items = []
                for _ in range(n_items):
                    data = f.readline().strip().split('\t')
                    length, width, demand = map(int, data)
                    items.append({
                        'length': length,
                        'width': width,
                        'demand': demand
                    })
                instance['items'] = items
                
                instances.append(instance)
        
        return instances

    def display_problem_results(self, results: Dict):
        """Display results for a single problem"""
        print(f"    Best Result: {results.get('best_result', 'N/A')}")
        print(f"    Best Fitness: {results.get('best_fitness', 'N/A'):.4f}")
        print(f"    Average Time: {results.get('avg_time', 'N/A'):.2f}s")
        print(f"    Demand Fulfilled: {results.get('avg_demand_fulfilled', 'N/A'):.2%}")

    def display_results(self, results: List[Dict]):
      """Display results in a formatted table"""
      print("\nPerformance Evaluation Results:")
      print("-" * 100)
      print(f"{'Problem No.':<12} {'Best':<8} {'Best':<8} {'Average':<8} {'Average':<8} {'Average':<8} {'Average':<8} {'Demand':<8}")
      print(f"{'':12} {'result':<8} {'fitness':<8} {'result':<8} {'fitness':<8} {'time(s)':<8} {'trials':<8} {'fulfilled':<8}")
      print("-" * 100)
      
      for r in results:
          print(f"{r['problem_no']:<12} "
                f"{r['best_result']:<8.0f} "
                f"{r['best_fitness']:<8.4f} "
                f"{r['avg_result']:<8.2f} "
                f"{r['avg_fitness']:<8.4f} "
                f"{r['avg_time']:<8.4f} "
                f"{r['avg_trials']:<8.1f} "
                f"{r['avg_demand_fulfilled']:<8.2%}")
                
    def export_results_to_csv(self, results: List[Dict], filename: str = "evaluation_results.csv"):
      """Exports the evaluation results to a CSV file."""
      if not results:
          self.logger.warning("No results to export.")
          return

      with open(filename, 'w', newline='') as csvfile:
          fieldnames = ['problem_no', 'seed', 'dimensions', 'item_dims', 'n_large_obj', 'n_items',
                        'best_result', 'best_fitness', 'avg_result', 'avg_fitness',
                        'avg_time', 'avg_trials', 'avg_demand_fulfilled']
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

          writer.writeheader()
          for r in results:
              row_data = {
                  'problem_no': r['problem_no'],
                  'seed': r['params']['seed'],
                  'dimensions': str(r['params']['dimensions']),
                  'item_dims': str(r['params']['item_dims']),
                  'n_large_obj': str(r['params']['n_large_obj']),
                  'n_items': str(r['params']['n_items']),
                  'best_result': r['best_result'],
                  'best_fitness': r['best_fitness'],
                  'avg_result': r['avg_result'],
                  'avg_fitness': r['avg_fitness'],
                  'avg_time': r['avg_time'],
                  'avg_trials': r['avg_trials'],
                  'avg_demand_fulfilled': r['avg_demand_fulfilled']
              }
              writer.writerow(row_data)
      self.logger.info(f"Results exported to '{filename}'")


def main():
    metric = AutomatedMetric()
    results = metric.run_performance_evaluation(n_problems=10)
    
    if results:
      metric.display_results(results)
      metric.export_results_to_csv(results)

if __name__ == "__main__":
    main()