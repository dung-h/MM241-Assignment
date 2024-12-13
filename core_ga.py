from multiprocessing import Queue
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
import time
import logging
from queue import Empty
from multiprocessing import Process, Manager, cpu_count
import copy

@dataclass
class Product:
    id: int
    length: float
    width: float
    quantity: int
    
@dataclass
class Stock:
    id: int
    length: float
    width: float

@dataclass
class Placement:
    product_id: int
    x: float
    y: float
    rotated: bool  # True if 90-degree rotation is applied
    # We'll store the product reference separately to avoid serialization issues
    _product: Product = None
    
    @property
    def product(self):
        return self._product
    
    @product.setter
    def product(self, value):
        self._product = value

class CuttingStockGA:  
    def __init__(
        self,
        population_size: int = 40,      # Reduced from 50 - smaller but more focused population
        generations: int = 75,          # Increased from 50 - more time to optimize
        crossover_rate: float = 0.85,   # Increased from 0.8 - more combination of good solutions
        mutation_rate: float = 0.25,    # Increased from 0.2 - more exploration
        elite_size: int = 3,            # Increased from 2 - preserve more good solutions
        tournament_size: int = 4        # Adjusted for population size
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # Tuned placement parameters
        self.min_gap = 0            # Smaller gap between pieces
        self.position_step = 0.00001       # Finer position adjustments
        self.strip_height_factor = 0.5  # Allow slight overlap in strip heights
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def is_valid_placement(self, placement: Placement, stock: Stock, 
                      existing_placements: List[Placement], products: List[Product]) -> bool:
        """
        Enhanced placement validation with strict boundary checking and comprehensive validation.
        
        Args:
            placement: The placement to validate
            stock: The stock piece
            existing_placements: List of existing placements
            products: List of available products
            
        Returns:
            bool: True if placement is valid, False otherwise
        """
        try:
            # 1. Basic null checks
            if not placement or not stock:
                self.logger.error("Null placement or stock")
                return False
                
            # 2. Ensure we have a valid product reference
            if not placement.product:
                product = next((p for p in products if p.id == placement.product_id), None)
                if not product:
                    self.logger.error(f"No product found for id {placement.product_id}")
                    return False
                placement.product = product
                
            # 3. Calculate dimensions based on rotation
            length = placement.product.width if placement.rotated else placement.product.length
            width = placement.product.length if placement.rotated else placement.product.width
            
            # 4. Add safety margin for floating point calculations
            margin = 0.0001  # Small safety margin
            
            # 5. Strict boundary validation with detailed logging
            if placement.x < 0 or placement.y < 0:
                self.logger.debug(f"Negative coordinates: ({placement.x}, {placement.y})")
                return False
                
            if placement.x + length > stock.length + margin:
                self.logger.debug(
                    f"X-axis overflow: pos={placement.x}, length={length}, "
                    f"stock_length={stock.length}, overflow={placement.x + length - stock.length}"
                )
                return False
                
            if placement.y + width > stock.width + margin:
                self.logger.debug(
                    f"Y-axis overflow: pos={placement.y}, width={width}, "
                    f"stock_width={stock.width}, overflow={placement.y + width - stock.width}"
                )
                return False
                
            # 6. Enhanced overlap detection with safety buffer
            buffer = self.min_gap  # Minimum spacing between pieces
            for existing in existing_placements:
                if not existing.product:
                    continue
                    
                exist_length = existing.product.width if existing.rotated else existing.product.length
                exist_width = existing.product.length if existing.rotated else existing.product.width
                
                # Check for overlap with buffer zone
                if not (
                    placement.x + length + buffer <= existing.x or
                    existing.x + exist_length + buffer <= placement.x or
                    placement.y + width + buffer <= existing.y or
                    existing.y + exist_width + buffer <= placement.y
                ):
                    self.logger.debug(
                        f"Overlap detected between placement at ({placement.x}, {placement.y}) and "
                        f"existing placement at ({existing.x}, {existing.y})"
                    )
                    return False
            
            # 7. Final dimension sanity check
            if length <= 0 or width <= 0:
                self.logger.error(f"Invalid dimensions: length={length}, width={width}")
                return False
                
            if length > stock.length or width > stock.width:
                self.logger.debug(
                    f"Piece too large for stock: piece=({length}, {width}), "
                    f"stock=({stock.length}, {stock.width})"
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
            
   
    def load_data(self, stock_file: str, demand_file: str) -> Tuple[List[Stock], List[Product]]:
        """Load and validate input data"""
        try:
            # Read CSV files
            stocks_df = pd.read_csv(stock_file)
            demands_df = pd.read_csv(demand_file)
            
            # Validate columns
            required_stock_cols = {'id', 'length', 'width'}
            required_demand_cols = {'id', 'length', 'width', 'quantity'}
            
            if not all(col in stocks_df.columns for col in required_stock_cols):
                raise ValueError(f"Stock file missing columns. Required: {required_stock_cols}")
            if not all(col in demands_df.columns for col in required_demand_cols):
                raise ValueError(f"Demand file missing columns. Required: {required_demand_cols}")
            
            # Convert to objects
            stocks = []
            for _, row in stocks_df.iterrows():
                if row['length'] <= 0 or row['width'] <= 0:
                    self.logger.warning(f"Skipping invalid stock: {row}")
                    continue
                stocks.append(Stock(int(row['id']), float(row['length']), float(row['width'])))
            
            products = []
            for _, row in demands_df.iterrows():
                if row['length'] <= 0 or row['width'] <= 0 or row['quantity'] <= 0:
                    self.logger.warning(f"Skipping invalid product: {row}")
                    continue
                products.append(Product(
                    int(row['id']), float(row['length']), float(row['width']), int(row['quantity'])
                ))
            
            return stocks, products
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def check_overlap(self, placement: Placement, existing_placements: List[Placement], 
                     products: List[Product]) -> bool:
        """Check if placement overlaps with existing placements"""
        new_prod = products[placement.product_id]
        new_l = new_prod.width if placement.rotated else new_prod.length
        new_w = new_prod.length if placement.rotated else new_prod.width
        
        new_rect = (placement.x, placement.y, new_l, new_w)
        
        for exist in existing_placements:
            exist_prod = products[exist.product_id]
            exist_l = exist_prod.width if exist.rotated else exist_prod.length
            exist_w = exist_prod.length if exist.rotated else exist_prod.width
            
            exist_rect = (exist.x, exist.y, exist_l, exist_w)
            
            # Check overlap
            if not (new_rect[0] >= exist_rect[0] + exist_rect[2] or
                   exist_rect[0] >= new_rect[0] + new_rect[2] or
                   new_rect[1] >= exist_rect[1] + exist_rect[3] or
                   exist_rect[1] >= new_rect[1] + new_rect[3]):
                return True
                
        return False
    
    def crossover(self, parent1: Dict, parent2: Dict, stocks: List[Stock], products: List[Product]) -> Dict:
        """Modified crossover operator with area-based merging"""
        if random.random() > self.crossover_rate:
            return dict(parent1)
        
        child = {}
        all_stock_ids = set(parent1.keys()) | set(parent2.keys())
        
        for stock_id in all_stock_ids:
            if stock_id in parent1 and stock_id in parent2:
                placements1 = parent1[stock_id]
                placements2 = parent2[stock_id]
                
                # Use a list to track all occupied areas
                occupied_areas = []

                # Copy pieces from the first parent to child
                child_placements = copy.deepcopy(placements1)
                for p in child_placements:
                    width = p.product.width if p.rotated else p.product.length
                    length = p.product.length if p.rotated else p.product.width
                    occupied_areas.append((p.x, p.y, length, width))
                
                # Try to fit pieces from the second parent
                for p in placements2:
                    new_piece = copy.deepcopy(p)
                    best_x = -1
                    best_y = -1
                    min_y = float('inf')
                    
                    for y_type in [0,1]:
                        stock = next(s for s in stocks if s.id == stock_id)
                        width = new_piece.product.width if new_piece.rotated else new_piece.product.length
                        length = new_piece.product.length if new_piece.rotated else new_piece.product.width
                        for y in range(0 if y_type == 0 else int(stock.width - width)):
                            for x in range(0, int(stock.length - length), 1):
                                valid = True
                                # Check overlap against placed pieces
                                for exist_x, exist_y, exist_width, exist_length in occupied_areas:
                                    if not (x + length + self.min_gap <= exist_x or
                                            exist_x + exist_length + self.min_gap <= x or
                                            y + width + self.min_gap <= exist_y or
                                            exist_y + exist_width + self.min_gap <= y):
                                        valid = False
                                        break
                                if valid:
                                    if y < min_y:
                                        min_y = y
                                        best_x = x
                                        best_y = y
                    if best_x >= 0 and best_y >= 0:
                         new_piece.x = best_x
                         new_piece.y = best_y
                         child_placements.append(new_piece)
                         width = new_piece.product.width if new_piece.rotated else new_piece.product.length
                         length = new_piece.product.length if new_piece.rotated else new_piece.product.width
                         occupied_areas.append((best_x, best_y, length, width))
                if child_placements:
                   child[stock_id] = child_placements
        return child

    def _create_basic_chromosome(self, stocks: List[Stock], products: List[Product]) -> Dict:
        """Create initial solution with corner-first strategy"""
        largest_stock = max(stocks, key=lambda s: s.length * s.width)
        chromosome = {largest_stock.id: []}
        
        # Sort products by area and quantity
        sorted_products = sorted(
            [(p, p.length * p.width * p.quantity) for p in products],
            key=lambda x: x[1],
            reverse=True
        )
        
        remaining = {p.id: p.quantity for p in products}
        occupied_space = []

        for product, _ in sorted_products:
            while remaining[product.id] > 0:
                best_x = -1
                best_y = -1
                best_rotation = False
                min_waste = float('inf')
                
                # Try both orientations
                for rotated in [False, True]:
                    length = product.width if rotated else product.length
                    width = product.length if rotated else product.width
                    
                    # Try corners first
                    corners = [(0, 0), (0, largest_stock.width - width),
                            (largest_stock.length - length, 0),
                            (largest_stock.length - length, largest_stock.width - width)]
                    
                    for x, y in corners:
                        valid = True
                        for ex_x, ex_y, ex_w, ex_h in occupied_space:
                            if not (x + length + self.min_gap <= ex_x or
                                    ex_x + ex_w + self.min_gap <= x or
                                    y + width + self.min_gap <= ex_y or
                                    ex_y + ex_h + self.min_gap <= y):
                                valid = False
                                break
                        
                        if valid and x >= 0 and y >= 0 and x + length <= largest_stock.length and y + width <= largest_stock.width:
                            # Calculate waste area (gaps created)
                            waste = self._calculate_waste(x, y, length, width, occupied_space, largest_stock)
                            if waste < min_waste:
                                min_waste = waste
                                best_x = x
                                best_y = y
                                best_rotation = rotated
                
                if best_x >= 0:
                    placement = Placement(
                        product_id=product.id,
                        x=best_x,
                        y=best_y,
                        rotated=best_rotation
                    )
                    placement.product = product
                    chromosome[largest_stock.id].append(placement)
                    
                    # Update occupied space
                    length = product.width if best_rotation else product.length
                    width = product.length if best_rotation else product.width
                    occupied_space.append((best_x, best_y, length, width))
                    remaining[product.id] -= 1
                else:
                    break
        
        return chromosome if any(placements for placements in chromosome.values()) else None

    def _calculate_waste(self, x: float, y: float, length: float, width: float, 
                        occupied_space: List[Tuple], stock: Stock) -> float:
        """Calculate waste area created by placement"""
        total_waste = 0
        gaps = []
        
        # Check gaps with stock edges
        gaps.extend([
            (0, y, x, width),  # Left gap
            (x + length, y, stock.length - (x + length), width),  # Right gap
            (x, 0, length, y),  # Bottom gap
            (x, y + width, length, stock.width - (y + width))  # Top gap
        ])
        
        # Calculate unusable areas
        for gap in gaps:
            if gap[2] > 0 and gap[3] > 0:  # If gap has positive dimensions
                if gap[2] * gap[3] < self.min_gap * 10:  # If gap is too small to be useful
                    total_waste += gap[2] * gap[3]
        
        return total_waste

    def mutate(self, chromosome: Dict, stocks: List[Stock], products: List[Product]) -> Dict:
        """Modified mutation operator with improved relocation"""
        if random.random() > self.mutation_rate:
            return chromosome
            
        mutated = copy.deepcopy(chromosome)
        if not mutated:
            return mutated
            
        stock_id = random.choice(list(mutated.keys()))
        placements = mutated[stock_id]
        
        if not placements or len(placements) < 2:
            return mutated
            
        # Choose mutation type
        mutation_type = random.choice(['shift', 'rotate', 'swap', 'relocate'])
        
        if mutation_type == 'shift':
            # Shift a piece horizontally and vertically
            idx = random.randrange(len(placements))
            placement = placements[idx]
            original_x = placement.x
            original_y = placement.y
            
            # Try shifting left or right and up or down
            shift_x = random.uniform(-1.0, 1.0)
            shift_y = random.uniform(-1.0, 1.0)
            
            placement.x += shift_x
            placement.y += shift_y
            
            if not self.is_valid_placement(
                placement,
                next(s for s in stocks if s.id == stock_id),
                [p for i, p in enumerate(placements) if i != idx],
                products
            ):
                placement.x = original_x
                placement.y = original_y
            
        elif mutation_type == 'rotate':
            # Try rotating a piece
            idx = random.randrange(len(placements))
            placement = placements[idx]
            placement.rotated = not placement.rotated
            
            if not self.is_valid_placement(
                placement,
                next(s for s in stocks if s.id == stock_id),
                [p for i, p in enumerate(placements) if i != idx],
                products
            ):
                placement.rotated = not placement.rotated
                
        elif mutation_type == 'swap':
            # Swap two pieces
            if len(placements) >= 2:
                idx1, idx2 = random.sample(range(len(placements)), 2)
                p1, p2 = placements[idx1], placements[idx2]
                p1.x, p2.x = p2.x, p1.x
                p1.y, p2.y = p2.y, p1.y
                
                # Validate both pieces
                others = [p for i, p in enumerate(placements) if i not in (idx1, idx2)]
                stock = next(s for s in stocks if s.id == stock_id)
                
                if not (self.is_valid_placement(p1, stock, others + [p2], products) and
                    self.is_valid_placement(p2, stock, others + [p1], products)):
                    p1.x, p2.x = p2.x, p1.x
                    p1.y, p2.y = p2.y, p1.y
            
        elif mutation_type == 'relocate':
            # Relocate a piece to a new position based on surrounding free spaces
            if len(placements) >= 2:
                idx = random.randrange(len(placements))
                placement = placements[idx]
                original_x = placement.x
                original_y = placement.y
                
                stock = next(s for s in stocks if s.id == stock_id)
                
                best_x = -1
                best_y = -1
                
                for y_offset in [-1, 0, 1]:  # Check locations around each existing placement
                    for x_offset in [-1,0,1]:
                        
                        new_x = original_x + x_offset * (placement.product.length if not placement.rotated else placement.product.width )
                        new_y = original_y + y_offset * (placement.product.width if not placement.rotated else placement.product.length)

                        if new_x < 0 or new_y < 0 or new_x + (placement.product.length if not placement.rotated else placement.product.width) > stock.length or new_y + (placement.product.width if not placement.rotated else placement.product.length) > stock.width:
                            continue
                            
                        placement.x = new_x
                        placement.y = new_y

                        if self.is_valid_placement(
                                placement,
                                stock,
                                [p for i, p in enumerate(placements) if i != idx],
                                products
                            ):
                            best_x = new_x
                            best_y = new_y
                            break
                    if best_x != -1:
                        break
                        
                if best_x != -1 and best_y != -1:
                        placement.x = best_x
                        placement.y = best_y
                else:
                        placement.x = original_x
                        placement.y = original_y
    
        return mutated

    def initialize_population(self, stocks: List[Stock], products: List[Product], strategy=None) -> List[Dict]:
        """Initialize population with improved placement strategies"""
        population = []
        
        # Sort stocks by area
        sorted_stocks = sorted(stocks, key=lambda s: s.length * s.width, reverse=True)
        
        # Try different product sorting strategies for diversity
        sorting_strategies = [
            # Strategy 1: Sort by area * quantity (default)
            lambda p: p.length * p.width * p.quantity,
            # Strategy 2: Sort by area only
            lambda p: p.length * p.width,
            # Strategy 3: Sort by perimeter
            lambda p: 2 * (p.length + p.width),
            # Strategy 4: Sort by quantity
            lambda p: p.quantity,
            # Strategy 5: Sort by aspect ratio
            lambda p: max(p.length/p.width, p.width/p.length)
        ]
        
        # Try to create solutions with each strategy
        for sort_key in sorting_strategies:
            sorted_products = sorted(
                [(p, sort_key(p)) for p in products],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create multiple solutions per strategy with different starting positions
            for starting_corner in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                chromosome = {}
                remaining = {p.id: p.quantity for p in products}
                
                for stock in sorted_stocks:
                    if all(q == 0 for q in remaining.values()):
                        break
                        
                    placements = []
                    occupied_spaces = []
                    
                    # Try to place each product
                    for product, _ in sorted_products:
                        while remaining[product.id] > 0:
                            best_x = -1
                            best_y = -1
                            best_rotated = False
                            min_waste = float('inf')
                            
                            # Try both orientations
                            for rotated in [False, True]:
                                length = product.width if rotated else product.length
                                width = product.length if rotated else product.width
                                
                                # Start from specified corner
                                start_x = stock.length - length if starting_corner[0] else 0
                                start_y = stock.width - width if starting_corner[1] else 0
                                
                                # Try positions with small steps
                                step = min(1.0, min(length, width) / 2)
                                for x in np.arange(start_x, stock.length - length + 0.1, step):
                                    for y in np.arange(start_y, stock.width - width + 0.1, step):
                                        valid = True
                                        
                                        # Check overlap with existing placements
                                        for ex_x, ex_y, ex_l, ex_w in occupied_spaces:
                                            if not (x + length + self.min_gap <= ex_x or
                                                    ex_x + ex_l + self.min_gap <= x or
                                                    y + width + self.min_gap <= ex_y or
                                                    ex_y + ex_w + self.min_gap <= y):
                                                valid = False
                                                break
                                        
                                        if valid:
                                            # Calculate waste space
                                            waste = 0
                                            # Check gaps with edges and other pieces
                                            for ex_x, ex_y, ex_l, ex_w in occupied_spaces:
                                                gap_x = abs(x - (ex_x + ex_l))
                                                gap_y = abs(y - (ex_y + ex_w))
                                                if gap_x < length and gap_y < width:
                                                    waste += gap_x * gap_y
                                            
                                            if waste < min_waste:
                                                min_waste = waste
                                                best_x = x
                                                best_y = y
                                                best_rotated = rotated
                            
                            if best_x >= 0:
                                # Create placement
                                placement = Placement(
                                    product_id=product.id,
                                    x=best_x,
                                    y=best_y,
                                    rotated=best_rotated
                                )
                                placement.product = product
                                placements.append(placement)
                                
                                # Update occupied spaces
                                length = product.width if best_rotated else product.length
                                width = product.length if best_rotated else product.width
                                occupied_spaces.append((best_x, best_y, length, width))
                                
                                remaining[product.id] -= 1
                            else:
                                break  # If no valid position found, try next product
                    
                    if placements:
                        chromosome[stock.id] = placements
                
                # Add chromosome if it has valid placements
                if chromosome and any(placements for placements in chromosome.values()):
                    population.append(chromosome)
        
        # If we don't have enough solutions, create variations of existing ones
        while len(population) < self.population_size:
            if not population:
                # If no valid solutions found, create a basic one
                basic = self._create_basic_chromosome(stocks, products)
                if basic:
                    population.append(basic)
                else:
                    break
            else:
                # Create variation of existing solution
                parent = random.choice(population)
                variant = copy.deepcopy(parent)
                
                # Modify placements slightly
                for stock_placements in variant.values():
                    for placement in stock_placements:
                        # Small random adjustments to position
                        placement.x += random.uniform(-0.5, 0.5)
                        placement.y += random.uniform(-0.5, 0.5)
                        # Occasionally flip rotation
                        if random.random() < 0.1:
                            placement.rotated = not placement.rotated
                        
                        # Ensure placement remains valid
                        placement.x = max(0, min(placement.x, stocks[0].length - 
                            (placement.product.width if placement.rotated else placement.product.length)))
                        placement.y = max(0, min(placement.y, stocks[0].width - 
                            (placement.product.length if placement.rotated else placement.product.width)))
                
                population.append(variant)
        
        return population[:self.population_size]

    def _get_max_strip_height(self, placements: List[Placement], x: float, products: List[Product]) -> float:
        """Helper method to identify horizontal strips in placements"""
        max_height = 0
        for p in placements:
            prod = products[p.product_id]
            
            length = prod.width if p.rotated else prod.length
            if p.x <= x < p.x + length:
                height = p.y + (prod.length if p.rotated else prod.width)
                max_height = max(max_height, height)
        
        return max_height
    
    def tournament_select(self, population: List[Dict], fitness_values: List[float]) -> Dict:
        """Tournament selection with added validation"""
        if not population or not fitness_values:
            self.logger.error("Empty population or fitness_values in tournament_select")
            return None  # Or handle appropriately
        
        if len(population) < self.tournament_size:
            self.logger.warning(f"Population size is less than tournament size, using smaller size: {len(population)}")
            tournament_size = len(population)
        else:
            tournament_size = self.tournament_size
            
        if tournament_size == 0:
            return None
        
        self.logger.debug(f"tournament_select population size: {len(population)}, fitness values size: {len(fitness_values)}")
        self.logger.debug(f"tournament_select fitness values : {fitness_values}")
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        if max(tournament_indices) >= len(fitness_values):
            self.logger.error(f"tournament_select error: tournament_indices are out of bounds")
            return None
            
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        if winner_idx >= len(population):
             self.logger.error(f"winner_idx {winner_idx} is out of bounds for len(population) {len(population)}, population size: {len(population)}")
             return None
        self.logger.debug(f"Tournament indices: {tournament_indices}, max index: {len(population)}, winner_idx: {winner_idx}")
        
        return copy.deepcopy(population[winner_idx])
        
    def calculate_fitness(self, chromosome: Dict, stocks: List[Stock], products: List[Product]) -> Tuple[float, float, float]:
        """Tuned fitness calculation with stronger penalties"""
        try:
            placed_quantities = {p.id: 0 for p in products}
            required_quantities = {p.id: p.quantity for p in products}
            
            total_area_used = 0
            total_stock_area = 0
            
            # Calculate metrics with weighted importance
            for stock_id, placements in chromosome.items():
                stock = next(s for s in stocks if s.id == stock_id)
                stock_area = stock.length * stock.width
                total_stock_area += stock_area
                
                for placement in placements:
                    product = next(p for p in products if p.id == placement.product_id)
                    placed_quantities[product.id] += 1
                    total_area_used += product.length * product.width
            
            # Calculate weighted fulfillment ratio with higher penalties
            fulfillment_scores = []
            total_pieces = sum(p.quantity for p in products)
            
            for p_id in placed_quantities:
                if required_quantities[p_id] > 0:
                    # Increased weight for unfulfilled items
                    product = next(p for p in products if p.id == p_id)
                    piece_area = product.length * product.width
                    piece_weight = (piece_area * required_quantities[p_id]) / total_pieces
                    
                    ratio = placed_quantities[p_id] / required_quantities[p_id]
                    # Exponential penalty for unfulfilled items
                    penalty = (1 - ratio) ** 2  
                    fulfillment_scores.append((ratio - penalty) * piece_weight)
            
            overall_fulfillment = sum(fulfillment_scores) / len(fulfillment_scores)
            
            # Calculate utilization with increased penalties
            utilization = total_area_used / total_stock_area if total_stock_area > 0 else 0
            
            # Heavier stock penalty
            stock_penalty = (len(chromosome) - 1) * 0.2  # Increased from 0.1
            
            # Combined fitness score with adjusted weights
            fitness = (
                3000 * overall_fulfillment +    # Increased from 2000
                600 * utilization -             # Reduced from 800
                500 * stock_penalty +           # Increased penalty
                100 * (1 - len(chromosome)/len(stocks))
            )
            
            # Calculate overall fulfillment ratio
            total_demands = sum(req for req in required_quantities.values())
            total_placed = sum(placed for placed in placed_quantities.values())
            fulfillment_ratio = total_placed / total_demands if total_demands > 0 else 0

            return max(0, fitness), overall_fulfillment, fulfillment_ratio
            
        except Exception as e:
            self.logger.error(f"Fitness calculation error: {str(e)}")
            return 0, 0, 0

    def optimize(self, stocks: List[Stock], products: List[Product]) -> Tuple[Dict, float, Dict]:
            """Optimized main method with early stopping"""
            start_time = time.time()
            
            try:
                population = self.initialize_population(stocks, products)
                if not population:
                    raise ValueError("Failed to create initial population")
                
                best_solution = None
                best_fitness = float('-inf')
                best_fulfillment = 0
                best_fulfillment_ratio = 0
                generations_without_improvement = 0
                
                for generation in range(self.generations):
                    # Calculate fitness
                    fitness_values = []
                    fulfillment_ratios = []
                    fulfillment_values = []
                    
                    for chrom in population:
                        fitness, fulfillment, ratio = self.calculate_fitness(chrom, stocks, products)
                        fitness_values.append(fitness)
                        fulfillment_ratios.append(fulfillment)
                        fulfillment_values.append(ratio)
                    
                    # Update best solution
                    max_fitness_idx = np.argmax(fitness_values)
                    if fitness_values[max_fitness_idx] > best_fitness:
                        best_fitness = fitness_values[max_fitness_idx]
                        best_fulfillment = fulfillment_ratios[max_fitness_idx]
                        best_fulfillment_ratio = fulfillment_values[max_fitness_idx]
                        best_solution = copy.deepcopy(population[max_fitness_idx])
                        generations_without_improvement = 0
                        
                        self.logger.info(f"Generation {generation}: "
                                       f"Fitness = {best_fitness:.2f}, "
                                       f"Fulfillment = {best_fulfillment:.2%}, "
                                       f"Ratio = {best_fulfillment_ratio:.2%}")
                    else:
                        generations_without_improvement += 1
                    
                    # Early stopping if solution is perfect or no improvement
                    if best_fulfillment_ratio > 0.999 or generations_without_improvement >= 15:
                        break
                    
                    # Create next generation
                    population = self.create_next_generation(population, fitness_values, stocks, products)
                
                stats = {
                    'execution_time': time.time() - start_time,
                    'generations_completed': generation + 1,
                    'best_fitness': best_fitness,
                    'fulfillment_ratio': best_fulfillment,
                    'overall_fulfillment_ratio': best_fulfillment_ratio,
                    'stocks_used': len(best_solution) if best_solution else 0
                }
                
                return best_solution, best_fitness, stats
                
            except Exception as e:
                self.logger.error(f"Optimization failed: {str(e)}")
                return None, float('-inf'), {}
        
    def create_next_generation(self, population: List[Dict], fitness_values: List[float],
                             stocks: List[Stock], products: List[Product]) -> List[Dict]:
        """Tuned next generation creation"""
        new_population = []
        
        # Enhanced elitism - keep best solutions
        sorted_indices = np.argsort(fitness_values)[::-1]
        for i in range(self.elite_size):
            if i < len(sorted_indices):
                elite = copy.deepcopy(population[sorted_indices[i]])
                new_population.append(elite)
                
                # Add slight variation of elite solutions
                if i < 2:  # Only for top 2 solutions
                    variant = copy.deepcopy(elite)
                    for placements in variant.values():
                        for p in placements:
                            p.x += random.uniform(-0.1, 0.1)
                            p.y += random.uniform(-0.1, 0.1)
                    new_population.append(variant)
        
        # Fill rest of population
        while len(new_population) < self.population_size:
            # Tournament selection
            tournament_size = min(self.tournament_size, len(population))
            idx1 = max(random.sample(range(len(population)), tournament_size),
                      key=lambda i: fitness_values[i])
            idx2 = max(random.sample(range(len(population)), tournament_size),
                      key=lambda i: fitness_values[i])
            
            parent1 = population[idx1]
            parent2 = population[idx2]
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2, stocks, products)
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
            
            # Mutation with dynamic rate
            mutation_rate = self.mutation_rate * (1 + 0.5 * (1 - fitness_values[idx1]/max(fitness_values)))
            if random.random() < mutation_rate:
                child = self.mutate(child, stocks, products)
            
            if child:
                new_population.append(child)
        
        return new_population
    
    def _identify_strips(self, placements: List[Placement]) -> List[List[Placement]]:
        """Helper method to identify horizontal strips in placements"""
        if not placements:
            return []
            
        # Group placements by y coordinate
        strips = {}
        for p in placements:
            y_key = round(p.y, 1)  # Round to handle floating point
            if y_key not in strips:
                strips[y_key] = []
            strips[y_key].append(p)
        
        # Sort strips by y coordinate and sort pieces within strips by x coordinate
        sorted_strips = []
        for y in sorted(strips.keys()):
            strip = sorted(strips[y], key=lambda p: p.x)
            sorted_strips.append(strip)
        
        return sorted_strips

class ParallelCuttingStockGA(CuttingStockGA):
    def __init__(
        self,
        num_islands: int = 8,           # Increased from 4 - more parallel exploration
        island_population: int = 50,     # Reduced per island but more total with more islands
        migration_interval: int = 8,     # More frequent migration
        migration_size: int = 4,         # Increased migration size
        generations: int = 60,           # Adjusted for faster convergence
        crossover_rate: float = 0.85,    # Higher crossover for more combination
        mutation_rate: float = 0.25,     # Higher mutation for better exploration
        elite_size: int = 2,
        tournament_size: int = 3
    ):
        super().__init__(
            population_size=island_population,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_size=tournament_size
        )
        
        # PGA specific parameters
        self.num_islands = num_islands if num_islands else min(cpu_count(), 6)
        self.island_population = island_population
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        
        # Migration strategy parameters
        self.migration_selection_pressure = 0.7  # Top 70% solutions considered for migration
        self.migration_acceptance_rate = 0.8    # 80% chance to accept migrant
        self.diversity_threshold = 0.15        # Minimum difference for diversity

    def manage_migration(self, num_islands: int, migration_queues: List[Queue],
                        migration_in_queues: List[Queue]):
        """Enhanced migration management"""
        try:
            generation_counter = 0
            islands_ready = [False] * num_islands
            active_queues = set(range(num_islands))
            
            while True:
                try:
                    # Collect available migrations
                    migrations = []
                    for i in list(active_queues):
                        if migration_queues[i] is None:
                            active_queues.remove(i)
                            continue
                            
                        try:
                            while True:
                                source_id, migrants, gen = migration_queues[i].get_nowait()
                                migrations.append((source_id, migrants, gen))
                        except Empty:
                            continue
                    
                    # Process migrations with diversity check
                    if migrations:
                        current_gen = []
                        for source_id, migrants, gen in migrations:
                            if gen == generation_counter:
                                islands_ready[source_id] = True
                                current_gen.append((source_id, migrants))
                        
                        # Distribute migrations considering diversity
                        if current_gen:
                            for target_id in active_queues:
                                for source_id, migrants in current_gen:
                                    if target_id != source_id:
                                        try:
                                            if random.random() < self.migration_acceptance_rate:
                                                migration_in_queues[target_id].put_nowait(migrants)
                                        except Exception:
                                            continue
                    
                    # Check generation completion
                    if all(not (i in active_queues) or ready 
                          for i, ready in enumerate(islands_ready)):
                        generation_counter += 1
                        islands_ready = [False] * num_islands
                    
                    if not active_queues:
                        break
                        
                    time.sleep(0.05)  # Reduced sleep time
                    
                except Exception as e:
                    self.logger.warning(f"Migration manager warning: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Migration manager error: {str(e)}")

    def run_island(self, island_id: int, stocks: list, products: list, result_queue: Queue):
        """Enhanced island execution"""
        try:
            self.logger.info(f"Starting Island {island_id}")
            
            # Initialize with different strategies per island
            strategies = [
                ('area_first', lambda p: p.length * p.width),
                ('length_first', lambda p: p.length),
                ('width_first', lambda p: p.width),
                ('aspect_ratio', lambda p: p.length / p.width),
                ('quantity_first', lambda p: p.quantity),
                ('hybrid', lambda p: (p.length * p.width) * p.quantity)
            ]
            
            # Assign different strategy to each island
            island_strategy = strategies[island_id % len(strategies)]
            
            population = self.initialize_population(stocks, products, strategy=island_strategy)
            
            if not population:
                raise ValueError(f"Island {island_id} could not create initial population")
                
            self.logger.info(f"Island {island_id} initial population created, size: {len(population)}")
            
            best_solution = None
            best_fitness = float('-inf')
            best_fulfillment_ratio = 0
            generations_without_improvement = 0
            
            for generation in range(self.generations):
                try:
                    # Calculate fitness with early stopping check
                    fitness_values = []
                    fulfillment_values = []
                    for chrom in population:
                        if not chrom:
                            continue
                        try:
                            fitness, _, ratio = self.calculate_fitness(chrom, stocks, products)
                            fitness_values.append(fitness)
                            fulfillment_values.append(ratio)
                            
                            # Early stopping if perfect solution found
                            if fitness > 1990:  # Near perfect score
                                result_queue.put((island_id, chrom, fitness, ratio))
                                return
                                
                        except Exception as e:
                            self.logger.error(f"Island {island_id} fitness calculation failed: {e}")
                            fitness_values.append(0) # Append 0 when the fitness calculation fails
                            fulfillment_values.append(0)
                    
                    # Update best solution
                    max_fitness_idx = np.argmax(fitness_values)
                    if fitness_values[max_fitness_idx] > best_fitness:
                        best_fitness = fitness_values[max_fitness_idx]
                        best_solution = copy.deepcopy(population[max_fitness_idx])
                        best_fulfillment_ratio = fulfillment_values[max_fitness_idx]
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1
                    
                    # Early stopping check
                    if generations_without_improvement >= 10:
                        break
                    
                    # Create next generation with island-specific parameters
                    mutation_rate = self.mutation_rate * (1 + 0.1 * island_id)  # Different rates per island
                    population = self.create_next_generation(population, fitness_values, stocks, products) #Removed mutation_rate
                    
                except Exception as e:
                    self.logger.error(f"Island {island_id}: Generation {generation} failed: {str(e)}")
                    continue
            
            result_queue.put((island_id, best_solution, best_fitness, best_fulfillment_ratio))
            
        except Exception as e:
            self.logger.error(f"Island {island_id} failed: {str(e)}")
            result_queue.put((island_id, None, float('-inf'), 0))
    
    def optimize(self, stocks: list, products: list):
            """Main optimization method"""
            start_time = time.time()
            
            try:
                # Validate input
                if not stocks or not products:
                    raise ValueError("Empty stocks or products list")
                
                # Create Manager and queues
                with Manager() as manager:
                    result_queue = manager.Queue()
                    
                    # Start island processes
                    processes = []
                    for i in range(self.num_islands):
                        p = Process(
                            target=self.run_island,
                            args=(i, stocks, products, result_queue)
                        )
                        p.daemon = True
                        processes.append(p)
                        p.start()
                    
                    # Collect results with timeout
                    results = []
                    timeout = max(300, self.generations * 2)
                    end_time = time.time() + timeout
                    
                    while len(results) < self.num_islands and time.time() < end_time:
                        try:
                            result = result_queue.get(timeout=1.0)
                            if result:
                                results.append(result)
                        except Empty:
                            continue
                    
                    # Find best solution
                    best_solution = None
                    best_fitness = float('-inf')
                    best_fulfillment_ratio = 0
                    
                    for island_id, solution, fitness, ratio in results:
                        if solution and fitness > best_fitness:
                            best_fitness = fitness
                            best_solution = copy.deepcopy(solution)
                            best_fulfillment_ratio = ratio
                    
                    # Calculate additional statistics including average utilization
                    total_area_used = 0
                    total_stock_area = 0
                    
                    if best_solution:
                        for stock_id, placements in best_solution.items():
                            # Get stock dimensions
                            stock = next(s for s in stocks if s.id == stock_id)
                            stock_area = stock.length * stock.width
                            total_stock_area += stock_area
                            
                            # Calculate used area from placements
                            for placement in placements:
                                product = next(p for p in products if p.id == placement.product_id)
                                total_area_used += product.length * product.width
                    
                    # Calculate average utilization
                    average_utilization = (total_area_used / total_stock_area * 100) if total_stock_area > 0 else 0
                    
                    # Prepare stats
                    stats = {
                        'execution_time': time.time() - start_time,
                        'generations': self.generations,
                        'islands': self.num_islands,
                        'best_fitness': best_fitness,
                        'total_area_used': total_area_used,
                        'average_utilization': average_utilization,
                         'overall_fulfillment_ratio': best_fulfillment_ratio # Add this to stats
                    }
                    
                    return best_solution, best_fitness, stats
                    
            except Exception as e:
                self.logger.error(f"Optimization failed: {str(e)}")
                return None, float('-inf'), {}
            finally:
                # Clean up processes
                for p in processes:
                    if p and p.is_alive():
                        try:
                            p.terminate()
                        except:
                            pass
    
    def calculate_fitness_with_logging(self, chromosome, stocks, products):
        """Enhanced fitness calculation with logging"""
        try:
            self.logger.debug(f"Calculating fitness for chromosome with {len(chromosome)} stocks")
            fitness = super().calculate_fitness(chromosome, stocks, products)
            self.logger.debug(f"Fitness calculated: {fitness}")
            return fitness
        except Exception as e:
            self.logger.error(f"Fitness calculation error: {str(e)}")
            return 0
        
    def create_next_generation(self, population: List[Dict], fitness_values: List[float],
                             stocks: List[Stock], products: List[Product]) -> List[Dict]:
        """Tuned next generation creation"""
        new_population = []
        
        # Enhanced elitism - keep best solutions
        sorted_indices = np.argsort(fitness_values)[::-1]
        for i in range(self.elite_size):
            if i < len(sorted_indices):
                elite = copy.deepcopy(population[sorted_indices[i]])
                new_population.append(elite)
                
                # Add slight variation of elite solutions
                if i < 2:  # Only for top 2 solutions
                    variant = copy.deepcopy(elite)
                    for placements in variant.values():
                        for p in placements:
                            p.x += random.uniform(-0.1, 0.1)
                            p.y += random.uniform(-0.1, 0.1)
                    new_population.append(variant)
        
        # Fill rest of population
        while len(new_population) < self.population_size:
            # Tournament selection
            tournament_size = min(self.tournament_size, len(population))
            idx1 = max(random.sample(range(len(population)), tournament_size),
                      key=lambda i: fitness_values[i])
            idx2 = max(random.sample(range(len(population)), tournament_size),
                      key=lambda i: fitness_values[i])
            
            parent1 = population[idx1]
            parent2 = population[idx2]
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2, stocks, products)
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
            
            # Mutation with dynamic rate
            mutation_rate = self.mutation_rate * (1 + 0.5 * (1 - fitness_values[idx1]/max(fitness_values)))
            if random.random() < mutation_rate:
                child = self.mutate(child, stocks, products)
            
            if child:
                new_population.append(child)
        
        return new_population