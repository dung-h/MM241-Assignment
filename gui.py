import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QTabWidget, QPushButton, QFileDialog, QTableWidget, 
    QTableWidgetItem, QLabel, QMessageBox, QHeaderView, QSpinBox, QDoubleSpinBox)
from PySide6.QtGui import QDropEvent, QDragEnterEvent
import pandas as pd
from typing import Tuple, Optional
import logging
import os
from core_ga import CuttingStockGA, Stock, Product, ParallelCuttingStockGA, Placement
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QWheelEvent, QFont
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QFrame, QScrollArea
import random
from multiprocessing import Pool, Manager, Process, Queue, cpu_count
import numpy as np
import logging
from typing import List, Dict, Tuple
import time
import copy
from dataclasses import dataclass, asdict
import random

import os
from datetime import datetime
from typing import Dict, List
import json
import csv
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import pandas as pd
from PySide6.QtWidgets import (
    QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox, QTabWidget,
    QPushButton, QMessageBox, QGraphicsDropShadowEffect
)
from PySide6.QtGui import (
    QDropEvent, QDragEnterEvent, QPainter, QPen, QBrush, QColor, QPainterPath,
    QCursor, QMouseEvent
)

class ResultsWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization Results")
        self.resize(1200, 800)
        # Store solution data
        self.current_solution = None
        self.stocks = None
        self.products = None
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel for cutting pattern
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Controls layout
        controls = QHBoxLayout()
        self.zoom_out_btn = QPushButton("-")
        self.zoom_in_btn = QPushButton("+")
        self.grid_btn = QPushButton("Grid")
        self.grid_btn.setCheckable(True)
        self.measurements_btn = QPushButton("Measurements")
        self.measurements_btn.setCheckable(True)
        
        controls.addWidget(self.zoom_out_btn)
        controls.addWidget(self.zoom_in_btn)
        controls.addWidget(self.grid_btn)
        controls.addWidget(self.measurements_btn)
        controls.addStretch()
        
        left_layout.addLayout(controls)
        
        # Pattern view
        self.pattern_view = QGraphicsView()
        self.pattern_view.setRenderHint(QPainter.Antialiasing)
        self.scene = QGraphicsScene()
        self.pattern_view.setScene(self.scene)
        left_layout.addWidget(self.pattern_view)
        
        # Stats panel
        stats_panel = QHBoxLayout()
        left_layout.addLayout(stats_panel)
        
        main_layout.addWidget(left_panel, stretch=7)
        
        # Right panel for legend
        legend_panel = QFrame()
        legend_panel.setFrameStyle(QFrame.Panel | QFrame.Raised)
        legend_layout = QVBoxLayout(legend_panel)
        
        # Legend title
        legend_title = QLabel("Product Legend")
        legend_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        legend_layout.addWidget(legend_title)
        
        # Create scrollable area for legend items
        legend_scroll = QScrollArea()
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        legend_content = QWidget()
        self.legend_items_layout = QVBoxLayout(legend_content)
        legend_scroll.setWidget(legend_content)
        legend_layout.addWidget(legend_scroll)
        
        main_layout.addWidget(legend_panel, stretch=3)

    def initialize_display(self, solution, stocks, products):
        """Initialize the display with solution data"""
        self.current_solution = solution
        self.stocks = stocks
        self.products = products
        self.display_results()

    def display_results(self):
        """Display cutting pattern results with scaling"""
        if not self.current_solution or not self.stocks or not self.products:
            return

        self.scene.clear()
        
        # Calculate optimal scale factor
        view_width = self.pattern_view.viewport().width()
        view_height = self.pattern_view.viewport().height()
        
        # Get maximum stock dimensions
        max_stock_length = max(s.length for s in self.stocks)
        max_stock_width = max(s.width for s in self.stocks)
        
        # Calculate grid layout
        stocks_count = len(self.current_solution)
        cols = min(3, stocks_count)  # Maximum 3 stocks per row
        rows = (stocks_count + cols - 1) // cols
        
        # Calculate scale factor
        margin = 50
        spacing = 30
        available_width = (view_width - margin * 2 - spacing * (cols - 1)) / cols
        available_height = (view_height - margin * 2 - spacing * (rows - 1)) / rows
        scale_factor = min(
            available_width / max_stock_length,
            available_height / max_stock_width
        ) * 0.8
        
        # Draw stocks and placements
        colors = self._generate_colors(len(self.products))
        current_row = 0
        current_col = 0
        
        for stock_id, placements in self.current_solution.items():
            stock = next(s for s in self.stocks if s.id == stock_id)
            x_pos = margin + (max_stock_length * scale_factor + spacing) * current_col
            y_pos = margin + (max_stock_width * scale_factor + spacing) * current_row
            
            # Draw stock rectangle
            self.scene.addRect(
                x_pos, y_pos,
                stock.length * scale_factor,
                stock.width * scale_factor,
                QPen(Qt.black, 2),
                QBrush(QColor(240, 240, 240))
            )
            
            # Add stock label
            stock_label = self.scene.addText(f"Stock {stock_id}")
            stock_label.setPos(x_pos, y_pos - 20)
            
            # Draw placements
            for placement in placements:
                if placement.rotated:
                    width = placement.product.width * scale_factor
                    height = placement.product.length * scale_factor
                else:
                    width = placement.product.length * scale_factor
                    height = placement.product.width * scale_factor
                
                x = x_pos + placement.x * scale_factor
                y = y_pos + placement.y * scale_factor
                
                color = colors[placement.product.id % len(colors)]
                self.scene.addRect(
                    x, y, width, height,
                    QPen(Qt.black),
                    QBrush(color)
                )
                
                # Simplified label
                label = f"P{placement.product.id}"
                if placement.rotated:
                    label += "R"
                text = self.scene.addText(label)
                text.setDefaultTextColor(Qt.white if color.value() < 128 else Qt.black)
                font = text.font()
                font.setPointSize(8)
                text.setFont(font)
                text.setPos(
                    x + width/2 - text.boundingRect().width()/2,
                    y + height/2 - text.boundingRect().height()/2
                )
            
            # Update grid position
            current_col += 1
            if current_col >= cols:
                current_col = 0
                current_row += 1
        
        # Update legend
        self.update_legend(self.products, colors)
        
        # Fit view
        self.pattern_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _generate_colors(self, n):
        """Generate visually distinct colors"""
        colors = []
        golden_ratio = 0.618033988749895
        h = 0.1
        for i in range(n):
            h = (h + golden_ratio) % 1
            color = QColor.fromHsvF(h, 0.85, 0.90)
            colors.append(color)
        return colors
   
    def update_legend(self, products, colors):
        """Update the legend with product information"""
        # Clear existing legend items
        for i in reversed(range(self.legend_items_layout.count())): 
            self.legend_items_layout.itemAt(i).widget().deleteLater()
            
        # Add legend items for each product
        for product in products:
            # Create frame for legend item
            item_frame = QFrame()
            item_frame.setFrameStyle(QFrame.Panel | QFrame.Plain)
            item_layout = QHBoxLayout(item_frame)
            
            # Color sample
            color_sample = QFrame()
            color_sample.setFixedSize(20, 20)
            color_sample.setStyleSheet(
                f"background-color: {colors[product.id % len(colors)].name()};"
                f"border: 1px solid black;"
            )
            item_layout.addWidget(color_sample)
            
            # Product information
            info_text = QLabel(
                f"Product {product.id}\n"
                f"Dimensions: {product.length} x {product.width}\n"
                f"Quantity: {product.quantity}"
            )
            item_layout.addWidget(info_text)
            item_layout.addStretch()
            
            self.legend_items_layout.addWidget(item_frame)
            
        # Add stretch at the end
        self.legend_items_layout.addStretch()
    

    def draw_rotation_indicator(self, x: float, y: float, width: float, height: float):
        """
        Draw a clearer rotation indicator for rotated pieces
        
        Args:
            x, y: Top-left coordinates of the piece
            width, height: Dimensions of the piece
        """
        # Make arrow smaller and more subtle
        arrow_size = min(width, height) * 0.15
        
        # Create arrow in top-right corner
        arrow_path = QPainterPath()
        
        # Calculate start point
        start_x = x + width - arrow_size - 2  # Slight padding from edge
        start_y = y + 2  # Slight padding from top
        
        # Draw curved arrow
        arrow_path.moveTo(start_x, start_y + arrow_size)  # Start point
        arrow_path.arcTo(
            start_x - arrow_size,
            start_y,
            arrow_size * 2,
            arrow_size * 2,
            0, -90
        )
        
        # Add arrowhead
        arrow_path.moveTo(start_x + arrow_size/2, start_y + arrow_size/2)  # Arrow tip
        arrow_path.lineTo(start_x + arrow_size/2, start_y)  # Vertical line
        arrow_path.lineTo(start_x + arrow_size, start_y + arrow_size/2)  # Horizontal line
        
        # Draw with thinner, more subtle line
        pen = QPen(Qt.black, 1)  # Thinner line
        self.scene.addPath(arrow_path, pen)

        
    def update_statistics(self, stock, placements):
        # Stock info
        stock_text = f"Dimensions: {stock.length:.1f} x {stock.width:.1f}\n"
        stock_text += f"Total Area: {stock.length * stock.width:.1f}"
        self.stock_info.setText(stock_text)
        
        # Product info
        used_area = sum(p.product.length * p.product.width for p in placements)
        product_text = f"Products Placed: {len(placements)}\n"
        product_text += f"Used Area: {used_area:.1f}"
        self.product_info.setText(product_text)
        
        # Utilization info
        utilization = (used_area / (stock.length * stock.width)) * 100
        util_text = f"Area Utilization: {utilization:.1f}%\n"
        util_text += f"Remaining Area: {(stock.length * stock.width - used_area):.1f}"
        self.util_info.setText(util_text)
    

class ExportHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        
    def export_all(self, solution: Dict, stats: Dict, base_path: str) -> Dict[str, str]:
        """
        Export all data: cutting patterns, statistics, and solution details
        Returns dict with paths to exported files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(base_path, f"cutting_stock_export_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)
        
        exported_files = {}
        
        # Get ID mappings from main window
        id_mappings = getattr(self.main_window, 'current_id_mappings', {'stock': {}, 'demand': {}})
        
        # Export cutting patterns as images
        patterns_dir = os.path.join(export_dir, "patterns")
        os.makedirs(patterns_dir, exist_ok=True)
        exported_files['patterns'] = self.export_patterns(solution, patterns_dir, id_mappings)
        
        # Export statistics
        stats_path = os.path.join(export_dir, "statistics.csv")
        self.export_statistics(stats, stats_path)
        exported_files['statistics'] = stats_path
        
        # Export detailed solution
        solution_path = os.path.join(export_dir, "solution.json")
        self.export_solution(solution, stats, solution_path, id_mappings)
        exported_files['solution'] = solution_path
        
        # Export summary Excel file
        summary_path = os.path.join(export_dir, "summary.xlsx")
        self.export_summary_excel(solution, stats, summary_path, id_mappings)
        exported_files['summary'] = summary_path
        
        return exported_files

    def export_patterns(self, solution: Dict, output_dir: str, id_mappings: Dict) -> List[str]:
        """Export cutting patterns as images with None handling"""
        exported_paths = []
        
        for stock_id, placements in solution.items():
            # Get stock with None check
            stock = next((s for s in self.main_window.stocks if s.id == stock_id), None)
            if stock is None:
                self.logger.error(f"Could not find stock with id {stock_id}")
                continue
            
            # Filter out any None placements and ensure valid product references
            valid_placements = []
            for placement in placements:
                if placement is None:
                    continue
                    
                # Ensure placement has a valid product reference
                if not hasattr(placement, 'product_id'):
                    continue
                    
                valid_placements.append(placement)
            
            # Display pattern with valid placements
            self.main_window.visualization.display_pattern(
                stock=stock,
                placements=valid_placements,
                stock_id=stock_id,
                products=self.main_window.products,
                id_mapping=id_mappings.get('demand', {})
            )
            
            # Capture and save image
            scene = self.main_window.visualization.view.scene()
            if scene is None:
                continue
                
            scene_rect = scene.sceneRect()
            image = QPixmap(scene_rect.width(), scene_rect.height())
            image.fill(Qt.white)
            
            painter = QPainter(image)
            scene.render(painter)
            painter.end()
            
            # Save image with safe ID handling
            try:
                stock_display_id = id_mappings.get('stock', {}).get(stock_id, stock_id)
                if stock_display_id is None:
                    stock_display_id = "unknown"
                image_path = os.path.join(output_dir, f"stock_{stock_display_id}.png")
                image.save(image_path, "PNG")
                exported_paths.append(image_path)
            except Exception as e:
                self.logger.error(f"Failed to save pattern image: {str(e)}")
                continue
        
        return exported_paths

    def export_solution(self, solution: Dict, stats: Dict, output_path: str, id_mappings: Dict):
        """Export solution with safe None handling"""
        try:
            export_data = {
                'solution': self._serialize_solution(solution, id_mappings),
                'statistics': stats,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_stocks': len(solution),
                    'total_placements': sum(len(placements) for placements in solution.values() if placements)
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to export solution: {str(e)}")

    def _serialize_solution(self, solution: Dict, id_mappings: Dict) -> Dict:
        """Convert solution to serializable format with None handling"""
        serialized = {}
        demand_mapping = id_mappings.get('demand', {})
        
        for stock_id, placements in solution.items():
            if placements is None:
                continue
                
            serialized_placements = []
            for p in placements:
                if p is None or not hasattr(p, 'product') or p.product is None:
                    continue
                    
                try:
                    serialized_placements.append({
                        'product_id': demand_mapping.get(p.product.id, p.product.id),
                        'x': p.x if hasattr(p, 'x') else 0,
                        'y': p.y if hasattr(p, 'y') else 0,
                        'rotated': p.rotated if hasattr(p, 'rotated') else False,
                        'product_length': getattr(p.product, 'length', 0),
                        'product_width': getattr(p.product, 'width', 0)
                    })
                except AttributeError:
                    continue
            
            if serialized_placements:
                serialized[stock_id] = serialized_placements
        
        return serialized

    def export_statistics(self, stats: Dict, output_path: str):
        """Export statistics to CSV"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in stats.items():
                writer.writerow([key, value])
                
    def export_summary_excel(self, solution: Dict, stats: Dict, output_path: str, id_mappings: Dict):
        """Export comprehensive summary to Excel with proper ID mappings"""
        try:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Statistics sheet
                stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Stock utilization sheet
                stock_data = []
                stock_mapping = id_mappings.get('stock', {})
                for stock_id, placements in solution.items():
                    stock = next(s for s in self.main_window.stocks if s.id == stock_id)
                    total_area = stock.length * stock.width
                    used_area = sum(
                        p.product.length * p.product.width 
                        for p in placements 
                        if hasattr(p, 'product') and p.product is not None
                    )
                    utilization = (used_area / total_area) * 100 if total_area > 0 else 0
                    
                    stock_data.append({
                        'Stock ID': stock_mapping.get(stock_id, stock_id),
                        'Length': stock.length,
                        'Width': stock.width,
                        'Total Area': total_area,
                        'Used Area': used_area,
                        'Utilization %': round(utilization, 2),
                        'Placements': len(placements)
                    })
                
                stock_df = pd.DataFrame(stock_data)
                stock_df.to_excel(writer, sheet_name='Stock Utilization', index=False)
                
                # Detailed placements sheet
                placement_data = []
                demand_mapping = id_mappings.get('demand', {})
                for stock_id, placements in solution.items():
                    for p in placements:
                        if not hasattr(p, 'product') or p.product is None:
                            continue
                        placement_data.append({
                            'Stock ID': stock_mapping.get(stock_id, stock_id),
                            'Product ID': demand_mapping.get(p.product.id, p.product.id),
                            'X Position': p.x,
                            'Y Position': p.y,
                            'Rotated': p.rotated,
                            'Product Length': p.product.length,
                            'Product Width': p.product.width
                        })
                
                placement_df = pd.DataFrame(placement_data)
                placement_df.to_excel(writer, sheet_name='Detailed Placements', index=False)
                
        except Exception as e:
            self.main_window.logger.error(f"Failed to export summary Excel: {str(e)}")
            raise

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # Set up zooming
        self.zoom = 1.0
        self.zoom_factor = 1.15
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        
    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            # Zoom
            angle = event.angleDelta().y()
            if angle > 0:
                factor = self.zoom_factor
            else:
                factor = 1 / self.zoom_factor
                
            self.zoom *= factor
            self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom))
            self.scale(factor, factor)
        else:
            # Normal scroll
            super().wheelEvent(event)

class StockPatternView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.scale_factor = 1.0
        self.grid_size = 1.0
        self.show_measurements = True
        self.show_grid = True
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Toolbar for controls
        toolbar = QHBoxLayout()
        
        # Zoom controls
        self.zoom_in_btn = QPushButton("+")
        self.zoom_out_btn = QPushButton("-")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom(1.2))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom(0.8))
        
        # Grid toggle
        self.grid_toggle = QPushButton("Grid")
        self.grid_toggle.setCheckable(True)
        self.grid_toggle.setChecked(True)
        self.grid_toggle.clicked.connect(self.toggle_grid)
        
        # Measurements toggle
        self.measure_toggle = QPushButton("Measurements")
        self.measure_toggle.setCheckable(True)
        self.measure_toggle.setChecked(True)
        self.measure_toggle.clicked.connect(self.toggle_measurements)
        
        # Add controls to toolbar
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.grid_toggle)
        toolbar.addWidget(self.measure_toggle)
        toolbar.addStretch()
        
        layout.addLayout(toolbar)
        
        # Graphics view for cutting pattern
        self.view = ZoomableGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        layout.addWidget(self.view)
        
        # Information panel
        self.info_panel = QWidget()
        info_layout = QHBoxLayout(self.info_panel)
        
        # Stock information
        stock_info = QVBoxLayout()
        self.stock_label = QLabel("Stock Information")
        self.stock_label.setStyleSheet("font-weight: bold;")
        self.stock_details = QLabel()
        stock_info.addWidget(self.stock_label)
        stock_info.addWidget(self.stock_details)
        info_layout.addLayout(stock_info)
        
        # Product information
        product_info = QVBoxLayout()
        self.product_label = QLabel("Product Information")
        self.product_label.setStyleSheet("font-weight: bold;")
        self.product_details = QLabel()
        product_info.addWidget(self.product_label)
        product_info.addWidget(self.product_details)
        info_layout.addLayout(product_info)
        
        # Utilization information
        util_info = QVBoxLayout()
        self.util_label = QLabel("Utilization")
        self.util_label.setStyleSheet("font-weight: bold;")
        self.util_details = QLabel()
        util_info.addWidget(self.util_label)
        util_info.addWidget(self.util_details)
        info_layout.addLayout(util_info)
        
        layout.addWidget(self.info_panel)
        
        # Add tabbed view for multiple stock sheets
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
    def display_pattern(self, stock, placements, stock_id, products, id_mapping):
        """Display enhanced cutting pattern"""
        self.scene.clear()
        
        # Set up the scene with margins for measurements
        margin = 50
        stock_width = stock.width * self.scale_factor
        stock_height = stock.length * self.scale_factor
        
        # Draw background grid if enabled
        if self.show_grid:
            self.draw_grid(stock_width, stock_height, margin)
        
        # Draw stock rectangle
        stock_rect = self.scene.addRect(
            margin, margin, 
            stock_width, stock_height,
            QPen(Qt.black, 2),
            QBrush(QColor(240, 240, 240))
        )
        
        # Draw measurements if enabled
        if self.show_measurements:
            self.draw_measurements(stock_width, stock_height, margin)
        
        # Generate colors for products
        colors = self._generate_colors(len(products))
        color_map = {p.id: colors[i % len(colors)] for i, p in enumerate(products)}
        
        # Draw placed products
        for placement in placements:
            if not placement.product:
                continue
                
            width = placement.product.width if not placement.rotated else placement.product.length
            height = placement.product.length if not placement.rotated else placement.product.width
            
            # Scale dimensions
            scaled_width = width * self.scale_factor
            scaled_height = height * self.scale_factor
            scaled_x = placement.x * self.scale_factor + margin
            scaled_y = placement.y * self.scale_factor + margin
            
            # Product rectangle with shadow effect
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)
            shadow.setOffset(3, 3)
            
            rect_item = self.scene.addRect(
                scaled_x, scaled_y, 
                scaled_width, scaled_height,
                QPen(Qt.black),
                QBrush(color_map[placement.product.id])
            )
            rect_item.setGraphicsEffect(shadow)
            
            # Add product label
            display_id = id_mapping.get(placement.product.id, f"P{placement.product.id}")
            text_item = self.scene.addText(f"P{display_id}\n{width}x{height}")
            text_item.setDefaultTextColor(Qt.black)
            
            # Use scene coordinates to properly position text
            text_x = scaled_x + scaled_width / 2 - text_item.boundingRect().width() / 2
            text_y = scaled_y + scaled_height / 2 - text_item.boundingRect().height() / 2
            text_item.setPos(text_x, text_y)

            
            # Add rotation indicator if rotated
            if placement.rotated:
                self.draw_rotation_indicator(scaled_x, scaled_y, scaled_width, scaled_height)
        
        # Update information panels
        self.update_info_panels(stock, placements, products)
        
        # Fit view to scene
        self.view.setSceneRect(self.scene.itemsBoundingRect())
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
    def draw_grid(self, width, height, margin):
        """Draw background grid"""
        pen = QPen(QColor(200, 200, 200), 1, Qt.DashLine)
        
        # Draw vertical lines
        for x in range(0, int(width) + 1, int(self.grid_size * self.scale_factor)):
            self.scene.addLine(x + margin, margin, x + margin, height + margin, pen)
            
        # Draw horizontal lines
        for y in range(0, int(height) + 1, int(self.grid_size * self.scale_factor)):
            self.scene.addLine(margin, y + margin, width + margin, y + margin, pen)
            
    def draw_measurements(self, width, height, margin):
        """Draw measurements and axes"""
        # Draw axes
        self.scene.addLine(margin/2, margin, margin/2, height + margin + margin/2, 
                          QPen(Qt.black, 2))
        self.scene.addLine(margin, height + margin + margin/2, 
                          width + margin + margin/2, height + margin + margin/2, 
                          QPen(Qt.black, 2))
        
        # Add measurement labels
        for i in range(0, int(width/self.scale_factor) + 1, 5):
            x = i * self.scale_factor + margin
            text = self.scene.addText(str(i))
            text.setPos(x - text.boundingRect().width()/2, 
                       height + margin + margin/4)
            
        for i in range(0, int(height/self.scale_factor) + 1, 5):
            y = i * self.scale_factor + margin
            text = self.scene.addText(str(i))
            text.setPos(margin/4 - text.boundingRect().width()/2, 
                       y - text.boundingRect().height()/2)
                       
    def draw_rotation_indicator(self, x: float, y: float, width: float, height: float):
        """
        Draw a clearer rotation indicator for rotated pieces
        
        Args:
            x, y: Top-left coordinates of the piece
            width, height: Dimensions of the piece
        """
        # Make arrow smaller and more subtle
        arrow_size = min(width, height) * 0.15
        
        # Create arrow in top-right corner
        arrow_path = QPainterPath()
        
        # Calculate start point
        start_x = x + width - arrow_size - 2  # Slight padding from edge
        start_y = y + 2  # Slight padding from top
        
        # Draw curved arrow
        arrow_path.moveTo(start_x, start_y + arrow_size)  # Start point
        arrow_path.arcTo(
            start_x - arrow_size,
            start_y,
            arrow_size * 2,
            arrow_size * 2,
            0, -90
        )
        
        # Add arrowhead
        arrow_path.moveTo(start_x + arrow_size/2, start_y + arrow_size/2)  # Arrow tip
        arrow_path.lineTo(start_x + arrow_size/2, start_y)  # Vertical line
        arrow_path.lineTo(start_x + arrow_size, start_y + arrow_size/2)  # Horizontal line
        
        # Draw with thinner, more subtle line
        pen = QPen(Qt.black, 1)  # Thinner line
        self.scene.addPath(arrow_path, pen)
        
    def update_info_panels(self, stock, placements, products):
        """Update information panels with current data"""
        # Stock information
        stock_info = f"Dimensions: {stock.length} x {stock.width}\n"
        stock_info += f"Total Area: {stock.length * stock.width}"
        self.stock_details.setText(stock_info)
        
        # Product information
        used_area = sum(p.product.length * p.product.width for p in placements)
        product_info = f"Products Placed: {len(placements)}\n"
        product_info += f"Used Area: {used_area:.2f}"
        self.product_details.setText(product_info)
        
        # Utilization information
        utilization = (used_area / (stock.length * stock.width)) * 100
        util_info = f"Area Utilization: {utilization:.1f}%\n"
        util_info += f"Remaining Area: {(stock.length * stock.width - used_area):.2f}"
        self.util_details.setText(util_info)
        
    def zoom(self, factor):
        """Handle zoom operations"""
        self.scale_factor *= factor
        self.view.scale(factor, factor)
        
    def toggle_grid(self, enabled):
        """Toggle grid visibility"""
        self.show_grid = enabled
        self.refresh_display()
        
    def toggle_measurements(self, enabled):
        """Toggle measurements visibility"""
        self.show_measurements = enabled
        self.refresh_display()
        
    def refresh_display(self):
        """Refresh the current display"""
        if hasattr(self, 'current_display_data'):
            self.display_pattern(**self.current_display_data)
            
    def _generate_colors(self, n):
        """Generate visually distinct colors"""
        colors = []
        golden_ratio = 0.618033988749895
        h = 0.1
        
        for i in range(n):
            h = (h + golden_ratio) % 1
            # Use HSV color space for better distinction
            color = QColor.fromHsvF(h, 0.8, 0.95)
            colors.append(color)
            
        return colors
    
class DataTable(QTableWidget):
    def __init__(self, headers: list, parent=None):
        super().__init__(parent)
        self.headers = headers
        self.id_mapping = {}  # Store mapping between display ID and internal ID
        self.setup_table()

    def add_data_row(self, values=None):
        """Add a new data row with internal continuous IDs"""
        self.removeRow(self.rowCount() - 1)
        row_position = self.rowCount()
        self.insertRow(row_position)
        
        # Generate internal ID (0-based continuous)
        internal_id = row_position
        display_id = values[0] if values and len(values) > 0 else str(row_position)
        
        # Store mapping
        self.id_mapping[internal_id] = display_id
        
        # Add cells
        for col, header in enumerate(self.headers):
            if header == 'id':
                # Show display ID but store internal ID
                item = QTableWidgetItem(str(display_id))
                item.setData(Qt.UserRole, internal_id)  # Store internal ID
                self.setItem(row_position, col, item)
            elif header in ['length', 'width']:
                spinbox = QDoubleSpinBox()
                spinbox.setRange(0.1, 1000.0)
                spinbox.setDecimals(2)
                value = values[col] if values and col < len(values) else 0
                spinbox.setValue(float(value))
                self.setCellWidget(row_position, col, spinbox)
            elif header == 'quantity':
                spinbox = QSpinBox()
                spinbox.setRange(1, 1000)
                value = values[col] if values and col < len(values) else 1
                spinbox.setValue(int(value))
                self.setCellWidget(row_position, col, spinbox)
        
        self.add_button_row()

    def get_data(self) -> list:
        """Get data with internal continuous IDs"""
        data = []
        for row in range(self.rowCount() - 1):
            row_data = {}
            valid_row = True
            
            for col, header in enumerate(self.headers):
                if header == 'id':
                    # Use internal ID (row number) for GA
                    row_data[header] = row
                else:
                    widget = self.cellWidget(row, col)
                    if widget:
                        value = widget.value()
                        if value <= 0:
                            valid_row = False
                            break
                        row_data[header] = value
                    else:
                        valid_row = False
                        break
                    
            if valid_row and row_data:
                data.append(row_data)
                
        return data
        
    def setup_table(self):
        self.setColumnCount(len(self.headers))
        self.setHorizontalHeaderLabels(self.headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAcceptDrops(True)
        
        # Add button row at the bottom
        self.add_button_row()
        
    def add_button_row(self):
        """Add a row with '+ Add New' text"""
        row_position = self.rowCount()
        self.insertRow(row_position)
        
        # Create a cell with "+ Add New" text
        add_item = QTableWidgetItem("+ Add New")
        add_item.setTextAlignment(Qt.AlignCenter)
        self.setItem(row_position, 0, add_item)
        
        # Merge cells in the button row
        self.setSpan(row_position, 0, 1, self.columnCount())
  
    def load_data(self, filepath: str):
        """Load data from file"""
        try:
            df = pd.read_csv(filepath)
            required_cols = set(self.headers)
            
            if not all(col in df.columns for col in required_cols):
                QMessageBox.warning(self, "Invalid File", 
                    f"File must contain columns: {', '.join(required_cols)}")
                return

            # Clear existing data (except button row)
            while self.rowCount() > 1:
                self.removeRow(0)
                
            # Load data
            for _, row in df.iterrows():
                values = [row[header] for header in self.headers]
                self.add_data_row(values)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cutting Stock Optimizer")
        self.setMinimumSize(800, 600)
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize file paths
        self.stock_file_path = None
        self.demand_file_path = None
        
        # Setup UI
        self.setup_ui()
        
        # Initialize GA
        self.ga = CuttingStockGA()

    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget for input methods
        tab_widget = QTabWidget()
        
        # File input tab
        self.file_tab = QWidget()
        file_layout = QVBoxLayout(self.file_tab)
        
        # Stock file section
        stock_section = QVBoxLayout()
        stock_header = QHBoxLayout()
        
        stock_label = QLabel("Stock File:")
        self.stock_file_btn = QPushButton("Select Stock File")
        self.stock_file_btn.clicked.connect(lambda: self.select_file('stock'))
        
        stock_header.addWidget(stock_label)
        stock_header.addWidget(self.stock_file_btn)
        stock_section.addLayout(stock_header)
        
        # Add status label for stock file
        self.stock_status_label = QLabel("No file selected")
        self.stock_status_label.setStyleSheet("color: gray; margin-left: 20px;")
        stock_section.addWidget(self.stock_status_label)
        
        file_layout.addLayout(stock_section)
        
        # Add some spacing
        file_layout.addSpacing(10)
        
        # Demand file section
        demand_section = QVBoxLayout()
        demand_header = QHBoxLayout()
        
        demand_label = QLabel("Demand File:")
        self.demand_file_btn = QPushButton("Select Demand File")
        self.demand_file_btn.clicked.connect(lambda: self.select_file('demand'))
        
        demand_header.addWidget(demand_label)
        demand_header.addWidget(self.demand_file_btn)
        demand_section.addLayout(demand_header)
        
        # Add status label for demand file
        self.demand_status_label = QLabel("No file selected")
        self.demand_status_label.setStyleSheet("color: gray; margin-left: 20px;")
        demand_section.addWidget(self.demand_status_label)
        
        file_layout.addLayout(demand_section)
        
        # Add preview tables
        file_layout.addSpacing(20)
        preview_label = QLabel("Data Preview:")
        file_layout.addWidget(preview_label)
        
        # Stock preview
        self.file_stock_table = DataTable(['id', 'length', 'width'])
        stock_preview_label = QLabel("Stock Data:")
        file_layout.addWidget(stock_preview_label)
        file_layout.addWidget(self.file_stock_table)
        
        # Demand preview
        self.file_demand_table = DataTable(['id', 'length', 'width', 'quantity'])
        demand_preview_label = QLabel("Demand Data:")
        file_layout.addWidget(demand_preview_label)
        file_layout.addWidget(self.file_demand_table)
        
        tab_widget.addTab(self.file_tab, "File Input")
        
        # Manual input tab
        manual_tab = QWidget()
        manual_layout = QVBoxLayout(manual_tab)
        
        # Stock table
        stock_table_label = QLabel("Stocks:")
        self.stock_table = DataTable(['id', 'length', 'width'])
        manual_layout.addWidget(stock_table_label)
        manual_layout.addWidget(self.stock_table)
        
        # Demand table
        demand_table_label = QLabel("Demands:")
        self.demand_table = DataTable(['id', 'length', 'width', 'quantity'])
        manual_layout.addWidget(demand_table_label)
        manual_layout.addWidget(self.demand_table)
        
        tab_widget.addTab(manual_tab, "Manual Input")
        
        layout.addWidget(tab_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Optimization")
        self.run_button.clicked.connect(self.run_optimization)
        self.run_button.setEnabled(False)  # Disabled by default
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.export_button)
        layout.addLayout(button_layout)
        
        # Results area
        self.results_label = QLabel("Results will appear here")
        self.results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.results_label)
        self.visualization = StockPatternView()
        layout.addWidget(self.visualization)
    
        # Adjust layout ratios (40% input, 60% visualization)
        layout.setStretch(0, 40)  # Input section
        layout.setStretch(1, 60)  # Visualization section

    def process_solution(self, solution: Dict, products: List[Product]) -> Dict:
        """Process solution to include product information"""
        processed_solution = {}
        
        for stock_id, placements in solution.items():
            processed_placements = []
            for placement in placements:
                # Find corresponding product
                product = next((p for p in products if p.id == placement.product_id), None)
                if product:
                    # Create new placement with product information
                    new_placement = Placement(
                        product_id=placement.product_id,
                        x=placement.x,
                        y=placement.y,
                        rotated=placement.rotated
                    )
                    new_placement.product = product
                    processed_placements.append(new_placement)
            if processed_placements:
                processed_solution[stock_id] = processed_placements
                
        return processed_solution

    def export_results(self):
        """Handle the export button click"""
        if not hasattr(self, 'current_solution') or not hasattr(self, 'current_stats'):
            QMessageBox.warning(self, "No Results", 
                "Please run the optimization first before exporting results.")
            return
            
        # Get export directory from user
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not export_dir:
            return
            
        try:
            # Create and use export handler
            export_handler = ExportHandler(self)
            exported_files = export_handler.export_all(
                self.current_solution,
                self.current_stats,
                export_dir
            )
            
            # Show success message with export details
            message = "Export completed successfully!\n\nFiles created:\n"
            for export_type, path in exported_files.items():
                message += f"\n{export_type.title()}: {os.path.basename(path)}"
            
            QMessageBox.information(self, "Export Success", message)
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
    
    def display_results(self, solution, stats):
        """Display results with proper ID mapping"""
        if not solution:
            return
            
        # Map internal IDs back to original IDs for display
        mapped_solution = {}
        for stock_id, placements in solution.items():
            original_stock_id = next(k for k, v in self.stock_id_map.items() if v == stock_id)
            mapped_placements = []
            for placement in placements:
                original_product_id = next(k for k, v in self.product_id_map.items() 
                                        if v == placement.product_id)
                new_placement = copy.deepcopy(placement)
                new_placement.product_id = original_product_id
                mapped_placements.append(new_placement)
            mapped_solution[original_stock_id] = mapped_placements

        # Create and show results window with mapped IDs
        self.results_window = ResultsWindow(self)
        self.results_window.initialize_display(solution, self.stocks, self.products)
        self.results_window.show()
        # Store solution and stats for export
        self.current_solution = mapped_solution
        self.current_stats = stats
        self.export_button.setEnabled(True)
    
    def select_file(self, file_type: str):
        # Open file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {file_type.capitalize()} File",
            "",
            "CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        
        if filepath:
            try:
                # Load data based on file extension
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:  # Excel files
                    df = pd.read_excel(filepath)
                
                # Verify columns
                required_cols = {'id', 'length', 'width'}
                if file_type == 'demand':
                    required_cols.add('quantity')
                    
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns. File must contain: {', '.join(required_cols)}")
                
                # Validate numeric data
                for col in ['length', 'width']:
                    if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                        raise ValueError(f"Column '{col}' must contain only numeric values")
                    if (df[col] <= 0).any():
                        raise ValueError(f"Column '{col}' must contain only positive values")
                
                if file_type == 'demand':
                    if not pd.to_numeric(df['quantity'], errors='coerce').notnull().all():
                        raise ValueError("Column 'quantity' must contain only numeric values")
                    if (df['quantity'] <= 0).any():
                        raise ValueError("Column 'quantity' must contain only positive values")
                
                # If all validation passes, update tables and status
                if file_type == 'stock':
                    self.stock_file_path = filepath
                    self.stock_table.load_data(filepath)
                    self.stock_status_label.setText(f"✓ Loaded: {os.path.basename(filepath)}")
                    self.stock_status_label.setStyleSheet("color: green; margin-left: 20px;")
                    self.stock_file_btn.setText("Change Stock File")
                    
                    # Update preview if in file input tab
                    self.file_stock_table.load_data(filepath)
                else:
                    self.demand_file_path = filepath
                    self.demand_table.load_data(filepath)
                    self.demand_status_label.setText(f"✓ Loaded: {os.path.basename(filepath)}")
                    self.demand_status_label.setStyleSheet("color: green; margin-left: 20px;")
                    self.demand_file_btn.setText("Change Demand File")
                    
                    # Update preview if in file input tab
                    self.file_demand_table.load_data(filepath)
                
                # Enable run button if both files are loaded
                if hasattr(self, 'stock_file_path') and hasattr(self, 'demand_file_path'):
                    self.run_button.setEnabled(True)
                    
            except Exception as e:
                # Clear file path on error
                if file_type == 'stock':
                    self.stock_file_path = None
                    self.stock_status_label.setText(f"✗ Error loading file")
                    self.stock_status_label.setStyleSheet("color: red; margin-left: 20px;")
                else:
                    self.demand_file_path = None
                    self.demand_status_label.setText(f"✗ Error loading file")
                    self.demand_status_label.setStyleSheet("color: red; margin-left: 20px;")
                    
                # Show detailed error message
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
                
                # Disable run button if either file is invalid
                self.run_button.setEnabled(False)
        
    def validate_data(self) -> Tuple[Optional[list], Optional[list]]:
        """Validate and return stock and demand data with enhanced debugging"""
        print("\n=== Starting Data Validation ===")
        current_tab = self.findChild(QTabWidget).currentWidget()
        print(f"Current tab is file tab: {current_tab == self.file_tab}")
        
        # Determine which tables to use based on current tab
        if current_tab == self.file_tab:
            stock_table = self.file_stock_table
            demand_table = self.file_demand_table
            print("Using file input tables")
        else:
            stock_table = self.stock_table
            demand_table = self.demand_table
            print("Using manual input tables")
        
        # Get data from appropriate tables
        print("\nGetting stock data...")
        stocks_data = stock_table.get_data()
        print(f"Retrieved stock data: {stocks_data}")
        
        print("\nGetting demand data...")
        demands_data = demand_table.get_data()
        print(f"Retrieved demand data: {demands_data}")
        
        # Validate stocks
        if not stocks_data:
            print("No valid stock data found")
            QMessageBox.warning(self, "Invalid Data", "No valid stock data provided")
            return None, None
        
        # Validate each stock entry
        self.stock_id_map = {}  # Original ID -> Internal ID
        self.product_id_map = {}  # Original ID -> Internal ID
        
        # Create validated data with proper ID mapping
        valid_stocks = []
        for i, stock in enumerate(stocks_data):
            try:
                original_id = int(stock['id'])
                self.stock_id_map[original_id] = i  # Map original ID to internal index
                
                valid_stocks.append({
                    'id': i,  # Use internal sequential ID
                    'length': float(stock.get('length', 0)),
                    'width': float(stock.get('width', 0)),
                    'original_id': original_id  # Store original ID
                })
            except (ValueError, KeyError) as e:
                self.logger.error(f"Error processing stock {stock}: {e}")
                
        valid_demands = []
        for i, demand in enumerate(demands_data):
            try:
                original_id = int(demand['id'])
                self.product_id_map[original_id] = i  # Map original ID to internal index
                
                valid_demands.append({
                    'id': i,  # Use internal sequential ID
                    'length': float(demand.get('length', 0)),
                    'width': float(demand.get('width', 0)),
                    'quantity': int(demand.get('quantity', 0)),
                    'original_id': original_id  # Store original ID
                })
            except (ValueError, KeyError) as e:
                self.logger.error(f"Error processing demand {demand}: {e}")

        return valid_stocks, valid_demands


    def run_optimization(self):
        """Run optimization with enhanced data conversion"""
        try:
            # Validate and get data
            stocks_data, demands_data = self.validate_data()
            if not stocks_data or not demands_data:
                return
                
            # Convert to GA format with explicit object creation
            self.stocks = [
                Stock(
                    id=int(s['id']),
                    length=float(s['length']),
                    width=float(s['width'])
                ) 
                for s in stocks_data
            ]
            
            self.products = [
                Product(
                    id=int(d['id']),
                    length=float(d['length']),
                    width=float(d['width']),
                    quantity=int(d['quantity'])
                ) 
                for d in demands_data
            ]
            
            # Log converted objects for verification
            self.logger.info(f"Processing {len(self.stocks)} stocks and {len(self.products)} products")
            for stock in self.stocks:
                self.logger.debug(f"Stock: id={stock.id}, length={stock.length}, width={stock.width}")
            for product in self.products:
                self.logger.debug(f"Product: id={product.id}, length={product.length}, width={product.width}, quantity={product.quantity}")
            
            # Initialize and run GA
            self.ga = ParallelCuttingStockGA()
            # Run optimization
            solution, fitness, stats = self.ga.optimize(self.stocks, self.products)
            
            if solution:
                self.current_solution = solution
                self.current_stats = stats
                self.display_results(solution, stats)
            else:
                raise ValueError("No valid solution found")
                
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            QMessageBox.critical(self, "Optimization Error", 
                f"Failed to complete optimization: {str(e)}\n\n"
                "Please verify your input data and try again with different parameters.")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()