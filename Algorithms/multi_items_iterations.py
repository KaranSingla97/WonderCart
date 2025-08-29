#!/usr/bin/env python3
"""
Enhanced Supermarket Path Planner — Comprehensive Benchmarking System

Features:
- Systematic variation of item counts (2-15 items)
- Multiple item selection strategies
- 100+ iterations per configuration
- Comprehensive metrics collection
- Statistical analysis and visualization
- Performance comparison across methods

Requirements:
    Python 3.8+
    pip install networkx matplotlib pandas numpy seaborn scipy

Run:
    python supermarket_benchmark.py
"""

import math
import random
import itertools
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import seaborn as sns
from scipy import stats
import time
import json

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ---------- Enhanced Configuration ----------
WALKING_SPEED_M_S = 1.1
TIME_BUDGET_MIN = 45
ALLOWED_EXTRA_FRACTION = 0.15
VISIT_OVERHEAD_SEC = 8

# Enhanced objective weights
PREF_WEIGHT = 1.2
COUNT_BONUS = 0.025
TIME_EFFICIENCY_WEIGHT = 0.3
PATH_SMOOTHNESS_WEIGHT = 0.1

# Penalty system
REVISIT_PENALTY_PER = 0.08
REVISIT_TIME_PENALTY_MIN = 0.7
DIRECTION_CHANGE_PENALTY = 0.02

# Benchmarking configuration
BENCHMARK_CONFIG = {
    'item_counts': list(range(2, 16)),  # 2-15 items
    'iterations_per_config': 100,
    'item_selection_methods': [
        'random',
        'high_preference',
        'balanced',
        'category_diverse',
        'time_efficient',
        'mixed'
    ],
    'random_seeds': list(range(42, 142))  # 100 different seeds
}

# Output configuration
OUT_DIR = "benchmark_results"
os.makedirs(OUT_DIR, exist_ok=True)

def flog(msg, level="INFO"):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    emoji = {"INFO":"ℹ️", "STEP":"🔄", "WARN":"⚠️", "DONE":"✅", "FUN":"🚀", "METRIC":"📊", "BENCH":"🧪"}.get(level, "ℹ️")
    print(f"{emoji} [{now}] {msg}")

# ---------- Enhanced Store Layout (Same as original) ----------
def create_enhanced_store_layout():
    """Create a more realistic store layout with better connectivity"""
    aisle_x = [0, 6, 12, 18, 24, 30, 36, 42]
    aisle_y = [0, 6, 12, 18, 24]
    
    nodes = {}
    node_id = 0
    
    for y in aisle_y:
        for x in aisle_x:
            name = f"N{node_id}"
            nodes[name] = (x, y)
            node_id += 1
    
    nodes["Entrance"] = (-6, 12)
    nodes["Checkout"] = (48, 6)
    
    item_positions = {
        "Milk": (6, 6), "Cheese": (6, 12), "Yogurt": (6, 18),
        "Eggs": (12, 6), "Butter": (12, 12),
        "Apples": (18, 18), "Bananas": (18, 12), "Tomatoes": (18, 6),
        "Lettuce": (18, 0), "Carrots": (24, 18),
        "Chicken": (24, 12), "Beef": (24, 6), "Fish": (24, 0),
        "Bread": (30, 18), "Pastries": (30, 12),
        "Cereal": (36, 18), "Pasta": (36, 12), "Rice": (36, 6),
        "Chips": (36, 0), "Cookies": (42, 18),
        "Coffee": (0, 18), "Tea": (0, 12), "Soda": (42, 12),
        "Juice": (42, 6), "Water": (0, 6),
        "Soap": (0, 0), "Detergent": (42, 0), "Shampoo": (6, 0),
        "FrozenPizza": (12, 24), "IceCream": (30, 24),
        "OliveOil": (30, 6), "Vinegar": (30, 0),
        # Additional items for more variety
        "Salmon": (24, 18), "Pork": (18, 24), "Turkey": (12, 18),
        "Orange": (0, 24), "Grapes": (6, 24), "Spinach": (36, 24),
        "Broccoli": (42, 24), "Onions": (30, 0), "Potatoes": (0, 18),
        "Flour": (18, 0), "Sugar": (24, 0), "Salt": (30, 12),
        "Pepper": (36, 0), "Honey": (42, 0), "Jam": (0, 0)
    }
    
    for item, coord in list(item_positions.items()):
        found = None
        for n, c in nodes.items():
            if c == coord:
                found = n
                break
        if not found:
            found = f"X_{item}"
            nodes[found] = coord
        item_positions[item] = found
    
    return nodes, item_positions

def create_item_metadata():
    """Extended item metadata for comprehensive testing"""
    base_items = {
        "Milk": {"category":"Dairy","visit_min":1.2,"price":3.5,"rec_score":0.95, "popularity":0.9},
        "Bread": {"category":"Bakery","visit_min":1.0,"price":2.5,"rec_score":0.88, "popularity":0.85},
        "Eggs": {"category":"Dairy","visit_min":1.1,"price":4.2,"rec_score":0.90, "popularity":0.8},
        "Cheese": {"category":"Dairy","visit_min":1.6,"price":5.8,"rec_score":0.75, "popularity":0.7},
        "Yogurt": {"category":"Dairy","visit_min":0.9,"price":2.2,"rec_score":0.65, "popularity":0.6},
        "Butter": {"category":"Dairy","visit_min":0.8,"price":4.0,"rec_score":0.60, "popularity":0.5},
        
        "Apples": {"category":"Produce","visit_min":1.2,"price":2.8,"rec_score":0.82, "popularity":0.85},
        "Bananas": {"category":"Produce","visit_min":0.8,"price":1.5,"rec_score":0.78, "popularity":0.9},
        "Tomatoes": {"category":"Produce","visit_min":1.0,"price":3.0,"rec_score":0.70, "popularity":0.75},
        "Lettuce": {"category":"Produce","visit_min":0.9,"price":2.0,"rec_score":0.65, "popularity":0.6},
        "Carrots": {"category":"Produce","visit_min":0.7,"price":1.8,"rec_score":0.68, "popularity":0.65},
        
        "Chicken": {"category":"Meat","visit_min":2.2,"price":8.5,"rec_score":0.72, "popularity":0.8},
        "Beef": {"category":"Meat","visit_min":2.5,"price":12.0,"rec_score":0.65, "popularity":0.7},
        "Fish": {"category":"Meat","visit_min":2.0,"price":9.5,"rec_score":0.68, "popularity":0.6},
        
        "Pastries": {"category":"Bakery","visit_min":1.3,"price":4.5,"rec_score":0.55, "popularity":0.4},
        
        "Cereal": {"category":"Grocery","visit_min":1.4,"price":4.8,"rec_score":0.58, "popularity":0.7},
        "Pasta": {"category":"Grocery","visit_min":1.0,"price":2.8,"rec_score":0.62, "popularity":0.75},
        "Rice": {"category":"Grocery","visit_min":1.1,"price":3.5,"rec_score":0.60, "popularity":0.65},
        "Chips": {"category":"Snacks","visit_min":0.8,"price":3.2,"rec_score":0.45, "popularity":0.8},
        "Cookies": {"category":"Snacks","visit_min":0.9,"price":3.8,"rec_score":0.42, "popularity":0.6},
        
        "Coffee": {"category":"Beverages","visit_min":1.5,"price":8.0,"rec_score":0.85, "popularity":0.9},
        "Tea": {"category":"Beverages","visit_min":1.2,"price":5.5,"rec_score":0.55, "popularity":0.5},
        "Soda": {"category":"Beverages","visit_min":0.6,"price":2.5,"rec_score":0.35, "popularity":0.8},
        "Juice": {"category":"Beverages","visit_min":0.8,"price":4.0,"rec_score":0.50, "popularity":0.7},
        "Water": {"category":"Beverages","visit_min":0.5,"price":1.5,"rec_score":0.90, "popularity":0.95},
        
        "Soap": {"category":"Household","visit_min":0.7,"price":3.5,"rec_score":0.40, "popularity":0.3},
        "Detergent": {"category":"Household","visit_min":1.2,"price":7.5,"rec_score":0.35, "popularity":0.25},
        "Shampoo": {"category":"Household","visit_min":1.0,"price":6.0,"rec_score":0.38, "popularity":0.4},
        
        "FrozenPizza": {"category":"Frozen","visit_min":1.3,"price":5.5,"rec_score":0.48, "popularity":0.6},
        "IceCream": {"category":"Frozen","visit_min":1.1,"price":4.8,"rec_score":0.52, "popularity":0.7},
        
        "OliveOil": {"category":"Condiments","visit_min":0.9,"price":7.2,"rec_score":0.58, "popularity":0.45},
        "Vinegar": {"category":"Condiments","visit_min":0.6,"price":3.0,"rec_score":0.25, "popularity":0.2}
    }
    
    # Add extended items
    extended_items = {
        "Salmon": {"category":"Meat","visit_min":2.3,"price":15.0,"rec_score":0.78, "popularity":0.65},
        "Pork": {"category":"Meat","visit_min":2.1,"price":9.8,"rec_score":0.68, "popularity":0.72},
        "Turkey": {"category":"Meat","visit_min":2.4,"price":11.5,"rec_score":0.70, "popularity":0.58},
        "Orange": {"category":"Produce","visit_min":0.9,"price":2.2,"rec_score":0.75, "popularity":0.8},
        "Grapes": {"category":"Produce","visit_min":1.1,"price":3.8,"rec_score":0.68, "popularity":0.7},
        "Spinach": {"category":"Produce","visit_min":0.8,"price":2.5,"rec_score":0.72, "popularity":0.5},
        "Broccoli": {"category":"Produce","visit_min":0.9,"price":2.8,"rec_score":0.69, "popularity":0.55},
        "Onions": {"category":"Produce","visit_min":0.6,"price":1.5,"rec_score":0.58, "popularity":0.85},
        "Potatoes": {"category":"Produce","visit_min":1.0,"price":2.0,"rec_score":0.65, "popularity":0.9},
        "Flour": {"category":"Grocery","visit_min":0.8,"price":2.5,"rec_score":0.45, "popularity":0.4},
        "Sugar": {"category":"Grocery","visit_min":0.7,"price":2.8,"rec_score":0.42, "popularity":0.6},
        "Salt": {"category":"Condiments","visit_min":0.5,"price":1.2,"rec_score":0.38, "popularity":0.95},
        "Pepper": {"category":"Condiments","visit_min":0.5,"price":3.5,"rec_score":0.35, "popularity":0.8},
        "Honey": {"category":"Condiments","visit_min":0.8,"price":6.5,"rec_score":0.55, "popularity":0.45},
        "Jam": {"category":"Condiments","visit_min":0.7,"price":4.2,"rec_score":0.48, "popularity":0.5}
    }
    
    base_items.update(extended_items)
    return base_items

# Initialize layout
nodes, item_positions = create_enhanced_store_layout()
items_meta = create_item_metadata()
item_to_node = {item: item_positions.get(item, "N0") for item in items_meta.keys()}

# Filter items that have valid positions
items_meta = {k: v for k, v in items_meta.items() if k in item_positions}
item_to_node = {item: item_positions[item] for item in items_meta.keys()}

# ---------- Graph and Matrix Setup ----------
def build_enhanced_graph():
    """Build graph with better connectivity and realistic constraints"""
    G = nx.Graph()
    
    for n, coord in nodes.items():
        G.add_node(n, pos=coord)
    
    for u, v in itertools.combinations(nodes.keys(), 2):
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        dist = math.hypot(x1-x2, y1-y2)
        
        connect = False
        if abs(x1-x2) < 0.1 and abs(y1-y2) <= 18:
            connect = True
        elif abs(y1-y2) < 0.1 and abs(x1-x2) <= 12:
            connect = True
        elif dist <= 8.5:
            connect = True
            
        if connect:
            base_time = dist / WALKING_SPEED_M_S
            realistic_time = base_time * (1 + random.uniform(-0.1, 0.1))
            G.add_edge(u, v, distance=dist, time_sec=realistic_time, weight=realistic_time)
    
    for special in ["Entrance", "Checkout"]:
        for n in list(nodes.keys()):
            if n in ("Entrance", "Checkout"):
                continue
            dist = math.hypot(nodes[special][0]-nodes[n][0], nodes[special][1]-nodes[n][1])
            if dist <= 20:
                realistic_time = (dist / WALKING_SPEED_M_S) * (1 + random.uniform(-0.05, 0.05))
                G.add_edge(special, n, distance=dist, time_sec=realistic_time, weight=realistic_time)
    
    return G

def compute_distance_matrices():
    """Compute comprehensive distance and time matrices"""
    all_nodes = list(G.nodes())
    time_matrix = pd.DataFrame(index=all_nodes, columns=all_nodes, dtype=float)
    dist_matrix = pd.DataFrame(index=all_nodes, columns=all_nodes, dtype=float)
    
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                time_matrix.loc[i, j] = 0.0
                dist_matrix.loc[i, j] = 0.0
                continue
                
            try:
                path = nx.shortest_path(G, i, j, weight='time_sec')
                time_sec = sum(G[path[k]][path[k+1]]['time_sec'] for k in range(len(path)-1))
                distance = sum(G[path[k]][path[k+1]]['distance'] for k in range(len(path)-1))
                time_matrix.loc[i, j] = time_sec / 60.0
                dist_matrix.loc[i, j] = distance
            except nx.NetworkXNoPath:
                time_matrix.loc[i, j] = float('inf')
                dist_matrix.loc[i, j] = float('inf')
    
    return time_matrix, dist_matrix

G = build_enhanced_graph()
time_matrix_min, distance_matrix = compute_distance_matrices()

# ---------- Item Selection Strategies ----------
class ItemSelector:
    """Class containing different item selection strategies"""
    
    @staticmethod
    def random_selection(all_items, count, seed=None):
        """Random selection of items"""
        if seed is not None:
            random.seed(seed)
        return random.sample(list(all_items), min(count, len(all_items)))
    
    @staticmethod
    def high_preference_selection(all_items, count, seed=None):
        """Select items with highest recommendation scores"""
        if seed is not None:
            random.seed(seed)
        sorted_items = sorted(all_items, 
                            key=lambda x: items_meta[x]["rec_score"] * items_meta[x]["popularity"], 
                            reverse=True)
        return sorted_items[:count]
    
    @staticmethod
    def balanced_selection(all_items, count, seed=None):
        """Balanced selection considering multiple factors"""
        if seed is not None:
            random.seed(seed)
        
        def balanced_score(item):
            meta = items_meta[item]
            return (meta["rec_score"] * 0.4 + 
                   meta["popularity"] * 0.3 + 
                   (1/max(meta["visit_min"], 0.1)) * 0.2 +
                   (1/max(meta["price"], 1)) * 0.1)
        
        sorted_items = sorted(all_items, key=balanced_score, reverse=True)
        return sorted_items[:count]
    
    @staticmethod
    def category_diverse_selection(all_items, count, seed=None):
        """Select items ensuring category diversity"""
        if seed is not None:
            random.seed(seed)
        
        categories = list(set(items_meta[item]["category"] for item in all_items))
        items_per_category = max(1, count // len(categories))
        selected = []
        
        for category in categories:
            cat_items = [item for item in all_items if items_meta[item]["category"] == category]
            cat_items.sort(key=lambda x: items_meta[x]["rec_score"], reverse=True)
            selected.extend(cat_items[:items_per_category])
            if len(selected) >= count:
                break
        
        # Fill remaining slots with high-preference items
        remaining_count = count - len(selected)
        if remaining_count > 0:
            remaining_items = [item for item in all_items if item not in selected]
            remaining_items.sort(key=lambda x: items_meta[x]["rec_score"], reverse=True)
            selected.extend(remaining_items[:remaining_count])
        
        return selected[:count]
    
    @staticmethod
    def time_efficient_selection(all_items, count, seed=None):
        """Select items that are quick to visit"""
        if seed is not None:
            random.seed(seed)
        
        def efficiency_score(item):
            meta = items_meta[item]
            return meta["rec_score"] / max(meta["visit_min"], 0.1)
        
        sorted_items = sorted(all_items, key=efficiency_score, reverse=True)
        return sorted_items[:count]
    
    @staticmethod
    def mixed_selection(all_items, count, seed=None):
        """Mixed strategy combining different approaches"""
        if seed is not None:
            random.seed(seed)
        
        methods = [
            ItemSelector.random_selection,
            ItemSelector.high_preference_selection,
            ItemSelector.balanced_selection,
            ItemSelector.category_diverse_selection
        ]
        
        items_per_method = max(1, count // len(methods))
        selected = []
        used_items = set()
        
        for method in methods:
            method_items = method(all_items, items_per_method + 2, seed)  # Get a few extra
            new_items = [item for item in method_items if item not in used_items]
            selected.extend(new_items[:items_per_method])
            used_items.update(new_items[:items_per_method])
            if len(selected) >= count:
                break
        
        # Fill remaining slots
        remaining_count = count - len(selected)
        if remaining_count > 0:
            remaining_items = [item for item in all_items if item not in used_items]
            selected.extend(random.sample(remaining_items, min(remaining_count, len(remaining_items))))
        
        return selected[:count]

# ---------- Core Optimization Functions ----------
def compute_path_smoothness(route_nodes):
    """Calculate path smoothness"""
    if len(route_nodes) < 3:
        return 0.0
    
    direction_changes = 0
    for i in range(2, len(route_nodes)):
        p1, p2, p3 = nodes[route_nodes[i-2]], nodes[route_nodes[i-1]], nodes[route_nodes[i]]
        
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
            continue
            
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))
            angle = math.acos(cos_angle)
            
            if angle > math.pi / 6:
                direction_changes += 1
    
    return direction_changes

def enhanced_objective_function(route_nodes, must_items):
    """Enhanced multi-objective optimization function"""
    item_counts = defaultdict(int)
    for nd in route_nodes:
        for item, item_node in item_to_node.items():
            if item_node == nd:
                item_counts[item] += 1
    
    missing_musts = [m for m in must_items if item_to_node[m] not in route_nodes]
    if missing_musts:
        return 0.0
    
    total_time = compute_route_time_minutes(route_nodes)
    time_efficiency = max(0, 1 - (total_time / TIME_BUDGET_MIN)) if total_time <= TIME_BUDGET_MIN else 0
    
    pref_satisfaction = sum(
        items_meta[item]["rec_score"] * items_meta[item]["popularity"] * count
        for item, count in item_counts.items()
    ) / max(total_time, 1)
    
    unique_items = len(item_counts)
    diversity_bonus = COUNT_BONUS * unique_items
    
    smoothness_penalty = DIRECTION_CHANGE_PENALTY * compute_path_smoothness(route_nodes)
    revisit_penalty = sum(max(0, count - 1) for count in item_counts.values()) * REVISIT_PENALTY_PER
    
    objective = (
        PREF_WEIGHT * pref_satisfaction +
        TIME_EFFICIENCY_WEIGHT * time_efficiency +
        diversity_bonus -
        smoothness_penalty -
        revisit_penalty
    )
    
    return max(0, objective)

def compute_route_time_minutes(route_nodes):
    """Enhanced route time calculation"""
    if len(route_nodes) < 2:
        return 0.0
    
    total_time = 0.0
    item_visit_counts = defaultdict(int)
    
    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        total_time += float(time_matrix_min.loc[u, v])
        
        for item, item_node in item_to_node.items():
            if item_node == v:
                visit_time = items_meta[item]["visit_min"] + VISIT_OVERHEAD_SEC / 60.0
                crowd_factor = 1 + (items_meta[item]["popularity"] * 0.1)
                total_time += visit_time * crowd_factor
                item_visit_counts[item] += 1
    
    revisit_count = sum(max(0, count - 1) for count in item_visit_counts.values())
    total_time += revisit_count * REVISIT_TIME_PENALTY_MIN
    
    return total_time

def enhanced_two_opt_with_or_opt(waypoint_sequence, max_iterations=200):
    """Enhanced 2-opt with Or-opt moves (reduced iterations for benchmarking)"""
    def route_cost(sequence):
        return sum(time_matrix_min.loc[sequence[i], sequence[i+1]] 
                  for i in range(len(sequence)-1))
    
    current = waypoint_sequence[:]
    current_cost = route_cost(current)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # 2-opt moves
        for i in range(1, len(current) - 2):
            for j in range(i + 1, len(current) - 1):
                if j - i == 1:
                    continue
                
                new_sequence = (current[:i] + 
                               current[i:j+1][::-1] + 
                               current[j+1:])
                new_cost = route_cost(new_sequence)
                
                if new_cost < current_cost - 1e-9:
                    current = new_sequence
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    
    return current

def nodes_from_waypoints(waypoint_sequence):
    """Convert waypoint sequence to full node path"""
    if len(waypoint_sequence) < 2:
        return waypoint_sequence
    
    full_path = [waypoint_sequence[0]]
    for i in range(len(waypoint_sequence) - 1):
        try:
            segment = nx.shortest_path(G, waypoint_sequence[i], waypoint_sequence[i + 1], weight='time_sec')
            full_path.extend(segment[1:])
        except nx.NetworkXNoPath:
            full_path.append(waypoint_sequence[i + 1])
    
    return full_path

def optimize_route(must_items):
    """Streamlined route optimization for benchmarking"""
    # Build initial route using nearest neighbor
    waypoints = [item_to_node[item] for item in must_items]
    current = "Entrance"
    route = ["Entrance"]
    remaining = waypoints[:]
    
    while remaining:
        nearest = min(remaining, key=lambda x: time_matrix_min.loc[current, x])
        route.append(nearest)
        current = nearest
        remaining.remove(nearest)
    
    route.append("Checkout")
    
    # Optimize waypoints
    optimized_waypoints = enhanced_two_opt_with_or_opt(route)
    optimized_route = nodes_from_waypoints(optimized_waypoints)
    
    return optimized_route

# ---------- Benchmarking System ----------
# def run_single_benchmark(item_count, selection_method, seed):
#     """Run a single benchmark iteration"""
#     # Set seed for reproducibility
#     random.seed(seed)
#     np.random.seed(seed)
    
#     # Select items using the specified method
#     all_items = list(items_meta.keys())
#     selector_methods = {
#         'random': ItemSelector.random_selection,
#         'high_preference': ItemSelector.high_preference_selection,
#         'balanced': ItemSelector.balanced_selection,
#         'category_diverse': ItemSelector.category_diverse_selection,
#         'time_efficient': ItemSelector.time_efficient_selection,
#         'mixed': ItemSelector.mixed_selection
#     }
    
#     selected_items = selector_methods[selection_method](all_items, item_count, seed)
    
#     # Ensure we have valid items
#     if not selected_items:
#         return None
    
#     # Record start time for performance measurement
#     start_time = time.time()
    
#     # Optimize route
#     try:
#         optimized_route = optimize_route(selected_items)
#         optimization_time = time.time() - start_time
        
#         # Calculate metrics
#         route_time = compute_route_time_minutes(optimized_route)
#         objective_score = enhanced_objective_function(optimized_route, selected_items)
#         path_smoothness = compute_path_smoothness(optimized_route)
        
#         # Count visited items
#         node_to_item = {v: k for k, v in item_to_node.items()}
#         visited_items = [node_to_item[n] for n in optimized_route if n in node_to_item]
#         unique_visited = len(set(visited_items))
        
#         # Category analysis
#         visited_categories = set(items_meta[item]["category"] for item in visited_items)
#         category_diversity = len(visited_categories)
        
#         # Calculate efficiency metrics
#         time_efficiency = max(0, 1 - (route_time / TIME_BUDGET_MIN)) if route_time <= TIME_BUDGET_MIN else 0
#         items_per_minute = unique_visited / max(route_time, 0.1)
        
#         # Cost analysis
#         total_cost = sum(items_meta[item]["price"] for item in visited_items)
#         avg_recommendation_score = np.mean([items_meta[item]["rec_score"] for item in visited_items])
        
#         # Return comprehensive metrics
#         return {
#             'item_count': item_count,
#             'selection_method': selection_method,
#             'seed': seed,
#             'selected_items': selected_items,
#             'visited_items': visited_items,
#             'route_time_min': route_time,
#             'objective_score': objective_score,
#             'path_smoothness': path_smoothness,
#             'optimization_time_sec': optimization_time,
#             'unique_items_visited': unique_visited,
#             'category_diversity': category_diversity,
#             'time_efficiency': time_efficiency,
#             'items_per_minute': items_per_minute,
#             'total_cost': total_cost,
#             'avg_recommendation_score': avg_recommendation_score,
#             'route_length_nodes': len(optimized_route),
#             'success': True
#         }
    
#     except Exception as e:
#         flog(f"Error in benchmark (items={item_count}, method={selection_method}, seed={seed}): {str(e)}", "WARN")
#         return {
#             'item_count': item_count,
#             'selection_method': selection_method,
#             'seed': seed,
#             'error': str(e),
#             'success': False
#         }

def run_single_benchmark(item_count, selection_method, seed):
    """Run a single benchmark iteration"""
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Select items using the specified method
    all_items = list(items_meta.keys())
    selector_methods = {
        'random': ItemSelector.random_selection,
        'high_preference': ItemSelector.high_preference_selection,
        'balanced': ItemSelector.balanced_selection,
        'category_diverse': ItemSelector.category_diverse_selection,
        'time_efficient': ItemSelector.time_efficient_selection,
        'mixed': ItemSelector.mixed_selection
    }
    
    selected_items = selector_methods[selection_method](all_items, item_count, seed)
    
    # Ensure we have valid items
    if not selected_items:
        return None
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Optimize route
    try:
        optimized_route = optimize_route(selected_items)
        optimization_time = time.time() - start_time
        
        # Calculate metrics
        route_time = compute_route_time_minutes(optimized_route)
        objective_score = enhanced_objective_function(optimized_route, selected_items)
        path_smoothness = compute_path_smoothness(optimized_route)
        
        # Movement and distance metrics
        movement_time = 0.0
        walk_distance_total = 0.0
        for i in range(len(optimized_route) - 1):
            u, v = optimized_route[i], optimized_route[i + 1]
            movement_time += float(time_matrix_min.loc[u, v])
            walk_distance_total += float(distance_matrix.loc[u, v])
        
        # Count visited items
        node_to_item = {v: k for k, v in item_to_node.items()}
        visited_items = [node_to_item[n] for n in optimized_route if n in node_to_item]
        unique_visited = len(set(visited_items))
        
        # Category analysis
        visited_categories = set(items_meta[item]["category"] for item in visited_items)
        category_diversity = len(visited_categories)
        
        # Calculate efficiency metrics
        time_efficiency = max(0, 1 - (route_time / TIME_BUDGET_MIN)) if route_time <= TIME_BUDGET_MIN else 0
        items_per_minute = unique_visited / max(route_time, 0.1)
        
        # Cost analysis
        total_cost = sum(items_meta[item]["price"] for item in visited_items)
        avg_recommendation_score = np.mean([items_meta[item]["rec_score"] for item in visited_items]) if visited_items else 0.0
        
        # Simple extra metrics (non-invasive):
        visit_time_total = max(0.0, route_time - movement_time)
        avg_visit_time_per_item = visit_time_total / max(unique_visited, 1)
        revisit_rate = (len(visited_items) - unique_visited) / max(len(visited_items), 1)
        budget_overrun = max(0.0, route_time - TIME_BUDGET_MIN)
        
        # Additional simple metrics to add (kept non-invasive)
        # route_distance_m: total walking distance along the optimized route
        total_walk_distance = 0.0
        for i in range(len(optimized_route) - 1):
            u, v = optimized_route[i], optimized_route[i + 1]
            try:
                total_walk_distance += float(distance_matrix.loc[u, v])
            except Exception:
                total_walk_distance += 0.0
        
        # total_visit_time_min: sum of per-item visit times (incl. overhead & simple crowd factor)
        total_visit_time = 0.0
        for item in visited_items:
            vt = items_meta[item]["visit_min"] + VISIT_OVERHEAD_SEC / 60.0
            crowd_factor = 1 + (items_meta[item]["popularity"] * 0.1)
            total_visit_time += vt * crowd_factor
        
        avg_visit_time_per_item_min = total_visit_time / max(unique_visited, 1)
        percent_time_budget_used = route_time / TIME_BUDGET_MIN * 100.0
        
        # high_pref_items_pct: percent of visited items with rec_score >= 0.75
        high_pref_count = sum(1 for it in visited_items if items_meta[it]["rec_score"] >= 0.75)
        high_pref_items_pct = high_pref_count / max(unique_visited, 1)
        
        # Return comprehensive metrics (keeps original keys; adds the new ones)
        return {
            'item_count': item_count,
            'selection_method': selection_method,
            'seed': seed,
            'selected_items': selected_items,
            'visited_items': visited_items,
            'route_time_min': route_time,
            'objective_score': objective_score,
            'path_smoothness': path_smoothness,
            'optimization_time_sec': optimization_time,
            'unique_items_visited': unique_visited,
            'category_diversity': category_diversity,
            'time_efficiency': time_efficiency,
            'items_per_minute': items_per_minute,
            'total_cost': total_cost,
            'avg_recommendation_score': avg_recommendation_score,
            'route_length_nodes': len(optimized_route),
            # --- NEW METRICS ---
            'movement_time_min': movement_time,
            'walk_distance_total': walk_distance_total,
            'visit_time_total_min': visit_time_total,
            'avg_visit_time_per_item': avg_visit_time_per_item,
            'revisit_rate': revisit_rate,
            'budget_overrun_min': budget_overrun,
            'route_distance_m': total_walk_distance,
            'total_visit_time_min': total_visit_time,
            'avg_visit_time_per_item_min': avg_visit_time_per_item_min,
            'percent_time_budget_used': percent_time_budget_used,
            'high_pref_items_pct': high_pref_items_pct,
            'success': True
        }
    
    except Exception as e:
        flog(f"Error in benchmark (items={item_count}, method={selection_method}, seed={seed}): {str(e)}", "WARN")
        return {
            'item_count': item_count,
            'selection_method': selection_method,
            'seed': seed,
            'error': str(e),
            'success': False
        }


def run_comprehensive_benchmark():
    """Run comprehensive benchmarking across all configurations"""
    flog("Starting comprehensive benchmarking system...", "FUN")
    flog("=" * 80, "INFO")
    
    all_results = []
    total_iterations = (len(BENCHMARK_CONFIG['item_counts']) * 
                       len(BENCHMARK_CONFIG['item_selection_methods']) * 
                       BENCHMARK_CONFIG['iterations_per_config'])
    
    flog(f"Configuration:", "BENCH")
    flog(f"  Item counts: {BENCHMARK_CONFIG['item_counts']}", "INFO")
    flog(f"  Selection methods: {BENCHMARK_CONFIG['item_selection_methods']}", "INFO")
    flog(f"  Iterations per config: {BENCHMARK_CONFIG['iterations_per_config']}", "INFO")
    flog(f"  Total iterations: {total_iterations:,}", "INFO")
    
    current_iteration = 0
    start_time = time.time()
    
    # Run benchmarks
    for item_count in BENCHMARK_CONFIG['item_counts']:
        for method in BENCHMARK_CONFIG['item_selection_methods']:
            method_start_time = time.time()
            method_results = []
            
            flog(f"Running {BENCHMARK_CONFIG['iterations_per_config']} iterations: {item_count} items, {method} method", "STEP")
            
            for i in range(BENCHMARK_CONFIG['iterations_per_config']):
                seed = BENCHMARK_CONFIG['random_seeds'][i]
                result = run_single_benchmark(item_count, method, seed)
                
                if result:
                    all_results.append(result)
                    method_results.append(result)
                
                current_iteration += 1
                
                # Progress reporting
                if current_iteration % 50 == 0:
                    elapsed = time.time() - start_time
                    progress = current_iteration / total_iterations * 100
                    eta = elapsed / current_iteration * (total_iterations - current_iteration)
                    flog(f"Progress: {progress:.1f}% ({current_iteration:,}/{total_iterations:,}) - ETA: {eta/60:.1f}min", "BENCH")
            
            # Method summary
            if method_results:
                successful = [r for r in method_results if r['success']]
                if successful:
                    avg_time = np.mean([r['route_time_min'] for r in successful])
                    avg_score = np.mean([r['objective_score'] for r in successful])
                    method_duration = time.time() - method_start_time
                    flog(f"  {method} method completed: {len(successful)}/{len(method_results)} successful, "
                         f"avg_time={avg_time:.2f}min, avg_score={avg_score:.4f} ({method_duration:.1f}s)", "METRIC")
    
    total_time = time.time() - start_time
    successful_results = [r for r in all_results if r['success']]
    
    flog("=" * 80, "DONE")
    flog(f"Benchmarking completed!", "FUN")
    flog(f"  Total time: {total_time/60:.1f} minutes", "METRIC")
    flog(f"  Successful iterations: {len(successful_results):,}/{len(all_results):,} ({len(successful_results)/len(all_results)*100:.1f}%)", "METRIC")
    
    return successful_results

# def create_comprehensive_analysis(results):
#     """Create comprehensive analysis and visualizations"""
#     flog("Creating comprehensive analysis...", "STEP")
    
#     # Convert to DataFrame
#     df = pd.DataFrame(results)
    
#     # Save raw results
#     raw_results_file = os.path.join(OUT_DIR, "raw_benchmark_results.csv")
#     df.to_csv(raw_results_file, index=False)
#     flog(f"Raw results saved: {raw_results_file}", "DONE")
    
#     # Statistical analysis
#     stats_by_method = df.groupby(['item_count', 'selection_method']).agg({
#         'route_time_min': ['mean', 'std', 'min', 'max', 'median'],
#         'objective_score': ['mean', 'std', 'min', 'max', 'median'],
#         'unique_items_visited': ['mean', 'std', 'min', 'max', 'median'],
#         'time_efficiency': ['mean', 'std', 'min', 'max', 'median'],
#         'items_per_minute': ['mean', 'std', 'min', 'max', 'median'],
#         'category_diversity': ['mean', 'std', 'min', 'max', 'median'],
#         'optimization_time_sec': ['mean', 'std', 'min', 'max', 'median'],
#         'total_cost': ['mean', 'std', 'min', 'max', 'median'],
#         'avg_recommendation_score': ['mean', 'std', 'min', 'max', 'median']
#     }).round(4)
    
#     # Flatten column names
#     stats_by_method.columns = ['_'.join(col).strip() for col in stats_by_method.columns.values]
#     stats_file = os.path.join(OUT_DIR, "statistical_analysis.csv")
#     stats_by_method.to_csv(stats_file)
#     flog(f"Statistical analysis saved: {stats_file}", "DONE")
    
#     # Create visualizations
#     create_performance_visualizations(df)
#     create_method_comparison_analysis(df)
#     create_scalability_analysis(df)
#     create_efficiency_analysis(df)
    
#     return df, stats_by_method

def create_comprehensive_analysis(results):
    """Create comprehensive analysis and visualizations"""
    flog("Creating comprehensive analysis...", "STEP")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    raw_results_file = os.path.join(OUT_DIR, "raw_benchmark_results.csv")
    df.to_csv(raw_results_file, index=False)
    flog(f"Raw results saved: {raw_results_file}", "DONE")
    
    # Statistical analysis (includes the new simple metrics)
    stats_by_method = df.groupby(['item_count', 'selection_method']).agg({
        'route_time_min': ['mean', 'std', 'min', 'max', 'median'],
        'objective_score': ['mean', 'std', 'min', 'max', 'median'],
        'unique_items_visited': ['mean', 'std', 'min', 'max', 'median'],
        'time_efficiency': ['mean', 'std', 'min', 'max', 'median'],
        'items_per_minute': ['mean', 'std', 'min', 'max', 'median'],
        'category_diversity': ['mean', 'std', 'min', 'max', 'median'],
        'optimization_time_sec': ['mean', 'std', 'min', 'max', 'median'],
        'total_cost': ['mean', 'std', 'min', 'max', 'median'],
        'avg_recommendation_score': ['mean', 'std', 'min', 'max', 'median'],
        # ---- new aggregated fields ----
        'route_distance_m': ['mean', 'std', 'min', 'max', 'median'],
        'total_visit_time_min': ['mean', 'std', 'min', 'max', 'median'],
        'avg_visit_time_per_item_min': ['mean', 'std', 'min', 'max', 'median'],
        'revisit_rate': ['mean', 'std', 'min', 'max', 'median'],
        'percent_time_budget_used': ['mean', 'std', 'min', 'max', 'median'],
        'high_pref_items_pct': ['mean', 'std', 'min', 'max', 'median'],
        'movement_time_min': ['mean', 'std', 'min', 'max', 'median'],
        'walk_distance_total': ['mean', 'std', 'min', 'max', 'median'],
        'visit_time_total_min': ['mean', 'std', 'min', 'max', 'median'],
        'avg_visit_time_per_item': ['mean', 'std', 'min', 'max', 'median'],
        'budget_overrun_min': ['mean', 'std', 'min', 'max', 'median']
    }).round(4)
    
    # Flatten column names
    stats_by_method.columns = ['_'.join(col).strip() for col in stats_by_method.columns.values]
    stats_file = os.path.join(OUT_DIR, "statistical_analysis.csv")
    stats_by_method.to_csv(stats_file)
    flog(f"Statistical analysis saved: {stats_file}", "DONE")
    
    # Create visualizations
    create_performance_visualizations(df)
    create_method_comparison_analysis(df)
    create_scalability_analysis(df)
    create_efficiency_analysis(df)
    
    return df, stats_by_method


def create_performance_visualizations(df):
    """Create comprehensive performance visualizations"""
    flog("Creating performance visualizations...", "STEP")
    
    # 1. Route Time by Item Count and Method
    plt.figure(figsize=(16, 10))
    
    # Main plot
    plt.subplot(2, 2, 1)
    methods = df['selection_method'].unique()
    colors = sns.color_palette("husl", len(methods))
    
    for i, method in enumerate(methods):
        method_data = df[df['selection_method'] == method]
        grouped = method_data.groupby('item_count')['route_time_min'].agg(['mean', 'std'])
        
        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                    marker='o', linewidth=2, markersize=6, label=method, color=colors[i])
    
    plt.xlabel('Number of Items', fontsize=12)
    plt.ylabel('Average Route Time (minutes)', fontsize=12)
    plt.title('Route Time vs Item Count by Selection Method', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. Objective Score Distribution
    plt.subplot(2, 2, 2)
    df.boxplot(column='objective_score', by='selection_method', ax=plt.gca())
    plt.title('Objective Score Distribution by Method')
    plt.xlabel('Selection Method')
    plt.ylabel('Objective Score')
    plt.xticks(rotation=45)
    
    # 3. Time Efficiency vs Items Collected
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(df['unique_items_visited'], df['time_efficiency'], 
                         c=df['item_count'], cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Planned Item Count')
    plt.xlabel('Unique Items Visited')
    plt.ylabel('Time Efficiency')
    plt.title('Time Efficiency vs Items Collected')
    
    # 4. Method Performance Heatmap
    plt.subplot(2, 2, 4)
    pivot_data = df.groupby(['item_count', 'selection_method'])['objective_score'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Avg Objective Score'})
    plt.title('Average Objective Score Heatmap')
    plt.xlabel('Selection Method')
    plt.ylabel('Item Count')
    
    plt.tight_layout()
    viz_file = os.path.join(OUT_DIR, "performance_overview.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Performance visualization saved: {viz_file}", "DONE")

def create_method_comparison_analysis(df):
    """Create detailed method comparison analysis"""
    flog("Creating method comparison analysis...", "STEP")
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    methods = df['selection_method'].unique()
    
    # 1. Route Time Comparison
    ax = axes[0, 0]
    df.boxplot(column='route_time_min', by='selection_method', ax=ax)
    ax.set_title('Route Time Distribution by Method')
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Route Time (minutes)')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Items per Minute Efficiency
    ax = axes[0, 1]
    df.boxplot(column='items_per_minute', by='selection_method', ax=ax)
    ax.set_title('Items per Minute Efficiency')
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Items per Minute')
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Category Diversity
    ax = axes[1, 0]
    df.boxplot(column='category_diversity', by='selection_method', ax=ax)
    ax.set_title('Category Diversity')
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Number of Categories')
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Optimization Time
    ax = axes[1, 1]
    df.boxplot(column='optimization_time_sec', by='selection_method', ax=ax)
    ax.set_title('Optimization Time')
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    
    # 5. Total Cost Analysis
    ax = axes[2, 0]
    df.boxplot(column='total_cost', by='selection_method', ax=ax)
    ax.set_title('Total Shopping Cost')
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Total Cost ($)')
    ax.tick_params(axis='x', rotation=45)
    
    # 6. Recommendation Quality
    ax = axes[2, 1]
    df.boxplot(column='avg_recommendation_score', by='selection_method', ax=ax)
    ax.set_title('Average Recommendation Score')
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Recommendation Score')
    ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Comprehensive Method Comparison Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    method_file = os.path.join(OUT_DIR, "method_comparison.png")
    plt.savefig(method_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Method comparison saved: {method_file}", "DONE")

def create_scalability_analysis(df):
    """Create scalability analysis"""
    flog("Creating scalability analysis...", "STEP")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Time Complexity Analysis
    ax = axes[0, 0]
    item_counts = sorted(df['item_count'].unique())
    
    for method in df['selection_method'].unique():
        method_data = df[df['selection_method'] == method]
        opt_times = []
        std_errors = []
        
        for count in item_counts:
            times = method_data[method_data['item_count'] == count]['optimization_time_sec']
            opt_times.append(times.mean())
            std_errors.append(times.std())
        
        ax.errorbar(item_counts, opt_times, yerr=std_errors, marker='o', 
                   linewidth=2, label=method, markersize=6)
    
    ax.set_xlabel('Number of Items')
    ax.set_ylabel('Optimization Time (seconds)')
    ax.set_title('Optimization Time Scalability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Route Quality vs Problem Size
    ax = axes[0, 1]
    for method in df['selection_method'].unique():
        method_data = df[df['selection_method'] == method]
        quality_by_size = method_data.groupby('item_count')['objective_score'].mean()
        ax.plot(quality_by_size.index, quality_by_size.values, marker='o', 
               linewidth=2, label=method, markersize=6)
    
    ax.set_xlabel('Number of Items')
    ax.set_ylabel('Average Objective Score')
    ax.set_title('Route Quality vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Success Rate by Item Count
    ax = axes[1, 0]
    success_rates = []
    for count in item_counts:
        count_data = df[df['item_count'] == count]
        success_rate = len(count_data) / (len(df['selection_method'].unique()) * BENCHMARK_CONFIG['iterations_per_config'])
        success_rates.append(success_rate * 100)
    
    ax.bar(item_counts, success_rates, alpha=0.7, color='steelblue')
    ax.set_xlabel('Number of Items')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Problem Size')
    ax.grid(True, alpha=0.3)
    
    # 4. Items Visited vs Items Planned
    ax = axes[1, 1]
    ax.scatter(df['item_count'], df['unique_items_visited'], alpha=0.5, s=20)
    # Add diagonal line for perfect match
    max_items = max(df['item_count'].max(), df['unique_items_visited'].max())
    ax.plot([0, max_items], [0, max_items], 'r--', alpha=0.8, label='Perfect Match')
    ax.set_xlabel('Planned Items')
    ax.set_ylabel('Items Actually Visited')
    ax.set_title('Planning vs Execution Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Scalability and Efficiency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    scale_file = os.path.join(OUT_DIR, "scalability_analysis.png")
    plt.savefig(scale_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Scalability analysis saved: {scale_file}", "DONE")

def create_efficiency_analysis(df):
    """Create efficiency analysis"""
    flog("Creating efficiency analysis...", "STEP")
    
    plt.figure(figsize=(20, 12))
    
    # 1. Pareto Frontier Analysis
    plt.subplot(2, 3, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    method_colors = dict(zip(df['selection_method'].unique(), colors))
    
    for method in df['selection_method'].unique():
        method_data = df[df['selection_method'] == method]
        plt.scatter(method_data['route_time_min'], method_data['unique_items_visited'], 
                   alpha=0.6, label=method, color=method_colors[method], s=30)
    
    plt.xlabel('Route Time (minutes)')
    plt.ylabel('Items Visited')
    plt.title('Time vs Items Pareto Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Cost-Benefit Analysis
    plt.subplot(2, 3, 2)
    for method in df['selection_method'].unique():
        method_data = df[df['selection_method'] == method]
        plt.scatter(method_data['total_cost'], method_data['avg_recommendation_score'], 
                   alpha=0.6, label=method, color=method_colors[method], s=30)
    
    plt.xlabel('Total Cost ($)')
    plt.ylabel('Avg Recommendation Score')
    plt.title('Cost vs Quality Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Method Ranking by Item Count
    plt.subplot(2, 3, 3)
    ranking_data = []
    for count in sorted(df['item_count'].unique()):
        count_data = df[df['item_count'] == count]
        method_scores = count_data.groupby('selection_method')['objective_score'].mean().sort_values(ascending=False)
        for rank, (method, score) in enumerate(method_scores.items()):
            ranking_data.append({'item_count': count, 'method': method, 'rank': rank + 1, 'score': score})
    
    rank_df = pd.DataFrame(ranking_data)
    pivot_ranks = rank_df.pivot(index='item_count', columns='method', values='rank')
    
    for method in pivot_ranks.columns:
        plt.plot(pivot_ranks.index, pivot_ranks[method], marker='o', 
                linewidth=2, label=method, markersize=6)
    
    plt.xlabel('Number of Items')
    plt.ylabel('Method Rank (1=Best)')
    plt.title('Method Performance Ranking')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 4. Efficiency Distribution
    plt.subplot(2, 3, 4)
    df['efficiency_score'] = df['unique_items_visited'] / df['route_time_min']
    df.boxplot(column='efficiency_score', by='selection_method', ax=plt.gca())
    plt.title('Overall Efficiency Distribution')
    plt.xlabel('Selection Method')
    plt.ylabel('Items per Minute')
    plt.xticks(rotation=45)
    
    # 5. Time Budget Utilization
    plt.subplot(2, 3, 5)
    df['time_utilization'] = df['route_time_min'] / TIME_BUDGET_MIN * 100
    df.boxplot(column='time_utilization', by='selection_method', ax=plt.gca())
    plt.title('Time Budget Utilization')
    plt.xlabel('Selection Method')
    plt.ylabel('Budget Utilization (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Budget Limit')
    
    # 6. Quality vs Speed Trade-off
    plt.subplot(2, 3, 6)
    for method in df['selection_method'].unique():
        method_data = df[df['selection_method'] == method]
        plt.scatter(method_data['optimization_time_sec'], method_data['objective_score'], 
                   alpha=0.6, label=method, color=method_colors[method], s=30)
    
    plt.xlabel('Optimization Time (seconds)')
    plt.ylabel('Objective Score')
    plt.title('Quality vs Optimization Speed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Efficiency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    eff_file = os.path.join(OUT_DIR, "efficiency_analysis.png")
    plt.savefig(eff_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Efficiency analysis saved: {eff_file}", "DONE")

# def create_executive_summary(df, stats_by_method):
#     """Create executive summary with key findings"""
#     flog("Creating executive summary...", "STEP")
    
#     # Key metrics calculation
#     best_method_overall = str(df.groupby('selection_method')['objective_score'].mean().idxmax())
#     fastest_method = str(df.groupby('selection_method')['optimization_time_sec'].mean().idxmin())
#     most_efficient = str(df.groupby('selection_method')['items_per_minute'].mean().idxmax())
    
#     # Method performance by item count
#     method_performance = {}
#     for count in sorted(df['item_count'].unique()):
#         count_data = df[df['item_count'] == count]
#         best_for_count = str(count_data.groupby('selection_method')['objective_score'].mean().idxmax())
#         method_performance[str(count)] = best_for_count
    
#     # Statistical significance testing
#     try:
#         from scipy.stats import kruskal
#         methods = df['selection_method'].unique()
#         method_groups = [df[df['selection_method'] == method]['objective_score'] for method in methods]
#         kruskal_stat, kruskal_p = kruskal(*method_groups)
#         statistical_significance = bool(kruskal_p < 0.05)
#         kruskal_p_value = float(kruskal_p)
#     except ImportError:
#         flog("scipy not available, skipping statistical significance test", "WARN")
#         statistical_significance = False
#         kruskal_p_value = 1.0
    
#     # Create summary report - ensuring all values are JSON serializable
#     summary = {
#         'benchmark_overview': {
#             'total_iterations': int(len(df)),
#             'item_count_range': f"{int(df['item_count'].min())}-{int(df['item_count'].max())}",
#             'methods_tested': [str(method) for method in df['selection_method'].unique()],
#             'success_rate': float(len(df) / (len(BENCHMARK_CONFIG['item_counts']) * 
#                                      len(BENCHMARK_CONFIG['item_selection_methods']) * 
#                                      BENCHMARK_CONFIG['iterations_per_config']) * 100)
#         },
#         'key_findings': {
#             'best_overall_method': best_method_overall,
#             'fastest_optimization': fastest_method,
#             'most_item_efficient': most_efficient,
#             'statistical_significance': statistical_significance,
#             'kruskal_p_value': kruskal_p_value
#         },
#         'performance_by_size': method_performance,
#         'average_metrics': {
#             'route_time_min': float(df['route_time_min'].mean()),
#             'items_visited': float(df['unique_items_visited'].mean()),
#             'optimization_time_sec': float(df['optimization_time_sec'].mean()),
#             'objective_score': float(df['objective_score'].mean())
#         },
#         'method_comparison': {}
#     }
    
#     # Detailed method comparison - ensuring all values are JSON serializable
#     for method in methods:
#         method_data = df[df['selection_method'] == method]
#         summary['method_comparison'][str(method)] = {
#             'avg_route_time': float(method_data['route_time_min'].mean()),
#             'avg_items_visited': float(method_data['unique_items_visited'].mean()),
#             'avg_objective_score': float(method_data['objective_score'].mean()),
#             'avg_optimization_time': float(method_data['optimization_time_sec'].mean()),
#             'success_rate': float(len(method_data) / BENCHMARK_CONFIG['iterations_per_config'] / len(BENCHMARK_CONFIG['item_counts']) * 100)
#         }
    
#     # Save summary with proper error handling
#     summary_file = os.path.join(OUT_DIR, "executive_summary.json")
#     try:
#         with open(summary_file, 'w') as f:
#             json.dump(summary, f, indent=2, default=str)  # Added default=str for any remaining type issues
#         flog(f"Executive summary saved: {summary_file}", "DONE")
#     except Exception as e:
#         flog(f"Error saving JSON summary: {e}", "WARN")
#         # Save as text instead
#         text_summary_file = os.path.join(OUT_DIR, "executive_summary.txt")
#         with open(text_summary_file, 'w') as f:
#             f.write("EXECUTIVE SUMMARY\n")
#             f.write("="*50 + "\n\n")
#             f.write(f"Best Overall Method: {best_method_overall}\n")
#             f.write(f"Fastest Method: {fastest_method}\n")
#             f.write(f"Most Efficient: {most_efficient}\n")
#             f.write(f"Total Iterations: {len(df)}\n")
#             f.write(f"Statistical Significance: {statistical_significance}\n")
#         flog(f"Text summary saved instead: {text_summary_file}", "DONE")
    
#     # Print key findings
#     print("\n" + "="*80)
#     print("🎯 EXECUTIVE SUMMARY - KEY FINDINGS")
#     print("="*80)
#     print(f"📊 Total Benchmark Iterations: {len(df):,}")
#     print(f"📈 Success Rate: {summary['benchmark_overview']['success_rate']:.1f}%")
#     print(f"🏆 Best Overall Method: {best_method_overall}")
#     print(f"⚡ Fastest Optimization: {fastest_method}")
#     print(f"🎯 Most Item Efficient: {most_efficient}")
#     print(f"📊 Statistical Significance: {'Yes' if statistical_significance else 'No'} (p={kruskal_p_value:.4f})")
    
#     print(f"\n📈 AVERAGE PERFORMANCE METRICS:")
#     print(f"  Route Time: {summary['average_metrics']['route_time_min']:.2f} minutes")
#     print(f"  Items Visited: {summary['average_metrics']['items_visited']:.1f}")
#     print(f"  Optimization Time: {summary['average_metrics']['optimization_time_sec']:.3f} seconds")
#     print(f"  Objective Score: {summary['average_metrics']['objective_score']:.4f}")
    
#     print(f"\n🏅 METHOD RANKINGS BY ITEM COUNT:")
#     for count, method in method_performance.items():
#         print(f"  {count} items: {method}")
    
#     print("="*80)
    
#     return summary

def create_executive_summary(df, stats_by_method):
    """Create executive summary with key findings"""
    flog("Creating executive summary...", "STEP")
    
    # Key metrics calculation
    best_method_overall = str(df.groupby('selection_method')['objective_score'].mean().idxmax())
    fastest_method = str(df.groupby('selection_method')['optimization_time_sec'].mean().idxmin())
    most_efficient = str(df.groupby('selection_method')['items_per_minute'].mean().idxmax())
    
    methods = df['selection_method'].unique()
    
    # Method performance by item count
    method_performance = {}
    for count in sorted(df['item_count'].unique()):
        count_data = df[df['item_count'] == count]
        best_for_count = str(count_data.groupby('selection_method')['objective_score'].mean().idxmax())
        method_performance[str(count)] = best_for_count
    
    # Statistical significance testing
    try:
        from scipy.stats import kruskal
        method_groups = [df[df['selection_method'] == method]['objective_score'] for method in methods]
        kruskal_stat, kruskal_p = kruskal(*method_groups)
        statistical_significance = bool(kruskal_p < 0.05)
        kruskal_p_value = float(kruskal_p)
    except Exception:
        flog("scipy not available or test failed, skipping statistical significance test", "WARN")
        statistical_significance = False
        kruskal_p_value = 1.0
    
    # Create summary report - ensuring all values are JSON serializable
    summary = {
        'benchmark_overview': {
            'total_iterations': int(len(df)),
            'item_count_range': f"{int(df['item_count'].min())}-{int(df['item_count'].max())}",
            'methods_tested': [str(method) for method in methods],
            'success_rate': float(len(df) / (len(BENCHMARK_CONFIG['item_counts']) * 
                                     len(BENCHMARK_CONFIG['item_selection_methods']) * 
                                     BENCHMARK_CONFIG['iterations_per_config']) * 100)
        },
        'key_findings': {
            'best_overall_method': best_method_overall,
            'fastest_optimization': fastest_method,
            'most_item_efficient': most_efficient,
            'statistical_significance': statistical_significance,
            'kruskal_p_value': kruskal_p_value
        },
        'performance_by_size': method_performance,
        'average_metrics': {
            'route_time_min': float(df['route_time_min'].mean()),
            'items_visited': float(df['unique_items_visited'].mean()),
            'optimization_time_sec': float(df['optimization_time_sec'].mean()),
            'objective_score': float(df['objective_score'].mean()),
            # Include the new simple averages
            'avg_walk_distance_m': float(df['route_distance_m'].mean()) if 'route_distance_m' in df.columns else None,
            'avg_movement_time_min': float(df['movement_time_min'].mean()) if 'movement_time_min' in df.columns else None,
            'avg_visit_time_per_item_min': float(df['avg_visit_time_per_item_min'].mean()) if 'avg_visit_time_per_item_min' in df.columns else None,
            'avg_revisit_rate': float(df['revisit_rate'].mean()) if 'revisit_rate' in df.columns else None,
            'avg_budget_overrun_min': float(df['budget_overrun_min'].mean()) if 'budget_overrun_min' in df.columns else None
        },
        'method_comparison': {}
    }
    
    # Detailed method comparison - ensuring all values are JSON serializable
    for method in methods:
        method_data = df[df['selection_method'] == method]
        summary['method_comparison'][str(method)] = {
            'avg_route_time': float(method_data['route_time_min'].mean()),
            'avg_items_visited': float(method_data['unique_items_visited'].mean()),
            'avg_objective_score': float(method_data['objective_score'].mean()),
            'avg_optimization_time': float(method_data['optimization_time_sec'].mean()),
            'success_rate': float(len(method_data) / BENCHMARK_CONFIG['iterations_per_config'] / len(BENCHMARK_CONFIG['item_counts']) * 100)
        }
    
    # Save summary with proper error handling
    summary_file = os.path.join(OUT_DIR, "executive_summary.json")
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        flog(f"Executive summary saved: {summary_file}", "DONE")
    except Exception as e:
        flog(f"Error saving JSON summary: {e}", "WARN")
        text_summary_file = os.path.join(OUT_DIR, "executive_summary.txt")
        with open(text_summary_file, 'w') as f:
            f.write("EXECUTIVE SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Best Overall Method: {best_method_overall}\n")
            f.write(f"Fastest Method: {fastest_method}\n")
            f.write(f"Most Efficient: {most_efficient}\n")
            f.write(f"Total Iterations: {len(df)}\n")
            f.write(f"Statistical Significance: {statistical_significance}\n")
        flog(f"Text summary saved instead: {text_summary_file}", "DONE")
    
    # Print key findings
    print("\n" + "="*80)
    print("🎯 EXECUTIVE SUMMARY - KEY FINDINGS")
    print("="*80)
    print(f"📊 Total Benchmark Iterations: {len(df):,}")
    print(f"📈 Success Rate: {summary['benchmark_overview']['success_rate']:.1f}%")
    print(f"🏆 Best Overall Method: {best_method_overall}")
    print(f"⚡ Fastest Optimization: {fastest_method}")
    print(f"🎯 Most Item Efficient: {most_efficient}")
    print(f"📊 Statistical Significance: {'Yes' if statistical_significance else 'No'} (p={kruskal_p_value:.4f})")
    
    print(f"\n📈 AVERAGE PERFORMANCE METRICS:")
    print(f"  Route Time: {summary['average_metrics']['route_time_min']:.2f} minutes")
    print(f"  Items Visited: {summary['average_metrics']['items_visited']:.1f}")
    print(f"  Optimization Time: {summary['average_metrics']['optimization_time_sec']:.3f} seconds")
    print(f"  Objective Score: {summary['average_metrics']['objective_score']:.4f}")
    if summary['average_metrics'].get('avg_walk_distance_m') is not None:
        print(f"  Avg Walk Distance: {summary['average_metrics']['avg_walk_distance_m']:.2f} meters")
    if summary['average_metrics'].get('avg_visit_time_per_item_min') is not None:
        print(f"  Avg Visit Time / Item: {summary['average_metrics']['avg_visit_time_per_item_min']:.2f} minutes")
    if summary['average_metrics'].get('avg_revisit_rate') is not None:
        print(f"  Avg Revisit Rate: {summary['average_metrics']['avg_revisit_rate']:.2%}")
    
    print(f"\n🏅 METHOD RANKINGS BY ITEM COUNT:")
    for count, method in method_performance.items():
        print(f"  {count} items: {method}")
    
    print("="*80)
    
    return summary


# ---------- Main Execution ----------
def main():
    """Main execution function for comprehensive benchmarking"""
    flog("🧪 Enhanced Supermarket Planner - Comprehensive Benchmarking System", "FUN")
    flog("=" * 80, "INFO")
    
    # Check if we have enough items for testing
    available_items = len(items_meta)
    max_items = max(BENCHMARK_CONFIG['item_counts'])
    
    if available_items < max_items:
        flog(f"Warning: Only {available_items} items available, but testing up to {max_items}", "WARN")
        BENCHMARK_CONFIG['item_counts'] = [c for c in BENCHMARK_CONFIG['item_counts'] if c <= available_items]
        flog(f"Adjusted item counts: {BENCHMARK_CONFIG['item_counts']}", "INFO")
    
    flog(f"Store layout: {len(nodes)} nodes, {len(items_meta)} items", "INFO")
    flog(f"Output directory: {OUT_DIR}", "INFO")
    
    try:
        # Run comprehensive benchmark
        results = run_comprehensive_benchmark()
        
        if not results:
            flog("No successful results obtained. Exiting.", "WARN")
            return
        
        # Create comprehensive analysis
        df, stats = create_comprehensive_analysis(results)
        
        # Create executive summary
        summary = create_executive_summary(df, stats)
        
        flog("=" * 80, "DONE")
        flog("🎉 Comprehensive benchmarking completed successfully!", "FUN")
        flog(f"📁 All results and analysis saved in: {OUT_DIR}", "DONE")
        flog("📊 Files generated:", "INFO")
        flog("  • raw_benchmark_results.csv - Raw benchmark data", "INFO")
        flog("  • statistical_analysis.csv - Statistical summaries", "INFO")
        flog("  • executive_summary.json - Key findings and recommendations", "INFO")
        flog("  • performance_overview.png - Main performance visualization", "INFO")
        flog("  • method_comparison.png - Detailed method comparison", "INFO")
        flog("  • scalability_analysis.png - Scalability analysis", "INFO")
        flog("  • efficiency_analysis.png - Efficiency analysis", "INFO")
        flog("=" * 80, "DONE")
        
    except KeyboardInterrupt:
        flog("Benchmark interrupted by user", "WARN")
    except Exception as e:
        flog(f"Error during benchmarking: {str(e)}", "WARN")
        raise

if __name__ == "__main__":
    main()