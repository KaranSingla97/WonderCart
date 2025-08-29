#!/usr/bin/env python3
"""
Enhanced Supermarket Path Planner — PPT-Ready with Advanced Optimization

Features:
- Multi-objective optimization with better path quality
- Advanced 2-opt with Or-opt improvements
- Comprehensive metrics and analytics
- Professional visualization for presentations
- Comparison dashboard with efficiency metrics

Requirements:
    Python 3.8+
    pip install networkx matplotlib pandas numpy seaborn

Run:
    python optimized_supermarket_planner.py
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

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
random.seed(12345)
np.random.seed(12345)

# ---------- Enhanced Configuration ----------
WALKING_SPEED_M_S = 1.1
TIME_BUDGET_MIN = 45
ALLOWED_EXTRA_FRACTION = 0.15  # Reduced for tighter optimization
VISIT_OVERHEAD_SEC = 8  # More realistic scanning time

# Enhanced objective weights
PREF_WEIGHT = 1.2
COUNT_BONUS = 0.025
TIME_EFFICIENCY_WEIGHT = 0.3  # New: reward time efficiency
PATH_SMOOTHNESS_WEIGHT = 0.1  # New: reward smoother paths

# Penalty system
REVISIT_PENALTY_PER = 0.08
REVISIT_TIME_PENALTY_MIN = 0.7
DIRECTION_CHANGE_PENALTY = 0.02  # New: penalize excessive direction changes

# Output configuration
OUT_DIR = "enhanced_planner_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def flog(msg, level="INFO"):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    emoji = {"INFO":"ℹ️", "STEP":"🔄", "WARN":"⚠️", "DONE":"✅", "FUN":"🚀", "METRIC":"📊"}.get(level, "ℹ️")
    print(f"{emoji} [{now}] {msg}")

# ---------- Enhanced Store Layout ----------
def create_enhanced_store_layout():
    """Create a more realistic store layout with better connectivity"""
    # Main grid points
    aisle_x = [0, 6, 12, 18, 24, 30, 36, 42]  # Added more points
    aisle_y = [0, 6, 12, 18, 24]  # Added top row
    
    nodes = {}
    node_id = 0
    
    # Create main grid
    for y in aisle_y:
        for x in aisle_x:
            name = f"N{node_id}"
            nodes[name] = (x, y)
            node_id += 1
    
    # Special locations
    nodes["Entrance"] = (-6, 12)  # Moved to center-left
    nodes["Checkout"] = (48, 6)   # Moved to right-center
    
    # Enhanced item positions with better distribution
    item_positions = {
        # Dairy section (left side)
        "Milk": (6, 6), "Cheese": (6, 12), "Yogurt": (6, 18),
        "Eggs": (12, 6), "Butter": (12, 12),
        
        # Produce (center-left)
        "Apples": (18, 18), "Bananas": (18, 12), "Tomatoes": (18, 6),
        "Lettuce": (18, 0), "Carrots": (24, 18),
        
        # Meat (center)
        "Chicken": (24, 12), "Beef": (24, 6), "Fish": (24, 0),
        
        # Bakery (center-right)
        "Bread": (30, 18), "Pastries": (30, 12),
        
        # Packaged goods (right side)
        "Cereal": (36, 18), "Pasta": (36, 12), "Rice": (36, 6),
        "Chips": (36, 0), "Cookies": (42, 18),
        
        # Beverages (distributed)
        "Coffee": (0, 18), "Tea": (0, 12), "Soda": (42, 12),
        "Juice": (42, 6), "Water": (0, 6),
        
        # Household (back and edges)
        "Soap": (0, 0), "Detergent": (42, 0), "Shampoo": (6, 0),
        
        # Frozen (top row)
        "FrozenPizza": (12, 24), "IceCream": (30, 24),
        
        # Condiments/Oil
        "OliveOil": (30, 6), "Vinegar": (30, 0)
    }
    
    # Map items to nodes
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

# Enhanced item metadata
def create_item_metadata():
    return {
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

# Initialize enhanced layout
nodes, item_positions = create_enhanced_store_layout()
items_meta = create_item_metadata()
item_to_node = {item: item_positions[item] for item in items_meta.keys()}

# ---------- Enhanced Graph Construction ----------
def build_enhanced_graph():
    """Build graph with better connectivity and realistic constraints"""
    G = nx.Graph()
    
    # Add nodes
    for n, coord in nodes.items():
        G.add_node(n, pos=coord)
    
    # Add edges with realistic movement constraints
    for u, v in itertools.combinations(nodes.keys(), 2):
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        dist = math.hypot(x1-x2, y1-y2)
        
        # Connect adjacent nodes (horizontal, vertical, and some diagonals)
        connect = False
        if abs(x1-x2) < 0.1 and abs(y1-y2) <= 18:  # Vertical connection
            connect = True
        elif abs(y1-y2) < 0.1 and abs(x1-x2) <= 12:  # Horizontal connection  
            connect = True
        elif dist <= 8.5:  # Short diagonal connections
            connect = True
            
        if connect:
            # Add slight randomness for more realistic travel times
            base_time = dist / WALKING_SPEED_M_S
            realistic_time = base_time * (1 + random.uniform(-0.1, 0.1))
            G.add_edge(u, v, distance=dist, time_sec=realistic_time, weight=realistic_time)
    
    # Special connections for entrance/checkout
    for special in ["Entrance", "Checkout"]:
        for n in list(nodes.keys()):
            if n in ("Entrance", "Checkout"):
                continue
            dist = math.hypot(nodes[special][0]-nodes[n][0], nodes[special][1]-nodes[n][1])
            if dist <= 20:  # Increased connection range
                realistic_time = (dist / WALKING_SPEED_M_S) * (1 + random.uniform(-0.05, 0.05))
                G.add_edge(special, n, distance=dist, time_sec=realistic_time, weight=realistic_time)
    
    return G

G = build_enhanced_graph()

# ---------- Enhanced Distance Matrix ----------
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
                time_matrix.loc[i, j] = time_sec / 60.0  # Convert to minutes
                dist_matrix.loc[i, j] = distance
            except nx.NetworkXNoPath:
                time_matrix.loc[i, j] = float('inf')
                dist_matrix.loc[i, j] = float('inf')
    
    return time_matrix, dist_matrix

time_matrix_min, distance_matrix = compute_distance_matrices()

# ---------- Enhanced Optimization Functions ----------

def compute_path_smoothness(route_nodes):
    """Calculate path smoothness (fewer direction changes = smoother)"""
    if len(route_nodes) < 3:
        return 0.0
    
    direction_changes = 0
    for i in range(2, len(route_nodes)):
        p1, p2, p3 = nodes[route_nodes[i-2]], nodes[route_nodes[i-1]], nodes[route_nodes[i]]
        
        # Calculate vectors
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Skip if either vector is zero
        if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
            continue
            
        # Calculate angle change
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
            angle = math.acos(cos_angle)
            
            # Count significant direction changes (> 30 degrees)
            if angle > math.pi / 6:
                direction_changes += 1
    
    return direction_changes

def enhanced_objective_function(route_nodes, must_items):
    """Enhanced multi-objective optimization function"""
    # Basic item counting and satisfaction
    item_counts = defaultdict(int)
    for nd in route_nodes:
        for item, item_node in item_to_node.items():
            if item_node == nd:
                item_counts[item] += 1
    
    # Check for missing must-items
    missing_musts = [m for m in must_items if item_to_node[m] not in route_nodes]
    if missing_musts:
        return 0.0  # Heavily penalize missing must-items
    
    # Calculate components
    total_time = compute_route_time_minutes(route_nodes)
    time_efficiency = max(0, 1 - (total_time / TIME_BUDGET_MIN)) if total_time <= TIME_BUDGET_MIN else 0
    
    # Preference satisfaction
    pref_satisfaction = sum(
        items_meta[item]["rec_score"] * items_meta[item]["popularity"] * count
        for item, count in item_counts.items()
    ) / max(total_time, 1)
    
    # Item diversity bonus
    unique_items = len(item_counts)
    diversity_bonus = COUNT_BONUS * unique_items
    
    # Path quality
    smoothness_penalty = DIRECTION_CHANGE_PENALTY * compute_path_smoothness(route_nodes)
    
    # Revisit penalties
    revisit_penalty = sum(max(0, count - 1) for count in item_counts.values()) * REVISIT_PENALTY_PER
    
    # Combined objective
    objective = (
        PREF_WEIGHT * pref_satisfaction +
        TIME_EFFICIENCY_WEIGHT * time_efficiency +
        diversity_bonus -
        smoothness_penalty -
        revisit_penalty
    )
    
    return max(0, objective)

def compute_route_time_minutes(route_nodes):
    """Enhanced route time calculation with realistic factors"""
    if len(route_nodes) < 2:
        return 0.0
    
    total_time = 0.0
    item_visit_counts = defaultdict(int)
    
    # Travel and visit times
    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        total_time += float(time_matrix_min.loc[u, v])
        
        # Add visit time for items
        for item, item_node in item_to_node.items():
            if item_node == v:
                visit_time = items_meta[item]["visit_min"] + VISIT_OVERHEAD_SEC / 60.0
                # Add crowd delay factor based on popularity
                crowd_factor = 1 + (items_meta[item]["popularity"] * 0.1)
                total_time += visit_time * crowd_factor
                item_visit_counts[item] += 1
    
    # Revisit penalties
    revisit_count = sum(max(0, count - 1) for count in item_visit_counts.values())
    total_time += revisit_count * REVISIT_TIME_PENALTY_MIN
    
    return total_time

# ---------- Advanced Optimization Algorithms ----------

def enhanced_two_opt_with_or_opt(waypoint_sequence, max_iterations=1000):
    """Enhanced 2-opt with Or-opt moves for better optimization"""
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
                
                # Create 2-opt swap
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
        
        # Or-opt moves (relocate segments of length 1, 2, or 3)
        if not improved:
            for segment_len in [1, 2, 3]:
                for i in range(1, len(current) - segment_len):
                    segment = current[i:i + segment_len]
                    remaining = current[:i] + current[i + segment_len:]
                    
                    for j in range(1, len(remaining)):
                        new_sequence = (remaining[:j] + segment + remaining[j:])
                        new_cost = route_cost(new_sequence)
                        
                        if new_cost < current_cost - 1e-9:
                            current = new_sequence
                            current_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return current

def intelligent_recommendation_insertion(route_nodes, candidate_items, budget_fraction=ALLOWED_EXTRA_FRACTION):
    """Smart recommendation insertion using multiple criteria"""
    current_time = compute_route_time_minutes(route_nodes)
    max_allowed_time = min(current_time * (1 + budget_fraction), TIME_BUDGET_MIN)
    
    # Score candidates by multiple factors
    candidate_scores = []
    for item in candidate_items:
        if item_to_node[item] in route_nodes:
            continue
            
        # Calculate insertion benefit
        rec_score = items_meta[item]["rec_score"]
        popularity = items_meta[item]["popularity"]
        visit_time = items_meta[item]["visit_min"]
        
        # Benefit score considering recommendation value and efficiency
        benefit = (rec_score * popularity) / max(visit_time, 0.1)
        candidate_scores.append((item, benefit))
    
    # Sort by benefit score
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    inserted_items = []
    current_route = route_nodes[:]
    
    for item, _ in candidate_scores:
        item_node = item_to_node[item]
        best_position = None
        min_extra_time = float('inf')
        
        # Find best insertion position
        for pos in range(1, len(current_route)):
            prev_node = current_route[pos - 1]
            next_node = current_route[pos]
            
            # Calculate extra time for insertion
            extra_travel = (time_matrix_min.loc[prev_node, item_node] +
                           time_matrix_min.loc[item_node, next_node] -
                           time_matrix_min.loc[prev_node, next_node])
            
            extra_visit = (items_meta[item]["visit_min"] + VISIT_OVERHEAD_SEC / 60.0) * \
                         (1 + items_meta[item]["popularity"] * 0.1)
            
            total_extra = extra_travel + extra_visit
            
            if total_extra < min_extra_time:
                min_extra_time = total_extra
                best_position = pos
        
        # Insert if within budget
        if best_position is not None:
            new_time = current_time + min_extra_time
            if new_time <= max_allowed_time:
                current_route.insert(best_position, item_node)
                current_time = new_time
                inserted_items.append((item, best_position, min_extra_time))
                flog(f"Inserted {item} at position {best_position} (+{min_extra_time:.3f}min)", "STEP")
    
    return current_route, inserted_items

def nodes_from_waypoints(waypoint_sequence):
    """Convert waypoint sequence to full node path"""
    if len(waypoint_sequence) < 2:
        return waypoint_sequence
    
    full_path = [waypoint_sequence[0]]
    for i in range(len(waypoint_sequence) - 1):
        try:
            segment = nx.shortest_path(G, waypoint_sequence[i], waypoint_sequence[i + 1], weight='time_sec')
            full_path.extend(segment[1:])  # Skip first node to avoid duplication
        except nx.NetworkXNoPath:
            flog(f"No path found between {waypoint_sequence[i]} and {waypoint_sequence[i+1]}", "WARN")
            full_path.append(waypoint_sequence[i + 1])  # Add directly
    
    return full_path

# ---------- Professional Visualization ----------

def create_professional_plot(route_nodes, title, filename, route_color="#FF6B35", comparison_route=None):
    """Create professional-quality plots for presentation"""
    plt.figure(figsize=(14, 9))
    pos = {n: nodes[n] for n in nodes}
    
    # Background grid
    nx.draw(G, pos, node_size=8, node_color="#E8E8E8", edge_color="#F0F0F0", 
            with_labels=False, width=0.5, alpha=0.6)
    
    # Draw item categories with different colors
    category_colors = {
        "Dairy": "#4CAF50", "Produce": "#8BC34A", "Meat": "#F44336",
        "Bakery": "#FF9800", "Grocery": "#2196F3", "Beverages": "#9C27B0",
        "Household": "#607D8B", "Snacks": "#E91E63", "Frozen": "#00BCD4",
        "Condiments": "#795548"
    }
    
    for category, color in category_colors.items():
        category_items = [item for item, meta in items_meta.items() if meta["category"] == category]
        category_nodes = [item_to_node[item] for item in category_items]
        nx.draw_networkx_nodes(G, pos, nodelist=category_nodes, 
                             node_color=color, node_size=150, alpha=0.8,
                             edgecolors="white", linewidths=1.5)
    
    # Special locations
    nx.draw_networkx_nodes(G, pos, nodelist=["Entrance"], 
                          node_color="#2E7D32", node_size=400, 
                          edgecolors="white", linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=["Checkout"], 
                          node_color="#C62828", node_size=400,
                          edgecolors="white", linewidths=2)
    
    # Draw comparison route if provided (dashed line)
    if comparison_route:
        comp_coords = [nodes[n] for n in comparison_route]
        comp_xs, comp_ys = zip(*comp_coords)
        plt.plot(comp_xs, comp_ys, color="#757575", linewidth=4, 
                linestyle="--", alpha=0.7, label="Baseline Route")
    
    # Draw main route
    route_coords = [nodes[n] for n in route_nodes]
    route_xs, route_ys = zip(*route_coords)
    plt.plot(route_xs, route_ys, color=route_color, linewidth=5, 
             solid_capstyle='round', alpha=0.9, label="Optimized Route")
    
    # Add step numbers for visited items
    step_num = 0
    node_to_item = {v: k for k, v in item_to_node.items()}
    
    for i, node in enumerate(route_nodes):
        if node in node_to_item:
            step_num += 1
            x, y = nodes[node]
            plt.scatter(x, y, s=300, color=route_color, edgecolors="white", 
                       linewidth=2, zorder=10)
            plt.text(x, y, str(step_num), ha='center', va='center', 
                    fontsize=10, fontweight='bold', color="white", zorder=11)
    
    # Labels for items
    item_labels = {item_to_node[item]: item for item in items_meta.keys()}
    nx.draw_networkx_labels(G, pos, item_labels, font_size=7, font_weight='bold')
    
    # Special location labels
    special_labels = {"Entrance": "START", "Checkout": "END"}
    nx.draw_networkx_labels(G, pos, special_labels, font_size=10, 
                           font_weight='bold', font_color='white')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.axis('off')
    
    if comparison_route:
        plt.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Professional plot saved: {filename}", "DONE")

def create_metrics_dashboard(shortest_route, optimized_route, must_items):
    """Create comprehensive metrics dashboard"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate metrics
    shortest_time = compute_route_time_minutes(shortest_route)
    optimized_time = compute_route_time_minutes(optimized_route)
    shortest_score = enhanced_objective_function(shortest_route, must_items)
    optimized_score = enhanced_objective_function(optimized_route, must_items)
    
    # Get visited items for each route
    node_to_item = {v: k for k, v in item_to_node.items()}
    shortest_items = [node_to_item[n] for n in shortest_route if n in node_to_item]
    optimized_items = [node_to_item[n] for n in optimized_route if n in node_to_item]
    
    # 1. Time Comparison Bar Chart
    times = [shortest_time, optimized_time]
    labels = ['Shortest Path', 'Optimized Path']
    colors = ['#757575', '#FF6B35']
    
    bars1 = ax1.bar(labels, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Time (minutes)', fontsize=12)
    ax1.set_title('Route Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(times) * 1.2)
    
    # Add value labels on bars
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.02,
                f'{time:.2f}min', ha='center', va='bottom', fontweight='bold')
    
    # 2. Items Collected Comparison
    item_counts = [len(shortest_items), len(optimized_items)]
    bars2 = ax2.bar(labels, item_counts, color=colors, alpha=0.8)
    ax2.set_ylabel('Number of Items', fontsize=12)
    ax2.set_title('Items Collected Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(item_counts) * 1.2)
    
    for bar, count in zip(bars2, item_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(item_counts)*0.02,
                f'{count} items', ha='center', va='bottom', fontweight='bold')
    
    # 3. Efficiency Metrics Radar Chart
    metrics = ['Time Efficiency', 'Item Diversity', 'Route Smoothness', 'Satisfaction Score']
    
    # Calculate normalized metrics (0-1 scale)
    time_eff_short = max(0, 1 - (shortest_time / TIME_BUDGET_MIN))
    time_eff_opt = max(0, 1 - (optimized_time / TIME_BUDGET_MIN))
    
    diversity_short = len(shortest_items) / len(items_meta)
    diversity_opt = len(optimized_items) / len(items_meta)
    
    smoothness_short = max(0, 1 - (compute_path_smoothness(shortest_route) / 20))
    smoothness_opt = max(0, 1 - (compute_path_smoothness(optimized_route) / 20))
    
    sat_short = min(shortest_score, 1)
    sat_opt = min(optimized_score, 1)
    
    shortest_metrics = [time_eff_short, diversity_short, smoothness_short, sat_short]
    optimized_metrics = [time_eff_opt, diversity_opt, smoothness_opt, sat_opt]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    shortest_metrics += shortest_metrics[:1]
    optimized_metrics += optimized_metrics[:1]
    
    ax3.plot(angles, shortest_metrics, 'o-', linewidth=2, label='Shortest Path', color='#757575')
    ax3.fill(angles, shortest_metrics, alpha=0.25, color='#757575')
    ax3.plot(angles, optimized_metrics, 'o-', linewidth=2, label='Optimized Path', color='#FF6B35')
    ax3.fill(angles, optimized_metrics, alpha=0.25, color='#FF6B35')
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics)
    ax3.set_ylim(0, 1)
    ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    
    # 4. Category Coverage Analysis
    categories = list(set(items_meta[item]["category"] for item in items_meta.keys()))
    shortest_categories = list(set(items_meta[item]["category"] for item in shortest_items))
    optimized_categories = list(set(items_meta[item]["category"] for item in optimized_items))
    
    category_data = {
        'Category': categories,
        'Shortest Path': [1 if cat in shortest_categories else 0 for cat in categories],
        'Optimized Path': [1 if cat in optimized_categories else 0 for cat in categories]
    }
    
    df_categories = pd.DataFrame(category_data)
    x_pos = np.arange(len(categories))
    
    ax4.bar(x_pos - 0.2, df_categories['Shortest Path'], 0.4, 
            label='Shortest Path', color='#757575', alpha=0.8)
    ax4.bar(x_pos + 0.2, df_categories['Optimized Path'], 0.4, 
            label='Optimized Path', color='#FF6B35', alpha=0.8)
    
    ax4.set_xlabel('Product Categories', fontsize=12)
    ax4.set_ylabel('Coverage (0=No, 1=Yes)', fontsize=12)
    ax4.set_title('Category Coverage Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.legend()
    ax4.set_ylim(0, 1.2)
    
    plt.tight_layout()
    dashboard_file = os.path.join(OUT_DIR, "metrics_dashboard.png")
    plt.savefig(dashboard_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Metrics dashboard saved: {dashboard_file}", "METRIC")
    
    return {
        'shortest_time': shortest_time,
        'optimized_time': optimized_time,
        'shortest_items': len(shortest_items),
        'optimized_items': len(optimized_items),
        'time_improvement': ((shortest_time - optimized_time) / shortest_time * 100) if shortest_time > 0 else 0,
        'item_improvement': len(optimized_items) - len(shortest_items)
    }

def create_summary_table(metrics, shortest_route, optimized_route, must_items):
    """Create a comprehensive summary table"""
    node_to_item = {v: k for k, v in item_to_node.items()}
    shortest_items = [node_to_item[n] for n in shortest_route if n in node_to_item]
    optimized_items = [node_to_item[n] for n in optimized_route if n in node_to_item]
    
    # Calculate additional metrics
    shortest_score = enhanced_objective_function(shortest_route, must_items)
    optimized_score = enhanced_objective_function(optimized_route, must_items)
    
    summary_data = {
        'Metric': [
            'Total Time (minutes)',
            'Items Collected',
            'Must Items Covered',
            'Recommendation Items',
            'Satisfaction Score',
            'Time Efficiency (%)',
            'Route Smoothness',
            'Category Coverage'
        ],
        'Shortest Path': [
            f"{metrics['shortest_time']:.2f}",
            f"{metrics['shortest_items']}",
            f"{len([i for i in must_items if i in shortest_items])}/{len(must_items)}",
            f"{len([i for i in shortest_items if i not in must_items])}",
            f"{shortest_score:.4f}",
            f"{max(0, (1 - metrics['shortest_time']/TIME_BUDGET_MIN) * 100):.1f}%",
            f"{max(0, 1 - compute_path_smoothness(shortest_route)/20):.2f}",
            f"{len(set(items_meta[item]['category'] for item in shortest_items))}"
        ],
        'Optimized Path': [
            f"{metrics['optimized_time']:.2f}",
            f"{metrics['optimized_items']}",
            f"{len([i for i in must_items if i in optimized_items])}/{len(must_items)}",
            f"{len([i for i in optimized_items if i not in must_items])}",
            f"{optimized_score:.4f}",
            f"{max(0, (1 - metrics['optimized_time']/TIME_BUDGET_MIN) * 100):.1f}%",
            f"{max(0, 1 - compute_path_smoothness(optimized_route)/20):.2f}",
            f"{len(set(items_meta[item]['category'] for item in optimized_items))}"
        ],
        'Improvement': [
            f"{metrics['time_improvement']:+.1f}%" if metrics['time_improvement'] != 0 else "0%",
            f"{metrics['item_improvement']:+d}",
            "✓" if len([i for i in must_items if i in optimized_items]) >= len([i for i in must_items if i in shortest_items]) else "✗",
            f"{len([i for i in optimized_items if i not in must_items]) - len([i for i in shortest_items if i not in must_items]):+d}",
            f"{((optimized_score - shortest_score) / max(shortest_score, 0.001) * 100):+.1f}%" if shortest_score > 0 else "+∞%",
            f"{((1 - metrics['optimized_time']/TIME_BUDGET_MIN) - (1 - metrics['shortest_time']/TIME_BUDGET_MIN)) * 100:+.1f}%",
            f"{(1 - compute_path_smoothness(optimized_route)/20) - (1 - compute_path_smoothness(shortest_route)/20):+.2f}",
            f"{len(set(items_meta[item]['category'] for item in optimized_items)) - len(set(items_meta[item]['category'] for item in shortest_items)):+d}"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_file = os.path.join(OUT_DIR, "performance_summary.csv")
    df_summary.to_csv(summary_file, index=False)
    flog(f"Performance summary saved: {summary_file}", "METRIC")
    
    return df_summary

# ---------- Main Optimization Pipeline ----------

def build_baseline_route(must_items, start_node="Entrance", end_node="Checkout"):
    """Build baseline route using nearest neighbor heuristic"""
    waypoints = [item_to_node[item] for item in must_items]
    current = start_node
    route = [start_node]
    remaining = waypoints[:]
    
    while remaining:
        nearest = min(remaining, key=lambda x: time_matrix_min.loc[current, x])
        path_segment = nx.shortest_path(G, current, nearest, weight='time_sec')
        route.extend(path_segment[1:])  # Skip first node to avoid duplication
        current = nearest
        remaining.remove(nearest)
    
    # Path to checkout
    final_segment = nx.shortest_path(G, current, end_node, weight='time_sec')
    route.extend(final_segment[1:])
    
    return route

def optimize_route_comprehensive(must_items, recommendation_candidates):
    """Comprehensive route optimization pipeline"""
    flog("Starting comprehensive route optimization...", "FUN")
    
    # Step 1: Build baseline route
    baseline_route = build_baseline_route(must_items)
    flog(f"Baseline route built with {len(baseline_route)} nodes", "STEP")
    
    # Step 2: Extract waypoints and optimize with enhanced 2-opt
    visited_item_nodes = []
    for node in baseline_route:
        if node in item_to_node.values() and node not in visited_item_nodes:
            visited_item_nodes.append(node)
    
    waypoints = ["Entrance"] + visited_item_nodes + ["Checkout"]
    optimized_waypoints = enhanced_two_opt_with_or_opt(waypoints)
    optimized_route = nodes_from_waypoints(optimized_waypoints)
    flog(f"Route optimized with 2-opt+Or-opt", "STEP")
    
    # Step 3: Intelligent recommendation insertion
    route_with_recs, inserted_recs = intelligent_recommendation_insertion(
        optimized_route, recommendation_candidates
    )
    flog(f"Added {len(inserted_recs)} recommendations", "STEP")
    
    # Step 4: Final refinement
    # Re-extract waypoints and optimize again
    final_item_nodes = []
    for node in route_with_recs:
        if node in item_to_node.values() and node not in final_item_nodes:
            final_item_nodes.append(node)
    
    final_waypoints = ["Entrance"] + final_item_nodes + ["Checkout"]
    final_optimized_waypoints = enhanced_two_opt_with_or_opt(final_waypoints)
    final_route = nodes_from_waypoints(final_optimized_waypoints)
    
    flog("Comprehensive optimization completed", "DONE")
    return baseline_route, final_route, inserted_recs

def shortest_path_baseline(must_items, start="Entrance", end="Checkout"):
    """Generate shortest path visiting all must items (TSP solution)"""
    must_nodes = [item_to_node[item] for item in must_items]
    
    if len(must_nodes) > 8:
        flog("Too many must-items for exact TSP; using greedy approximation", "WARN")
        return build_baseline_route(must_items, start, end)
    
    best_route = None
    best_time = float('inf')
    
    # Try all permutations
    for perm in itertools.permutations(must_nodes):
        waypoints = [start] + list(perm) + [end]
        route = nodes_from_waypoints(waypoints)
        route_time = sum(time_matrix_min.loc[route[i], route[i+1]] 
                        for i in range(len(route)-1))
        
        if route_time < best_time:
            best_time = route_time
            best_route = route
    
    return best_route if best_route else build_baseline_route(must_items, start, end)

# ---------- Main Execution ----------

def main():
    """Main execution function for PPT-ready demonstration"""
    flog("🚀 Enhanced Supermarket Path Planner - PPT Ready Version", "FUN")
    flog("=" * 60, "INFO")
    
    # Configuration
    must_items = ["Milk", "Bread", "Eggs", "Coffee", "Chicken", "Apples"]
    all_items = list(items_meta.keys())
    recommendation_candidates = [item for item in all_items if item not in must_items]
    
    flog(f"Must-have items: {must_items}", "INFO")
    flog(f"Available recommendations: {len(recommendation_candidates)} items", "INFO")
    flog(f"Time budget: {TIME_BUDGET_MIN} minutes", "INFO")
    
    # Generate routes
    flog("Generating shortest path baseline...", "STEP")
    shortest_route = shortest_path_baseline(must_items)
    
    flog("Running comprehensive optimization...", "STEP")
    baseline_route, optimized_route, recommendations = optimize_route_comprehensive(
        must_items, recommendation_candidates
    )
    
    # Calculate metrics
    metrics = create_metrics_dashboard(shortest_route, optimized_route, must_items)
    summary_df = create_summary_table(metrics, shortest_route, optimized_route, must_items)
    
    # Generate visualizations
    flog("Creating professional visualizations...", "STEP")
    
    # Individual route plots
    shortest_file = os.path.join(OUT_DIR, "route_shortest_path.png")
    create_professional_plot(shortest_route, 
                           "Shortest Path Route (Baseline - Must Items Only)", 
                           shortest_file, "#757575")
    
    optimized_file = os.path.join(OUT_DIR, "route_optimized_enhanced.png")
    create_professional_plot(optimized_route, 
                           "Enhanced Optimized Route (Must Items + Smart Recommendations)", 
                           optimized_file, "#FF6B35")
    
    # Comparison plot
    comparison_file = os.path.join(OUT_DIR, "route_comparison.png")
    create_professional_plot(optimized_route, 
                           "Route Comparison: Shortest vs Enhanced Optimization", 
                           comparison_file, "#FF6B35", shortest_route)
    
    # Print comprehensive results
    flog("=" * 60, "INFO")
    flog("FINAL RESULTS SUMMARY", "METRIC")
    flog("=" * 60, "INFO")
    
    node_to_item = {v: k for k, v in item_to_node.items()}
    shortest_items = [node_to_item[n] for n in shortest_route if n in node_to_item]
    optimized_items = [node_to_item[n] for n in optimized_route if n in node_to_item]
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Shortest Path Time:    {metrics['shortest_time']:.2f} minutes")
    print(f"   Optimized Route Time:  {metrics['optimized_time']:.2f} minutes")
    print(f"   Time Improvement:      {metrics['time_improvement']:+.1f}%")
    print(f"   Items (Shortest):      {metrics['shortest_items']} items")
    print(f"   Items (Optimized):     {metrics['optimized_items']} items")
    print(f"   Additional Items:      {metrics['item_improvement']:+d}")
    
    print(f"\n🛒 ITEM SEQUENCES:")
    print(f"   Shortest Route:  {' → '.join(shortest_items)}")
    print(f"   Optimized Route: {' → '.join(optimized_items)}")
    
    if recommendations:
        print(f"\n💡 SMART RECOMMENDATIONS ADDED:")
        for item, pos, time_cost in recommendations:
            print(f"   • {item} (position {pos}, +{time_cost:.2f}min)")
    
    print(f"\n📈 EFFICIENCY GAINS:")
    efficiency_gain = (metrics['optimized_items'] - metrics['shortest_items']) / max(metrics['optimized_time'] - metrics['shortest_time'], 0.1)
    print(f"   Items per Extra Minute: {efficiency_gain:.2f}")
    
    satisfaction_short = enhanced_objective_function(shortest_route, must_items)
    satisfaction_opt = enhanced_objective_function(optimized_route, must_items)
    if satisfaction_short > 0:
        satisfaction_improvement = ((satisfaction_opt - satisfaction_short) / satisfaction_short) * 100
        print(f"   Satisfaction Improvement: {satisfaction_improvement:+.1f}%")
    
    print(f"\n📁 OUTPUT FILES GENERATED:")
    print(f"   • {shortest_file}")
    print(f"   • {optimized_file}")
    print(f"   • {comparison_file}")
    print(f"   • {os.path.join(OUT_DIR, 'metrics_dashboard.png')}")
    print(f"   • {os.path.join(OUT_DIR, 'performance_summary.csv')}")
    
    flog("=" * 60, "DONE")
    flog("🎉 PPT-ready analysis complete! All outputs saved in: " + OUT_DIR, "FUN")
    flog("Ready for presentation with professional visualizations and metrics!", "DONE")

if __name__ == "__main__":
    main()