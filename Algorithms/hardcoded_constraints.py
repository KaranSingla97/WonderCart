#!/usr/bin/env python3
"""
Enhanced Supermarket Path Planner — Two-Plot Benchmark Run (corrected)

- Guarantees: recommended items inserted < 50% of must-items each iteration (strict cap)
- Records and plots the number of recommendations inserted per iteration
- Produces three plots:
    - Time series: shortest vs optimized time
    - Items series: must-items, optimized items (must+recs), and recommendations inserted
    - Recommendations vs must-count scatter
- Saves per-iteration logs to enhanced_planner_outputs/benchmark_iteration_logs.csv

Run:
    python enhanced_planner_benchmarked_corrected.py

Notes:
- This script builds a small store graph and runs combinatorial routing optimizations; for large ITERATIONS the run may be slow
  because of repeated shortest-path and permutation computations. For quick tests set ITERATIONS to a small value (e.g. 4).
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
sns.set_style("whitegrid")

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

random.seed(12345)
np.random.seed(12345)

OUT_DIR = "enhanced_planner_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Configuration ----------
WALKING_SPEED_M_S = 1.1
TIME_BUDGET_MIN = 45
ALLOWED_EXTRA_FRACTION = 0.15        # legacy budget for recs
VISIT_OVERHEAD_SEC = 8

# New config controlling max allowed extra time relative to shortest path
MAX_EXTRA_FRACTION_REL_TO_SHORTEST = 0.08

# Benchmark configuration (variable-must-count)
ITERATIONS = 100
MUST_COUNT = None
MUST_COUNT_MIN = 3
MUST_COUNT_MAX = 8

# Objective weights / penalties (unchanged)
PREF_WEIGHT = 1.2
COUNT_BONUS = 0.025
TIME_EFFICIENCY_WEIGHT = 0.3
PATH_SMOOTHNESS_WEIGHT = 0.1
REVISIT_PENALTY_PER = 0.08
REVISIT_TIME_PENALTY_MIN = 0.7
DIRECTION_CHANGE_PENALTY = 0.02


def flog(msg, level="INFO"):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    emoji = {"INFO":"ℹ️", "STEP":"🔄", "WARN":"⚠️", "DONE":"✅", "FUN":"🚀", "METRIC":"📊"}.get(level, "ℹ️")
    print(f"{emoji} [{now}] {msg}")

# ---------- Store layout + metadata ----------
def create_enhanced_store_layout():
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
        "OliveOil": (30, 6), "Vinegar": (30, 0)
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

# Initialize layout and metadata
nodes, item_positions = create_enhanced_store_layout()
items_meta = create_item_metadata()
item_to_node = {item: item_positions[item] for item in items_meta.keys()}

# ---------- Graph building ----------
def build_enhanced_graph():
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

G = build_enhanced_graph()

# ---------- Distance matrices ----------
def compute_distance_matrices():
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

time_matrix_min, distance_matrix = compute_distance_matrices()

# ---------- Professional plot function ----------
def create_professional_plot(route_nodes, title, filename, route_color="#FF6B35"):
    """Create a professional visualization of the route."""
    plt.figure(figsize=(14, 10))
    
    # Plot all nodes
    for node, (x, y) in nodes.items():
        if node in ["Entrance", "Checkout"]:
            plt.scatter(x, y, c='red' if node == "Entrance" else 'green', s=200, marker='s', 
                       edgecolors='black', linewidth=2, zorder=5)
            plt.annotate(node, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=12, fontweight='bold')
        else:
            plt.scatter(x, y, c='lightgray', s=50, alpha=0.6, zorder=2)
    
    # Plot items
    node_to_item = {v: k for k, v in item_to_node.items()}
    for node, (x, y) in nodes.items():
        if node in node_to_item:
            item = node_to_item[node]
            plt.scatter(x, y, c='blue', s=100, marker='o', alpha=0.8, zorder=3)
            plt.annotate(item, (x, y), xytext=(5, -15), textcoords='offset points', 
                        fontsize=8, ha='left')
    
    # Plot route
    if len(route_nodes) > 1:
        route_coords = [nodes[node] for node in route_nodes]
        route_x, route_y = zip(*route_coords)
        plt.plot(route_x, route_y, color=route_color, linewidth=3, alpha=0.8, zorder=4)
        
        # Add arrows to show direction
        for i in range(len(route_coords) - 1):
            x1, y1 = route_coords[i]
            x2, y2 = route_coords[i + 1]
            plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=route_color, lw=2),
                        zorder=4)
    
    # Highlight route nodes
    for i, node in enumerate(route_nodes):
        x, y = nodes[node]
        if node not in ["Entrance", "Checkout"]:
            plt.scatter(x, y, c=route_color, s=150, marker='o', 
                       edgecolors='black', linewidth=1, zorder=6)
            plt.annotate(str(i), (x, y), ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=10, zorder=7)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('X Coordinate (meters)', fontsize=12)
    plt.ylabel('Y Coordinate (meters)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ---------- Utility functions ----------
def compute_path_smoothness(route_nodes):
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
    pref_satisfaction = sum(items_meta[item]["rec_score"] * items_meta[item]["popularity"] * count for item, count in item_counts.items()) / max(total_time, 1)
    unique_items = len(item_counts)
    diversity_bonus = COUNT_BONUS * unique_items
    smoothness_penalty = DIRECTION_CHANGE_PENALTY * compute_path_smoothness(route_nodes)
    revisit_penalty = sum(max(0, count - 1) for count in item_counts.values()) * REVISIT_PENALTY_PER
    objective = (PREF_WEIGHT * pref_satisfaction + TIME_EFFICIENCY_WEIGHT * time_efficiency + diversity_bonus - smoothness_penalty - revisit_penalty)
    return max(0, objective)


def compute_route_time_minutes(route_nodes):
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


def enhanced_two_opt_with_or_opt(waypoint_sequence, max_iterations=1000):
    def route_cost(sequence):
        if len(sequence) < 2:
            return 0
        return sum(time_matrix_min.loc[sequence[i], sequence[i+1]] for i in range(len(sequence)-1))
    
    current = waypoint_sequence[:]
    current_cost = route_cost(current)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # 2-opt improvement
        for i in range(1, len(current) - 2):
            for j in range(i + 1, len(current) - 1):
                if j - i == 1:
                    continue
                new_sequence = (current[:i] + current[i:j+1][::-1] + current[j+1:])
                new_cost = route_cost(new_sequence)
                if new_cost < current_cost - 1e-9:
                    current = new_sequence
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        # Or-opt improvement
        if not improved:
            for segment_len in [1, 2, 3]:
                if segment_len >= len(current) - 2:
                    continue
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


def nodes_from_waypoints(waypoint_sequence):
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

# ---------- Baseline and optimization pipelines ----------
def build_baseline_route(must_items, start_node="Entrance", end_node="Checkout"):
    waypoints = [item_to_node[item] for item in must_items]
    current = start_node
    route = [start_node]
    remaining = waypoints[:]
    while remaining:
        nearest = min(remaining, key=lambda x: time_matrix_min.loc[current, x])
        path_segment = nx.shortest_path(G, current, nearest, weight='time_sec')
        route.extend(path_segment[1:])
        current = nearest
        remaining.remove(nearest)
    final_segment = nx.shortest_path(G, current, end_node, weight='time_sec')
    route.extend(final_segment[1:])
    return route


def shortest_path_baseline(must_items, start="Entrance", end="Checkout"):
    must_nodes = [item_to_node[item] for item in must_items]
    if len(must_nodes) > 8:
        return build_baseline_route(must_items, start, end)
    best_route = None
    best_time = float('inf')
    for perm in itertools.permutations(must_nodes):
        waypoints = [start] + list(perm) + [end]
        route = nodes_from_waypoints(waypoints)
        route_time = sum(time_matrix_min.loc[route[i], route[i+1]] for i in range(len(route)-1))
        if route_time < best_time:
            best_time = route_time
            best_route = route
    return best_route if best_route else build_baseline_route(must_items, start, end)

# ---------- Recommendation insertion: strict <50% cap + time guard ----------
def intelligent_recommendation_insertion(
    route_nodes,
    candidate_items,
    budget_fraction=ALLOWED_EXTRA_FRACTION,
    reference_time=None,
    max_extra_fraction_rel_to_reference=MAX_EXTRA_FRACTION_REL_TO_SHORTEST,
    max_recs_allowed=None
):
    """
    Insert recommended items greedily but:
      - do not exceed max_recs_allowed (if provided)
      - do not exceed allowed time derived from reference_time and budget_fraction
      - max_recs_allowed is computed with strict less-than 50% if invoked via optimize_route_comprehensive
    """
    # If no recommendations allowed, return original route
    if max_recs_allowed is not None and max_recs_allowed <= 0:
        return route_nodes[:], []
    
    current_time = compute_route_time_minutes(route_nodes)
    allowed_by_budget = min(current_time * (1 + budget_fraction), TIME_BUDGET_MIN)
    if reference_time is not None:
        allowed_by_reference = min(reference_time * (1 + max_extra_fraction_rel_to_reference), TIME_BUDGET_MIN)
        max_allowed_time = min(allowed_by_budget, allowed_by_reference)
    else:
        max_allowed_time = allowed_by_budget

    # Filter out items already in route and sort by benefit score
    candidate_scores = []
    for item in candidate_items:
        if item_to_node[item] in route_nodes:
            continue
        rec_score = items_meta[item]["rec_score"]
        popularity = items_meta[item]["popularity"]
        visit_time = items_meta[item]["visit_min"]
        benefit = (rec_score * popularity) / max(visit_time, 0.1)
        candidate_scores.append((item, benefit))
    
    # Sort by benefit score (higher is better)
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    inserted_items = []
    current_route = route_nodes[:]
    recs_inserted_count = 0
    max_recs = 999 if max_recs_allowed is None else int(max_recs_allowed)

    for item, benefit_score in candidate_scores:
        # Stop if we hit the recommendation cap
        if recs_inserted_count >= max_recs:
            break

        item_node = item_to_node[item]
        if item_node in current_route:
            continue
            
        # Find best insertion position
        best_position = None
        min_extra_time = float('inf')
        
        for pos in range(1, len(current_route)):
            prev_node = current_route[pos - 1]
            next_node = current_route[pos]
            
            # Calculate extra travel time
            extra_travel = (time_matrix_min.loc[prev_node, item_node] + 
                          time_matrix_min.loc[item_node, next_node] - 
                          time_matrix_min.loc[prev_node, next_node])
            
            # Calculate visit time with crowd factor
            extra_visit = (items_meta[item]["visit_min"] + VISIT_OVERHEAD_SEC / 60.0) * (1 + items_meta[item]["popularity"] * 0.1)
            
            total_extra = extra_travel + extra_visit
            
            if total_extra < min_extra_time:
                min_extra_time = total_extra
                best_position = pos
        
        if best_position is None:
            continue
            
        # Check if adding this item keeps us within time budget
        projected_time = current_time + min_extra_time
        if projected_time <= max_allowed_time + 1e-9:
            current_route.insert(best_position, item_node)
            current_time = projected_time
            inserted_items.append((item, best_position, min_extra_time))
            recs_inserted_count += 1
        # else: skip this item to stay within time budget

    return current_route, inserted_items


def optimize_route_comprehensive(must_items, recommendation_candidates, max_extra_fraction_for_recs=None, reference_time=None):
    baseline_route = build_baseline_route(must_items)
    visited_item_nodes = []
    for node in baseline_route:
        if node in item_to_node.values() and node not in visited_item_nodes:
            visited_item_nodes.append(node)
    waypoints = ["Entrance"] + visited_item_nodes + ["Checkout"]
    optimized_waypoints = enhanced_two_opt_with_or_opt(waypoints)
    optimized_route = nodes_from_waypoints(optimized_waypoints)

    # compute strict-cap: recommendations must be strictly less than 50% of must_items
    # AND cap at max 3 recommendations total
    must_count = len(must_items)
    theoretical_cap = max(0, int(math.floor(0.5 * must_count - 1e-9)))
    strict_cap = min(theoretical_cap, 3)  # Never more than 3 recommendations

    budget_fraction = ALLOWED_EXTRA_FRACTION if max_extra_fraction_for_recs is None else max_extra_fraction_for_recs
    route_with_recs, inserted_recs = intelligent_recommendation_insertion(
        optimized_route,
        recommendation_candidates,
        budget_fraction=budget_fraction,
        reference_time=reference_time,
        max_extra_fraction_rel_to_reference=MAX_EXTRA_FRACTION_REL_TO_SHORTEST,
        max_recs_allowed=strict_cap
    )

    final_item_nodes = []
    for node in route_with_recs:
        if node in item_to_node.values() and node not in final_item_nodes:
            final_item_nodes.append(node)
    final_waypoints = ["Entrance"] + final_item_nodes + ["Checkout"]
    final_optimized_waypoints = enhanced_two_opt_with_or_opt(final_waypoints)
    final_route = nodes_from_waypoints(final_optimized_waypoints)
    return baseline_route, final_route, inserted_recs, strict_cap

# ---------- BENCHMARK + logs ----------
def run_two_benchmarks(iterations=ITERATIONS, must_count=MUST_COUNT):
    flog(f"Running {iterations} benchmark iterations (two plots + logs; strict <50% rec cap)...", "FUN")
    all_items = list(items_meta.keys())
    n_items = len(all_items)

    if must_count is None:
        if MUST_COUNT_MIN < 1 or MUST_COUNT_MAX < MUST_COUNT_MIN:
            raise ValueError("Invalid MUST_COUNT_MIN / MUST_COUNT_MAX configuration.")
        if MUST_COUNT_MAX >= n_items:
            raise ValueError("MUST_COUNT_MAX must be smaller than total distinct items in store.")
    else:
        if must_count >= n_items:
            raise ValueError("must_count must be smaller than total distinct items in store.")

    times_shortest = []
    times_optimized = []
    optimized_items_counts = []
    must_counts_per_iter = []
    recommendations_inserted_counts = []

    iteration_logs = []

    last_shortest_route = None
    last_optimized_route = None
    last_recommendations = None

    for i in range(iterations):
        if must_count is None:
            must_count_iter = random.randint(MUST_COUNT_MIN, MUST_COUNT_MAX)
        else:
            must_count_iter = must_count

        must_items_iter = random.sample(all_items, must_count_iter)
        recommendation_candidates = [it for it in all_items if it not in must_items_iter]

        shortest_route = shortest_path_baseline(must_items_iter)
        shortest_time = compute_route_time_minutes(shortest_route)

        # optimize and return strict_cap
        _, optimized_route, recommendations, strict_cap = optimize_route_comprehensive(
            must_items_iter,
            recommendation_candidates,
            max_extra_fraction_for_recs=None,
            reference_time=shortest_time
        )

        optimized_time = compute_route_time_minutes(optimized_route)
        node_to_item = {v: k for k, v in item_to_node.items()}
        opt_items = [node_to_item[n] for n in optimized_route if n in node_to_item]

        times_shortest.append(shortest_time)
        times_optimized.append(optimized_time)
        optimized_items_counts.append(len(opt_items))
        must_counts_per_iter.append(must_count_iter)

        inserted_items_list = [it[0] for it in recommendations] if recommendations else []
        recommendations_inserted_counts.append(len(inserted_items_list))
        iteration_logs.append({
            'iteration': i + 1,
            'must_count': must_count_iter,
            'must_items': "|".join(must_items_iter),
            'recommended_items_inserted': "|".join(inserted_items_list),
            'num_recommendations_inserted': len(inserted_items_list),
            'max_recs_allowed_strict_lt_50pct': strict_cap,
            'shortest_time_min': shortest_time,
            'optimized_time_min': optimized_time,
            'optimized_items_count': len(opt_items),
            'optimized_items': "|".join(opt_items)
        })

        last_shortest_route = shortest_route
        last_optimized_route = optimized_route
        last_recommendations = recommendations

        if (i + 1) % 10 == 0:
            flog(f"Iter {i+1}: must_count={must_count_iter}, shortest={shortest_time:.2f}min, optimized={optimized_time:.2f}min, recs_inserted={len(inserted_items_list)}, strict_cap={strict_cap}", "INFO")

    logs_df = pd.DataFrame(iteration_logs)
    logs_csv = os.path.join(OUT_DIR, "benchmark_iteration_logs.csv")
    logs_df.to_csv(logs_csv, index=False)
    flog(f"Saved iteration logs CSV: {logs_csv}", "METRIC")

    xs = list(range(1, iterations + 1))

    # Time series plot
    # Walmart-inspired palette
    palette = ['#FFC220','#0071CE']  # Blue, Yellow

    # --- Time Series Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(xs, times_shortest, label='Shortest Path Time (must-only)',
            linewidth=2, color=palette[0], linestyle='-')
    plt.plot(xs, times_optimized, label='Optimized Path Time (with recs)',
            linewidth=2, color=palette[1], linestyle='-')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Time (minutes)', fontsize=12)
    plt.title(f'Benchmark Time Series: Shortest vs Optimized over {iterations} Iterations',
            fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    time_series_file = os.path.join(OUT_DIR, 'benchmark_time_series.png')
    plt.tight_layout()
    plt.savefig(time_series_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Saved time series plot: {time_series_file}", "METRIC")

    # --- Items Series Plot ---
    plt.figure(figsize=(12, 6))
    combined_counts = [a + b for a, b in zip(recommendations_inserted_counts, must_counts_per_iter)]

    # Must-items only (blue dashed)
    plt.plot(xs, must_counts_per_iter,
            label='Per-iteration must-items (initial)',
            linewidth=2, linestyle='--', color=palette[0])

    # Must-items + recommendations (yellow solid)
    plt.plot(xs, combined_counts,
            label='Optimized Route: # Items Collected (must + recs)',
            linewidth=2, linestyle='-', color=palette[1])

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Number of Items', fontsize=12)
    plt.title(f'Benchmark: Items Collected vs Iteration', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    items_series_file = os.path.join(OUT_DIR, 'benchmark_items_series_corrected.png')
    plt.tight_layout()
    plt.savefig(items_series_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Saved items series plot: {items_series_file}", "METRIC")

    # --- Scatter Plot: Recommendations vs Must-items ---
    plt.figure(figsize=(10, 6))
    plt.scatter(must_counts_per_iter, recommendations_inserted_counts,
                s=80, alpha=0.9, edgecolors='k', facecolor=palette[1])  # Yellow points
    for xi, yi, it in zip(must_counts_per_iter, recommendations_inserted_counts, xs):
        plt.annotate(str(it), (xi + 0.08, yi + 0.08), fontsize=8, color=palette[0])  # Blue labels
    plt.xlabel('Initial Must-items Count', fontsize=12)
    plt.ylabel('Number of Recommendations Inserted', fontsize=12)
    plt.title('Recommendations Inserted vs Initial Must-items (per iteration)',
            fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    recs_vs_must_file = os.path.join(OUT_DIR, 'recommendations_vs_must_count.png')
    plt.savefig(recs_vs_must_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    flog(f"Saved recommendations vs must-count plot: {recs_vs_must_file}", "METRIC")


    return {
        'time_series_file': time_series_file,
        'items_series_file': items_series_file,
        'recs_vs_must_file': recs_vs_must_file,
        'logs_csv': logs_csv,
        'times_shortest': times_shortest,
        'times_optimized': times_optimized,
        'optimized_items_counts': optimized_items_counts,
        'must_counts_per_iter': must_counts_per_iter,
        'recommendations_inserted_counts': recommendations_inserted_counts,
        'last_shortest_route': last_shortest_route,
        'last_optimized_route': last_optimized_route,
        'last_recommendations': last_recommendations
    }

# ---------- Main ----------
def main():
    flog("🚀 Enhanced Supermarket Planner — Two-Plot Benchmark Run (plots corrected)", "FUN")
    flog("=" * 60, "INFO")
    
    try:
        results = run_two_benchmarks(ITERATIONS, MUST_COUNT)
        flog(f"\nBENCHMARKS COMPLETE — OUTPUTS SAVED:\n   - {results['time_series_file']}\n   - {results['items_series_file']}\n   - {results['recs_vs_must_file']}\n   - {results['logs_csv']}\n", "DONE")

        
            
    except Exception as e:
        flog(f"Error during benchmark execution: {e}", "WARN")
        raise

    flog("=" * 60, "DONE")
    flog(f"🎉 Completed. Check the enhanced_planner_outputs folder for images and logs.", "FUN")

if __name__ == "__main__":
    main()
