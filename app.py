from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import heapq
import traceback
import os
import requests # Added for making HTTP requests to Gemini
from groq import Groq


# gemini_key = os.getenv('GEMINI_API_KEY')
groq_key = "groq_api_key"

# Configure Gemini client
client = Groq(api_key = groq_key)
model_name = "qwen/qwen3-32b"


app = Flask(__name__)
CORS(app)

# --- 1. LOAD FLOOR 1 PLAN AND DEPARTMENT DATA ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', 'json')
    if not os.path.isdir(json_path):
        json_path = os.path.join(script_dir, 'json')

    with open(os.path.join(json_path, 'floor1.json'), 'r') as f:
        floor_plan = json.load(f)

    with open(os.path.join(json_path, 'ids.json'), 'r') as f:
        department_data = json.load(f)

except Exception as e:
    print(f"Error loading data: {e}. Ensure floor1.json and ids.json are in the /json folder.")
    exit()

# --- 2. A* (A-STAR) SHORTEST PATH ALGORITHM ---
path_cache = {}

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def a_star_with_path(start_node, end_node):
    start_pos = (start_node[1], start_node[2])
    end_pos = (end_node[1], end_node[2])
    
    cache_key = (start_pos, end_pos)
    if cache_key in path_cache:
        return path_cache[cache_key]

    pq = [(0, start_pos)]
    g_score = {start_pos: 0}
    parent = {}

    while pq:
        _, current_pos = heapq.heappop(pq)
        if current_pos == end_pos:
            break
        y, x = current_pos
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor_pos = (y + dy, x + dx)
            ny, nx = neighbor_pos
            if 0 <= ny < len(floor_plan) and 0 <= nx < len(floor_plan[0]):
                tile_value = floor_plan[ny][nx]
                is_traversable = tile_value == 0 or tile_value >= 100 or neighbor_pos == end_pos
                if is_traversable:
                    tentative_g_score = g_score[current_pos] + 1
                    if tentative_g_score < g_score.get(neighbor_pos, float('inf')):
                        parent[neighbor_pos] = current_pos
                        g_score[neighbor_pos] = tentative_g_score
                        f_score = tentative_g_score + manhattan_distance(neighbor_pos, end_pos)
                        heapq.heappush(pq, (f_score, neighbor_pos))
    path = []
    if end_pos in parent or end_pos == start_pos:
        crawl = end_pos
        while crawl is not None:
            path.append((1, crawl[0], crawl[1]))
            crawl = parent.get(crawl)
        path.reverse()
    final_path = path if path and path[0] == start_node else []
    path_cache[cache_key] = final_path
    # print(final_path)
    return final_path

# --- 3. FIND ACCESSIBLE WALKING POINTS ---
def find_all_accessible_points(dept_id):
    accessible_points = set()
    n, m = len(floor_plan), len(floor_plan[0])
    for r_idx, row in enumerate(floor_plan):
        for c_idx, cell_id in enumerate(row):
            if cell_id == dept_id:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < n and 0 <= nc < m and floor_plan[nr][nc] == 0:
                        accessible_points.add((1, nr, nc))
    return list(accessible_points)

# --- 4. NEW: GEMINI RECOMMENDATION ENDPOINT ---
@app.route('/get-recommendation', methods=['POST'])
def get_recommendation_endpoint():
    try:
        data = request.get_json()
        main_dept_name = data.get('main_department')
        reco_dept_name = data.get('reco_department')
        print(main_dept_name, reco_dept_name)
        if not main_dept_name or not reco_dept_name:
            return jsonify({"error": "Missing department names"}), 400

        prompt = f"""
        You are a helpful and slightly persuasive Walmart shopping assistant.
        A shopper is on their way to the '{main_dept_name}' department.
        The '{reco_dept_name}' department is very close by.
        Generate a short, friendly, and enticing message (30-40 words) to suggest they visit the '{reco_dept_name}' department as well.
        Make it sound like a helpful tip and try to sell {reco_dept_name}. Start with something like 'Psst! While you're heading to...' or 'Quick tip! Since you're near...'.
        Add some emojis too
        """

        # NOTE: This calls a local LLaMA endpoint as specified.
        # In a real scenario, you would replace this with a call to the Gemini API.

        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.3,
            reasoning_effort="none",
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_output = completion.choices[0].message.content
        return jsonify({"recommendation_text": raw_output})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "An unexpected server error occurred."}), 500


# --- 5. MAIN ROUTE PLANNING ENDPOINT ---
@app.route('/plan-route', methods=['POST'])
def plan_route_endpoint():
    try:
        global path_cache
        path_cache.clear()

        data = request.get_json()
        department_ids = [int(d) for d in data['departments']]
        start_node = (1, 182, 92)

        waypoints_map = {dept_id: find_all_accessible_points(dept_id) for dept_id in department_ids}
        valid_waypoints_map = {k: v for k, v in waypoints_map.items() if v}
        unreachable_depts = list(set(department_ids) - set(valid_waypoints_map.keys()))
        if unreachable_depts:
            print(f"Departments with no access points: {unreachable_depts}")

        similar_dept_map = {
            dept_id: [int(sid) for sid in department_data.get(str(dept_id), {}).get("similar_ids", []) if str(sid) in department_data]
            for dept_id in department_ids
        }

        ordered_checkpoints = [start_node]
        ordered_department_ids = []
        unvisited_dept_ids = set(valid_waypoints_map.keys())
        last_checkpoint = start_node

        while unvisited_dept_ids:
            best_next_checkpoint, best_dept_id, min_dist = None, None, float('inf')
            for dept_id in unvisited_dept_ids:
                for access_point in valid_waypoints_map[dept_id]:
                    path = a_star_with_path(last_checkpoint, access_point)
                    if path and len(path) < min_dist:
                        min_dist = len(path)
                        best_next_checkpoint = access_point
                        best_dept_id = dept_id
            if best_dept_id is not None:
                ordered_checkpoints.append(best_next_checkpoint)
                ordered_department_ids.append(best_dept_id)
                last_checkpoint = best_next_checkpoint
                unvisited_dept_ids.remove(best_dept_id)
            else:
                print(f"Could not find a path to any remaining depts: {unvisited_dept_ids}. Halting.")
                break

        path_segments = {}
        recommendations = {}
        current_checkpoint = start_node

        for i, target_dept_id in enumerate(ordered_department_ids):
            target_checkpoint = ordered_checkpoints[i + 1]
            key = f"{current_checkpoint[0]},{current_checkpoint[1]},{current_checkpoint[2]}-{target_checkpoint[0]},{target_checkpoint[1]},{target_checkpoint[2]}"
            main_path = a_star_with_path(current_checkpoint, target_checkpoint)
            path_segments[key] = main_path
            if not main_path:
                current_checkpoint = target_checkpoint
                continue
            best_reco, min_detour_len = None, float('inf')
            for similar_id in similar_dept_map.get(target_dept_id, []):
                access_points = find_all_accessible_points(similar_id)
                if not access_points: continue
                reco_ap = access_points[0]
                path_to_reco = a_star_with_path(current_checkpoint, reco_ap)
                path_from_reco = a_star_with_path(reco_ap, target_checkpoint)
                if path_to_reco and path_from_reco:
                    detour_len = len(path_to_reco) + len(path_from_reco)
                    if (detour_len - len(main_path)) < 25 and detour_len < min_detour_len:
                        min_detour_len = detour_len
                        best_reco = {
                            "dept_id": similar_id,
                            "dept_name": department_data.get(str(similar_id), {}).get('name', 'Unknown'),
                            "path_to": path_to_reco,
                            "path_from": path_from_reco,
                        }
            if best_reco:
                recommendations[key] = best_reco
            current_checkpoint = target_checkpoint
        return jsonify({
            "ordered_department_ids": ordered_department_ids,
            "checkpoints": ordered_checkpoints,
            "path_segments": path_segments,
            "recommendations": recommendations,
            "unreachable_departments": unreachable_depts
        })
    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "An unexpected server error occurred."}), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        history = data.get('history', [])

        with open('info.txt', 'r', encoding='utf-8') as file:
            info = file.read()

        if not history:
            return jsonify({"error": "No chat history provided"}), 400

        system_prompt = {
            "role": "system",
            "content": f"""
        You are a helpful and friendly Walmart shopping assistant. 🛍️

        Your job is to assist customers with questions about:
        - Products and their availability
        - Prices and similar items
        - Store departments and layout
        - In-store deals or tips

        Be persuasive but not pushy. Keep replies concise and friendly. Use emojis where helpful 🎯.

        Here is some knowledge about this store you can use to answer questions (like prices, stock, or related products):
        ---
        {info}
        ---

        If the question isn’t answered by the info above, reply based on your normal persuasive results.
        """
        }
        
        messages_to_send = [system_prompt] + history

        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.3,
            messages=messages_to_send,
            reasoning_effort="none",
        )
        
        response_text = completion.choices[0].message.content
        return jsonify({"response": response_text})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Failed to get response from AI model"}), 500



# --- 6. RUN FLASK SERVER ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
