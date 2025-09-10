
## Features    

1. **Optimized Route Planning**:
   - Algorithms for shortest and optimized routes.
   - Visualizations for route comparisons.

2. **Performance Metrics**:
   - Detailed analysis of shopping efficiency.
   - Metrics dashboards and statistical summaries.

3. **Visualization Outputs**:
   - Professional-quality plots for presentations.
   - Scalability and method comparison analyses.

4. **Interactive Frontend**:
   - Navigation UI for guiding users through the store.
   - Zoomable and interactive floor plans.

5. **Backend API**:
   - Flask-based server for handling route planning and data processing.

## Setup Instructions

1. **Install Dependencies**:
   - Python dependencies: Install using `pip install -r requirements.txt` (create a `requirements.txt` if not present).

2. **Run the Backend**:
   - Navigate to the project directory and run:
     ```bash
     python app.py
     ```

3. **Run the Frontend**:
   - Open `pages/index.html` in a browser.

4. **Data Configuration**:
   - Ensure JSON files in the `json/` directory are correctly configured for your store layout and item metadata.

## Outputs

- **Benchmark Results**:
  - Located in `Algorithms/benchmark_results_100_multi_items/`.
  - Includes visualizations like `efficiency_analysis.png`, `method_comparison.png`, and more.

- **Enhanced Planner Outputs**:
  - Located in `Algorithms/enhanced_planner_outputs/` and `Algorithms/enhanced_planner_outputs_single/`.
  - Includes metrics dashboards, route comparisons, and performance summaries.

## Key Files

- `app.py`: Backend server for handling requests and processing data.
- `app.js`: Frontend logic for route planning and navigation.
- `multi_items_iterations.py`: Core algorithm for multi-item route optimization.
- `path _single_ comprehensive.py`: Comprehensive route optimization and visualization.

## Contributing

Feel free to contribute by submitting issues or pull requests. Ensure your code follows the project's style guidelines.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
