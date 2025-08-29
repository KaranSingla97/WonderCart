document.addEventListener('DOMContentLoaded', function () {
    const mapContainer = document.getElementById('map-container');
    const controlsContainer = document.getElementById('controls-container');
    const resultsContainer = document.getElementById('results-container');

    const API_URL = 'http://127.0.0.1:5001'; // Base URL for the backend

    // --- STATE MANAGEMENT ---
    let departmentData = null;
    let floorPlans = {};
    let colorMap = {};
    let animationFrameId = null;

    let appState = {
        mode: 'selection',
        currentFloor: 1,
        checkpoints: [],
        path_segments: {},
        ordered_department_ids: [],
        recommendations: {},
        current_segment_index: 0,
        unreachable_departments: [],
        zoomLevel: 1, // Added for zoom functionality
    };

    // --- HELPER FUNCTIONS ---
    // Helper function to compare arrays
    function arraysEqual(arr1, arr2) {
        if (arr1.length !== arr2.length) return false;
        return arr1.every((val, index) => val === arr2[index]);
    }

    // --- ZOOM FUNCTIONALITY ---
    function zoomMap(amount) {
        appState.zoomLevel += amount;
        if (appState.zoomLevel < 0.5) appState.zoomLevel = 0.5;
        if (appState.zoomLevel > 3) appState.zoomLevel = 3;

        if (mapContainer) {
            mapContainer.style.transform = `scale(${appState.zoomLevel})`;
        }
    }

    // --- DATA FETCHING & INITIALIZATION ---
    async function fetchData() {
        try {
            // FIX: Reverted fetch paths to be relative to the HTML document's location.
            const [floor1Res, idsRes, colorRes] = await Promise.all([
                fetch('../json/floor1.json'),
                fetch('../json/ids.json'),
                fetch('../json/color2.json')
            ]);
            if (!floor1Res.ok || !idsRes.ok || !colorRes.ok) throw new Error('Network response was not ok.');
            
            floorPlans[1] = await floor1Res.json();
            departmentData = await idsRes.json();
            const rawColorData = await colorRes.json();

            colorMap = Object.entries(rawColorData).reduce((acc, [color, id]) => {
                acc[id] = color;
                return acc;
            }, {});
            
            return true;
        } catch (error) {
            console.error('Failed to fetch store data:', error);
            mapContainer.innerHTML = `<p class="text-red-500 text-center col-span-full p-8">Error: Could not load store data.</p>`;
            return false;
        }
    }

    // --- UI RENDERING ---
    function renderControls() {
        controlsContainer.innerHTML = '';
        if (appState.mode === 'selection') {
            renderSelectionUI();
        } else {
            renderNavigationUI();
        }
    }
    
    function renderSelectionUI() {
        const heading = document.createElement('h2');
        heading.className = 'text-xl font-bold text-gray-700 mb-2 border-b pb-2';
        heading.textContent = 'Select Departments';

        const legend = document.createElement('div');
        legend.className = 'space-y-1 flex-grow overflow-y-auto pr-2';
        Object.keys(departmentData).forEach(id => {
            if (parseInt(id) < 3 || parseInt(id) >= 100) return;
            const label = document.createElement('label');
            label.className = 'flex items-center p-2 rounded-md cursor-pointer transition-all duration-200 hover:bg-gray-200';
            const colorStyle = colorMap[id] ? `style="background-color: ${colorMap[id]}"` : '';
            label.innerHTML = `
                <input type="checkbox" class="form-checkbox h-5 w-5 rounded text-blue-600" value="${id}">
                <div class="w-5 h-5 rounded-sm ml-3 mr-2 flex-shrink-0" ${colorStyle}></div>
                <span class="text-sm font-medium text-gray-700">${departmentData[id].name}</span>
            `;
            legend.appendChild(label);
        });

        const button = document.createElement('button');
        button.id = 'plan-route-btn';
        button.className = 'w-full mt-4 bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors';
        button.textContent = 'Start Shopping';
        button.addEventListener('click', planRouteFromSelection);
        controlsContainer.append(heading, legend, button);
    }

    function renderNavigationUI() {
        const currentDeptId = appState.ordered_department_ids[appState.current_segment_index];
        const currentDeptInfo = departmentData[currentDeptId];
        
        const heading = document.createElement('h2');
        heading.className = 'text-xl font-bold text-gray-700 mb-2 border-b pb-2';
        heading.textContent = 'Navigation';

        const statusDiv = document.createElement('div');
        statusDiv.className = 'text-center my-4 p-3 bg-blue-100 rounded-lg';
        statusDiv.innerHTML = `
            <p class="text-sm text-gray-600">Next Stop (${appState.current_segment_index + 1}/${appState.ordered_department_ids.length}):</p>
            <p class="text-lg font-bold text-blue-800">${currentDeptInfo ? currentDeptInfo.name : 'Finish'}</p>
        `;

        const navButtons = document.createElement('div');
        navButtons.className = 'grid grid-cols-2 gap-2 mt-4';
        navButtons.innerHTML = `
            <button id="prev-btn" class="w-full bg-gray-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-gray-700">Previous</button>
            <button id="next-btn" class="w-full bg-gray-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-gray-700">Next</button>
        `;

        const finishButton = document.createElement('button');
        finishButton.id = 'finish-btn';
        finishButton.className = 'w-full mt-4 bg-red-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-red-700';
        finishButton.textContent = 'Finish Shopping';
        finishButton.addEventListener('click', resetApp);

        controlsContainer.append(heading, statusDiv, navButtons, finishButton);
        
        document.getElementById('prev-btn').addEventListener('click', navigateToPrevious);
        document.getElementById('next-btn').addEventListener('click', navigateToNext);
        
        document.getElementById('prev-btn').disabled = appState.current_segment_index === 0;
        document.getElementById('next-btn').disabled = appState.current_segment_index >= appState.ordered_department_ids.length - 1;
    }

    function generateMap() {
        const floorPlanMatrix = floorPlans[1];
        if (!floorPlanMatrix) return;
        const columns = floorPlanMatrix[0].length;
        mapContainer.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
        mapContainer.innerHTML = '<svg id="path-overlay" class="absolute top-0 left-0 w-full h-full pointer-events-none"></svg>';
        
        floorPlanMatrix.flat().forEach(id => {
            const cell = document.createElement('div');
            const deptInfo = departmentData[id] || {};
            cell.className = `map-grid-cell aspect-square flex items-center justify-center text-xs md:text-lg`;
            const color = colorMap[id] || '#FFFFFF';
            cell.style.backgroundColor = color;
            const hex = color.replace('#', '');
            const r = parseInt(hex.substring(0, 2), 16);
            const g = parseInt(hex.substring(2, 4), 16);
            const b = parseInt(hex.substring(4, 6), 16);
            cell.style.color = (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5 ? 'white' : 'black';
            if (id != 0) {
                cell.textContent = ''; // or maybe use initials like deptInfo.name[0]
            }
            mapContainer.appendChild(cell);
        });
    }

    // --- APP LOGIC & NAVIGATION ---
    async function planRouteFromSelection() {
        const selectedCheckboxes = controlsContainer.querySelectorAll('input[type="checkbox"]:checked');
        const departments = Array.from(selectedCheckboxes).map(cb => cb.value);
        if (departments.length === 0) {
            resultsContainer.innerHTML = `<p class="text-red-500">Please select at least one department.</p>`;
            return;
        }
        await executeRoutePlanning(departments);
    }
    
    async function executeRoutePlanning(departments) {
        controlsContainer.innerHTML = `<div class="text-center p-4"><h3 class="font-semibold text-lg">Calculating optimal route...</h3></div>`;
        try {
            const response = await fetch(`${API_URL}/plan-route`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ departments })
            });
            if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            
            appState = {
                ...appState,
                mode: 'navigation',
                checkpoints: result.checkpoints,
                path_segments: result.path_segments,
                ordered_department_ids: result.ordered_department_ids,
                recommendations: result.recommendations || {},
                unreachable_departments: result.unreachable_departments || [],
                current_segment_index: 0
            };
            
            if (appState.ordered_department_ids.length === 0) {
                 resultsContainer.innerHTML = `<p class="text-orange-600 font-semibold">Could not find a route for the selected items. They may be unreachable.</p>`;
                 setTimeout(resetApp, 40000);
                 return;
            }

            renderControls();
            drawCurrentSegment();
        } catch (error) {
            console.error('Error planning route:', error);
            resultsContainer.innerHTML = `<p class="text-red-500">Error: Could not plan route. Is the Python server running?</p>`;
        }
    }

    function navigateToNext() {
        if (appState.current_segment_index < appState.ordered_department_ids.length - 1) {
            appState.current_segment_index++;
            renderControls();
            drawCurrentSegment();
        }
    }

    function navigateToPrevious() {
        if (appState.current_segment_index > 0) {
            appState.current_segment_index--;
            renderControls();
            drawCurrentSegment();
        }
    }

    async function drawCurrentSegment() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }

        const startCheckpoint = appState.checkpoints[appState.current_segment_index];
        const endCheckpoint = appState.checkpoints[appState.current_segment_index + 1];
        
        const svg = document.getElementById('path-overlay');
        svg.innerHTML = ''; 
        resultsContainer.innerHTML = '';
        
        if (!startCheckpoint || !endCheckpoint) return;

        const pathKey = `${startCheckpoint.join(',')}-${endCheckpoint.join(',')}`;
        const recommendation = appState.recommendations[pathKey];
        
        // FIX: Ensure path starts from the correct checkpoint
        let pathToDraw = appState.path_segments[pathKey];
        
        // Handle special case for department ID 9 or any path inconsistencies
        if (!pathToDraw || pathToDraw.length === 0) {
            console.warn(`No path found for segment ${pathKey}`);
            return;
        }

        // FIX: Verify and correct starting point
        const expectedStartPoint = [startCheckpoint[0], startCheckpoint[1], startCheckpoint[2]];
        const actualStartPoint = pathToDraw[0];
        
        // Check if the path starts from the correct position
        if (!arraysEqual(expectedStartPoint, actualStartPoint)) {
            console.warn(`Path starting point mismatch. Expected: ${expectedStartPoint}, Got: ${actualStartPoint}`);
            
            // Attempt to correct the path by prepending the correct start point
            if (pathToDraw.length > 1) {
                pathToDraw = [expectedStartPoint, ...pathToDraw.slice(1)];
            } else {
                pathToDraw = [expectedStartPoint, endCheckpoint];
            }
        }

        const floorPlanMatrix = floorPlans[appState.currentFloor];
        const rows = floorPlanMatrix.length;
        const cols = floorPlanMatrix[0].length;
        const cellWidth = svg.clientWidth / cols;
        const cellHeight = svg.clientHeight / rows;

        // Draw start and end markers with correct positions
        [startCheckpoint, endCheckpoint].forEach((p, index) => {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', (p[2] + 0.5) * cellWidth);
            circle.setAttribute('cy', (p[1] + 0.5) * cellHeight);
            circle.setAttribute('r', '10');
            circle.setAttribute('stroke', 'white');
            circle.setAttribute('stroke-width', '2');
            circle.setAttribute('fill', index === 0 ? '#16a34a' : '#2563eb');
            svg.appendChild(circle);
        });

        // Create polyline with corrected path
        const points = pathToDraw.map(p => `${(p[2] + 0.5) * cellWidth},${(p[1] + 0.5) * cellHeight}`).join(' ');
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', points);
        polyline.setAttribute('fill', 'none');
        polyline.setAttribute('stroke', '#1d4ed8');
        polyline.setAttribute('stroke-width', '5');
        polyline.setAttribute('stroke-linecap', 'round');
        polyline.setAttribute('stroke-linejoin', 'round');
        svg.appendChild(polyline);

        const length = polyline.getTotalLength();
        if (length > 0) {
            polyline.style.strokeDasharray = length;
            polyline.style.strokeDashoffset = length;
            polyline.getBoundingClientRect();
            polyline.style.transition = 'stroke-dashoffset 1.5s ease-in-out';
            polyline.style.strokeDashoffset = 0;
        }

        // Create animated marker
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        marker.setAttribute('r', '8');
        marker.setAttribute('fill', '#ef4444');
        svg.appendChild(marker);

        // FIX: Start animation from the correct first point
        if (length > 0) {
            const startPoint = polyline.getPointAtLength(0);
            marker.setAttribute('cx', startPoint.x);
            marker.setAttribute('cy', startPoint.y);
        }

        let startTime = null;
        const animationDuration = 1500;

        function animationStep(timestamp) {
            if (!startTime) startTime = timestamp;
            const progress = timestamp - startTime;
            const percentage = Math.min(progress / animationDuration, 1);
            
            if (length > 0) {
                const point = polyline.getPointAtLength(length * percentage);
                marker.setAttribute('cx', point.x);
                marker.setAttribute('cy', point.y);
            }
            
            if (percentage < 1) {
                animationFrameId = requestAnimationFrame(animationStep);
            }
        }
        animationFrameId = requestAnimationFrame(animationStep);

        // Handle special popup for department ID 9 (Sports/Dairy section)
        const currentDeptId = appState.ordered_department_ids[appState.current_segment_index];
        if (currentDeptId === 9) {
            // Remove any existing popup
            const existingPopup = document.getElementById('dairy-popup');
            if (existingPopup) existingPopup.remove();
        
            // Create Sports Promotion Popup
            const dairyPopup = document.createElement('div');
            dairyPopup.id = 'dairy-popup';
            dairyPopup.className = `
                fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50
                bg-white border border-gray-300 shadow-lg rounded-md p-6 max-w-lg w-[90%]
                animate-fade-in
            `;
        
            dairyPopup.innerHTML = `
                <div class="flex justify-between items-start mb-4">
                    <h3 class="text-lg font-bold text-gray-800">🏀 Game On: Sports Gear Spotlight</h3>
                    <button class="text-gray-500 hover:text-red-600 font-bold text-lg" onclick="this.parentElement.parentElement.remove()">&times;</button>
                </div>
                <p class="text-sm text-gray-700 mb-2">Ready to elevate your game?</p>
                <p class="text-sm text-gray-700 mb-4">
                    Explore our latest line of premium sporting goods — from basketballs and sneakers to high-performance gear that meets pro standards. Whether you're training or competing, we've got you covered!
                </p>
                <div class="w-full aspect-video bg-black rounded overflow-hidden">
                    <iframe 
                        class="w-full h-full"
                        src="https://www.youtube.com/embed/FzaS0V_FCrI"
                        title="Sporting Goods Promo"
                        frameborder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowfullscreen>
                    </iframe>
                </div>
            `;
        
            document.body.appendChild(dairyPopup);
        
            // Auto-remove after 15 seconds
            setTimeout(() => {
                if (dairyPopup && dairyPopup.parentNode) {
                    dairyPopup.remove();
                }
            }, 150000);
        }

        // Handle AI recommendations
        if (recommendation) {
            try {
                const mainDeptId = appState.ordered_department_ids[appState.current_segment_index];
                const mainDeptName = departmentData[mainDeptId].name;
        
                const recoResponse = await fetch(`${API_URL}/get-recommendation`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        main_department: mainDeptName,
                        reco_department: recommendation.dept_name,
                    })
                });
        
                if (!recoResponse.ok) throw new Error('Recommendation API failed');
                const recoResult = await recoResponse.json();
        
                createPopup(
                    "🛒 A quick tip from your shopping assistant!",
                    recoResult.recommendation_text
                );
        
            } catch (err) {
                console.error("Failed to fetch AI recommendation:", err);
        
                createPopup(
                    "🛍️ Suggestion Nearby",
                    `Don't forget to check out the ${recommendation.dept_name} department!`
                );
            }
        }
        
        function createPopup(title, message) {
            const existingPopup = document.getElementById('tip-popup');
            if (existingPopup) existingPopup.remove();
        
            const popup = document.createElement('div');
            popup.id = 'tip-popup';
            popup.className = `
                fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50
                bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800
                px-6 py-4 rounded-md shadow-lg max-w-lg w-[90%]
            `;
        
            popup.innerHTML = `
                <div class="flex justify-between items-start gap-4">
                    <div class="text-sm leading-relaxed">
                        <p class="font-bold mb-1">${title}</p>
                        <p>${message}</p>
                    </div>
                    <button class="text-yellow-800 hover:text-red-600 font-bold text-lg leading-none" onclick="this.closest('#tip-popup').remove()">&times;</button>
                </div>
            `;
        
            document.body.appendChild(popup);
        
            setTimeout(() => {
                if (popup && popup.parentNode) popup.remove();
            }, 70000);
        }
    }

    function resetApp() {
        window.location.href = 'index.html';
    }

    async function init() {
        const dataLoaded = await fetchData();
        if (dataLoaded) {
            generateMap();
            
            // Setup zoom controls
            const zoomInBtn = document.getElementById('zoom-in-btn');
            const zoomOutBtn = document.getElementById('zoom-out-btn');
            if (zoomInBtn && zoomOutBtn) {
                zoomInBtn.addEventListener('click', () => zoomMap(0.2));
                zoomOutBtn.addEventListener('click', () => zoomMap(-0.2));
            }

            window.addEventListener('resize', () => {
                if (appState.mode === 'navigation') drawCurrentSegment();
            });
            const urlParams = new URLSearchParams(window.location.search);
            const deptsFromURL = urlParams.get('departments');
            if (deptsFromURL && deptsFromURL.length > 0) {
                await executeRoutePlanning(deptsFromURL.split(','));
            } else {
                renderControls();
            }
        }
    }

    init();
});