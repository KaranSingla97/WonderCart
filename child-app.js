document.addEventListener('DOMContentLoaded', function () {
    // --- Element References ---
    const mapContainer = document.getElementById('map-container');
    const controlsContainer = document.getElementById('controls-container');
    const resultsContainer = document.getElementById('results-container');
    const pointsDisplay = document.getElementById('points-display');

    // --- Constants ---
    const API_URL = 'http://127.0.0.1:5001';

    // --- State Management ---
    let departmentData = null;
    let floorPlans = {};
    let colorMap = {};
    let animationFrameId = null;

    let gameState = {
        points: 0,
        currentFloor: 1,
        checkpoints: [],
        path_segments: {},
        ordered_department_ids: [],
        current_segment_index: 0,
        zoomLevel: 1, // FIX: Added for zoom functionality
    };

    // --- Sound Effects and Speech ---
    const synth = window.speechSynthesis;

    function speak(text) {
        if (synth.speaking) {
            synth.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.2;   // Slightly faster for playful tone
        utterance.pitch = 1.5;  // Higher pitch sounds more cheerful

        // Try to find a friendly female voice
        const voices = synth.getVoices();
        const femaleVoice = voices.find(voice =>
            /female|Google US English|Jenny|Samantha|en-US/i.test(voice.name) && voice.lang.includes("en")
        );

        if (femaleVoice) {
            utterance.voice = femaleVoice;
        } else {
            console.warn("No female voice found. Using default.");
        }

        synth.speak(utterance);
    }

    // Ensure voices are loaded (important for Chrome)
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = () => {
            synth.getVoices(); // Populate the voices list
        };
    }

    
    // --- ZOOM FUNCTIONALITY ---
    /**
     * Zooms the map by a given amount.
     * @param {number} amount - The amount to increase/decrease the zoom level.
     */
    function zoomMap(amount) {
        gameState.zoomLevel += amount;
        if (gameState.zoomLevel < 1) gameState.zoomLevel = 1;
        if (gameState.zoomLevel > 3) gameState.zoomLevel = 3;

        if (mapContainer) {
            mapContainer.style.transform = `scale(${gameState.zoomLevel})`;
        }
    }

    // --- Data Fetching ---
    async function fetchData() {
        try {
            const [floor1Res, idsRes, colorRes] = await Promise.all([
                fetch('../json/floor1.json'),
                fetch('../json/ids_1.json'),
                fetch('../json/color2.json')
            ]);
            if (!floor1Res.ok || !idsRes.ok || !colorRes.ok) {
                throw new Error('Network response was not ok.');
            }
            
            floorPlans[1] = await floor1Res.json();
            departmentData = await idsRes.json();
            const rawColorData = await colorRes.json();
            
            colorMap = Object.entries(rawColorData).reduce((acc, [color, id]) => {
                acc[id] = color;
                return acc;
            }, {});

            return true;
        } catch (error) {
            console.error("Failed to fetch initial data:", error);
            controlsContainer.innerHTML = `<p class="text-red-500 font-bold">Could not load map data! Please try again.</p>`;
            return false;
        }
    }

    // --- UI Rendering ---
    function renderControls() {
        const isLastItem = gameState.current_segment_index >= gameState.ordered_department_ids.length - 1;
        const currentDeptId = gameState.ordered_department_ids[gameState.current_segment_index];
        const currentDeptInfo = departmentData[currentDeptId];
        
        const nextButtonText = isLastItem ? "Finish!" : "Found It!";
        const nextButtonClass = isLastItem ? "bg-pink-500 hover:bg-pink-600" : "bg-green-500 hover:bg-green-600";

        controlsContainer.innerHTML = `
            <i class="fas fa-robot text-6xl sm:text-8xl text-yellow-400 mb-4 animate-bounce"></i>
            <div class="text-center my-4 p-3 bg-green-100 rounded-lg w-full">
                <p class="text-base sm:text-lg text-gray-600">Let's find the...</p>
                <p class="text-xl sm:text-2xl font-bold text-green-800">${currentDeptInfo ? currentDeptInfo.name : 'Finish Line!'}</p>
            </div>
            <div class="grid grid-cols-2 gap-4 mt-4 w-full">
                <button id="prev-btn" class="w-full bg-blue-500 text-white font-bold py-3 px-4 rounded-lg nav-button text-lg" ${gameState.current_segment_index === 0 ? 'disabled' : ''}>Back</button>
                <button id="next-btn" class="w-full ${nextButtonClass} text-white font-bold py-3 px-4 rounded-lg nav-button text-lg">${nextButtonText}</button>
            </div>
            <a href="index.html" class="w-full mt-4 bg-red-500 text-white font-bold py-3 px-4 rounded-lg nav-button">End Adventure</a>
        `;

        document.getElementById('prev-btn').addEventListener('click', navigateToPrevious);
        document.getElementById('next-btn').addEventListener('click', navigateToNext);
    }

    function generateMap() {
        const floorPlanMatrix = floorPlans[1];
        if (!floorPlanMatrix) return;
        const columns = floorPlanMatrix[0].length;
        mapContainer.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
        mapContainer.innerHTML = ''; 
        
        floorPlanMatrix.flat().forEach(id => {
            const cell = document.createElement('div');
            cell.className = `aspect-square`;
            cell.style.backgroundColor = colorMap[id] || '#FFFFFF';
            mapContainer.appendChild(cell);
        });
        
        const svgOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svgOverlay.id = 'path-overlay';
        svgOverlay.setAttribute('class', 'absolute top-0 left-0 w-full h-full pointer-events-none');
        mapContainer.appendChild(svgOverlay);
    }

    // --- Game Logic ---
    async function planRoute(departments) {
        controlsContainer.innerHTML = `<p class="text-gray-600 font-bold">GreenBot is planning the best path...</p>`;
        try {
            const response = await fetch(`${API_URL}/plan-route`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ departments })
            });
            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            const result = await response.json();
            gameState = { ...gameState, ...result };

            if (!gameState.ordered_department_ids || gameState.ordered_department_ids.length === 0) {
                 resultsContainer.innerHTML = `<p class="text-red-500 font-bold">GreenBot can't find a path! Let's try other treasures.</p>`;
                 setTimeout(() => window.location.href = 'child-products.html', 4000);
                 return;
            }
            
            speak(`Okay! First, let's go find the ${departmentData[gameState.ordered_department_ids[0]].name}`);
            renderControls();
            drawCurrentSegment();
        } catch (error) {
            console.error('Error planning route:', error);
            resultsContainer.innerHTML = `<p class="text-red-500 font-bold">Oops! The map seems to be broken. Please try again.</p>`;
        }
    }

    function navigateToNext() {
        const isLastItem = gameState.current_segment_index >= gameState.ordered_department_ids.length - 1;
        gameState.points += 100;
        updatePoints();
        showPointsFeedback("+100 Points! 🎉");
        speak("Great job!");

        if (isLastItem) {
            setTimeout(showRewards, 1000);
        } else {
            gameState.current_segment_index++;
            renderControls();
            drawCurrentSegment();
            const nextDeptName = departmentData[gameState.ordered_department_ids[gameState.current_segment_index]].name;
            speak(`Awesome! Now, let's find the ${nextDeptName}`);
        }
    }

    function navigateToPrevious() {
        if (gameState.current_segment_index > 0) {
            gameState.points = Math.max(0, gameState.points - 50);
            updatePoints();
            showPointsFeedback("-50 Points", false);
            speak("Whoops, wrong way!");
            gameState.current_segment_index--;
            renderControls();
            drawCurrentSegment();
        }
    }
    
    function updatePoints() {
        pointsDisplay.textContent = gameState.points;
        pointsDisplay.classList.add('animate-ping');
        setTimeout(() => pointsDisplay.classList.remove('animate-ping'), 500);
    }

    function showPointsFeedback(message, isPositive = true) {
        const color = isPositive ? 'text-green-500' : 'text-red-500';
        resultsContainer.innerHTML = `<p class="text-2xl font-bold ${color} animate-bounce">${message}</p>`;
        setTimeout(() => resultsContainer.innerHTML = '', 1500);
    }

    function drawCurrentSegment() {
        if (animationFrameId) cancelAnimationFrame(animationFrameId);

        const startCheckpoint = gameState.checkpoints[gameState.current_segment_index];
        const endCheckpoint = gameState.checkpoints[gameState.current_segment_index + 1];
        
        const svg = document.getElementById('path-overlay');
        if (!svg) return;
        svg.innerHTML = ''; 
        
        if (!startCheckpoint || !endCheckpoint) return;

        const pathKey = `${startCheckpoint.join(',')}-${endCheckpoint.join(',')}`;
        const pathToDraw = gameState.path_segments[pathKey];
        
        if (!pathToDraw || pathToDraw.length === 0) return;

        const rows = floorPlans[1].length;
        const cols = floorPlans[1][0].length;
        const cellWidth = svg.clientWidth / cols;
        const cellHeight = svg.clientHeight / rows;

        const markerSize = Math.max(16, Math.min(cellWidth * 2, 32)); 
        const pathWidth = Math.max(4, Math.min(cellWidth * 0.7, 8)); 
        const markerDotRadius = Math.max(3, pathWidth * 0.8);

        [startCheckpoint, endCheckpoint].forEach((p, index) => {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', (p[2] + 0.5) * cellWidth);
            text.setAttribute('y', (p[1] + 0.5) * cellHeight);
            text.setAttribute('font-size', `${markerSize}px`);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('dominant-baseline', 'middle');
            text.textContent = index === 0 ? '🤖' : '⭐';
            svg.appendChild(text);
        });

        const points = pathToDraw.map(p => `${(p[2] + 0.5) * cellWidth},${(p[1] + 0.5) * cellHeight}`).join(' ');
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', points);
        polyline.setAttribute('fill', 'none');
        polyline.setAttribute('stroke', '#f59e0b');
        polyline.setAttribute('stroke-width', `${pathWidth}px`);
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

        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        marker.setAttribute('r', `${markerDotRadius}px`);
        marker.setAttribute('fill', '#ef4444');
        svg.appendChild(marker);

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
    }
    
    function showRewards() {
        const rewards = [
            { points: 500, name: "Free Ice Cream!", icon: "fa-ice-cream" },
            { points: 300, name: "Cool Toy", icon: "fa-gamepad" },
            { points: 0, name: "Sticker", icon: "fa-star" }
        ];

        let earnedReward = rewards.find(r => gameState.points >= r.points);

        speak(`Wow! You earned ${gameState.points} points and won a ${earnedReward.name}!`);
        
        const rewardsHtml = rewards.map(reward => {
            const isEarned = gameState.points >= reward.points;
            const isBestEarned = reward.name === earnedReward.name;
            
            return `
                <div class="relative p-2 sm:p-4 rounded-lg transition-all ${isBestEarned ? 'bg-yellow-300 border-2 sm:border-4 border-yellow-500 scale-105' : 'bg-gray-200'}">
                    ${isBestEarned ? '<div class="absolute -top-2 -right-2 bg-green-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">EARNED!</div>' : ''}
                    <i class="fas ${reward.icon} text-2xl sm:text-4xl mb-2 ${isEarned ? 'text-blue-600' : 'text-gray-400'}"></i>
                    <p class="font-bold text-xs sm:text-base ${isEarned ? 'text-gray-800' : 'text-gray-500'}">${reward.name}</p>
                    <p class="text-[10px] sm:text-xs ${isEarned ? 'text-gray-600' : 'text-gray-400'}">(${reward.points}+ pts)</p>
                </div>
            `;
        }).join('');

        document.body.innerHTML = `
            <div class="min-h-screen flex flex-col items-center justify-center text-center p-4 bg-blue-500 text-white" style="font-family: 'Fredoka One', cursive;">
                <div class="bg-white/20 p-4 sm:p-8 rounded-2xl shadow-lg w-full max-w-md">
                    <i class="fas fa-trophy text-6xl sm:text-8xl text-yellow-300 mb-4"></i>
                    <h1 class="text-3xl sm:text-5xl font-bold">Adventure Complete!</h1>
                    <p class="text-xl sm:text-3xl mt-2 sm:mt-4">You earned</p>
                    <p class="text-5xl sm:text-7xl font-bold my-1 sm:my-2">${gameState.points} Points!</p>
                    
                    <div class="mt-6 sm:mt-8 text-left bg-white text-blue-800 p-4 rounded-xl">
                        <h2 class="text-xl sm:text-2xl font-bold text-center mb-4">Prize List</h2>
                        <div class="grid grid-cols-3 gap-2">
                            ${rewardsHtml}
                        </div>
                    </div>

                    <a href="index.html" class="mt-6 sm:mt-8 inline-block bg-white text-blue-800 font-bold py-3 px-6 sm:py-4 sm:px-8 rounded-full text-lg sm:text-2xl nav-button">Go Home</a>
                </div>
            </div>
        `;
    }

    // --- Initialization ---
    async function init() {
        const dataLoaded = await fetchData();
        if (dataLoaded) {
            generateMap();

            // FIX: Setup zoom controls
            const zoomInBtn = document.getElementById('zoom-in-btn');
            const zoomOutBtn = document.getElementById('zoom-out-btn');
            zoomInBtn.addEventListener('click', () => zoomMap(0.2));
            zoomOutBtn.addEventListener('click', () => zoomMap(-0.2));

            const resizeObserver = new ResizeObserver(() => {
                if (gameState.ordered_department_ids.length > 0) {
                    drawCurrentSegment();
                }
            });
            resizeObserver.observe(mapContainer);

            const urlParams = new URLSearchParams(window.location.search);
            const deptsFromURL = urlParams.get('departments');
            if (deptsFromURL) {
                await planRoute(deptsFromURL.split(','));
            } else {
                controlsContainer.innerHTML = `<p class="text-gray-600">Start your adventure from the home page!</p><a href="index.html" class="mt-4 bg-blue-500 text-white font-bold py-3 px-4 rounded-lg nav-button">Go Back</a>`;
            }
        }
    }

    init();
});