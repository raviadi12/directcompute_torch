let dataset = {}; // Structure: { labelName: [ Float32Array, Float32Array ] }
let totalSamples = 0;
let isDrawing = false;
let ctx;

const labelSelect = document.getElementById('labelSelect');
const newLabelInput = document.getElementById('newLabelInput');
const createLabelBtn = document.getElementById('createLabelBtn');
const drawCanvas = document.getElementById('drawCanvas');
const addExampleBtn = document.getElementById('addExampleBtn');
const datasetList = document.getElementById('datasetList');
const warningBox = document.getElementById('imbalanceWarning');
const startTrainBtn = document.getElementById('startTrainBtn');
const exportBtn = document.getElementById('exportBtn');
const fileInput = document.getElementById('fileInput');
const statusLog = document.getElementById('trainStatus');
const chartCard = document.getElementById('chartCard');
const metricsChartCtx = document.getElementById('metricsChart').getContext('2d');
const importImagesBtn = document.getElementById('importImagesBtn');
const imageImportInput = document.getElementById('imageImportInput');
const inferenceUploadBtn = document.getElementById('inferenceUploadBtn');
const inferenceImportInput = document.getElementById('inferenceImportInput');

let metricsChart = null;
let globalTrainedNN = null;
let globalLabels = [];

// Canvas Drawing Logic
function initCanvas() {
    ctx = drawCanvas.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = "black"; // Background
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "white"; // Draw color
    ctx.lineWidth = Math.max(1, Math.floor(drawCanvas.width / 10)); // Scale stroke
}

// Ensure initial intrinsic width and height matches the resolution variable
drawCanvas.width = 28;
drawCanvas.height = 28;
initCanvas();

function getEventPos(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

drawCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const pos = getEventPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
});

drawCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const pos = getEventPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
});

drawCanvas.addEventListener('mouseup', () => {
    isDrawing = false;
    runInference();
});
drawCanvas.addEventListener('mouseout', () => {
    isDrawing = false;
    runInference();
});

document.getElementById('clearBtn').addEventListener('click', () => {
    initCanvas();
    if (globalTrainedNN) {
        document.getElementById('inferenceResult').innerText = `Predicted: None`;
    }
});

if (inferenceUploadBtn && inferenceImportInput) {
    inferenceUploadBtn.addEventListener('click', () => {
        inferenceImportInput.click();
    });

    inferenceImportInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const bitmap = await createImageBitmap(file);
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
            ctx.drawImage(bitmap, 0, 0, drawCanvas.width, drawCanvas.height);
            bitmap.close();

            runInference();
        } catch (err) {
            console.error("Failed to load inference image", err);
        }
        inferenceImportInput.value = '';
    });
}

function runInference() {
    if (!globalTrainedNN || globalLabels.length === 0) return;

    const xData = getCanvasData();
    // Fast CPU execution on active Canvas instance
    const preds = cpuNNForward(globalTrainedNN.layers, xData, 1);

    let maxIdx = 0; let maxVal = -Infinity;
    for (let i = 0; i < preds.length; i++) {
        if (preds[i] > maxVal) {
            maxVal = preds[i];
            maxIdx = i;
        }
    }

    const inferenceResult = document.getElementById('inferenceResult');
    // If array values entirely sum to 0 (Black screen check to suppress phantom draws limit)
    let isEmpty = true;
    for (let j = 0; j < xData.length; j++) { if (xData[j] > 0) { isEmpty = false; break; } }

    if (isEmpty) {
        inferenceResult.innerText = `Predicted: None`;
    } else if (maxVal > 0.3) {
        inferenceResult.innerText = `Predicted: ${globalLabels[maxIdx]} (${(maxVal * 100).toFixed(1)}%)`;
    } else {
        inferenceResult.innerText = `Predicted: Unsure...`;
    }
}

// Label Management
createLabelBtn.addEventListener('click', () => {
    const txt = newLabelInput.value.trim();
    if (txt === '' || dataset[txt]) return;

    dataset[txt] = [];

    const opt = document.createElement('option');
    opt.value = txt;
    opt.innerText = txt;
    labelSelect.appendChild(opt);

    labelSelect.value = txt;
    newLabelInput.value = '';

    updateUI();
});

labelSelect.addEventListener('change', updateUI);

let currentCanvasSize = 28;

function getCanvasSize() {
    return currentCanvasSize;
}

const imageSettingsOverlay = document.getElementById('imageSettingsOverlay');
const imageSettingsBtn = document.getElementById('imageSettingsBtn');
const canvasSizeInputNode = document.getElementById('canvasSizeInput');

if (imageSettingsBtn && imageSettingsOverlay) {
    imageSettingsBtn.addEventListener('click', () => {
        if (canvasSizeInputNode) canvasSizeInputNode.value = currentCanvasSize;
        imageSettingsOverlay.style.display = 'flex';
    });
}

function closeImageSettings() {
    if (imageSettingsOverlay) imageSettingsOverlay.style.display = 'none';
}

document.getElementById('isCloseBtn')?.addEventListener('click', closeImageSettings);
document.getElementById('isCancelBtn')?.addEventListener('click', closeImageSettings);
imageSettingsOverlay?.addEventListener('click', (e) => {
    if (e.target === imageSettingsOverlay) closeImageSettings();
});

document.getElementById('isApplyBtn')?.addEventListener('click', () => {
    if (!canvasSizeInputNode) return;
    let size = parseInt(canvasSizeInputNode.value);
    if (isNaN(size) || size < 8) size = 8;
    if (size > 128) size = 128;

    const applySize = (newSize) => {
        currentCanvasSize = newSize;
        drawCanvas.width = newSize;
        drawCanvas.height = newSize;
        initCanvas();
        closeImageSettings();
    };

    if (size !== currentCanvasSize) {
        if (totalSamples > 0) {
            if (confirm("Changing canvas resolution will clear your current dataset to maintain tensor consistency. Proceed?")) {
                dataset = {};
                totalSamples = 0;
                labelSelect.innerHTML = '<option value="" disabled selected>-- Select or Create --</option>';
                applySize(size);
                updateUI();
            } else {
                canvasSizeInputNode.value = currentCanvasSize;
            }
        } else {
            applySize(size);
        }
    } else {
        closeImageSettings();
    }
});

function getCanvasData() {
    const size = getCanvasSize();
    // Resize down for MNIST-like neural networks
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = size;
    tempCanvas.height = size;
    const tCtx = tempCanvas.getContext('2d');

    // Draw scaled down
    tCtx.drawImage(drawCanvas, 0, 0, size, size);

    // Extract grayscale pixels (0.0 to 1.0)
    const imgData = tCtx.getImageData(0, 0, size, size).data;
    const floatData = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) {
        // imgData has R,G,B,A => we grab R. Since background is black (0) and stroke is white (255), we normalize / 255.
        floatData[i] = imgData[i * 4] / 255.0;
    }
    return floatData;
}

addExampleBtn.addEventListener('click', () => {
    const lbl = labelSelect.value;
    if (!lbl || !dataset[lbl]) return;

    dataset[lbl].push(getCanvasData());
    totalSamples++;

    initCanvas(); // Clear
    updateUI();
});

importImagesBtn.addEventListener('click', () => {
    imageImportInput.click();
});

imageImportInput.addEventListener('change', async (e) => {
    const lbl = labelSelect.value;
    if (!lbl || !dataset[lbl]) return;

    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    statusLog.innerText = `Processing ${files.length} images using Hardware Acceleration...`;

    // Quick hidden canvas for GPU-accelerated scalable rasterization
    const size = getCanvasSize();
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = size;
    tempCanvas.height = size;
    const tCtx = tempCanvas.getContext('2d', { willReadFrequently: true });

    let count = 0;
    for (const file of files) {
        try {
            // createImageBitmap uniquely decodes the image asynchronously via GPU hardware acceleration
            const bitmap = await createImageBitmap(file);

            // Standardize background cleanly
            tCtx.fillStyle = 'black';
            tCtx.fillRect(0, 0, size, size);

            // Hardware-accelerated image scaling and interpolation onto the 28x28 mathematical coordinate grid
            tCtx.drawImage(bitmap, 0, 0, size, size);
            bitmap.close();

            const imgData = tCtx.getImageData(0, 0, size, size).data;
            const floatData = new Float32Array(size * size);

            // Extract grayscale array equivalent scaling correctly to Float32 Tensors inside Javascript
            for (let i = 0; i < size * size; i++) {
                const r = imgData[i * 4];
                const g = imgData[i * 4 + 1];
                const b = imgData[i * 4 + 2];
                // Human visually weighted luminance constants mapping down from 0-255 RGB depth
                const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                floatData[i] = luminance;
            }

            dataset[lbl].push(floatData);
            count++;
            totalSamples++;
        } catch (err) {
            console.error("Failed processing an image", err);
        }
    }

    imageImportInput.value = ''; // Reset input selection
    updateUI();
    statusLog.innerText = `Successfully imported ${count} images into "${lbl}". Ready to check parameters and train.`;
});

function updateUI() {
    // 1. Refresh Dataset List
    datasetList.innerHTML = '';
    const keys = Object.keys(dataset);

    if (keys.length === 0) {
        datasetList.innerHTML = '<div class="data-list-empty">No data added yet.</div>';
    } else {
        keys.forEach(k => {
            const count = dataset[k].length;
            const row = document.createElement('div');
            row.className = 'label-item';
            row.style.alignItems = 'center';
            row.innerHTML = `
                <div style="flex: 1;">
                    <span>${k}</span>
                    <span class="label-count" style="margin-left: 0.5rem;">(${count} samples)</span>
                </div>
                <button class="view-btn" style="width: auto; margin-top: 0; padding: 0.25rem 0.75rem;" data-lbl="${k}">View Grid</button>
            `;
            datasetList.appendChild(row);

            // Hook up grid gallery button
            row.querySelector('.view-btn').addEventListener('click', () => openGallery(k));
        });
    }

    // 2. Button states
    const hasLabels = keys.length > 0;
    const hasSelection = labelSelect.value !== "";
    const hasData = totalSamples > 0;

    addExampleBtn.disabled = !hasSelection;
    importImagesBtn.disabled = !hasSelection;
    exportBtn.disabled = !hasData;
    startTrainBtn.disabled = totalSamples < 10 || keys.length < 2; // Need at least 10 samples & 2 classes

    if (startTrainBtn.disabled && totalSamples > 0) {
        statusLog.innerText = `Need at least 2 categories and 10 total samples to train. (Currently: ${keys.length} categories, ${totalSamples} samples)`;
    } else if (hasData) {
        statusLog.innerText = `Ready to train Network on ${totalSamples} total samples across ${keys.length} categories.`;
    }

    // 3. Imbalance checks
    if (keys.length > 1 && totalSamples > 0) {
        const counts = keys.map(k => dataset[k].length);
        const max = Math.max(...counts);
        const min = Math.min(...counts);

        // If max group is 3x larger than min group, throw warning
        if (min === 0 || max / min > 3.0) {
            warningBox.style.display = 'block';
        } else {
            warningBox.style.display = 'none';
        }
    } else {
        warningBox.style.display = 'none';
    }
}

// Import/Export
exportBtn.addEventListener('click', () => {
    // Convert Float32Arrays to standard arrays for JSON serialization
    const exportable = {};
    for (const [key, val] of Object.entries(dataset)) {
        exportable[key] = val.map(arr => Array.from(arr));
    }

    const blob = new Blob([JSON.stringify(exportable)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dataset_snapshot.json`;
    a.click();
    URL.revokeObjectURL(url);

    statusLog.innerText = `Exported dataset with ${totalSamples} total samples.`;
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (evt) {
        try {
            const parsed = JSON.parse(evt.target.result);
            dataset = {};
            totalSamples = 0;
            labelSelect.innerHTML = '<option value="" disabled selected>-- Select or Create --</option>';

            for (const [key, valList] of Object.entries(parsed)) {
                dataset[key] = valList.map(a => new Float32Array(a));
                totalSamples += valList.length;

                const opt = document.createElement('option');
                opt.value = key;
                opt.innerText = key;
                labelSelect.appendChild(opt);
            }
            labelSelect.value = Object.keys(dataset)[0];
            updateUI();

            statusLog.innerText = `Imported dataset with ${totalSamples} samples.`;

        } catch (err) {
            alert("Error parsing JSON dataset file.");
        }
    };
    reader.readAsText(file);
});

// Train Network
startTrainBtn.addEventListener('click', async () => {
    try {
        if (!navigator.gpu) throw new Error("WebGPU Not Supported in this browser");

        startTrainBtn.disabled = true;
        statusLog.innerText = `Preparing data...`;

        const labels = Object.keys(dataset);
        const numClasses = labels.length;

        // Flatten inputs (X) and one-hot encode targets (Y)
        const size = getCanvasSize();
        const inputSize = size * size;
        const X = new Float32Array(totalSamples * inputSize); // size * size
        const Y = new Float32Array(totalSamples * numClasses);

        let sampleIdx = 0;
        for (let l = 0; l < numClasses; l++) {
            const labelKey = labels[l];
            const examples = dataset[labelKey];

            for (let e = 0; e < examples.length; e++) {
                // Copy floats
                X.set(examples[e], sampleIdx * inputSize);
                // One Hot Encoding
                Y[sampleIdx * numClasses + l] = 1.0;
                sampleIdx++;
            }
        }

        // Read training settings from UI
        const epochs = parseInt(document.getElementById('epochsInput').value) || 500;
        const userLR = parseFloat(document.getElementById('lrInput').value);
        const learningRate = isNaN(userLR) ? 0.5 : userLR;
        const valSplit = parseFloat(document.getElementById('valSplitInput').value) || 0;
        const accumSteps = parseInt(document.getElementById('accumStepsInput').value) || 1;

        // Shuffle helper
        function shuffleData(X, Y, totalSamples, inputSize, numClasses) {
            const indices = Array.from({ length: totalSamples }, (_, i) => i);
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }
            const sX = new Float32Array(X.length);
            const sY = new Float32Array(Y.length);
            for (let i = 0; i < totalSamples; i++) {
                const src = indices[i];
                sX.set(X.subarray(src * inputSize, src * inputSize + inputSize), i * inputSize);
                sY.set(Y.subarray(src * numClasses, src * numClasses + numClasses), i * numClasses);
            }
            return { X: sX, Y: sY };
        }

        const numValSamples = Math.floor(totalSamples * valSplit);
        const numTrainSamples = totalSamples - numValSamples;

        if (numTrainSamples === 0) {
            statusLog.innerText = "Error: Validation split too high (0 training samples left).";
            startTrainBtn.disabled = false;
            return;
        }

        let trainX = X, trainY = Y;
        let valX = null, valY = null;
        if (numValSamples > 0) {
            const shuffled = shuffleData(X, Y, totalSamples, inputSize, numClasses);
            trainX = shuffled.X.slice(0, numTrainSamples * inputSize);
            trainY = shuffled.Y.slice(0, numTrainSamples * numClasses);
            valX = shuffled.X.slice(numTrainSamples * inputSize);
            valY = shuffled.Y.slice(numTrainSamples * numClasses);
        }

        statusLog.innerText = `Requesting GPU/CPU Device...`;
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter && document.getElementById('deviceSelect').value === 'gpu') throw new Error("No WebGPU Adapter found!");
        let device = null;
        if (adapter) {
            const requiredLimits = {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
                maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize
            };
            device = await adapter.requestDevice({ requiredLimits });
        }

        const activationFunc = 'relu'; // Default hidden activation
        let deviceMode = document.getElementById('deviceSelect').value;
        const nn = new WebGPUNeuralNetwork(device);

        if (device && (deviceMode === 'gpu' || deviceMode === 'cnn_gpu')) await nn.init();

        let modelName = 'Dense Neural Network';
        let usedCustomArch = false;

        // Try custom architecture first
        if (modelArchitecture && deviceMode !== 'tfjs') {
            const isGPU = (deviceMode === 'gpu' || deviceMode === 'cnn_gpu');
            const hasCNN = modelArchitecture.some(l => l.type === 'conv2d' || l.type === 'maxpool2d' || l.type === 'averagepool2d' || l.type === 'batchnorm');
            usedCustomArch = buildModelFromArchitecture(nn, device, numClasses, activationFunc, hasCNN, isGPU);
            if (usedCustomArch) {
                modelName = `Custom ${hasCNN ? 'CNN' : 'MLP'} (${isGPU ? 'WebGPU' : 'CPU'})`;
                // Override deviceMode for training loop routing
                if (hasCNN && isGPU) deviceMode = 'cnn_gpu';
                else if (hasCNN && !isGPU) deviceMode = 'cnn';
                else if (!hasCNN && isGPU) deviceMode = 'gpu';
                else deviceMode = 'cpu';
            }
        }

        if (!usedCustomArch) {
            if (deviceMode === 'cpu') {
                modelName = 'Dense NN (CPU)';
            } else if (deviceMode === 'cnn') {
                modelName = 'Convolutional NN (CPU)';
                nn.layers.push(new Conv2DLayer(null, 1, size, size, 8, 3, 'relu'));
                const outW = size - 3 + 1;
                const poolOutW = Math.floor((outW - 2) / 2) + 1;
                if (poolOutW <= 0) throw new Error("Canvas too small for this CNN architecture.");
                nn.layers.push(new MaxPooling2DLayer(8, outW, outW, 2, 2));
                nn.layers.push(new FlattenLayer(8, poolOutW, poolOutW));
                nn.layers.push(new DenseLayer(null, 8 * poolOutW * poolOutW, numClasses, 'sigmoid'));
            } else if (deviceMode === 'cnn_gpu') {
                modelName = 'Convolutional NN (WebGPU)';
                nn.addConvLayer(1, size, size, 8, 3, 'relu');
                const outW = size - 3 + 1;
                const poolOutW = Math.floor((outW - 2) / 2) + 1;
                if (poolOutW <= 0) throw new Error("Canvas too small for this CNN architecture.");
                nn.addMaxPool2DLayer(2, 2);
                nn.addFlattenLayer(8, poolOutW, poolOutW);
                nn.addLayer(8 * poolOutW * poolOutW, numClasses, 'sigmoid');
            } else if (deviceMode !== 'tfjs') {
                nn.addLayer(inputSize, 128, 'relu');
                nn.addLayer(128, 64, 'relu');
                nn.addLayer(64, numClasses, 'sigmoid');
            }
        }

        let tfModel = null;
        let xs = null;
        let ys = null;
        let valXs = null;
        let valYs = null;

        if (deviceMode === 'tfjs') {
            modelName = 'Convolutional NN (TFJS WebGL)';
            await tf.setBackend('webgl');
            await tf.ready();
            tfModel = tf.sequential();

            let tfAct = activationFunc;
            if (tfAct === 'sigmoid') tfAct = 'sigmoid';
            else if (tfAct === 'relu') tfAct = 'relu';
            else if (tfAct === 'tanh') tfAct = 'tanh';

            tfModel.add(tf.layers.conv2d({
                inputShape: [size, size, 1],
                filters: 8,
                kernelSize: 3,
                activation: tfAct,
                dataFormat: 'channelsLast',
                padding: 'valid',
                useBias: true
            }));

            tfModel.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

            tfModel.add(tf.layers.flatten());
            tfModel.add(tf.layers.dense({
                units: numClasses,
                activation: 'sigmoid'
            }));

            tfModel.compile({
                optimizer: tf.train.sgd(0.05),
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });

            xs = tf.tensor4d(trainX, [numTrainSamples, size, size, 1]);
            ys = tf.tensor2d(trainY, [numTrainSamples, numClasses]);
            if (numValSamples > 0) {
                valXs = tf.tensor4d(valX, [numValSamples, size, size, 1]);
                valYs = tf.tensor2d(valY, [numValSamples, numClasses]);
            }
        }

        const batchSizeSetting = parseInt(document.getElementById('batchSizeInput').value) || 0;
        const batchSize = (batchSizeSetting === 0 || batchSizeSetting >= numTrainSamples) ? numTrainSamples : batchSizeSetting;
        const useAccum = accumSteps > 1 && (deviceMode === 'gpu' || deviceMode === 'cnn_gpu');

        // Initialize gradient accumulation buffers if needed
        if (useAccum) {
            nn.initGradAccum();
        }

        const effectiveBatch = batchSize * (useAccum ? accumSteps : 1);
        statusLog.innerText = `Training ${modelName}!\nTrain: ${numTrainSamples} | Val: ${numValSamples} · Batch: ${batchSize === numTrainSamples ? 'Full' : batchSize}${useAccum ? ' × ' + accumSteps + ' accum = ' + effectiveBatch : ''} · LR: ${learningRate} · Epochs: ${epochs}`;

        // Setup Chart
        chartCard.style.display = 'block';
        if (metricsChart) {
            metricsChart.destroy();
        }

        metricsChart = new Chart(metricsChartCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Train Loss',
                        borderColor: '#ff2d2d',
                        backgroundColor: 'rgba(255,45,45,0.08)',
                        data: [],
                        yAxisID: 'y',
                        tension: 0.3,
                        fill: true,
                        borderWidth: 1.5
                    },
                    {
                        label: 'Train Acc (%)',
                        borderColor: '#00e5c8',
                        backgroundColor: 'transparent',
                        data: [],
                        yAxisID: 'y1',
                        tension: 0.3,
                        borderWidth: 1.5
                    },
                    {
                        label: 'Val Loss',
                        borderColor: '#ff9500',
                        backgroundColor: 'rgba(255,149,0,0.08)',
                        data: [],
                        yAxisID: 'y',
                        tension: 0.3,
                        fill: false,
                        borderWidth: 1.5,
                        hidden: numValSamples === 0
                    },
                    {
                        label: 'Val Acc (%)',
                        borderColor: '#a78bfa',
                        backgroundColor: 'transparent',
                        data: [],
                        yAxisID: 'y1',
                        tension: 0.3,
                        fill: false,
                        borderWidth: 1.5,
                        hidden: numValSamples === 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        grid: { color: '#1a1a1a' },
                        ticks: { color: '#444', font: { family: "'Space Mono', monospace", size: 10 } }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: { color: '#1a1a1a' },
                        ticks: { color: '#444', font: { family: "'Space Mono', monospace", size: 10 } },
                        title: { display: true, text: 'LOSS', color: '#666', font: { family: "'Space Mono', monospace", size: 10 } }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        min: 0,
                        max: 100,
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#444', font: { family: "'Space Mono', monospace", size: 10 } },
                        title: { display: true, text: 'ACCURACY (%)', color: '#666', font: { family: "'Space Mono', monospace", size: 10 } }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#888', font: { family: "'Space Mono', monospace", size: 11 } } }
                }
            }
        });

        let initialTime = performance.now();
        // inputSize is defined above

        const numBatches = Math.ceil(numTrainSamples / batchSize);

        function evaluateMetrics(predsArr, actualY, batch, classes, actType) {
            let predCopy = new Float32Array(predsArr);
            if (actType === 4 || actType === 'softmax') {
                for (let m = 0; m < batch; m++) {
                    const base = m * classes;
                    let maxVal = predCopy[base];
                    for (let i = 1; i < classes; i++) maxVal = Math.max(maxVal, predCopy[base + i]);
                    let sumExp = 0;
                    for (let i = 0; i < classes; i++) {
                        predCopy[base + i] = Math.exp(predCopy[base + i] - maxVal);
                        sumExp += predCopy[base + i];
                    }
                    for (let i = 0; i < classes; i++) predCopy[base + i] /= sumExp;
                }
            }
            let loss = 0, correct = 0;
            for (let m = 0; m < batch; m++) {
                let pMaxIdx = -1, pMaxVal = -Infinity;
                let tMaxIdx = -1, tMaxVal = -Infinity;
                for (let c = 0; c < classes; c++) {
                    let idx = m * classes + c;
                    let p = predCopy[idx];
                    let t = actualY[idx];
                    p = Math.max(1e-7, Math.min(1 - 1e-7, p));
                    loss += -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
                    if (p > pMaxVal) { pMaxVal = p; pMaxIdx = c; }
                    if (t > tMaxVal) { tMaxVal = t; tMaxIdx = c; }
                }
                if (pMaxIdx === tMaxIdx) correct++;
            }
            return { loss: loss / batch, accuracy: correct / batch };
        }

        for (let epoch = 1; epoch <= epochs; epoch++) {
            const tStart = performance.now();
            let epochLoss = 0, epochAcc = 0;
            let valLossVal = 0, valAccVal = 0;

            if (deviceMode === 'tfjs') {
                const fitOptions = { epochs: 1, batchSize: batchSize, shuffle: true, verbose: 0 };
                if (numValSamples > 0) fitOptions.validationData = [valXs, valYs];
                const tfHistory = await tfModel.fit(xs, ys, fitOptions);
                epochLoss = tfHistory.history.loss[0];
                epochAcc = tfHistory.history.acc[0];
                if (numValSamples > 0) {
                    valLossVal = tfHistory.history.val_loss[0];
                    valAccVal = tfHistory.history.val_acc[0];
                }
            } else {
                let eX = trainX, eY = trainY;
                if (batchSize < numTrainSamples) {
                    const shuffled = shuffleData(trainX, trainY, numTrainSamples, inputSize, numClasses);
                    eX = shuffled.X;
                    eY = shuffled.Y;
                }

                let accumCounter = 0;
                if (useAccum) nn.zeroGradAccum();

                for (let b = 0; b < numBatches; b++) {
                    const start = b * batchSize;
                    const end = Math.min(start + batchSize, numTrainSamples);
                    const curBatch = end - start;
                    let bX = (batchSize === numTrainSamples) ? eX : eX.slice(start * inputSize, end * inputSize);
                    let bY = (batchSize === numTrainSamples) ? eY : eY.slice(start * numClasses, end * numClasses);

                    let loss, accuracy;
                    if (deviceMode === 'cpu' || deviceMode === 'cnn') {
                        const results = cpuNNTrainStep(nn.layers, bX, bY, curBatch, learningRate);
                        loss = results.loss;
                        accuracy = results.accuracy;
                    } else {
                        const results = await nn.trainStep(bX, bY, curBatch, learningRate, { accumulate: useAccum });
                        loss = results.loss;
                        accuracy = results.accuracy;
                    }

                    epochLoss += loss;
                    epochAcc += accuracy;

                    if (useAccum) {
                        accumCounter++;
                        if (accumCounter >= accumSteps) {
                            nn.applyGradAccum(learningRate, accumSteps);
                            accumCounter = 0;
                        }
                    }
                }
                if (useAccum && accumCounter > 0) nn.applyGradAccum(learningRate, accumCounter);
                epochLoss /= numBatches;
                epochAcc /= numBatches;

                if (numValSamples > 0) {
                    const outAct = nn.layers[nn.layers.length - 1].actType;
                    let preds;
                    if (deviceMode === 'cpu' || deviceMode === 'cnn') {
                        preds = cpuNNForward(nn.layers, valX, numValSamples);
                    } else {
                        const fwdRes = await nn.forward(valX, numValSamples);
                        preds = fwdRes.result;
                        for (const buf of fwdRes.activationBuffers) {
                            buf.destroy();
                        }
                    }
                    const valRes = evaluateMetrics(preds, valY, numValSamples, numClasses, outAct);
                    valLossVal = valRes.loss;
                    valAccVal = valRes.accuracy;
                }
            }

            const loss = epochLoss;
            const accuracy = epochAcc;
            const timeMs = (performance.now() - tStart).toFixed(2);

            if (epoch === 1 || epoch % 5 === 0) {
                metricsChart.data.labels.push(epoch);
                metricsChart.data.datasets[0].data.push(loss);
                metricsChart.data.datasets[1].data.push(accuracy * 100);
                if (numValSamples > 0) {
                    metricsChart.data.datasets[2].data.push(valLossVal);
                    metricsChart.data.datasets[3].data.push(valAccVal * 100);
                }
                metricsChart.update();

                let txt = `Training Epoch ${epoch}/${epochs}\nBatch: ${batchSize === numTrainSamples ? 'Full' : batchSize + '/' + numTrainSamples} · LR: ${learningRate}\nTime: ~${timeMs}ms · Train Acc: ${(accuracy * 100).toFixed(2)}% | Loss: ${loss.toFixed(4)}`;
                if (numValSamples > 0) {
                    txt += `\nVal Acc: ${(valAccVal * 100).toFixed(2)}% | Val Loss: ${valLossVal.toFixed(4)}`;
                }
                statusLog.innerText = txt;
            }

            if (epoch % 5 === 0) await new Promise(r => setTimeout(r, 0));

            if (accuracy >= 0.999 && (!numValSamples || valAccVal >= 0.999) && epoch > 20) {
                metricsChart.data.labels.push(epoch);
                metricsChart.data.datasets[0].data.push(loss);
                metricsChart.data.datasets[1].data.push(accuracy * 100);
                if (numValSamples > 0) {
                    metricsChart.data.datasets[2].data.push(valLossVal);
                    metricsChart.data.datasets[3].data.push(valAccVal * 100);
                }
                metricsChart.update();

                statusLog.innerText = `Early Stopping! Converged at Epoch ${epoch}/${epochs}\nFinal Train Acc: 100% · Loss: ${loss.toFixed(4)}` + (numValSamples > 0 ? `\nVal Acc: ${(valAccVal * 100).toFixed(2)}% | Val Loss: ${valLossVal.toFixed(4)}` : '');
                break;
            }
        }

        const totalTime = ((performance.now() - initialTime) / 1000).toFixed(2);
        statusLog.innerText += `\n\nTraining Complete in ${totalTime}s! Optimized ${deviceMode.toUpperCase()} network successfully adapted to your custom dataset.`;

        if (deviceMode === 'tfjs') {
            xs.dispose();
            ys.dispose();
        }

        if (deviceMode === 'gpu' || deviceMode === 'cnn_gpu') {
            await nn.syncWeightsToCPU();
        }

        // Cache globally for realtime Inference predictions
        globalTrainedNN = nn;
        globalLabels = Object.keys(dataset);

        document.getElementById('inferenceResult').style.display = 'block';
        document.getElementById('exportWeightsBtn').style.display = 'block';

        startTrainBtn.disabled = false;

    } catch (e) {
        statusLog.innerText = `Error: ${e.message}`;
        startTrainBtn.disabled = false;
    }
});

document.getElementById('exportWeightsBtn').addEventListener('click', () => {
    if (!globalTrainedNN) return;

    const exportData = {
        labels: globalLabels,
        layers: globalTrainedNN.layers.map(l => {
            let data = { type: l.type || 'dense', actType: l.actType, inputSize: l.inputSize, outputSize: l.outputSize };
            if (l.type === 'conv2d') {
                data.filters = Array.from(l.filters); data.bias = Array.from(l.bias);
                data.inC = l.inC; data.inH = l.inH; data.inW = l.inW;
                data.outC = l.outC; data.k = l.k;
            } else if (l.type === 'batchnorm') {
                data.gamma = Array.from(l.gamma); data.beta = Array.from(l.beta);
                data.runningMean = Array.from(l.runningMean); data.runningVar = Array.from(l.runningVar);
                data.inC = l.inC; data.spatial = l.spatial;
            } else if (l.type === 'maxpool2d' || l.type === 'averagepool2d' || l.type === 'flatten') {
                // No weights
            } else {
                // Dense by default
                data.weights = Array.from(l.weights); data.bias = Array.from(l.bias);
            }
            return data;
        })
    };

    const blob = new Blob([JSON.stringify(exportData)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `webgpu_model_weights.json`;
    a.click();
    URL.revokeObjectURL(url);

    statusLog.innerText = `Exported trained Neural Network model parameters!`;
});

document.getElementById('weightFileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (evt) {
        try {
            const parsed = JSON.parse(evt.target.result);
            if (!parsed.labels || !parsed.layers) throw new Error("Invalid weights format. File might be missing layers or labels arrays.");

            globalLabels = parsed.labels;

            // Construct a mock NN object containing just parameter arrays for the CPU inference logic
            globalTrainedNN = {
                layers: parsed.layers.map(lData => {
                    let l = { type: lData.type || 'dense', actType: lData.actType, inputSize: lData.inputSize, outputSize: lData.outputSize };
                    if (l.type === 'conv2d') {
                        l.filters = new Float32Array(lData.filters); l.bias = new Float32Array(lData.bias);
                        l.inC = lData.inC; l.inH = lData.inH; l.inW = lData.inW;
                        l.outC = lData.outC; l.k = lData.k;
                    } else if (l.type === 'batchnorm') {
                        l.gamma = new Float32Array(lData.gamma); l.beta = new Float32Array(lData.beta);
                        l.runningMean = new Float32Array(lData.runningMean); l.runningVar = new Float32Array(lData.runningVar);
                        l.inC = lData.inC; l.spatial = lData.spatial;
                    } else if (l.type === 'maxpool2d' || l.type === 'averagepool2d' || l.type === 'flatten') {
                        // Inherit structure parameters
                    } else {
                        l.weights = new Float32Array(lData.weights); l.bias = new Float32Array(lData.bias);
                    }
                    return l;
                })
            };

            document.getElementById('inferenceResult').style.display = 'block';
            document.getElementById('exportWeightsBtn').style.display = 'block';
            document.getElementById('inferenceResult').innerText = `Predicted: None`;

            statusLog.innerText = `Successfully imported trained parameters!\nCategories loaded: ${globalLabels.join(', ')}`;
        } catch (err) {
            alert("Error parsing Neural Network weights file: " + err.message);
        }
    };
    reader.readAsText(file);
});

// -- Modal Gallery Logic --
const modalOverlay = document.getElementById('galleryModal');
const modalTitle = document.getElementById('modalTitle');
const galleryGrid = document.getElementById('galleryGrid');
const closeModalBtn = document.getElementById('closeModalBtn');

function openGallery(label) {
    modalTitle.innerText = `Dataset: "${label}"`;
    galleryGrid.innerHTML = '';

    const examples = dataset[label];

    examples.forEach((floatArr, idx) => {
        const itemBox = document.createElement('div');
        itemBox.className = 'gallery-item';

        // Render float32 back to a Canvas
        const size = getCanvasSize();
        const c = document.createElement('canvas');
        c.width = size;
        c.height = size;
        const cCtx = c.getContext('2d');
        const imgData = cCtx.createImageData(size, size);

        for (let i = 0; i < size * size; i++) {
            const val = Math.round(floatArr[i] * 255);
            const pIdx = i * 4;
            imgData.data[pIdx] = val;     // R
            imgData.data[pIdx + 1] = val;   // G
            imgData.data[pIdx + 2] = val;   // B
            imgData.data[pIdx + 3] = 255;   // A (Opaque)
        }

        cCtx.putImageData(imgData, 0, 0);

        const delBtn = document.createElement('button');
        delBtn.className = 'delete-item-btn';
        delBtn.innerText = '🗑️ Delete';
        delBtn.onclick = () => {
            // Remove from array memory
            dataset[label].splice(idx, 1);
            totalSamples--;
            // Re-render gallery list to update indices
            openGallery(label);
            // Refresh main UI counts
            updateUI();
        };

        itemBox.appendChild(c);
        itemBox.appendChild(delBtn);
        galleryGrid.appendChild(itemBox);
    });

    modalOverlay.style.display = 'flex';
}

closeModalBtn.addEventListener('click', () => {
    modalOverlay.style.display = 'none';
});

// Close when clicking outside content box
modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) modalOverlay.style.display = 'none';
});

// =====================================================
// MODEL ARCHITECTURE EDITOR
// =====================================================
let modelArchitecture = null; // null = use defaults; array = custom architecture

const meOverlay = document.getElementById('modelEditorOverlay');
const meStack = document.getElementById('meStack');
const meSummary = document.getElementById('meSummary');

// Working copy of layers while editor is open
let editorLayers = [];

function getDefaultArchitecture(mode) {
    const act = 'relu';
    if (mode === 'cnn_gpu' || mode === 'cnn' || mode === 'tfjs') {
        return [
            { type: 'conv2d', filters: 8, kernel: 3, activation: act },
            { type: 'maxpool2d', pool: 2, stride: 2 },
            { type: 'flatten' },
            { type: 'dense', units: '__auto__', activation: 'sigmoid' }
        ];
    }
    // MLP default
    return [
        { type: 'dense', units: 128, activation: act },
        { type: 'dense', units: 64, activation: act },
        { type: 'dense', units: '__auto__', activation: 'sigmoid' }
    ];
}

function computeDimensions(layers) {
    const size = getCanvasSize();
    // Input channels
    let shape = { channels: 1, height: size, width: size, flat: size * size };
    // If the architecture has no spatial layers at all, treat input as already flat (MLP mode)
    const hasSpatialLayers = layers.some(l => l.type === 'conv2d' || l.type === 'maxpool2d' || l.type === 'averagepool2d' || l.type === 'batchnorm');
    let isSpatial = hasSpatialLayers;
    const resolved = [];

    for (let i = 0; i < layers.length; i++) {
        const L = layers[i];
        const info = { ...L, _inputShape: { ...shape }, _isSpatial: isSpatial, _error: null };

        if (L.type === 'conv2d') {
            if (!isSpatial) { info._error = 'Conv2D needs spatial input'; resolved.push(info); continue; }
            const k = L.kernel || 3;
            const f = L.filters || 8;
            const outH = shape.height - k + 1;
            const outW = shape.width - k + 1;
            if (outH <= 0 || outW <= 0) { info._error = `Kernel ${k} too large for ${shape.height}x${shape.width}`; resolved.push(info); continue; }
            shape = { channels: f, height: outH, width: outW, flat: f * outH * outW };
            info._outputShape = { ...shape };
        } else if (L.type === 'maxpool2d' || L.type === 'averagepool2d') {
            if (!isSpatial) { info._error = 'Pooling needs spatial input'; resolved.push(info); continue; }
            const p = L.pool || 2;
            const s = L.stride || p;
            const outH = Math.floor((shape.height - p) / s) + 1;
            const outW = Math.floor((shape.width - p) / s) + 1;
            shape = { channels: shape.channels, height: outH, width: outW, flat: shape.channels * outH * outW };
            info._outputShape = { ...shape };
        } else if (L.type === 'batchnorm') {
            if (!isSpatial) { info._error = 'BatchNorm (currently implemented) needs spatial input'; resolved.push(info); continue; }
            info._outputShape = { ...shape };
        } else if (L.type === 'flatten') {
            if (!isSpatial) { info._error = 'Already flat'; resolved.push(info); continue; }
            shape = { channels: 0, height: 0, width: 0, flat: shape.flat };
            isSpatial = false;
            info._outputShape = { ...shape };
        } else if (L.type === 'dense') {
            if (isSpatial) { info._error = 'Dense needs flat input (add Flatten first)'; resolved.push(info); continue; }
            const units = L.units === '__auto__' ? -1 : (L.units || 64);
            shape = { channels: 0, height: 0, width: 0, flat: units === -1 ? shape.flat : units };
            info._outputShape = { ...shape };
            info._isAutoUnits = L.units === '__auto__';
        }

        resolved.push(info);
    }
    return resolved;
}

function estimateParams(layers) {
    const size = getCanvasSize();
    let total = 0;
    let inFlat = size * size;
    const hasSpatialLayers = layers.some(l => l.type === 'conv2d' || l.type === 'maxpool2d' || l.type === 'averagepool2d' || l.type === 'batchnorm');
    let isSpatial = hasSpatialLayers;
    let c = 1, h = size, w = size;

    for (const L of layers) {
        if (L.type === 'conv2d' && isSpatial) {
            const k = L.kernel || 3;
            const f = L.filters || 8;
            total += c * k * k * f + f; // filters + biases
            const outH = h - k + 1, outW = w - k + 1;
            c = f; h = outH; w = outW;
            inFlat = c * h * w;
        } else if ((L.type === 'maxpool2d' || L.type === 'averagepool2d') && isSpatial) {
            const p = L.pool || 2;
            const s = L.stride || p;
            h = Math.floor((h - p) / s) + 1;
            w = Math.floor((w - p) / s) + 1;
            inFlat = c * h * w;
        } else if (L.type === 'batchnorm' && isSpatial) {
            total += c * 2; // gamma + beta
        } else if (L.type === 'flatten') {
            isSpatial = false;
        } else if (L.type === 'dense' && !isSpatial) {
            const units = L.units === '__auto__' ? 10 : (L.units || 64);
            total += inFlat * units + units;
            inFlat = units;
        }
    }
    return total;
}

let graph = null;
let graphCanvas = null;

function renderEditorStack() {
    if (!graph) {
        graph = new LGraph();
        graphCanvas = new LGraphCanvas("#graphCanvas", graph);
        graphCanvas.background_color = "#222";

        // Define LiteGraph node types for our layers
        function createNodeType(name, color, props) {
            function Node() {
                this.addInput("In", "tensor");
                this.addOutput("Out", "tensor");
                this.properties = { ...props };
                this.color = color;
                // Add widgets for properties
                for (let k in this.properties) {
                    if (k === 'activation') {
                        this.addWidget("combo", "Act", this.properties[k], (v) => { this.properties[k] = v; }, { values: ["relu", "sigmoid", "tanh", "softmax"] });
                    } else {
                        this.addWidget("number", k, this.properties[k], (v) => { this.properties[k] = v; }, { precision: 0, step: 10 });
                    }
                }
            }
            Node.title = name;
            LiteGraph.registerNodeType("layers/" + name.toLowerCase(), Node);
        }

        createNodeType("Conv2D", "#312e81", { filters: 8, kernel: 3, activation: "relu" });
        createNodeType("MaxPool2D", "#064e3b", { pool: 2, stride: 2 });
        createNodeType("AvgPool2D", "#1e3a8a", { pool: 2, stride: 2 });
        createNodeType("BatchNorm", "#581c87", {});
        createNodeType("Flatten", "#78350f", {});
        createNodeType("Dense", "#7f1d1d", { units: 64, activation: "relu" });

        // Input Node
        function InputNode() {
            this.addOutput("Out", "tensor");
            this.color = "#171717";
        }
        InputNode.title = "Input";
        LiteGraph.registerNodeType("layers/input", InputNode);
    }

    graph.clear();

    let prevNode = null;
    let startX = 50;

    // Create Input Node
    prevNode = LiteGraph.createNode("layers/input");
    prevNode.pos = [startX, 200];
    graph.add(prevNode);
    startX += 200;

    // Convert editorLayers into graph nodes
    editorLayers.forEach(layer => {
        let typeName = layer.type === 'conv2d' ? 'conv2d' :
            layer.type === 'maxpool2d' ? 'maxpool2d' :
                layer.type === 'averagepool2d' ? 'avgpool2d' :
                    layer.type === 'batchnorm' ? 'batchnorm' :
                        layer.type === 'flatten' ? 'flatten' : 'dense';

        let node = LiteGraph.createNode("layers/" + typeName);
        node.pos = [startX, 200];

        // Apply properties
        for (let k in layer) {
            if (k !== 'type' && k !== '_error' && k !== '_outputShape' && k !== '_isSpatial' && k !== '_inputShape' && k !== '_isAutoUnits') {
                node.properties[k] = layer[k];
                // Update widget display value
                if (node.widgets) {
                    let w = node.widgets.find(wg => wg.name === k || (k === 'activation' && wg.name === 'Act'));
                    if (w) w.value = layer[k];
                }
            }
        }

        graph.add(node);
        if (prevNode) {
            prevNode.connect(0, node, 0);
        }
        prevNode = node;
        startX += 200;
    });

    graph.start();
}

// Intercept palette dragging - LiteGraph has its own menu, but we can hook the palette
document.querySelectorAll('.palette-item').forEach(item => {
    item.addEventListener('click', (e) => {
        if (!graph) return;
        let typeName = item.dataset.layerType === 'averagepool2d' ? 'avgpool2d' : item.dataset.layerType;
        let node = LiteGraph.createNode("layers/" + typeName);
        node.pos = [graphCanvas.canvas.width / 2 - graphCanvas.ds.offset[0], 200 - graphCanvas.ds.offset[1]];
        graph.add(node);
    });

    // Remove old drag n drop logic hooks
    const clone = item.cloneNode(true);
    item.parentNode.replaceChild(clone, item);
    clone.addEventListener('click', (e) => {
        if (!graph) return;
        let typeName = clone.dataset.layerType === 'averagepool2d' ? 'avgpool2d' : clone.dataset.layerType;
        let node = LiteGraph.createNode("layers/" + typeName);
        node.pos = [graphCanvas.canvas.width / 2 - graphCanvas.ds.offset[0], (graphCanvas.canvas.height / 2 - graphCanvas.ds.offset[1])];
        graph.add(node);
    });
});

// Remove old drag target logic
meStack.removeEventListener('dragover', () => { });
meStack.removeEventListener('drop', () => { });

// Open editor
document.getElementById('editModelBtn').addEventListener('click', () => {
    const mode = document.getElementById('deviceSelect').value;
    if (modelArchitecture) {
        editorLayers = JSON.parse(JSON.stringify(modelArchitecture));
    } else {
        editorLayers = getDefaultArchitecture(mode);
    }
    renderEditorStack();

    // Auto resize canvas to fit container smoothly
    setTimeout(() => {
        if (graphCanvas) {
            const rect = meStack.getBoundingClientRect();
            graphCanvas.resize(rect.width, rect.height);
        }
    }, 10);
    meOverlay.classList.add('active');
});

// Resize hook
window.addEventListener("resize", () => {
    if (meOverlay.classList.contains('active') && graphCanvas) {
        const rect = meStack.getBoundingClientRect();
        graphCanvas.resize(rect.width, rect.height);
    }
});
document.getElementById('meCloseBtn').addEventListener('click', () => meOverlay.classList.remove('active'));
document.getElementById('meCancelBtn').addEventListener('click', () => meOverlay.classList.remove('active'));
meOverlay.addEventListener('click', (e) => {
    if (e.target === meOverlay) meOverlay.classList.remove('active');
});

// Reset
document.getElementById('meResetBtn').addEventListener('click', () => {
    const mode = document.getElementById('deviceSelect').value;
    editorLayers = getDefaultArchitecture(mode);
    renderEditorStack();
});

document.getElementById('meApplyBtn').addEventListener('click', () => {
    if (!graph) return;

    // Sort nodes topologically to build the layer sequence
    const nodesInOrder = graph.computeExecutionOrder(false);

    let newArch = [];
    for (const node of nodesInOrder) {
        if (node.title === "Input") continue; // skip dummy input node
        let typeInfo = node.type.replace("layers/", "");
        if (typeInfo === "avgpool2d") typeInfo = "averagepool2d";

        let layerProxy = { type: typeInfo };
        for (const k in node.properties) {
            layerProxy[k] = node.properties[k];
        }
        newArch.push(layerProxy);
    }

    // Fallback: check linear compatibility first
    editorLayers = newArch;
    const resolved = computeDimensions(editorLayers);
    const errors = resolved.filter(l => l._error);
    if (errors.length > 0) {
        alert(`Cannot apply: A node caused a dimension mismatch: ${errors[0]._error}`);
        return;
    }

    modelArchitecture = JSON.parse(JSON.stringify(editorLayers));
    meOverlay.classList.remove('active');

    const params = estimateParams(editorLayers);
    statusLog.innerText = `Custom graph architecture applied (${editorLayers.length} nodes). Ready to train.`;
    document.getElementById('meSummary').textContent = `${editorLayers.length} layers · ~${params.toLocaleString()} parameters`;
});

// Patch train button to use custom architecture
function buildModelFromArchitecture(nn, device, numClasses, activationFunc, isCNN, isGPU) {
    if (!modelArchitecture) return false; // use defaults

    const size = getCanvasSize();
    let c = 1, h = size, w = size;
    let isSpatial = true;
    let flatSize = size * size;

    for (const L of modelArchitecture) {
        if (L.type === 'conv2d') {
            const k = L.kernel || 3;
            const f = L.filters || 8;
            const act = L.activation || activationFunc;
            if (isGPU) {
                nn.addConvLayer(c, h, w, f, k, act);
            } else {
                nn.layers.push(new Conv2DLayer(null, c, h, w, f, k, act));
            }
            const outH = h - k + 1;
            const outW = w - k + 1;
            c = f; h = outH; w = outW;
            flatSize = c * h * w;
        } else if (L.type === 'maxpool2d') {
            const p = L.pool || 2;
            const s = L.stride || p;
            if (isGPU) {
                nn.addMaxPool2DLayer(p, s);
            } else {
                nn.layers.push(new MaxPooling2DLayer(c, h, w, p, s));
            }
            h = Math.floor((h - p) / s) + 1;
            w = Math.floor((w - p) / s) + 1;
            flatSize = c * h * w;
        } else if (L.type === 'averagepool2d') {
            const p = L.pool || 2;
            const s = L.stride || p;
            if (isGPU) {
                nn.addAveragePool2DLayer(p, s);
            } else {
                nn.layers.push(new AveragePooling2DLayer(c, h, w, p, s));
            }
            h = Math.floor((h - p) / s) + 1;
            w = Math.floor((w - p) / s) + 1;
            flatSize = c * h * w;
        } else if (L.type === 'flatten') {
            if (isGPU) {
                nn.addFlattenLayer(c, h, w);
            } else {
                nn.layers.push(new FlattenLayer(c, h, w));
            }
            isSpatial = false;
        } else if (L.type === 'batchnorm') {
            if (isGPU) {
                nn.addBatchNormLayer(c, h * w);
            } else {
                nn.layers.push(new BatchNormalizationLayer(null, c, h * w));
            }
        } else if (L.type === 'dense') {
            const units = L.units === '__auto__' ? numClasses : (L.units || 64);
            const act = L.units === '__auto__' ? 'sigmoid' : (L.activation || activationFunc);
            if (isGPU) {
                nn.addLayer(flatSize, units, act);
            } else {
                nn.layers.push(new DenseLayer(null, flatSize, units, act));
            }
            flatSize = units;
        }
    }

    return true;
}

// Expose globally so train.js train logic can use it
window._buildModelFromArchitecture = buildModelFromArchitecture;
window._modelArchitecture = () => modelArchitecture;

// =====================================================
// TRAINING SETTINGS
// =====================================================
const epochsInput = document.getElementById('epochsInput');
const lrInput = document.getElementById('lrInput');
const batchSizeInput = document.getElementById('batchSizeInput');

function getDefaultLR() {
    const mode = document.getElementById('deviceSelect').value;
    return (mode === 'cnn' || mode === 'cnn_gpu' || mode === 'tfjs') ? 0.05 : 0.5;
}

// Auto-adjust LR when backend changes
document.getElementById('deviceSelect').addEventListener('change', () => {
    lrInput.value = getDefaultLR();
});

// Set initial LR based on default backend
lrInput.value = getDefaultLR();

// Restore defaults
document.getElementById('restoreDefaultsBtn').addEventListener('click', () => {
    epochsInput.value = 500;
    lrInput.value = getDefaultLR();
    batchSizeInput.value = '0';
    document.getElementById('accumStepsInput').value = '1';
});

// =====================================================
// THEME TOGGLE
// =====================================================
const themeToggleBtn = document.getElementById('themeToggle');

function updateThemeIcon() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    themeToggleBtn.textContent = isLight ? '☾' : '☀';
    themeToggleBtn.title = isLight ? 'Switch to dark mode' : 'Switch to light mode';
}

updateThemeIcon();

themeToggleBtn.addEventListener('click', () => {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';

    // Add transition class for smooth animation
    document.body.classList.add('theme-transition');

    if (isLight) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('unik-theme', 'dark');
    } else {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('unik-theme', 'light');
    }

    updateThemeIcon();

    // Remove transition class after animation
    setTimeout(() => document.body.classList.remove('theme-transition'), 350);
});
