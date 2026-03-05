class FlattenLayer {
    constructor(inChannels, inHeight, inWidth) {
        this.type = 'flatten';
        this.inC = inChannels;
        this.inH = inHeight;
        this.inW = inWidth;
        this.inputSize = inChannels * inHeight * inWidth;
        this.outputSize = this.inputSize;
    }
}

class Conv2DLayer {
    constructor(device, inChannels, inHeight, inWidth, outChannels, kernelSize, activation = 'relu') {
        this.type = 'conv2d';
        this.device = device;
        this.inC = inChannels;
        this.inH = inHeight;
        this.inW = inWidth;
        this.outC = outChannels;
        this.k = kernelSize;
        this.outH = inHeight - kernelSize + 1;
        this.outW = inWidth - kernelSize + 1;
        this.inputSize = inChannels * inHeight * inWidth;
        this.outputSize = outChannels * this.outH * this.outW;
        this.actType = activation === 'relu' ? 1 : (activation === 'tanh' ? 2 : (activation === 'softmax' ? 4 : 0));

        const fanIn = this.inC * this.k * this.k;
        const fanOut = this.outC * this.k * this.k;
        const limit = Math.sqrt(6 / (fanIn + fanOut));

        this.filters = new Float32Array(this.outC * this.inC * this.k * this.k);
        for (let i = 0; i < this.filters.length; i++) this.filters[i] = (Math.random() * 2 - 1) * limit;

        this.bias = new Float32Array(this.outC);
        for (let i = 0; i < this.bias.length; i++) this.bias[i] = 0.0;

        if (device) {
            this.filterBuffer = device.createBuffer({
                size: this.filters.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            device.queue.writeBuffer(this.filterBuffer, 0, this.filters);

            this.biasBuffer = device.createBuffer({
                size: this.bias.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            device.queue.writeBuffer(this.biasBuffer, 0, this.bias);
        }
    }
}

class DenseLayer {
    constructor(device, inputSize, outputSize, activation = 'sigmoid') {
        this.device = device;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.actType = activation === 'relu' ? 1 : (activation === 'tanh' ? 2 : (activation === 'softmax' ? 4 : 0));

        // Weights & biases initialized uniformly
        this.weights = new Float32Array(inputSize * outputSize);
        const limit = Math.sqrt(6 / (inputSize + outputSize)); // Xavier init
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] = (Math.random() * 2 - 1) * limit;
        }

        this.bias = new Float32Array(outputSize);
        for (let i = 0; i < this.bias.length; i++) {
            this.bias[i] = 0.0;
        }

        if (device) {
            this.weightBuffer = device.createBuffer({
                size: this.weights.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            device.queue.writeBuffer(this.weightBuffer, 0, this.weights);

            this.biasBuffer = device.createBuffer({
                size: this.bias.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            device.queue.writeBuffer(this.biasBuffer, 0, this.bias);
        }
    }
}

class MaxPooling2DLayer {
    constructor(inChannels, inHeight, inWidth, poolSize, stride = null) {
        this.type = 'maxpool2d';
        this.inC = inChannels;
        this.inH = inHeight;
        this.inW = inWidth;
        this.p = poolSize;
        this.s = stride || poolSize;
        this.outH = Math.floor((inHeight - this.p) / this.s) + 1;
        this.outW = Math.floor((inWidth - this.p) / this.s) + 1;
        this.outC = inChannels;
        this.inputSize = inChannels * inHeight * inWidth;
        this.outputSize = inChannels * this.outH * this.outW;
        this.actType = 3; // linear/identity
    }
}

class AveragePooling2DLayer {
    constructor(inChannels, inHeight, inWidth, poolSize, stride = null) {
        this.type = 'averagepool2d';
        this.inC = inChannels;
        this.inH = inHeight;
        this.inW = inWidth;
        this.p = poolSize;
        this.s = stride || poolSize;
        this.outH = Math.floor((inHeight - this.p) / this.s) + 1;
        this.outW = Math.floor((inWidth - this.p) / this.s) + 1;
        this.outC = inChannels;
        this.inputSize = inChannels * inHeight * inWidth;
        this.outputSize = inChannels * this.outH * this.outW;
        this.actType = 3; // linear/identity
    }
}

class BatchNormalizationLayer {
    constructor(device, numFeatures, spatialSize = 1) {
        this.type = 'batchnorm';
        this.device = device;
        this.inC = numFeatures;
        this.spatial = spatialSize;
        this.inputSize = numFeatures * spatialSize;
        this.outputSize = numFeatures * spatialSize;
        this.actType = 3; // identity

        this.gamma = new Float32Array(numFeatures).fill(1.0);
        this.beta = new Float32Array(numFeatures).fill(0.0);
        this.runningMean = new Float32Array(numFeatures).fill(0.0);
        this.runningVar = new Float32Array(numFeatures).fill(1.0);

        if (device) {
            const usageParams = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
            this.gammaBuf = device.createBuffer({ size: this.gamma.byteLength, usage: usageParams });
            this.betaBuf = device.createBuffer({ size: this.beta.byteLength, usage: usageParams });
            this.runMeanBuf = device.createBuffer({ size: this.runningMean.byteLength, usage: usageParams });
            this.runVarBuf = device.createBuffer({ size: this.runningVar.byteLength, usage: usageParams });
            this.cacheMeanBuf = device.createBuffer({ size: this.runningMean.byteLength, usage: usageParams });
            this.cacheVarBuf = device.createBuffer({ size: this.runningVar.byteLength, usage: usageParams });

            device.queue.writeBuffer(this.gammaBuf, 0, this.gamma);
            device.queue.writeBuffer(this.betaBuf, 0, this.beta);
            device.queue.writeBuffer(this.runMeanBuf, 0, this.runningMean);
            device.queue.writeBuffer(this.runVarBuf, 0, this.runningVar);
        }
    }
}

class WebGPUNeuralNetwork {
    constructor(device) {
        this.device = device;
        this.layers = [];
        this.pipelineForward = null;
        this.pipelineBackpropWeights = null;
        this.pipelineBackpropErrors = null;
        this.pipelineInitialError = null;
        this.pipelineOptimizer = null;
        this.pipelineConvForward = null;
        this.pipelineConvBackpropFilters = null;
        this.pipelineConvBackpropInput = null;
    }

    async init() {
        // ===== CNN SHADERS =====
        const shaderConvForward = `
override inC: u32;
override inH: u32;
override inW: u32;
override outC: u32;
override ks: u32;
override outH: u32;
override outW: u32;
override actType: u32;

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> filters : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform> p : array<vec4<u32>, 3>;

const TILE_SIZE = 16u;
var<workgroup> subTileA: array<f32, 256>;
var<workgroup> subTileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let batch = p[0].x; 

    
    let M = outC; 
    let N = batch * outH * outW; 
    let K_dim = inC * ks * ks; 
    
    let localRow = lid.y;
    let localCol = lid.x;
    let flatLocalId = localRow * TILE_SIZE + localCol;
    
    let globalRow = wgid.y * TILE_SIZE + localRow; // oc
    
    let numWgTotalX = (N + TILE_SIZE - 1u) / TILE_SIZE;
    let strideWgX = numWg.x;
    
    // We stride over the N dimension using uniform wgid to satisfy workgroupBarrier rules.
    for (var wg_x = wgid.x; wg_x < numWgTotalX; wg_x = wg_x + strideWgX) {
        let globalCol = wg_x * TILE_SIZE + localCol;
        
        // 1. HOIST THE SPATIAL MATH OUTSIDE THE TILE LOOP
        var ow = 0u; var oh = 0u; var b_idx = 0u; var valid_col = false;
        if (globalCol < N) {
            ow = globalCol % outW;
            let tmp2 = globalCol / outW;
            oh = tmp2 % outH;
            b_idx = tmp2 / outH;
            valid_col = true;
        }

        var acc = 0.0;
        let numTiles = (K_dim + TILE_SIZE - 1u) / TILE_SIZE;
        
        for (var t = 0u; t < numTiles; t = t + 1u) {
            // Load A (Filters): M x K_dim
            let aRow = globalRow;
            let aCol = t * TILE_SIZE + localCol;
            if (aRow < M && aCol < K_dim) {
                subTileA[flatLocalId] = filters[aRow * K_dim + aCol];
            } else {
                subTileA[flatLocalId] = 0.0;
            }
            
            // Load B (Implicit Im2Col Input): K_dim x N
            let bRow = t * TILE_SIZE + localRow;
            
            if (bRow < K_dim && valid_col) {
                // 2. ONLY KERNEL MATH REMAINS IN THE LOOP
                let kw = bRow % ks;
                let tmp1 = bRow / ks;
                let kh = tmp1 % ks;
                let ic = tmp1 / ks;
                
                let ih = oh + kh;
                let iw = ow + kw;
                
                let in_idx = b_idx * (inC * inH * inW) + ic * (inH * inW) + ih * inW + iw;
                subTileB[flatLocalId] = input[in_idx];
            } else {
                subTileB[flatLocalId] = 0.0;
            }
            workgroupBarrier();
            
            // Dot product
            for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
                acc = acc + subTileA[localRow * TILE_SIZE + k] * subTileB[k * TILE_SIZE + localCol];
            }
            workgroupBarrier();
        }
        
        if (globalRow < M && valid_col) {
            let z = acc + bias[globalRow];
            var a: f32 = 0.0;
            if (actType == 1u) { a = max(0.0, z); }
            else if (actType == 2u) { a = tanh(z); }
            else if (actType == 4u) { a = z; } // Softmax: identity here
            else { a = 1.0 / (1.0 + exp(-z)); }
            
            let out_idx = b_idx * (outC * outH * outW) + globalRow * (outH * outW) + oh * outW + ow;
            output[out_idx] = a;
        }
    }
}
        `;

        const shaderConvBackpropFilters = `
override inC: u32;
override inH: u32;
override inW: u32;
override outC: u32;
override ks: u32;
override outH: u32;
override outW: u32;

@group(0) @binding(0) var<storage, read> dZ : array<f32>;
@group(0) @binding(1) var<storage, read> A_prev : array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_dF : array<f32>;
@group(0) @binding(3) var<storage, read_write> partial_dB : array<f32>;
@group(0) @binding(4) var<uniform> p : array<vec4<u32>, 3>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let batch = p[0].x; 
    let totalFilters = outC * inC * ks * ks;
    let totalThreads = batch * totalFilters;
    let stride = numWg.x * 64u;

    for (var idx = gid.x; idx < totalThreads; idx = idx + stride) {
        let fIdx = idx % totalFilters;
        let m = idx / totalFilters;
        
        let kw = fIdx % ks;
        let t1 = fIdx / ks;
        let kh = t1 % ks;
        let t2 = t1 / ks;
        let ic = t2 % inC;
        let oc = t2 / inC;

        var acc: f32 = 0.0;
        let m_dz_offset = m * (outC * outH * outW) + oc * (outH * outW);
        let m_in_offset = m * (inC * inH * inW) + ic * (inH * inW);
        for (var oh = 0u; oh < outH; oh = oh + 1u) {
            let oh_dz_offset = m_dz_offset + oh * outW;
            let oh_in_offset = m_in_offset + (oh + kh) * inW;
            for (var ow = 0u; ow < outW; ow = ow + 1u) {
                acc = acc + dZ[oh_dz_offset + ow] * A_prev[oh_in_offset + (ow + kw)];
            }
        }
        partial_dF[m * totalFilters + fIdx] = acc;

        // Bias gradient partial
        if (ic == 0u && kh == 0u && kw == 0u) {
            var bAcc: f32 = 0.0;
            for (var oh = 0u; oh < outH; oh = oh + 1u) {
                let oh_dz_offset = m_dz_offset + oh * outW;
                for (var ow = 0u; ow < outW; ow = ow + 1u) {
                    bAcc = bAcc + dZ[oh_dz_offset + ow];
                }
            }
            partial_dB[m * outC + oc] = bAcc;
        }
    }
}
        `;

        const shaderConvReduceFilters = `
override inC: u32;
override outC: u32;
override ks: u32;

@group(0) @binding(0) var<storage, read> partial_dF : array<f32>;
@group(0) @binding(1) var<storage, read> partial_dB : array<f32>;
@group(0) @binding(2) var<storage, read_write> dF : array<f32>;
@group(0) @binding(3) var<storage, read_write> dB : array<f32>;
@group(0) @binding(4) var<uniform> p : array<vec4<u32>, 3>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let batch = p[0].x; 
    let totalFilters = outC * inC * ks * ks;
    let stride = numWg.x * 64u;

    for (var fIdx = gid.x; fIdx < totalFilters; fIdx = fIdx + stride) {
        var sumF: f32 = 0.0;
        for (var m = 0u; m < batch; m = m + 1u) {
            sumF = sumF + partial_dF[m * totalFilters + fIdx];
        }
        dF[fIdx] = sumF / f32(batch);
    }
    
    // Also use this shader to reduce partial_dB
    // gid.x handles outC elements cleanly. 
    for (var oc = gid.x; oc < outC; oc = oc + stride) {
        var sumB: f32 = 0.0;
        for (var m = 0u; m < batch; m = m + 1u) {
            sumB = sumB + partial_dB[m * outC + oc];
        }
        dB[oc] = sumB / f32(batch);
    }
}
        `;

        const shaderConvBackpropInput = `
override inC: u32;
override inH: u32;
override inW: u32;
override outC: u32;
override ks: u32;
override outH: u32;
override outW: u32;
override prevActType: u32;

@group(0) @binding(0) var<storage, read> dZ : array<f32>;
@group(0) @binding(1) var<storage, read> filters : array<f32>;
@group(0) @binding(2) var<storage, read> A_prev : array<f32>;
@group(0) @binding(3) var<storage, read_write> dInput : array<f32>;
@group(0) @binding(4) var<uniform> p : array<vec4<u32>, 3>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let batch = p[0].x; 
    let totalIn = batch * inC * inH * inW;
    let stride = numWg.x * 64u;

    for (var idx = gid.x; idx < totalIn; idx = idx + stride) {
        let iw = idx % inW;
        let t1 = idx / inW;
        let ih = t1 % inH;
        let t2 = t1 / inH;
        let ic = t2 % inC;
        let m = t2 / inC;

        var acc: f32 = 0.0;
        let m_dz_offset = m * (outC * outH * outW);
        let ic_f_offset = ic * (ks * ks);

        for (var oc = 0u; oc < outC; oc = oc + 1u) {
            let oc_dz_offset = m_dz_offset + oc * (outH * outW);
            let oc_f_offset = oc * (inC * ks * ks) + ic_f_offset;
            for (var kh = 0u; kh < ks; kh = kh + 1u) {
                let oh_i = ih - kh;
                let kh_f_offset = oc_f_offset + kh * ks;
                if (oh_i < outH) { // unsigned check will fail naturally if it wraps, so we check < outH safely assuming two's complement underflow makes it > outH
                    let oh_dz_offset = oc_dz_offset + oh_i * outW;
                    for (var kw = 0u; kw < ks; kw = kw + 1u) {
                        let ow_i = iw - kw;
                        if (ow_i < outW) {
                            acc = acc + dZ[oh_dz_offset + ow_i] * filters[kh_f_offset + kw];
                        }
                    }
                }
            }
        }

        let a = A_prev[idx];
        var dA: f32 = 1.0;
        if (prevActType == 1u) { dA = select(0.0, 1.0, a > 0.0); }
        else if (prevActType == 2u) { dA = 1.0 - (a * a); }
        else if (prevActType == 0u) { dA = a * (1.0 - a); }
        // prevActType == 3u means identity/flatten (dA stays 1.0)
        dInput[idx] = acc * dA;
    }
}
        `;

        this.modConvFwd = this.device.createShaderModule({ code: shaderConvForward });
        this.modConvBF = this.device.createShaderModule({ code: shaderConvBackpropFilters });
        this.modConvRF = this.device.createShaderModule({ code: shaderConvReduceFilters });
        this.modConvBI = this.device.createShaderModule({ code: shaderConvBackpropInput });

        const shaderPoolForward = `
override inC: u32;
override inH: u32;
override inW: u32;
override p_size: u32;
override stride: u32;
override outH: u32;
override outW: u32;

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<storage, read_write> indices : array<u32>;
@group(0) @binding(3) var<uniform> p : vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let ow = gid.x;
    let oh = gid.y;
    if (ow >= outW || oh >= outH) { return; }

    let totalBC = p.y;  // batch * inC
    // Stride over Z dimension to handle totalBC > 65535
    for (var b_c = gid.z; b_c < totalBC; b_c = b_c + numWg.z) {
        let c = b_c % inC;
        let b = b_c / inC;

        var maxVal = -1e30;
        var maxIdx = 0xFFFFFFFFu;

        let in_b_c_offset = b * (inC * inH * inW) + c * (inH * inW);
        let start_h = oh * stride;
        let start_w = ow * stride;

        for (var ph = 0u; ph < p_size; ph = ph + 1u) {
            for (var pw = 0u; pw < p_size; pw = pw + 1u) {
                let ih = start_h + ph;
                let iw = start_w + pw;
                if (ih < inH && iw < inW) {
                    let in_idx = in_b_c_offset + ih * inW + iw;
                    let val = input[in_idx];
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = in_idx;
                    }
                }
            }
        }

        let out_idx = b_c * (outH * outW) + oh * outW + ow;
        output[out_idx] = select(maxVal, 0.0, maxIdx == 0xFFFFFFFFu);
        indices[out_idx] = maxIdx;
    }
}
        `;

        const shaderPoolBackprop = `
override inC: u32;
override outH: u32;
override outW: u32;

@group(0) @binding(0) var<storage, read> dZ : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<u32>;
@group(0) @binding(2) var<storage, read_write> dInput : array<f32>;
@group(0) @binding(3) var<uniform> p : vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let ow = gid.x;
    let oh = gid.y;
    if (ow >= outW || oh >= outH) { return; }

    let totalBC = p.y;  // batch * inC
    for (var b_c = gid.z; b_c < totalBC; b_c = b_c + numWg.z) {
        let out_idx = b_c * (outH * outW) + oh * outW + ow;
        let maxIdx = indices[out_idx];
        if (maxIdx != 0xFFFFFFFFu) {
            let grad = dZ[out_idx];
            dInput[maxIdx] = grad;
        }
    }
}
        `;

        this.modPoolFwd = this.device.createShaderModule({ code: shaderPoolForward });
        this.modPoolBwd = this.device.createShaderModule({ code: shaderPoolBackprop });

        const shaderAvgPoolForward = `
override inC: u32;
override inH: u32;
override inW: u32;
override p_size: u32;
override stride: u32;
override outH: u32;
override outW: u32;

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> p : vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let ow = gid.x;
    let oh = gid.y;
    if (ow >= outW || oh >= outH) { return; }

    let totalBC = p.y;
    for (var b_c = gid.z; b_c < totalBC; b_c = b_c + numWg.z) {
        let c = b_c % inC;
        let b = b_c / inC;

        var sumVal: f32 = 0.0;
        var count: f32 = 0.0;

        let in_b_c_offset = b * (inC * inH * inW) + c * (inH * inW);
        let start_h = oh * stride;
        let start_w = ow * stride;

        for (var ph = 0u; ph < p_size; ph = ph + 1u) {
            for (var pw = 0u; pw < p_size; pw = pw + 1u) {
                let ih = start_h + ph;
                let iw = start_w + pw;
                if (ih < inH && iw < inW) {
                    let in_idx = in_b_c_offset + ih * inW + iw;
                    sumVal = sumVal + input[in_idx];
                    count = count + 1.0;
                }
            }
        }

        let out_idx = b_c * (outH * outW) + oh * outW + ow;
        output[out_idx] = select(0.0, sumVal / count, count > 0.0);
    }
}
        `;

        const shaderAvgPoolBackprop = `
override inC: u32;
override inH: u32;
override inW: u32;
override p_size: u32;
override stride: u32;
override outH: u32;
override outW: u32;

@group(0) @binding(0) var<storage, read> dZ : array<f32>;
@group(0) @binding(1) var<storage, read_write> dInput : array<f32>;
@group(0) @binding(2) var<uniform> p : vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let iw = gid.x;
    let ih = gid.y;
    if (iw >= inW || ih >= inH) { return; }

    let totalBC = p.y;
    for (var b_c = gid.z; b_c < totalBC; b_c = b_c + numWg.z) {
        let c = b_c % inC;
        let b = b_c / inC;

        var gradSum: f32 = 0.0;
        
        for (var oh = 0u; oh < outH; oh = oh + 1u) {
            for (var ow = 0u; ow < outW; ow = ow + 1u) {
                let start_h = oh * stride;
                let start_w = ow * stride;
                if (ih >= start_h && ih < start_h + p_size && iw >= start_w && iw < start_w + p_size) {
                    var count: f32 = 0.0;
                    for (var ph = 0u; ph < p_size; ph = ph + 1u) {
                       for (var pw = 0u; pw < p_size; pw = pw + 1u) {
                           if (start_h + ph < inH && start_w + pw < inW) { count = count + 1.0; }
                       }
                    }
                    let out_idx = b_c * (outH * outW) + oh * outW + ow;
                    gradSum = gradSum + (dZ[out_idx] / count);
                }
            }
        }
        
        let in_idx = b * (inC * inH * inW) + c * (inH * inW) + ih * inW + iw;
        dInput[in_idx] = gradSum;
    }
}
        `;

        this.modAvgPoolFwd = this.device.createShaderModule({ code: shaderAvgPoolForward });
        this.modAvgPoolBwd = this.device.createShaderModule({ code: shaderAvgPoolBackprop });

        const shaderForward = `
@group(0) @binding(0) var<storage, read> matrixA : array<f32>; // X
@group(0) @binding(1) var<storage, read> matrixB : array<f32>; // W
@group(0) @binding(2) var<storage, read> vectorB : array<f32>; // b
@group(0) @binding(3) var<storage, read_write> matrixC : array<f32>; // Y
@group(0) @binding(4) var<uniform> dims : vec4<u32>; // M (Batch), K (Input), N (Output), 0

const TILE_SIZE = 16u;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;
    
    let numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;
    var acc: f32 = 0.0;
    
    for (var t = 0u; t < numTiles; t = t + 1u) {
        let tiledRowA = row;
        let tiledColA = t * TILE_SIZE + local_id.x;
        if (tiledRowA < M && tiledColA < K) {
            tileA[local_id.y][local_id.x] = matrixA[tiledRowA * K + tiledColA];
        } else {
            tileA[local_id.y][local_id.x] = 0.0;
        }
        
        let tiledRowB = t * TILE_SIZE + local_id.y;
        let tiledColB = col;
        if (tiledRowB < K && tiledColB < N) {
            tileB[local_id.y][local_id.x] = matrixB[tiledRowB * N + tiledColB];
        } else {
            tileB[local_id.y][local_id.x] = 0.0;
        }
        
        workgroupBarrier();
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + tileA[local_id.y][k] * tileB[k][local_id.x];
        }
        workgroupBarrier();
    }
    
    // Add bias and apply Activation
    if (row < M && col < N) {
        let bias = vectorB[col];
        let z = acc + bias;
        
        let actType = dims.w;
        var a: f32 = 0.0;
        if (actType == 1u) {
            a = max(0.0, z); // ReLU
        } else if (actType == 2u) {
            a = tanh(z); // Tanh
        } else if (actType == 4u) {
            a = z; // Softmax: identity here, applied separately
        } else {
            a = 1.0 / (1.0 + exp(-z)); // Sigmoid (Default 0)
        }
        
        matrixC[row * N + col] = a;
    }
}
        `;

        const shaderBackpropErrors = `
@group(0) @binding(0) var<storage, read> dZ : array<f32>; // Gradient from Next Layer (or loss) -> Shape: M x N
@group(0) @binding(1) var<storage, read> W : array<f32>; // Weights of Next Layer -> Shape: K x N 
@group(0) @binding(2) var<storage, read> A : array<f32>; // Activations of Current Layer -> Shape: M x K
@group(0) @binding(3) var<storage, read_write> dZ_prev : array<f32>; // Output Gradient -> Shape: M x K
@group(0) @binding(4) var<uniform> dims : vec4<u32>; // M (Batch), K (In), N (Out), 0

const TILE_SIZE = 16u;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // Note: We are calculating dZ_prev = (dZ * W^T) * (A * (1 - A))
    // We are essentially doing MatMul(dZ, W^T)
    let m = global_id.y; 
    let k_idx = global_id.x; 
    
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;
    
    let numTiles = (N + TILE_SIZE - 1u) / TILE_SIZE;
    var acc: f32 = 0.0;
    
    for (var t = 0u; t < numTiles; t = t + 1u) {
        let rA = m;
        let cA = t * TILE_SIZE + local_id.x;
        if (rA < M && cA < N) {
            tileA[local_id.y][local_id.x] = dZ[rA * N + cA];
        } else {
            tileA[local_id.y][local_id.x] = 0.0;
        }
        
        let rB = t * TILE_SIZE + local_id.y;
        let cB = k_idx;
        if (rB < N && cB < K) {
            // Read Transposed W
            tileB[local_id.y][local_id.x] = W[cB * N + rB];
        } else {
            tileB[local_id.y][local_id.x] = 0.0;
        }
        
        workgroupBarrier();
        for (var n = 0u; n < TILE_SIZE; n = n + 1u) {
            acc = acc + tileA[local_id.y][n] * tileB[n][local_id.x];
        }
        workgroupBarrier();
    }
    
    if (m < M && k_idx < K) {
        let a = A[m * K + k_idx];
        let actType = dims.w;
        var dA: f32 = 1.0;
        
        if (actType == 1u) {
            dA = select(0.0, 1.0, a > 0.0); // ReLU Derivative
        } else if (actType == 2u) {
            dA = 1.0 - (a * a); // Tanh Derivative
        } else if (actType == 0u) {
            dA = a * (1.0 - a); // Sigmoid Derivative
        }
        // actType == 3u or 4u: identity (dA stays 1.0) — softmax gradient handled by initial error
        
        dZ_prev[m * K + k_idx] = acc * dA;
    }
}
        `;

        const shaderBackpropWeights = `
@group(0) @binding(0) var<storage, read> dZ : array<f32>; // Shape: M x N
@group(0) @binding(1) var<storage, read> A_prev : array<f32>; // Shape: M x K
@group(0) @binding(2) var<storage, read_write> dW : array<f32>; // Output Weight Grads -> Shape: K x N
@group(0) @binding(3) var<storage, read_write> dB : array<f32>; // Output Bias Grads -> Shape: N
@group(0) @binding(4) var<uniform> dims : vec4<u32>; // M (Batch), K (In), N (Out), 0

const TILE_SIZE = 16u;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row_k = global_id.y; 
    let col_n = global_id.x; 
    
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;
    
    // MatMul(A_prev^T, dZ) -> Shape K x N
    // tileA will cache A_prev^T, tileB will cache dZ
    let numTiles = (M + TILE_SIZE - 1u) / TILE_SIZE;
    var accW: f32 = 0.0;
    
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load into tileA (A_prev^T)
        // row of A_prev^T -> K dimension -> row_k (global_id.y)
        // col of A_prev^T -> M dimension -> t * 16 + local_id.x
        let m_a = t * TILE_SIZE + local_id.x;
        let k_a = row_k;
        if (m_a < M && k_a < K) {
            tileA[local_id.y][local_id.x] = A_prev[m_a * K + k_a];
        } else {
            tileA[local_id.y][local_id.x] = 0.0;
        }
        
        // Load into tileB (dZ)
        // row of dZ -> M dimension -> t * 16 + local_id.y
        // col of dZ -> N dimension -> col_n (global_id.x)
        let m_b = t * TILE_SIZE + local_id.y;
        let n_b = col_n;
        if (m_b < M && n_b < N) {
            tileB[local_id.y][local_id.x] = dZ[m_b * N + n_b];
        } else {
            tileB[local_id.y][local_id.x] = 0.0;
        }
        
        workgroupBarrier();
        for (var n = 0u; n < TILE_SIZE; n = n + 1u) {
            accW = accW + tileA[local_id.y][n] * tileB[n][local_id.x];
        }
        workgroupBarrier();
    }
    
    // Average gradients over batch size
    if (row_k < K && col_n < N) {
        dW[row_k * N + col_n] = accW / f32(M);
    }
    
    // Bias gradient is column sum over dZ: sum(dZ, axis=0) / M
    if (row_k == 0u && col_n < N) {
        var accB: f32 = 0.0;
        for (var m = 0u; m < M; m = m + 1u) {
            accB = accB + dZ[m * N + col_n];
        }
        dB[col_n] = accB / f32(M);
    }
}
        `;

        const shaderInitialError = `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> Y : array<f32>;
@group(0) @binding(2) var<storage, read_write> dZ : array<f32>;
@group(0) @binding(3) var<uniform> info : vec4<u32>; // x: size

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let stride = numWg.x * 64u;
    for (var idx = gid.x; idx < info.x; idx = idx + stride) {
        dZ[idx] = A[idx] - Y[idx];
    }
}
        `;

        const shaderOptimizer = `
@group(0) @binding(0) var<storage, read_write> param : array<f32>;
@group(0) @binding(1) var<storage, read> grad : array<f32>;
@group(0) @binding(2) var<uniform> opt_params : vec4<f32>; // x: lr, y: size

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let stride = numWg.x * 64u;
    let size = u32(opt_params.y);
    for (var idx = gid.x; idx < size; idx = idx + stride) {
        param[idx] = param[idx] - opt_params.x * grad[idx];
    }
}
        `;

        const moduleForward = this.device.createShaderModule({ code: shaderForward });
        this.pipelineForward = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleForward, entryPoint: "main" }
        });

        const moduleBackpropErrors = this.device.createShaderModule({ code: shaderBackpropErrors });
        this.pipelineBackpropErrors = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleBackpropErrors, entryPoint: "main" }
        });

        const moduleBackpropWeights = this.device.createShaderModule({ code: shaderBackpropWeights });
        this.pipelineBackpropWeights = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleBackpropWeights, entryPoint: "main" }
        });

        const moduleInitialError = this.device.createShaderModule({ code: shaderInitialError });
        this.pipelineInitialError = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleInitialError, entryPoint: "main" }
        });

        const moduleOptimizer = this.device.createShaderModule({ code: shaderOptimizer });
        this.pipelineOptimizer = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleOptimizer, entryPoint: "main" }
        });

        // Gradient accumulation shader: accum[i] += grad[i]
        // Same bind group layout as optimizer for easy swapping
        const shaderGradAccum = `
@group(0) @binding(0) var<storage, read_write> accum : array<f32>;
@group(0) @binding(1) var<storage, read> grad : array<f32>;
@group(0) @binding(2) var<uniform> params : vec4<f32>; // y: size

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let stride = numWg.x * 64u;
    let size = u32(params.y);
    for (var idx = gid.x; idx < size; idx = idx + stride) {
        accum[idx] = accum[idx] + grad[idx];
    }
}
        `;

        const moduleGradAccum = this.device.createShaderModule({ code: shaderGradAccum });
        this.pipelineGradAccum = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleGradAccum, entryPoint: "main" }
        });

        // Softmax shader: applies softmax per sample across the output vector
        const shaderSoftmax = `
@group(0) @binding(0) var<storage, read_write> data : array<f32>;
@group(0) @binding(1) var<uniform> dims : vec2<u32>; // x: batchSize, y: outputSize

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let stride = numWg.x * 64u;
    let N = dims.y;

    for (var m = gid.x; m < dims.x; m = m + stride) {
        let base = m * N;

        // 1. Find max for numerical stability
        var maxVal = data[base];
        for (var i = 1u; i < N; i = i + 1u) {
            maxVal = max(maxVal, data[base + i]);
        }

        // 2. Compute exp(z - max) and sum
        var sumExp: f32 = 0.0;
        for (var i = 0u; i < N; i = i + 1u) {
            let e = exp(data[base + i] - maxVal);
            data[base + i] = e;
            sumExp = sumExp + e;
        }

        // 3. Normalize
        let invSum = 1.0 / sumExp;
        for (var i = 0u; i < N; i = i + 1u) {
            data[base + i] = data[base + i] * invSum;
        }
    }
}
        `;

        const moduleSoftmax = this.device.createShaderModule({ code: shaderSoftmax });
        this.pipelineSoftmax = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleSoftmax, entryPoint: "main" }
        });

        // BatchNorm Shader Forward
        const shaderBNForward = `
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<storage, read> gamma : array<f32>;
@group(0) @binding(3) var<storage, read> beta : array<f32>;
@group(0) @binding(4) var<storage, read_write> runMean : array<f32>;
@group(0) @binding(5) var<storage, read_write> runVar : array<f32>;
@group(0) @binding(6) var<storage, read_write> cacheMean : array<f32>;
@group(0) @binding(7) var<storage, read_write> cacheVar : array<f32>;
@group(0) @binding(8) var<uniform> params : vec4<u32>; // x: Batch Size, y: Channels (Features), z: Spatial Size, w: isTraining

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.y) { return; }
    
    let M = params.x;
    let spatial = params.z;
    let N_items = M * spatial;
    let isTraining = params.w;
    
    let eps = 1e-5;
    
    var mean: f32 = 0.0;
    var variance: f32 = 0.0;
    
    if (isTraining == 1u) {
        var sum: f32 = 0.0;
        for (var m = 0u; m < M; m = m + 1u) {
            for (var s = 0u; s < spatial; s = s + 1u) {
                let idx = m * (params.y * spatial) + c * spatial + s;
                sum = sum + input[idx];
            }
        }
        mean = sum / f32(N_items);
        
        var sumSq: f32 = 0.0;
        for (var m = 0u; m < M; m = m + 1u) {
            for (var s = 0u; s < spatial; s = s + 1u) {
                let idx = m * (params.y * spatial) + c * spatial + s;
                let diff = input[idx] - mean;
                sumSq = sumSq + diff * diff;
            }
        }
        variance = sumSq / f32(N_items);
        
        runMean[c] = 0.9 * runMean[c] + 0.1 * mean;
        let unbVar = sumSq / f32(max(1u, N_items - 1u));
        runVar[c] = 0.9 * runVar[c] + 0.1 * unbVar;
        
        cacheMean[c] = mean;
        cacheVar[c] = variance;
    } else {
        mean = runMean[c];
        variance = runVar[c];
    }
    
    let invStd = 1.0 / sqrt(variance + eps);
    let g = gamma[c];
    let b = beta[c];
    
    for (var m = 0u; m < M; m = m + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let idx = m * (params.y * spatial) + c * spatial + s;
            let xi = input[idx];
            let x_hat = (xi - mean) * invStd;
            output[idx] = g * x_hat + b;
        }
    }
}
        `;

        const moduleBNForward = this.device.createShaderModule({ code: shaderBNForward });
        this.pipelineBNForward = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleBNForward, entryPoint: "main" }
        });

        // BatchNorm Shader Backward
        const shaderBNBackward = `
@group(0) @binding(0) var<storage, read> dY : array<f32>;
@group(0) @binding(1) var<storage, read> X : array<f32>;
@group(0) @binding(2) var<storage, read> gamma : array<f32>;
@group(0) @binding(3) var<storage, read> cacheMean : array<f32>;
@group(0) @binding(4) var<storage, read> cacheVar : array<f32>;
@group(0) @binding(5) var<storage, read_write> dX : array<f32>;
@group(0) @binding(6) var<storage, read_write> dGamma : array<f32>;
@group(0) @binding(7) var<storage, read_write> dBeta : array<f32>;
@group(0) @binding(8) var<uniform> params : vec4<u32>; // x: M, y: C, z: spatial, w: 0

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.y) { return; }
    
    let M = params.x;
    let spatial = params.z;
    let N_items = f32(M * spatial);
    let eps = 1e-5;
    
    let mean = cacheMean[c];
    let variance = cacheVar[c];
    let invStd = 1.0 / sqrt(variance + eps);
    let g = gamma[c];
    
    var sum_dY: f32 = 0.0;
    var sum_dY_xhat: f32 = 0.0;
    
    for (var m = 0u; m < M; m = m + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let idx = m * (params.y * spatial) + c * spatial + s;
            let dyi = dY[idx];
            let xi = X[idx];
            let x_hat = (xi - mean) * invStd;
            
            sum_dY = sum_dY + dyi;
            sum_dY_xhat = sum_dY_xhat + dyi * x_hat;
        }
    }
    
    dGamma[c] = sum_dY_xhat / f32(M);
    dBeta[c] = sum_dY / f32(M);
    
    for (var m = 0u; m < M; m = m + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let idx = m * (params.y * spatial) + c * spatial + s;
            let dyi = dY[idx];
            let xi = X[idx];
            let x_hat = (xi - mean) * invStd;
            
            let dx_i = (g * invStd / N_items) * (N_items * dyi - sum_dY - x_hat * sum_dY_xhat);
            dX[idx] = dx_i;
        }
    }
}
        `;

        const moduleBNBackward = this.device.createShaderModule({ code: shaderBNBackward });
        this.pipelineBNBackward = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: moduleBNBackward, entryPoint: "main" }
        });
    }

    addBatchNormLayer(numFeatures, spatialSize = 1) {
        this.layers.push(new BatchNormalizationLayer(this.device, numFeatures, spatialSize));
    }

    addLayer(inputSize, outputSize, activation = 'sigmoid') {
        this.layers.push(new DenseLayer(this.device, inputSize, outputSize, activation));
    }

    addConvLayer(inC, inH, inW, outC, kernelSize, activation = 'relu') {
        this.layers.push(new Conv2DLayer(this.device, inC, inH, inW, outC, kernelSize, activation));
    }

    addFlattenLayer(inC, inH, inW) {
        this.layers.push(new FlattenLayer(inC, inH, inW));
    }

    addMaxPool2DLayer(poolSize, stride = null) {
        const prev = this.layers[this.layers.length - 1];
        if (prev && (prev.type === 'conv2d' || prev.type === 'maxpool2d' || prev.type === 'averagepool2d' || prev.type === 'batchnorm')) {
            this.layers.push(new MaxPooling2DLayer(prev.outC, prev.outH, prev.outW, poolSize, stride));
        } else {
            throw new Error("MaxPooling2D must follow a spatial layer.");
        }
    }

    addAveragePool2DLayer(poolSize, stride = null) {
        const prev = this.layers[this.layers.length - 1];
        if (prev && (prev.type === 'conv2d' || prev.type === 'maxpool2d' || prev.type === 'averagepool2d' || prev.type === 'batchnorm')) {
            this.layers.push(new AveragePooling2DLayer(prev.outC, prev.outH, prev.outW, poolSize, stride));
        } else {
            throw new Error("AveragePooling2D must follow a spatial layer.");
        }
    }

    async ensurePipelines() {
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            if (layer.type === 'conv2d' && !layer.pipelineFwd) {
                let prevActType = 3;
                if (i > 0) {
                    const prev = this.layers[i - 1];
                    if (prev.type === 'flatten') {
                        if (i > 1) prevActType = this.layers[i - 2].actType;
                    } else {
                        prevActType = prev.actType;
                    }
                }
                const c = { inC: layer.inC, inH: layer.inH, inW: layer.inW, outC: layer.outC, ks: layer.k, outH: layer.outH, outW: layer.outW };

                layer.pipelineFwd = await this.device.createComputePipelineAsync({
                    layout: "auto", compute: { module: this.modConvFwd, entryPoint: "main", constants: { ...c, actType: layer.actType } }
                });
                layer.pipelineBF = await this.device.createComputePipelineAsync({
                    layout: "auto", compute: { module: this.modConvBF, entryPoint: "main", constants: { ...c } }
                });
                layer.pipelineRF = await this.device.createComputePipelineAsync({
                    layout: "auto", compute: { module: this.modConvRF, entryPoint: "main", constants: { inC: c.inC, outC: c.outC, ks: c.ks } }
                });
                layer.pipelineBI = await this.device.createComputePipelineAsync({
                    layout: "auto", compute: { module: this.modConvBI, entryPoint: "main", constants: { ...c, prevActType: prevActType } }
                });
            } else if (layer.type === 'maxpool2d' && !layer.pipelineFwd) {
                const c = { inC: layer.inC, inH: layer.inH, inW: layer.inW, p_size: layer.p, stride: layer.s, outH: layer.outH, outW: layer.outW };
                layer.pipelineFwd = await this.device.createComputePipelineAsync({
                    layout: "auto", compute: { module: this.modPoolFwd, entryPoint: "main", constants: c }
                });
                layer.pipelineBwd = await this.device.createComputePipelineAsync({
                    layout: "auto", compute: { module: this.modPoolBwd, entryPoint: "main", constants: { inC: c.inC, outH: c.outH, outW: c.outW } }
                });
            } else if (layer.type === 'averagepool2d' && !layer.pipelineFwd) {
                const c = { inC: layer.inC, inH: layer.inH, inW: layer.inW, p_size: layer.p, stride: layer.s, outH: layer.outH, outW: layer.outW };
                layer.pipelineFwd = await this.device.createComputePipelineAsync({
                    layout: "auto", compute: { module: this.modAvgPoolFwd, entryPoint: "main", constants: c }
                });
                layer.pipelineBwd = await this.device.createComputePipelineAsync({
                    layout: "auto", compute: { module: this.modAvgPoolBwd, entryPoint: "main", constants: c }
                });
            }
        }
    }

    async forward(inputArray, batchSize) {
        await this.ensurePipelines();

        let buffersToClean = [];

        // Create initial Input Buffer
        let currentInputBuffer = this.device.createBuffer({
            size: inputArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(currentInputBuffer, 0, inputArray);
        buffersToClean.push(currentInputBuffer);

        const cmd = this.device.createCommandEncoder();

        let activationBuffers = [currentInputBuffer];

        // Loop sequentially through layers, chaining buffer outputs to inputs
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];

            if (layer.type === 'flatten') {
                // Flatten is identity on the flat buffer — just carry it forward
                activationBuffers.push(currentInputBuffer);
                continue;
            }

            const outputBuffer = this.device.createBuffer({
                size: batchSize * layer.outputSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            if (layer.type === 'conv2d') {
                const paramBuf = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([
                    batchSize, layer.inC, layer.inH, layer.inW,
                    layer.outC, layer.k, layer.outH, layer.outW,
                    layer.actType, 0, 0, 0
                ]));
                buffersToClean.push(paramBuf);

                const bindGroup = this.device.createBindGroup({
                    layout: layer.pipelineFwd.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: currentInputBuffer } },
                        { binding: 1, resource: { buffer: layer.filterBuffer } },
                        { binding: 2, resource: { buffer: layer.biasBuffer } },
                        { binding: 3, resource: { buffer: outputBuffer } },
                        { binding: 4, resource: { buffer: paramBuf } }
                    ]
                });

                const N = batchSize * layer.outH * layer.outW;
                const M = layer.outC;
                const pass = cmd.beginComputePass();
                pass.setPipeline(layer.pipelineFwd);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.min(Math.ceil(N / 16), 65535), Math.ceil(M / 16));
                pass.end();
            } else if (layer.type === 'maxpool2d') {
                const indicesBuffer = this.device.createBuffer({
                    size: batchSize * layer.outputSize * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                });
                buffersToClean.push(indicesBuffer);

                const totalBC = batchSize * layer.inC;
                const paramBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([batchSize, totalBC, 0, 0]));
                buffersToClean.push(paramBuf);

                const bindGroup = this.device.createBindGroup({
                    layout: layer.pipelineFwd.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: currentInputBuffer } },
                        { binding: 1, resource: { buffer: outputBuffer } },
                        { binding: 2, resource: { buffer: indicesBuffer } },
                        { binding: 3, resource: { buffer: paramBuf } }
                    ]
                });

                const pass = cmd.beginComputePass();
                pass.setPipeline(layer.pipelineFwd);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(layer.outW / 16), Math.ceil(layer.outH / 16), Math.min(totalBC, 65535));
                pass.end();
            } else if (layer.type === 'batchnorm') {
                const paramBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([batchSize, layer.inC, layer.spatial, 0])); // w: 0 for predict/inference
                buffersToClean.push(paramBuf);

                const bindGroup = this.device.createBindGroup({
                    layout: this.pipelineBNForward.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: currentInputBuffer } },
                        { binding: 1, resource: { buffer: outputBuffer } },
                        { binding: 2, resource: { buffer: layer.gammaBuf } },
                        { binding: 3, resource: { buffer: layer.betaBuf } },
                        { binding: 4, resource: { buffer: layer.runMeanBuf } },
                        { binding: 5, resource: { buffer: layer.runVarBuf } },
                        { binding: 6, resource: { buffer: layer.cacheMeanBuf } },
                        { binding: 7, resource: { buffer: layer.cacheVarBuf } },
                        { binding: 8, resource: { buffer: paramBuf } }
                    ]
                });

                const pass = cmd.beginComputePass();
                pass.setPipeline(this.pipelineBNForward);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(layer.inC / 64));
                pass.end();
            } else {
                // Dense layer (original path)
                const dimBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                this.device.queue.writeBuffer(dimBuf, 0, new Uint32Array([batchSize, layer.inputSize, layer.outputSize, layer.actType]));
                buffersToClean.push(dimBuf);

                const bindGroup = this.device.createBindGroup({
                    layout: this.pipelineForward.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: currentInputBuffer } },
                        { binding: 1, resource: { buffer: layer.weightBuffer } },
                        { binding: 2, resource: { buffer: layer.biasBuffer } },
                        { binding: 3, resource: { buffer: outputBuffer } },
                        { binding: 4, resource: { buffer: dimBuf } }
                    ]
                });

                const pass = cmd.beginComputePass();
                pass.setPipeline(this.pipelineForward);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(layer.outputSize / 16), Math.ceil(batchSize / 16));
                pass.end();
            }

            currentInputBuffer = outputBuffer;
            activationBuffers.push(currentInputBuffer);
        }

        const outLayer = this.layers[this.layers.length - 1];
        const outputSize = outLayer.outputSize;

        // Apply softmax on output layer if needed
        if (outLayer.actType === 4) {
            const smDimBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this.device.queue.writeBuffer(smDimBuf, 0, new Uint32Array([batchSize, outputSize]));
            buffersToClean.push(smDimBuf);
            const smBG = this.device.createBindGroup({
                layout: this.pipelineSoftmax.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: currentInputBuffer } },
                    { binding: 1, resource: { buffer: smDimBuf } }
                ]
            });
            const smPass = cmd.beginComputePass();
            smPass.setPipeline(this.pipelineSoftmax);
            smPass.setBindGroup(0, smBG);
            smPass.dispatchWorkgroups(Math.min(Math.ceil(batchSize / 64), 65535));
            smPass.end();
        }

        const readBuf = this.device.createBuffer({
            size: batchSize * outputSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        buffersToClean.push(readBuf);

        cmd.copyBufferToBuffer(currentInputBuffer, 0, readBuf, 0, readBuf.size);
        this.device.queue.submit([cmd.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const mapped = new Float32Array(readBuf.getMappedRange());
        const result = new Float32Array(mapped); // Copy out data
        readBuf.unmap();

        for (const buf of buffersToClean) {
            buf.destroy();
        }

        // Caller is responsible for eventually destroying activationBuffers if they want them for backprop
        return { result, activationBuffers };
    }

    _getBuf(key, size, usage) {
        if (!this._bufs) this._bufs = {};
        const existing = this._bufs[key];
        // Grow-only: reuse buffer if it's big enough, only reallocate when larger is needed
        if (!existing || existing.size < size) {
            if (existing) existing.destroy();
            this._bufs[key] = this.device.createBuffer({ size, usage });
            this._bgs = {}; // invalidate bind groups since buffer object changed
        }
        return this._bufs[key];
    }

    _getBG(key, layout, entries) {
        if (!this._bgs) this._bgs = {};
        if (!this._bgs[key]) {
            this._bgs[key] = this.device.createBindGroup({ layout, entries });
        }
        return this._bgs[key];
    }

    // Gradient descent + backpropagation framework
    async trainStep(X_data, Y_data, batchSize, learningRate = 0.05, { accumulate = false } = {}) {
        await this.ensurePipelines();

        // 1. FORWARD PASS 
        let currentInputBuffer = this._getBuf('inBuf', X_data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(currentInputBuffer, 0, X_data);

        const cmd = this.device.createCommandEncoder();
        let activationBuffers = [currentInputBuffer];

        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];

            if (layer.type === 'flatten') {
                activationBuffers.push(currentInputBuffer);
                continue;
            }

            const outputBuffer = this._getBuf(`out_${i}`, batchSize * layer.outputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

            const pass = cmd.beginComputePass();
            if (layer.type === 'conv2d') {
                const paramBuf = this._getBuf(`p_fwd_${i}`, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([
                    batchSize, layer.inC, layer.inH, layer.inW, // We leave these values around for alignment / uniform requirements
                    layer.outC, layer.k, layer.outH, layer.outW,
                    layer.actType, 0, 0, 0
                ]));

                const bindGroup = this._getBG(`bg_fwd_${i}`, layer.pipelineFwd.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: currentInputBuffer } },
                    { binding: 1, resource: { buffer: layer.filterBuffer } },
                    { binding: 2, resource: { buffer: layer.biasBuffer } },
                    { binding: 3, resource: { buffer: outputBuffer } },
                    { binding: 4, resource: { buffer: paramBuf } }
                ]);

                const N = batchSize * layer.outH * layer.outW;
                const M = layer.outC;
                pass.setPipeline(layer.pipelineFwd);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.min(Math.ceil(N / 16), 65535), Math.ceil(M / 16));
            } else if (layer.type === 'maxpool2d') {
                const indicesBuffer = this._getBuf(`indices_${i}`, batchSize * layer.outputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
                layer.indicesBuffer = indicesBuffer; // Save for backprop

                const totalBC = batchSize * layer.inC;
                const paramBuf = this._getBuf(`p_fwd_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([batchSize, totalBC, 0, 0]));

                const bindGroup = this._getBG(`bg_fwd_${i}`, layer.pipelineFwd.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: currentInputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                    { binding: 2, resource: { buffer: indicesBuffer } },
                    { binding: 3, resource: { buffer: paramBuf } }
                ]);

                pass.setPipeline(layer.pipelineFwd);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(layer.outW / 16), Math.ceil(layer.outH / 16), Math.min(totalBC, 65535));
            } else if (layer.type === 'averagepool2d') {
                const totalBC = batchSize * layer.inC;
                const paramBuf = this._getBuf(`p_avg_fwd_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([batchSize, totalBC, 0, 0]));

                const bindGroup = this._getBG(`bg_avg_fwd_${i}`, layer.pipelineFwd.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: currentInputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                    { binding: 2, resource: { buffer: paramBuf } }
                ]);

                pass.setPipeline(layer.pipelineFwd);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(layer.outW / 16), Math.ceil(layer.outH / 16), Math.min(totalBC, 65535));
            } else if (layer.type === 'batchnorm') {
                const paramBuf = this._getBuf(`p_bn_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([batchSize, layer.inC, layer.spatial, 1])); // w: 1 for training

                // We need cacheMean and cacheVar for backward pass, but since we reuse buffers across batches, we allocate them via _getBuf
                const cacheMeanBuf = this._getBuf(`bn_c_mean_${i}`, layer.inC * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
                const cacheVarBuf = this._getBuf(`bn_c_var_${i}`, layer.inC * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

                layer.cacheMeanBuf = cacheMeanBuf; // save for backprop
                layer.cacheVarBuf = cacheVarBuf;   // save for backprop

                const bindGroup = this._getBG(`bg_bn_${i}`, this.pipelineBNForward.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: currentInputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                    { binding: 2, resource: { buffer: layer.gammaBuf } },
                    { binding: 3, resource: { buffer: layer.betaBuf } },
                    { binding: 4, resource: { buffer: layer.runMeanBuf } },
                    { binding: 5, resource: { buffer: layer.runVarBuf } },
                    { binding: 6, resource: { buffer: cacheMeanBuf } },
                    { binding: 7, resource: { buffer: cacheVarBuf } },
                    { binding: 8, resource: { buffer: paramBuf } }
                ]);

                pass.setPipeline(this.pipelineBNForward);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(layer.inC / 64));
            } else {
                const dimBuf = this._getBuf(`p_fwd_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(dimBuf, 0, new Uint32Array([batchSize, layer.inputSize, layer.outputSize, layer.actType]));

                const bindGroup = this._getBG(`bg_fwd_${i}`, this.pipelineForward.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: currentInputBuffer } },
                    { binding: 1, resource: { buffer: layer.weightBuffer } },
                    { binding: 2, resource: { buffer: layer.biasBuffer } },
                    { binding: 3, resource: { buffer: outputBuffer } },
                    { binding: 4, resource: { buffer: dimBuf } }
                ]);

                pass.setPipeline(this.pipelineForward);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(layer.outputSize / 16), Math.ceil(batchSize / 16));
            }
            pass.end();

            currentInputBuffer = outputBuffer;
            activationBuffers.push(currentInputBuffer);
        }

        const outLayer = this.layers[this.layers.length - 1];

        // Apply softmax on output layer if needed (actType 4)
        if (outLayer.actType === 4) {
            const smDimBuf = this._getBuf('sm_dims', 8, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            this.device.queue.writeBuffer(smDimBuf, 0, new Uint32Array([batchSize, outLayer.outputSize]));
            const smBG = this._getBG('bg_softmax', this.pipelineSoftmax.getBindGroupLayout(0), [
                { binding: 0, resource: { buffer: activationBuffers[activationBuffers.length - 1] } },
                { binding: 1, resource: { buffer: smDimBuf } }
            ]);
            const smPass = cmd.beginComputePass();
            smPass.setPipeline(this.pipelineSoftmax);
            smPass.setBindGroup(0, smBG);
            smPass.dispatchWorkgroups(Math.min(Math.ceil(batchSize / 64), 65535));
            smPass.end();
        }

        // 2. Compute Cost/Loss Gradient (dZ_final = A - Y)
        const initialDzBuffer = this._getBuf('init_dz', batchSize * outLayer.outputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

        const yBuffer = this._getBuf('y_buf', Y_data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(yBuffer, 0, Y_data);

        const outSize = batchSize * outLayer.outputSize;
        const initialErrInfoBuf = this._getBuf('init_err_info', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(initialErrInfoBuf, 0, new Uint32Array([outSize, 0, 0, 0]));

        const initErrBindGroup = this._getBG('bg_init_err', this.pipelineInitialError.getBindGroupLayout(0), [
            { binding: 0, resource: { buffer: activationBuffers[activationBuffers.length - 1] } },
            { binding: 1, resource: { buffer: yBuffer } },
            { binding: 2, resource: { buffer: initialDzBuffer } },
            { binding: 3, resource: { buffer: initialErrInfoBuf } }
        ]);

        const passInitErr = cmd.beginComputePass();
        passInitErr.setPipeline(this.pipelineInitialError);
        passInitErr.setBindGroup(0, initErrBindGroup);
        passInitErr.dispatchWorkgroups(Math.min(Math.ceil(outSize / 64), 65535));
        passInitErr.end();

        let currentDzBuffer = initialDzBuffer;

        // 3. BACKWARD PASS Loop
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];
            const A_prev = activationBuffers[i];

            if (layer.type === 'flatten') {
                // Flatten: gradient passes through unchanged
                continue;
            }
            if (layer.type === 'maxpool2d') {
                let nextDzBuffer = null;
                if (i > 0) {
                    nextDzBuffer = this._getBuf(`next_dz_${i}`, batchSize * layer.inputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
                    // Clear the next Dz buffer to 0s for atomic-less scatter
                    cmd.clearBuffer(nextDzBuffer, 0, batchSize * layer.inputSize * 4);

                    const totalBC = batchSize * layer.inC;
                    const paramBuf = this._getBuf(`p_igrad_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([batchSize, totalBC, 0, 0]));

                    const iGradBG = this._getBG(`bg_iGrad_${i}`, layer.pipelineBwd.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: currentDzBuffer } },
                        { binding: 1, resource: { buffer: layer.indicesBuffer } },
                        { binding: 2, resource: { buffer: nextDzBuffer } },
                        { binding: 3, resource: { buffer: paramBuf } }
                    ]);

                    const passIG = cmd.beginComputePass();
                    passIG.setPipeline(layer.pipelineBwd);
                    passIG.setBindGroup(0, iGradBG);
                    passIG.dispatchWorkgroups(Math.ceil(layer.outW / 16), Math.ceil(layer.outH / 16), Math.min(totalBC, 65535));
                    passIG.end();
                }
                currentDzBuffer = nextDzBuffer;
            } else if (layer.type === 'averagepool2d') {
                let nextDzBuffer = null;
                if (i > 0) {
                    nextDzBuffer = this._getBuf(`next_dz_avg_${i}`, batchSize * layer.inputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
                    // Clear buffer to 0s? Not actually needed because average pool backward writes to all active inputs directly in shader!

                    const totalBC = batchSize * layer.inC;
                    const paramBuf = this._getBuf(`p_igrad_avg_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([batchSize, totalBC, 0, 0]));

                    const iGradBG = this._getBG(`bg_iGrad_avg_${i}`, layer.pipelineBwd.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: currentDzBuffer } },
                        { binding: 1, resource: { buffer: nextDzBuffer } },
                        { binding: 2, resource: { buffer: paramBuf } }
                    ]);

                    const passIG = cmd.beginComputePass();
                    passIG.setPipeline(layer.pipelineBwd);
                    passIG.setBindGroup(0, iGradBG);
                    passIG.dispatchWorkgroups(Math.ceil(layer.inW / 16), Math.ceil(layer.inH / 16), Math.min(totalBC, 65535));
                    passIG.end();
                }
                currentDzBuffer = nextDzBuffer;
            } else if (layer.type === 'conv2d') {
                // --- Conv2D Backward ---
                const dFBuffer = this._getBuf(`dF_${i}`, layer.filters.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
                const dBBuffer = this._getBuf(`dB_${i}`, layer.bias.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

                const totalFilters = layer.outC * layer.inC * layer.k * layer.k;
                const partialDFBuffer = this._getBuf(`p_dF_${i}`, batchSize * totalFilters * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
                const partialDBBuffer = this._getBuf(`p_dB_${i}`, batchSize * layer.outC * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

                const paramBuf = this._getBuf(`p_bwd_${i}`, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([
                    batchSize, layer.inC, layer.inH, layer.inW,
                    layer.outC, layer.k, layer.outH, layer.outW,
                    0, 0, 0, 0
                ]));

                let actIdx = i; // Input to conv layer is activationBuffers[i]
                if (i > 0 && this.layers[i - 1].type === 'maxpool2d') {
                    // Let's actually always read activationBuffers[i] regardless.
                }

                // Filter gradients (partial per batch)
                const fGradBG = this._getBG(`bg_fGrad_${i}`, layer.pipelineBF.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: currentDzBuffer } },
                    { binding: 1, resource: { buffer: activationBuffers[i] } },
                    { binding: 2, resource: { buffer: partialDFBuffer } },
                    { binding: 3, resource: { buffer: partialDBBuffer } },
                    { binding: 4, resource: { buffer: paramBuf } }
                ]);

                const passFG = cmd.beginComputePass();
                passFG.setPipeline(layer.pipelineBF);
                passFG.setBindGroup(0, fGradBG);
                passFG.dispatchWorkgroups(Math.min(Math.ceil((batchSize * totalFilters) / 64), 65535));
                passFG.end();

                // Reduce Filter gradients across batch
                const rGradBG = this._getBG(`bg_rGrad_${i}`, layer.pipelineRF.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: partialDFBuffer } },
                    { binding: 1, resource: { buffer: partialDBBuffer } },
                    { binding: 2, resource: { buffer: dFBuffer } },
                    { binding: 3, resource: { buffer: dBBuffer } },
                    { binding: 4, resource: { buffer: paramBuf } }
                ]);

                const passRG = cmd.beginComputePass();
                passRG.setPipeline(layer.pipelineRF);
                passRG.setBindGroup(0, rGradBG);
                passRG.dispatchWorkgroups(Math.min(Math.ceil(totalFilters / 64), 65535));
                passRG.end();

                // Input gradients (if not first layer)
                let nextDzBuffer = null;
                if (i > 0) {
                    nextDzBuffer = this._getBuf(`next_dz_${i}`, batchSize * layer.inputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

                    let prevActType = 3; // identity by default
                    let lookback = 1;
                    while (i - lookback >= 0) {
                        const pLayer = this.layers[i - lookback];
                        if (pLayer.type === 'flatten' || pLayer.type === 'maxpool2d') {
                            lookback++;
                        } else {
                            prevActType = pLayer.actType;
                            break;
                        }
                    }

                    const paramBufI = this._getBuf(`p_igrad_${i}`, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(paramBufI, 0, new Uint32Array([
                        batchSize, layer.inC, layer.inH, layer.inW,
                        layer.outC, layer.k, layer.outH, layer.outW,
                        prevActType, 0, 0, 0
                    ]));

                    const iGradBG = this._getBG(`bg_iGrad_${i}`, layer.pipelineBI.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: currentDzBuffer } },
                        { binding: 1, resource: { buffer: layer.filterBuffer } },
                        { binding: 2, resource: { buffer: activationBuffers[i] } },
                        { binding: 3, resource: { buffer: nextDzBuffer } },
                        { binding: 4, resource: { buffer: paramBufI } }
                    ]);

                    const totalIn = batchSize * layer.inputSize;
                    const passIG = cmd.beginComputePass();
                    passIG.setPipeline(layer.pipelineBI);
                    passIG.setBindGroup(0, iGradBG);
                    passIG.dispatchWorkgroups(Math.min(Math.ceil(totalIn / 64), 65535));
                    passIG.end();
                }

                // Optimizer OR Accumulation for filters
                if (accumulate && layer.accumF) {
                    const metaF = this._getBuf(`mf_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaF, 0, new Float32Array([0, totalFilters, 0, 0]));
                    const accFG = this._getBG(`bg_accf_${i}`, this.pipelineGradAccum.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.accumF } },
                        { binding: 1, resource: { buffer: dFBuffer } },
                        { binding: 2, resource: { buffer: metaF } }
                    ]);
                    const pAF = cmd.beginComputePass();
                    pAF.setPipeline(this.pipelineGradAccum);
                    pAF.setBindGroup(0, accFG);
                    pAF.dispatchWorkgroups(Math.min(Math.ceil(totalFilters / 64), 65535));
                    pAF.end();

                    const metaB = this._getBuf(`mb_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaB, 0, new Float32Array([0, layer.outC, 0, 0]));
                    const accBG = this._getBG(`bg_accb_${i}`, this.pipelineGradAccum.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.accumB } },
                        { binding: 1, resource: { buffer: dBBuffer } },
                        { binding: 2, resource: { buffer: metaB } }
                    ]);
                    const pAB = cmd.beginComputePass();
                    pAB.setPipeline(this.pipelineGradAccum);
                    pAB.setBindGroup(0, accBG);
                    pAB.dispatchWorkgroups(Math.min(Math.ceil(layer.outC / 64), 65535));
                    pAB.end();
                } else {
                    // Normal optimizer
                    const metaFBuf = this._getBuf(`mf_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaFBuf, 0, new Float32Array([learningRate, totalFilters, 0, 0]));
                    const optFGroup = this._getBG(`bg_optf_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.filterBuffer } },
                        { binding: 1, resource: { buffer: dFBuffer } },
                        { binding: 2, resource: { buffer: metaFBuf } }
                    ]);
                    const passOptF = cmd.beginComputePass();
                    passOptF.setPipeline(this.pipelineOptimizer);
                    passOptF.setBindGroup(0, optFGroup);
                    passOptF.dispatchWorkgroups(Math.min(Math.ceil(totalFilters / 64), 65535));
                    passOptF.end();

                    const metaBBuf = this._getBuf(`mb_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaBBuf, 0, new Float32Array([learningRate, layer.outC, 0, 0]));
                    const optBGroup = this._getBG(`bg_optb_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.biasBuffer } },
                        { binding: 1, resource: { buffer: dBBuffer } },
                        { binding: 2, resource: { buffer: metaBBuf } }
                    ]);
                    const passOptB = cmd.beginComputePass();
                    passOptB.setPipeline(this.pipelineOptimizer);
                    passOptB.setBindGroup(0, optBGroup);
                    passOptB.dispatchWorkgroups(Math.min(Math.ceil(layer.outC / 64), 65535));
                    passOptB.end();
                }

                currentDzBuffer = nextDzBuffer;

                currentDzBuffer = nextDzBuffer;

            } else if (layer.type === 'batchnorm') {
                const paramBuf = this._getBuf(`p_bn_bwd_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([batchSize, layer.inC, layer.spatial, 0]));

                const dGammaBuffer = this._getBuf(`dGamma_${i}`, layer.inC * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
                const dBetaBuffer = this._getBuf(`dBeta_${i}`, layer.inC * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
                const nextDzBuffer = this._getBuf(`next_bn_dz_${i}`, batchSize * layer.inputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

                const bnBwdBG = this._getBG(`bg_bn_bwd_${i}`, this.pipelineBNBackward.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: currentDzBuffer } },
                    { binding: 1, resource: { buffer: A_prev } },
                    { binding: 2, resource: { buffer: layer.gammaBuf } },
                    { binding: 3, resource: { buffer: layer.cacheMeanBuf } },
                    { binding: 4, resource: { buffer: layer.cacheVarBuf } },
                    { binding: 5, resource: { buffer: nextDzBuffer } },
                    { binding: 6, resource: { buffer: dGammaBuffer } },
                    { binding: 7, resource: { buffer: dBetaBuffer } },
                    { binding: 8, resource: { buffer: paramBuf } }
                ]);

                const passBNBwd = cmd.beginComputePass();
                passBNBwd.setPipeline(this.pipelineBNBackward);
                passBNBwd.setBindGroup(0, bnBwdBG);
                passBNBwd.dispatchWorkgroups(Math.ceil(layer.inC / 64));
                passBNBwd.end();

                // Optimizer OR Accumulation for gamma/beta
                if (accumulate && layer.accumGamma) {
                    const metaG = this._getBuf(`mg_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaG, 0, new Float32Array([0, layer.inC, 0, 0]));
                    const accBGG = this._getBG(`bg_accg_${i}`, this.pipelineGradAccum.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.accumGamma } },
                        { binding: 1, resource: { buffer: dGammaBuffer } },
                        { binding: 2, resource: { buffer: metaG } }
                    ]);
                    const pAG = cmd.beginComputePass();
                    pAG.setPipeline(this.pipelineGradAccum);
                    pAG.setBindGroup(0, accBGG);
                    pAG.dispatchWorkgroups(Math.ceil(layer.inC / 64));
                    pAG.end();

                    const metaB = this._getBuf(`mbp_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaB, 0, new Float32Array([0, layer.inC, 0, 0]));
                    const accBGB = this._getBG(`bg_accbp_${i}`, this.pipelineGradAccum.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.accumBeta } },
                        { binding: 1, resource: { buffer: dBetaBuffer } },
                        { binding: 2, resource: { buffer: metaB } }
                    ]);
                    const pAB = cmd.beginComputePass();
                    pAB.setPipeline(this.pipelineGradAccum);
                    pAB.setBindGroup(0, accBGB);
                    pAB.dispatchWorkgroups(Math.ceil(layer.inC / 64));
                    pAB.end();
                } else {
                    const metaGBuf = this._getBuf(`optg_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaGBuf, 0, new Float32Array([learningRate, layer.inC, 0, 0]));
                    const optBGG = this._getBG(`bg_optg_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.gammaBuf } },
                        { binding: 1, resource: { buffer: dGammaBuffer } },
                        { binding: 2, resource: { buffer: metaGBuf } }
                    ]);
                    const pOG = cmd.beginComputePass();
                    pOG.setPipeline(this.pipelineOptimizer);
                    pOG.setBindGroup(0, optBGG);
                    pOG.dispatchWorkgroups(Math.ceil(layer.inC / 64));
                    pOG.end();

                    const metaBBuf = this._getBuf(`optbp_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaBBuf, 0, new Float32Array([learningRate, layer.inC, 0, 0]));
                    const optBGB = this._getBG(`bg_optbp_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.betaBuf } },
                        { binding: 1, resource: { buffer: dBetaBuffer } },
                        { binding: 2, resource: { buffer: metaBBuf } }
                    ]);
                    const pOB = cmd.beginComputePass();
                    pOB.setPipeline(this.pipelineOptimizer);
                    pOB.setBindGroup(0, optBGB);
                    pOB.dispatchWorkgroups(Math.ceil(layer.inC / 64));
                    pOB.end();
                }

                currentDzBuffer = nextDzBuffer;

            } else {
                // --- Dense Backward (original) ---
                const dWBuffer = this._getBuf(`dW_${i}`, layer.inputSize * layer.outputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
                const dBBuffer = this._getBuf(`dB_${i}`, layer.outputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

                const bpropDimsBuf = this._getBuf(`p_dw_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(bpropDimsBuf, 0, new Uint32Array([batchSize, layer.inputSize, layer.outputSize, layer.actType]));

                const weightsGradBindGroup = this._getBG(`bg_dw_${i}`, this.pipelineBackpropWeights.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: currentDzBuffer } },
                    { binding: 1, resource: { buffer: A_prev } },
                    { binding: 2, resource: { buffer: dWBuffer } },
                    { binding: 3, resource: { buffer: dBBuffer } },
                    { binding: 4, resource: { buffer: bpropDimsBuf } }
                ]);

                const passW = cmd.beginComputePass();
                passW.setPipeline(this.pipelineBackpropWeights);
                passW.setBindGroup(0, weightsGradBindGroup);
                passW.dispatchWorkgroups(Math.ceil(layer.outputSize / 16), Math.ceil(layer.inputSize / 16));
                passW.end();

                let nextDzBuffer = null;
                if (i > 0) {
                    nextDzBuffer = this._getBuf(`next_dense_dz_${i}`, batchSize * layer.inputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

                    let prevActType = 3; // identity
                    const prevLayer = this.layers[i - 1];
                    if (prevLayer.type === 'flatten') {
                        if (i > 1) prevActType = this.layers[i - 2].actType;
                    } else {
                        prevActType = prevLayer.actType;
                    }

                    const bpropErrDimsBuf = this._getBuf(`p_err_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(bpropErrDimsBuf, 0, new Uint32Array([batchSize, layer.inputSize, layer.outputSize, prevActType]));

                    const errBindGroup = this._getBG(`bg_err_${i}`, this.pipelineBackpropErrors.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: currentDzBuffer } },
                        { binding: 1, resource: { buffer: layer.weightBuffer } },
                        { binding: 2, resource: { buffer: A_prev } },
                        { binding: 3, resource: { buffer: nextDzBuffer } },
                        { binding: 4, resource: { buffer: bpropErrDimsBuf } }
                    ]);

                    const passErr = cmd.beginComputePass();
                    passErr.setPipeline(this.pipelineBackpropErrors);
                    passErr.setBindGroup(0, errBindGroup);
                    passErr.dispatchWorkgroups(Math.ceil(layer.inputSize / 16), Math.ceil(batchSize / 16));
                    passErr.end();
                }

                // Optimizer OR Accumulation for weights/biases
                if (accumulate && layer.accumW) {
                    const wSize = layer.inputSize * layer.outputSize;
                    const metaW = this._getBuf(`mw_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaW, 0, new Float32Array([0, wSize, 0, 0]));
                    const accWG = this._getBG(`bg_accw_${i}`, this.pipelineGradAccum.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.accumW } },
                        { binding: 1, resource: { buffer: dWBuffer } },
                        { binding: 2, resource: { buffer: metaW } }
                    ]);
                    const pAW = cmd.beginComputePass();
                    pAW.setPipeline(this.pipelineGradAccum);
                    pAW.setBindGroup(0, accWG);
                    pAW.dispatchWorkgroups(Math.min(Math.ceil(wSize / 64), 65535));
                    pAW.end();

                    const metaB = this._getBuf(`mdb_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaB, 0, new Float32Array([0, layer.outputSize, 0, 0]));
                    const accBG = this._getBG(`bg_accdb_${i}`, this.pipelineGradAccum.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.accumB } },
                        { binding: 1, resource: { buffer: dBBuffer } },
                        { binding: 2, resource: { buffer: metaB } }
                    ]);
                    const pAB = cmd.beginComputePass();
                    pAB.setPipeline(this.pipelineGradAccum);
                    pAB.setBindGroup(0, accBG);
                    pAB.dispatchWorkgroups(Math.min(Math.ceil(layer.outputSize / 64), 65535));
                    pAB.end();
                } else {
                    // Normal optimizer
                    const metaWBuf = this._getBuf(`mw_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaWBuf, 0, new Float32Array([learningRate, layer.inputSize * layer.outputSize, 0, 0]));
                    const optWGroup = this._getBG(`bg_optw_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.weightBuffer } },
                        { binding: 1, resource: { buffer: dWBuffer } },
                        { binding: 2, resource: { buffer: metaWBuf } }
                    ]);
                    const passOptW = cmd.beginComputePass();
                    passOptW.setPipeline(this.pipelineOptimizer);
                    passOptW.setBindGroup(0, optWGroup);
                    passOptW.dispatchWorkgroups(Math.min(Math.ceil((layer.inputSize * layer.outputSize) / 64), 65535));
                    passOptW.end();

                    const metaBBuf = this._getBuf(`mdb_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                    this.device.queue.writeBuffer(metaBBuf, 0, new Float32Array([learningRate, layer.outputSize, 0, 0]));
                    const optBGroup = this._getBG(`bg_optdb_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                        { binding: 0, resource: { buffer: layer.biasBuffer } },
                        { binding: 1, resource: { buffer: dBBuffer } },
                        { binding: 2, resource: { buffer: metaBBuf } }
                    ]);
                    const passOptB = cmd.beginComputePass();
                    passOptB.setPipeline(this.pipelineOptimizer);
                    passOptB.setBindGroup(0, optBGroup);
                    passOptB.dispatchWorkgroups(Math.min(Math.ceil(layer.outputSize / 64), 65535));
                    passOptB.end();
                }

                currentDzBuffer = nextDzBuffer;
            }
        }

        // 4. Fetch Predicted outputs to measure metrics
        const readSize = batchSize * outLayer.outputSize * 4;
        const readBuf = this._getBuf('readBufMetrics', readSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

        cmd.copyBufferToBuffer(activationBuffers[activationBuffers.length - 1], 0, readBuf, 0, readSize);
        this.device.queue.submit([cmd.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ, 0, readSize);
        const yPred = new Float32Array(readBuf.getMappedRange(0, readSize).slice(0));
        readBuf.unmap();


        let loss = 0;
        let correct = 0;
        for (let m = 0; m < batchSize; m++) {
            let predMaxIdx = -1, predMaxVal = -Infinity;
            let targetMaxIdx = -1, targetMaxVal = -Infinity;

            for (let n = 0; n < outLayer.outputSize; n++) {
                let idx = m * outLayer.outputSize + n;
                let yP = yPred[idx];
                let yT = Y_data[idx];

                loss += -(yT * Math.log(yP + 1e-7) + (1 - yT) * Math.log(1 - yP + 1e-7));

                if (yP > predMaxVal) { predMaxVal = yP; predMaxIdx = n; }
                if (yT > targetMaxVal) { targetMaxVal = yT; targetMaxIdx = n; }
            }
            if (predMaxIdx === targetMaxIdx) correct++;
        }

        loss /= batchSize;
        const accuracy = correct / batchSize;

        // Note: activationBuffers are now cached explicitly or implicitly.
        // We only destroy the ad-hoc input & read ones.
        // currentInputBuffer is cached out_x, EXCEPT the first one which is cached 'inBuf'
        // readBuf is NOT cached, so destroy it.
        // We do not destroy the rest to reuse memory without constant reallocation.

        return { loss, accuracy };
    }

    async syncWeightsToCPU() {
        for (const layer of this.layers) {
            if (layer.type === 'flatten' || layer.type === 'maxpool2d') continue;

            let wArr, wGPU, bArr, bGPU;
            if (layer.type === 'conv2d') {
                wArr = layer.filters;
                wGPU = layer.filterBuffer;
                bArr = layer.bias;
                bGPU = layer.biasBuffer;
            } else if (layer.type === 'batchnorm') {
                wArr = layer.gamma;
                wGPU = layer.gammaBuf;
                bArr = layer.beta;
                bGPU = layer.betaBuf;
            } else {
                wArr = layer.weights;
                wGPU = layer.weightBuffer;
                bArr = layer.bias;
                bGPU = layer.biasBuffer;
            }

            const wReadBuf = this.device.createBuffer({
                size: wArr.byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const bReadBuf = this.device.createBuffer({
                size: bArr.byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            const cmd = this.device.createCommandEncoder();
            cmd.copyBufferToBuffer(wGPU, 0, wReadBuf, 0, wReadBuf.size);
            cmd.copyBufferToBuffer(bGPU, 0, bReadBuf, 0, bReadBuf.size);
            this.device.queue.submit([cmd.finish()]);

            await wReadBuf.mapAsync(GPUMapMode.READ);
            const wMapped = new Float32Array(wReadBuf.getMappedRange());
            wArr.set(wMapped);
            wReadBuf.unmap();
            wReadBuf.destroy();

            await bReadBuf.mapAsync(GPUMapMode.READ);
            const bMapped = new Float32Array(bReadBuf.getMappedRange());
            bArr.set(bMapped);
            bReadBuf.unmap();
            bReadBuf.destroy();

            // Also sync BatchNorm running stats explicitly if it's a BatchNorm layer
            if (layer.type === 'batchnorm') {
                const rmReadBuf = this.device.createBuffer({
                    size: layer.runningMean.byteLength,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                const rvReadBuf = this.device.createBuffer({
                    size: layer.runningVar.byteLength,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });

                const cmdStats = this.device.createCommandEncoder();
                cmdStats.copyBufferToBuffer(layer.runMeanBuf, 0, rmReadBuf, 0, rmReadBuf.size);
                cmdStats.copyBufferToBuffer(layer.runVarBuf, 0, rvReadBuf, 0, rvReadBuf.size);
                this.device.queue.submit([cmdStats.finish()]);

                await rmReadBuf.mapAsync(GPUMapMode.READ);
                layer.runningMean.set(new Float32Array(rmReadBuf.getMappedRange()));
                rmReadBuf.unmap();
                rmReadBuf.destroy();

                await rvReadBuf.mapAsync(GPUMapMode.READ);
                layer.runningVar.set(new Float32Array(rvReadBuf.getMappedRange()));
                rvReadBuf.unmap();
                rvReadBuf.destroy();
            }
        }
    }

    // =====================================================
    // GRADIENT ACCUMULATION
    // =====================================================

    /**
     * Initialize persistent gradient accumulation buffers for all trainable layers.
     * Call once before starting an accumulation cycle.
     */
    initGradAccum() {
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        for (const layer of this.layers) {
            if (layer.type === 'flatten' || layer.type === 'maxpool2d') continue;

            if (layer.type === 'conv2d') {
                const filterSize = layer.filters.byteLength;
                const biasSize = layer.bias.byteLength;
                layer.accumF = this.device.createBuffer({ size: filterSize, usage });
                layer.accumB = this.device.createBuffer({ size: biasSize, usage });
            } else if (layer.type === 'batchnorm') {
                const paramSize = layer.inC * 4;
                layer.accumGamma = this.device.createBuffer({ size: paramSize, usage });
                layer.accumBeta = this.device.createBuffer({ size: paramSize, usage });
            } else {
                // Dense
                const wSize = layer.inputSize * layer.outputSize * 4;
                const bSize = layer.outputSize * 4;
                layer.accumW = this.device.createBuffer({ size: wSize, usage });
                layer.accumB = this.device.createBuffer({ size: bSize, usage });
            }
        }
        this.zeroGradAccum();
        this._gradAccumReady = true;
    }

    /**
     * Zero all gradient accumulation buffers.
     */
    zeroGradAccum() {
        const cmd = this.device.createCommandEncoder();
        for (const layer of this.layers) {
            if (layer.type === 'flatten' || layer.type === 'maxpool2d') continue;
            if (layer.type === 'conv2d') {
                cmd.clearBuffer(layer.accumF);
                cmd.clearBuffer(layer.accumB);
            } else if (layer.type === 'batchnorm') {
                cmd.clearBuffer(layer.accumGamma);
                cmd.clearBuffer(layer.accumBeta);
            } else {
                cmd.clearBuffer(layer.accumW);
                cmd.clearBuffer(layer.accumB);
            }
        }
        this.device.queue.submit([cmd.finish()]);
    }

    /**
     * Apply accumulated gradients: param -= (lr / accumSteps) * accumGrad
     * Then zero the accumulation buffers for the next cycle.
     */
    applyGradAccum(learningRate, accumSteps) {
        const lr = learningRate / accumSteps;
        const cmd = this.device.createCommandEncoder();

        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            if (layer.type === 'flatten' || layer.type === 'maxpool2d') continue;

            if (layer.type === 'conv2d') {
                const totalFilters = layer.outC * layer.inC * layer.k * layer.k;

                // Apply accumulated filter gradients
                const metaF = this._getBuf(`ga_mf_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(metaF, 0, new Float32Array([lr, totalFilters, 0, 0]));
                const bgF = this._getBG(`ga_bgf_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: layer.filterBuffer } },
                    { binding: 1, resource: { buffer: layer.accumF } },
                    { binding: 2, resource: { buffer: metaF } }
                ]);
                const pF = cmd.beginComputePass();
                pF.setPipeline(this.pipelineOptimizer);
                pF.setBindGroup(0, bgF);
                pF.dispatchWorkgroups(Math.min(Math.ceil(totalFilters / 64), 65535));
                pF.end();

                // Apply accumulated bias gradients
                const metaB = this._getBuf(`ga_mb_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(metaB, 0, new Float32Array([lr, layer.outC, 0, 0]));
                const bgB = this._getBG(`ga_bgb_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: layer.biasBuffer } },
                    { binding: 1, resource: { buffer: layer.accumB } },
                    { binding: 2, resource: { buffer: metaB } }
                ]);
                const pB = cmd.beginComputePass();
                pB.setPipeline(this.pipelineOptimizer);
                pB.setBindGroup(0, bgB);
                pB.dispatchWorkgroups(Math.min(Math.ceil(layer.outC / 64), 65535));
                pB.end();
            } else if (layer.type === 'batchnorm') {
                const metaG = this._getBuf(`ga_mg_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(metaG, 0, new Float32Array([lr, layer.inC, 0, 0]));
                const bgG = this._getBG(`ga_bgg_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: layer.gammaBuf } },
                    { binding: 1, resource: { buffer: layer.accumGamma } },
                    { binding: 2, resource: { buffer: metaG } }
                ]);
                const pG = cmd.beginComputePass();
                pG.setPipeline(this.pipelineOptimizer);
                pG.setBindGroup(0, bgG);
                pG.dispatchWorkgroups(Math.ceil(layer.inC / 64));
                pG.end();

                const metaBeta = this._getBuf(`ga_mbeta_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(metaBeta, 0, new Float32Array([lr, layer.inC, 0, 0]));
                const bgBeta = this._getBG(`ga_bgbeta_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: layer.betaBuf } },
                    { binding: 1, resource: { buffer: layer.accumBeta } },
                    { binding: 2, resource: { buffer: metaBeta } }
                ]);
                const pBeta = cmd.beginComputePass();
                pBeta.setPipeline(this.pipelineOptimizer);
                pBeta.setBindGroup(0, bgBeta);
                pBeta.dispatchWorkgroups(Math.ceil(layer.inC / 64));
                pBeta.end();
            } else {
                // Dense
                const wSize = layer.inputSize * layer.outputSize;

                // Apply accumulated weight gradients
                const metaW = this._getBuf(`ga_mw_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(metaW, 0, new Float32Array([lr, wSize, 0, 0]));
                const bgW = this._getBG(`ga_bgw_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: layer.weightBuffer } },
                    { binding: 1, resource: { buffer: layer.accumW } },
                    { binding: 2, resource: { buffer: metaW } }
                ]);
                const pW = cmd.beginComputePass();
                pW.setPipeline(this.pipelineOptimizer);
                pW.setBindGroup(0, bgW);
                pW.dispatchWorkgroups(Math.min(Math.ceil(wSize / 64), 65535));
                pW.end();

                // Apply accumulated bias gradients
                const metaB = this._getBuf(`ga_mdb_${i}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.device.queue.writeBuffer(metaB, 0, new Float32Array([lr, layer.outputSize, 0, 0]));
                const bgB = this._getBG(`ga_bgdb_${i}`, this.pipelineOptimizer.getBindGroupLayout(0), [
                    { binding: 0, resource: { buffer: layer.biasBuffer } },
                    { binding: 1, resource: { buffer: layer.accumB } },
                    { binding: 2, resource: { buffer: metaB } }
                ]);
                const pB = cmd.beginComputePass();
                pB.setPipeline(this.pipelineOptimizer);
                pB.setBindGroup(0, bgB);
                pB.dispatchWorkgroups(Math.min(Math.ceil(layer.outputSize / 64), 65535));
                pB.end();
            }
        }

        this.device.queue.submit([cmd.finish()]);
        this.zeroGradAccum();
    }
}

function cpuNNForward(layers, inputArray, batchSize) {
    let currentInput = inputArray;

    for (const layer of layers) {
        if (!layer.type || layer.type === 'dense') {
            const M = batchSize;
            const K = layer.inputSize;
            const N = layer.outputSize;
            const out = new Float32Array(M * N);

            for (let m = 0; m < M; m++) {
                for (let n = 0; n < N; n++) {
                    let sum = 0;
                    for (let k = 0; k < K; k++) {
                        sum += currentInput[m * K + k] * layer.weights[k * N + n];
                    }
                    let z = sum + layer.bias[n];

                    if (layer.actType === 1) out[m * N + n] = Math.max(0, z); // ReLU
                    else if (layer.actType === 2) out[m * N + n] = Math.tanh(z); // Tanh
                    else if (layer.actType === 4) out[m * N + n] = z; // Softmax: identity here
                    else out[m * N + n] = 1.0 / (1.0 + Math.exp(-z)); // Sigmoid
                }
            }
            currentInput = out;
        } else if (layer.type === 'conv2d') {
            const M = batchSize;
            const out = new Float32Array(M * layer.outputSize);
            for (let m = 0; m < M; m++) {
                for (let oc = 0; oc < layer.outC; oc++) {
                    for (let oh = 0; oh < layer.outH; oh++) {
                        for (let ow = 0; ow < layer.outW; ow++) {
                            let sum = 0;
                            for (let ic = 0; ic < layer.inC; ic++) {
                                for (let kh = 0; kh < layer.k; kh++) {
                                    for (let kw = 0; kw < layer.k; kw++) {
                                        let ih = oh + kh;
                                        let iw = ow + kw;
                                        let inVal = currentInput[m * layer.inputSize + ic * (layer.inH * layer.inW) + ih * layer.inW + iw];
                                        let fVal = layer.filters[oc * (layer.inC * layer.k * layer.k) + ic * (layer.k * layer.k) + kh * layer.k + kw];
                                        sum += inVal * fVal;
                                    }
                                }
                            }
                            let z = sum + layer.bias[oc];
                            let outIdx = m * layer.outputSize + oc * (layer.outH * layer.outW) + oh * layer.outW + ow;
                            if (layer.actType === 1) out[outIdx] = Math.max(0, z);
                            else if (layer.actType === 2) out[outIdx] = Math.tanh(z);
                            else if (layer.actType === 4) out[outIdx] = z;
                            else out[outIdx] = 1.0 / (1.0 + Math.exp(-z));
                        }
                    }
                }
            }
            currentInput = out;
        } else if (layer.type === 'maxpool2d') {
            const M = batchSize;
            const out = new Float32Array(M * layer.outputSize);
            for (let m = 0; m < M; m++) {
                for (let c = 0; c < layer.inC; c++) {
                    for (let oh = 0; oh < layer.outH; oh++) {
                        for (let ow = 0; ow < layer.outW; ow++) {
                            let maxVal = -Infinity;
                            let maxIdxOrig = -1;
                            let start_h = oh * layer.s;
                            let start_w = ow * layer.s;
                            let validRegion = false;
                            for (let ph = 0; ph < layer.p; ph++) {
                                for (let pw = 0; pw < layer.p; pw++) {
                                    let ih = start_h + ph;
                                    let iw = start_w + pw;
                                    if (ih < layer.inH && iw < layer.inW) {
                                        let inIdx = m * layer.inputSize + c * (layer.inH * layer.inW) + ih * layer.inW + iw;
                                        if (!validRegion || currentInput[inIdx] > maxVal) {
                                            maxVal = currentInput[inIdx];
                                            maxIdxOrig = inIdx;
                                            validRegion = true;
                                        }
                                    }
                                }
                            }
                            if (!validRegion) maxVal = 0; // Padding fallback
                            let outIdx = m * layer.outputSize + c * (layer.outH * layer.outW) + oh * layer.outW + ow;
                            out[outIdx] = maxVal;
                        }
                    }
                }
            }
            currentInput = out;
        } else if (layer.type === 'averagepool2d') {
            const M = batchSize;
            const out = new Float32Array(M * layer.outputSize);
            for (let m = 0; m < M; m++) {
                for (let c = 0; c < layer.inC; c++) {
                    for (let oh = 0; oh < layer.outH; oh++) {
                        for (let ow = 0; ow < layer.outW; ow++) {
                            let sumVal = 0;
                            let count = 0;
                            let start_h = oh * layer.s;
                            let start_w = ow * layer.s;
                            for (let ph = 0; ph < layer.p; ph++) {
                                for (let pw = 0; pw < layer.p; pw++) {
                                    let ih = start_h + ph;
                                    let iw = start_w + pw;
                                    if (ih < layer.inH && iw < layer.inW) {
                                        let inIdx = m * layer.inputSize + c * (layer.inH * layer.inW) + ih * layer.inW + iw;
                                        sumVal += currentInput[inIdx];
                                        count++;
                                    }
                                }
                            }
                            let outIdx = m * layer.outputSize + c * (layer.outH * layer.outW) + oh * layer.outW + ow;
                            out[outIdx] = count > 0 ? (sumVal / count) : 0;
                        }
                    }
                }
            }
            currentInput = out;
        } else if (layer.type === 'batchnorm') {
            const M = batchSize;
            const C = layer.inC;
            const spatial = layer.spatial;
            const out = new Float32Array(M * C * spatial);
            const eps = 1e-5;
            for (let c = 0; c < C; c++) {
                let mean = layer.runningMean[c];
                let variance = layer.runningVar[c];
                let invStd = 1.0 / Math.sqrt(variance + eps);
                let g = layer.gamma[c];
                let b = layer.beta[c];
                for (let m = 0; m < M; m++) {
                    for (let s = 0; s < spatial; s++) {
                        let idx = m * (C * spatial) + c * spatial + s;
                        out[idx] = g * (currentInput[idx] - mean) * invStd + b;
                    }
                }
            }
            currentInput = out;
        } else if (layer.type === 'flatten') {
            // Arrays are inherently flat
        }
    }

    // Apply softmax on final output if needed
    const outLayer = layers[layers.length - 1];
    if (outLayer && outLayer.actType === 4) {
        const N = outLayer.outputSize;
        for (let m = 0; m < batchSize; m++) {
            const base = m * N;
            let maxVal = currentInput[base];
            for (let i = 1; i < N; i++) maxVal = Math.max(maxVal, currentInput[base + i]);
            let sumExp = 0;
            for (let i = 0; i < N; i++) {
                currentInput[base + i] = Math.exp(currentInput[base + i] - maxVal);
                sumExp += currentInput[base + i];
            }
            for (let i = 0; i < N; i++) currentInput[base + i] /= sumExp;
        }
    }

    return currentInput;
}

function cpuNNTrainStep(layers, X_data, Y_data, batchSize, learningRate = 0.05) {
    let activations = [X_data];
    let currentInput = X_data;

    // 1. FORWARD PASS
    for (const layer of layers) {
        if (!layer.type || layer.type === 'dense') {
            const M = batchSize;
            const K = layer.inputSize;
            const N = layer.outputSize;
            const out = new Float32Array(M * N);

            for (let m = 0; m < M; m++) {
                for (let n = 0; n < N; n++) {
                    let sum = 0;
                    for (let k = 0; k < K; k++) sum += currentInput[m * K + k] * layer.weights[k * N + n];
                    let z = sum + layer.bias[n];

                    if (layer.actType === 1) out[m * N + n] = Math.max(0, z);
                    else if (layer.actType === 2) out[m * N + n] = Math.tanh(z);
                    else if (layer.actType === 4) out[m * N + n] = z;
                    else out[m * N + n] = 1.0 / (1.0 + Math.exp(-z));
                }
            }
            currentInput = out;
            activations.push(currentInput);
        } else if (layer.type === 'conv2d') {
            const M = batchSize;
            const out = new Float32Array(M * layer.outputSize);
            for (let m = 0; m < M; m++) {
                for (let oc = 0; oc < layer.outC; oc++) {
                    for (let oh = 0; oh < layer.outH; oh++) {
                        for (let ow = 0; ow < layer.outW; ow++) {
                            let sum = 0;
                            for (let ic = 0; ic < layer.inC; ic++) {
                                for (let kh = 0; kh < layer.k; kh++) {
                                    for (let kw = 0; kw < layer.k; kw++) {
                                        let ih = oh + kh;
                                        let iw = ow + kw;
                                        let inVal = currentInput[m * layer.inputSize + ic * (layer.inH * layer.inW) + ih * layer.inW + iw];
                                        let fVal = layer.filters[oc * (layer.inC * layer.k * layer.k) + ic * (layer.k * layer.k) + kh * layer.k + kw];
                                        sum += inVal * fVal;
                                    }
                                }
                            }
                            let z = sum + layer.bias[oc];
                            let outIdx = m * layer.outputSize + oc * (layer.outH * layer.outW) + oh * layer.outW + ow;
                            if (layer.actType === 1) out[outIdx] = Math.max(0, z);
                            else if (layer.actType === 2) out[outIdx] = Math.tanh(z);
                            else if (layer.actType === 4) out[outIdx] = z;
                            else out[outIdx] = 1.0 / (1.0 + Math.exp(-z));
                        }
                    }
                }
            }
            currentInput = out;
            activations.push(currentInput);
        } else if (layer.type === 'maxpool2d') {
            const M = batchSize;
            const out = new Float32Array(M * layer.outputSize);
            const indices = new Float32Array(M * layer.outputSize);
            for (let m = 0; m < M; m++) {
                for (let c = 0; c < layer.inC; c++) {
                    for (let oh = 0; oh < layer.outH; oh++) {
                        for (let ow = 0; ow < layer.outW; ow++) {
                            let maxVal = -Infinity;
                            let maxIdxOrig = -1;
                            let start_h = oh * layer.s;
                            let start_w = ow * layer.s;
                            let validRegion = false;
                            for (let ph = 0; ph < layer.p; ph++) {
                                for (let pw = 0; pw < layer.p; pw++) {
                                    let ih = start_h + ph;
                                    let iw = start_w + pw;
                                    if (ih < layer.inH && iw < layer.inW) {
                                        let inIdx = m * layer.inputSize + c * (layer.inH * layer.inW) + ih * layer.inW + iw;
                                        if (!validRegion || currentInput[inIdx] > maxVal) {
                                            maxVal = currentInput[inIdx];
                                            maxIdxOrig = inIdx;
                                            validRegion = true;
                                        }
                                    }
                                }
                            }
                            if (!validRegion) maxVal = 0; // Pure padding
                            let outIdx = m * layer.outputSize + c * (layer.outH * layer.outW) + oh * layer.outW + ow;
                            out[outIdx] = maxVal;
                            indices[outIdx] = maxIdxOrig;
                        }
                    }
                }
            }
            layer.indices = indices; // For backprop
            currentInput = out;
            activations.push(currentInput);
        } else if (layer.type === 'averagepool2d') {
            const M = batchSize;
            const out = new Float32Array(M * layer.outputSize);
            for (let m = 0; m < M; m++) {
                for (let c = 0; c < layer.inC; c++) {
                    for (let oh = 0; oh < layer.outH; oh++) {
                        for (let ow = 0; ow < layer.outW; ow++) {
                            let sumVal = 0;
                            let count = 0;
                            let start_h = oh * layer.s;
                            let start_w = ow * layer.s;
                            for (let ph = 0; ph < layer.p; ph++) {
                                for (let pw = 0; pw < layer.p; pw++) {
                                    let ih = start_h + ph;
                                    let iw = start_w + pw;
                                    if (ih < layer.inH && iw < layer.inW) {
                                        let inIdx = m * layer.inputSize + c * (layer.inH * layer.inW) + ih * layer.inW + iw;
                                        sumVal += currentInput[inIdx];
                                        count++;
                                    }
                                }
                            }
                            let outIdx = m * layer.outputSize + c * (layer.outH * layer.outW) + oh * layer.outW + ow;
                            out[outIdx] = count > 0 ? (sumVal / count) : 0;
                        }
                    }
                }
            }
            currentInput = out;
            activations.push(currentInput);
        } else if (layer.type === 'batchnorm') {
            const M = batchSize;
            const C = layer.inC;
            const spatial = layer.spatial;
            const N_items = M * spatial;
            const out = new Float32Array(M * C * spatial);
            const eps = 1e-5;

            layer.cacheMean = new Float32Array(C);
            layer.cacheVar = new Float32Array(C);

            for (let c = 0; c < C; c++) {
                let sum = 0;
                for (let m = 0; m < M; m++) {
                    for (let s = 0; s < spatial; s++) {
                        let idx = m * (C * spatial) + c * spatial + s;
                        sum += currentInput[idx];
                    }
                }
                let mean = sum / N_items;

                let sumSq = 0;
                for (let m = 0; m < M; m++) {
                    for (let s = 0; s < spatial; s++) {
                        let idx = m * (C * spatial) + c * spatial + s;
                        let diff = currentInput[idx] - mean;
                        sumSq += diff * diff;
                    }
                }
                let variance = sumSq / N_items;

                layer.runningMean[c] = 0.9 * layer.runningMean[c] + 0.1 * mean;
                let unbVar = sumSq / Math.max(1, N_items - 1);
                layer.runningVar[c] = 0.9 * layer.runningVar[c] + 0.1 * unbVar;

                layer.cacheMean[c] = mean;
                layer.cacheVar[c] = variance;

                let invStd = 1.0 / Math.sqrt(variance + eps);
                let g = layer.gamma[c];
                let b = layer.beta[c];
                for (let m = 0; m < M; m++) {
                    for (let s = 0; s < spatial; s++) {
                        let idx = m * (C * spatial) + c * spatial + s;
                        out[idx] = g * (currentInput[idx] - mean) * invStd + b;
                    }
                }
            }
            currentInput = out;
            activations.push(currentInput);
        } else if (layer.type === 'flatten') {
            activations.push(currentInput);
        }
    }

    const outLayer = layers[layers.length - 1];
    let pred = activations[activations.length - 1];

    // Apply softmax on final output if needed
    if (outLayer.actType === 4) {
        const N = outLayer.outputSize;
        pred = new Float32Array(pred); // Clone so we don't corrupt stored activations
        for (let m = 0; m < batchSize; m++) {
            const base = m * N;
            let maxVal = pred[base];
            for (let i = 1; i < N; i++) maxVal = Math.max(maxVal, pred[base + i]);
            let sumExp = 0;
            for (let i = 0; i < N; i++) {
                pred[base + i] = Math.exp(pred[base + i] - maxVal);
                sumExp += pred[base + i];
            }
            for (let i = 0; i < N; i++) pred[base + i] /= sumExp;
        }
        activations[activations.length - 1] = pred;
    }

    let currentDz = new Float32Array(batchSize * outLayer.outputSize);
    let loss = 0;
    let correct = 0;

    for (let m = 0; m < batchSize; m++) {
        let predMaxIdx = -1, predMaxVal = -Infinity;
        let targetMaxIdx = -1, targetMaxVal = -Infinity;
        for (let n = 0; n < outLayer.outputSize; n++) {
            let idx = m * outLayer.outputSize + n;
            let yP = pred[idx];
            let yT = Y_data[idx];
            currentDz[idx] = yP - yT;
            loss += -(yT * Math.log(yP + 1e-7) + (1 - yT) * Math.log(1 - yP + 1e-7));
            if (yP > predMaxVal) { predMaxVal = yP; predMaxIdx = n; }
            if (yT > targetMaxVal) { targetMaxVal = yT; targetMaxIdx = n; }
        }
        if (predMaxIdx === targetMaxIdx) correct++;
    }
    loss /= batchSize;
    const accuracy = correct / batchSize;

    // 3. BACKWARD PASS & Optimizer
    for (let i = layers.length - 1; i >= 0; i--) {
        const layer = layers[i];
        const A_prev = activations[i];

        if (!layer.type || layer.type === 'dense') {
            const K = layer.inputSize;
            const N = layer.outputSize;
            const dW = new Float32Array(K * N);
            const dB = new Float32Array(N);

            for (let m = 0; m < batchSize; m++) {
                for (let n = 0; n < N; n++) {
                    let dz = currentDz[m * N + n];
                    dB[n] += dz;
                    for (let k = 0; k < K; k++) dW[k * N + n] += A_prev[m * K + k] * dz;
                }
            }

            for (let j = 0; j < dW.length; j++) dW[j] /= batchSize;
            for (let j = 0; j < dB.length; j++) dB[j] /= batchSize;

            let nextDz = null;
            if (i > 0) {
                nextDz = new Float32Array(batchSize * K);
                let prevLayer = layers[i - 1];
                for (let m = 0; m < batchSize; m++) {
                    for (let k = 0; k < K; k++) {
                        let sum = 0;
                        for (let n = 0; n < N; n++) sum += currentDz[m * N + n] * layer.weights[k * N + n];
                        let a = A_prev[m * K + k];
                        let dA = 1.0;
                        if (prevLayer.type === 'flatten' || prevLayer.type === 'maxpool2d') {
                            let curr = i - 1;
                            let reallyPrev = null;
                            while (curr >= 0) {
                                if (layers[curr].type !== 'flatten' && layers[curr].type !== 'maxpool2d') {
                                    reallyPrev = layers[curr];
                                    break;
                                }
                                curr--;
                            }
                            if (reallyPrev) {
                                if (reallyPrev.actType === 1) dA = a > 0 ? 1.0 : 0.0;
                                else if (reallyPrev.actType === 2) dA = 1.0 - (a * a);
                                else if (reallyPrev.actType !== 4) dA = a * (1.0 - a);
                            }
                        } else if (!prevLayer.type || prevLayer.type === 'dense' || prevLayer.type === 'conv2d') {
                            if (prevLayer.actType === 1) dA = a > 0 ? 1.0 : 0.0;
                            else if (prevLayer.actType === 2) dA = 1.0 - (a * a);
                            else if (prevLayer.actType !== 4 && prevLayer.actType !== 3) dA = a * (1.0 - a);
                        }
                        nextDz[m * K + k] = sum * dA;
                    }
                }
            }

            for (let j = 0; j < layer.weights.length; j++) layer.weights[j] -= learningRate * dW[j];
            for (let j = 0; j < layer.bias.length; j++) layer.bias[j] -= learningRate * dB[j];
            if (layer.device) {
                layer.device.queue.writeBuffer(layer.weightBuffer, 0, layer.weights);
                layer.device.queue.writeBuffer(layer.biasBuffer, 0, layer.bias);
            }
            currentDz = nextDz;

        } else if (layer.type === 'conv2d') {
            const K_c = layer.inC;
            const H = layer.inH;
            const W = layer.inW;
            const F = layer.outC;
            const oH = layer.outH;
            const oW = layer.outW;
            const ks = layer.k;

            const dF = new Float32Array(layer.filters.length);
            const dB = new Float32Array(layer.bias.length);

            for (let m = 0; m < batchSize; m++) {
                for (let oc = 0; oc < F; oc++) {
                    for (let oh = 0; oh < oH; oh++) {
                        for (let ow = 0; ow < oW; ow++) {
                            let outIdx = m * (F * oH * oW) + oc * (oH * oW) + oh * oW + ow;
                            let dz = currentDz[outIdx];
                            dB[oc] += dz;
                            for (let ic = 0; ic < K_c; ic++) {
                                for (let kh = 0; kh < ks; kh++) {
                                    for (let kw = 0; kw < ks; kw++) {
                                        let inIdx = m * (K_c * H * W) + ic * (H * W) + (oh + kh) * W + (ow + kw);
                                        let fIdx = oc * (K_c * ks * ks) + ic * (ks * ks) + kh * ks + kw;
                                        dF[fIdx] += A_prev[inIdx] * dz;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (let j = 0; j < dF.length; j++) dF[j] /= batchSize;
            for (let j = 0; j < dB.length; j++) dB[j] /= batchSize;

            let nextDz = null;
            if (i > 0) {
                nextDz = new Float32Array(batchSize * K_c * H * W);
                let prevLayer = layers[i - 1];
                for (let m = 0; m < batchSize; m++) {
                    for (let oc = 0; oc < F; oc++) {
                        for (let oh = 0; oh < oH; oh++) {
                            for (let ow = 0; ow < oW; ow++) {
                                let outIdx = m * (F * oH * oW) + oc * (oH * oW) + oh * oW + ow;
                                let dz = currentDz[outIdx];
                                for (let ic = 0; ic < K_c; ic++) {
                                    for (let kh = 0; kh < ks; kh++) {
                                        for (let kw = 0; kw < ks; kw++) {
                                            let inIdx = m * (K_c * H * W) + ic * (H * W) + (oh + kh) * W + (ow + kw);
                                            let fIdx = oc * (K_c * ks * ks) + ic * (ks * ks) + kh * ks + kw;
                                            nextDz[inIdx] += dz * layer.filters[fIdx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                for (let j = 0; j < nextDz.length; j++) {
                    let a = A_prev[j];
                    let dA = 1.0;
                    if (prevLayer.type === 'flatten' || prevLayer.type === 'maxpool2d') {
                        let curr = i - 1;
                        let reallyPrev = null;
                        while (curr >= 0) {
                            if (layers[curr].type !== 'flatten' && layers[curr].type !== 'maxpool2d') {
                                reallyPrev = layers[curr];
                                break;
                            }
                            curr--;
                        }
                        if (reallyPrev) {
                            if (reallyPrev.actType === 1) dA = a > 0 ? 1.0 : 0.0;
                            else if (reallyPrev.actType === 2) dA = 1.0 - (a * a);
                            else if (reallyPrev.actType !== 4) dA = a * (1.0 - a);
                        }
                    } else if (!prevLayer.type || prevLayer.type === 'dense' || prevLayer.type === 'conv2d') {
                        if (prevLayer.actType === 1) dA = a > 0 ? 1.0 : 0.0;
                        else if (prevLayer.actType === 2) dA = 1.0 - (a * a);
                        else if (prevLayer.actType !== 4 && prevLayer.actType !== 3) dA = a * (1.0 - a);
                    }
                    nextDz[j] *= dA;
                }
            }

            for (let j = 0; j < layer.filters.length; j++) layer.filters[j] -= learningRate * dF[j];
            for (let j = 0; j < layer.bias.length; j++) layer.bias[j] -= learningRate * dB[j];
            currentDz = nextDz;

        } else if (layer.type === 'batchnorm') {
            let nextDz = null;
            if (i > 0) {
                nextDz = new Float32Array(batchSize * layer.inputSize);
                const M = batchSize;
                const C = layer.inC;
                const spatial = layer.spatial;
                const N_items = M * spatial;
                const eps = 1e-5;

                let prevLayer = layers[i - 1];

                for (let c = 0; c < C; c++) {
                    let mean = layer.cacheMean[c];
                    let variance = layer.cacheVar[c];
                    let invStd = 1.0 / Math.sqrt(variance + eps);
                    let g = layer.gamma[c];

                    let sum_dY = 0;
                    let sum_dY_xhat = 0;

                    for (let m = 0; m < M; m++) {
                        for (let s = 0; s < spatial; s++) {
                            let idx = m * (C * spatial) + c * spatial + s;
                            let dyi = currentDz[idx];
                            let x_hat = (A_prev[idx] - mean) * invStd;
                            sum_dY += dyi;
                            sum_dY_xhat += dyi * x_hat;
                        }
                    }

                    layer.gamma[c] -= learningRate * (sum_dY_xhat / M);
                    layer.beta[c] -= learningRate * (sum_dY / M);

                    for (let m = 0; m < M; m++) {
                        for (let s = 0; s < spatial; s++) {
                            let idx = m * (C * spatial) + c * spatial + s;
                            let dyi = currentDz[idx];
                            let x_hat = (A_prev[idx] - mean) * invStd;
                            let dx_i = (g * invStd / N_items) * (N_items * dyi - sum_dY - x_hat * sum_dY_xhat);

                            let a = A_prev[idx];
                            let dA = 1.0;
                            if (prevLayer.type === 'flatten' || prevLayer.type === 'maxpool2d') {
                                let curr = i - 1;
                                let reallyPrev = null;
                                while (curr >= 0) {
                                    if (layers[curr].type !== 'flatten' && layers[curr].type !== 'maxpool2d') {
                                        reallyPrev = layers[curr];
                                        break;
                                    }
                                    curr--;
                                }
                                if (reallyPrev) {
                                    if (reallyPrev.actType === 1) dA = a > 0 ? 1.0 : 0.0;
                                    else if (reallyPrev.actType === 2) dA = 1.0 - (a * a);
                                    else if (reallyPrev.actType !== 4 && reallyPrev.actType !== 3) dA = a * (1.0 - a);
                                }
                            } else if (!prevLayer.type || prevLayer.type === 'dense' || prevLayer.type === 'conv2d') {
                                if (prevLayer.actType === 1) dA = a > 0 ? 1.0 : 0.0;
                                else if (prevLayer.actType === 2) dA = 1.0 - (a * a);
                                else if (prevLayer.actType !== 4 && prevLayer.actType !== 3) dA = a * (1.0 - a);
                            }
                            nextDz[idx] = dx_i * dA;
                        }
                    }
                }
            } else {
                const M = batchSize;
                const C = layer.inC;
                const spatial = layer.spatial;
                const eps = 1e-5;
                for (let c = 0; c < C; c++) {
                    let mean = layer.cacheMean[c];
                    let variance = layer.cacheVar[c];
                    let invStd = 1.0 / Math.sqrt(variance + eps);

                    let sum_dY = 0;
                    let sum_dY_xhat = 0;
                    for (let m = 0; m < M; m++) {
                        for (let s = 0; s < spatial; s++) {
                            let idx = m * (C * spatial) + c * spatial + s;
                            let dyi = currentDz[idx];
                            let x_hat = (A_prev[idx] - mean) * invStd;
                            sum_dY += dyi;
                            sum_dY_xhat += dyi * x_hat;
                        }
                    }
                    layer.gamma[c] -= learningRate * (sum_dY_xhat / M);
                    layer.beta[c] -= learningRate * (sum_dY / M);
                }
            }
            if (nextDz) currentDz = nextDz;

        } else if (layer.type === 'flatten') {
            let nextDz = null;
            if (i > 0) {
                nextDz = new Float32Array(currentDz);
                let prevLayer = layers[i - 1];
                for (let j = 0; j < nextDz.length; j++) {
                    let a = A_prev[j];
                    let dA = 1.0;
                    if (prevLayer.type === 'flatten' || prevLayer.type === 'maxpool2d') {
                        // Flatten here implies we are dealing with gradients passing through it, 
                        // but "prevLayer" IS flatten. We need layer before that.
                        let curr = i - 1;
                        let reallyPrev = null;
                        while (curr >= 0) {
                            if (layers[curr].type !== 'flatten' && layers[curr].type !== 'maxpool2d') {
                                reallyPrev = layers[curr];
                                break;
                            }
                            curr--;
                        }
                        if (reallyPrev) {
                            if (reallyPrev.actType === 1) dA = a > 0 ? 1.0 : 0.0;
                            else if (reallyPrev.actType === 2) dA = 1.0 - (a * a);
                            else if (reallyPrev.actType !== 4 && reallyPrev.actType !== 3) dA = a * (1.0 - a);
                        }
                    } else if (!prevLayer.type || prevLayer.type === 'dense' || prevLayer.type === 'conv2d') {
                        if (prevLayer.actType === 1) dA = a > 0 ? 1.0 : 0.0;
                        else if (prevLayer.actType === 2) dA = 1.0 - (a * a);
                        else if (prevLayer.actType !== 4 && prevLayer.actType !== 3) dA = a * (1.0 - a);
                    }
                    nextDz[j] *= dA;
                }
            }
            currentDz = nextDz;
        } else if (layer.type === 'maxpool2d') {
            let nextDz = null;
            if (i > 0) {
                nextDz = new Float32Array(batchSize * layer.inputSize);
                for (let j = 0; j < currentDz.length; j++) {
                    let grad = currentDz[j];
                    let maxIdx = layer.indices[j];
                    if (maxIdx >= 0) {
                        nextDz[maxIdx] = grad; // Non-overlapping typically safe
                    }
                }
            }
            currentDz = nextDz;
        } else if (layer.type === 'averagepool2d') {
            let nextDz = null;
            if (i > 0) {
                const M = batchSize;
                nextDz = new Float32Array(M * layer.inputSize);
                for (let m = 0; m < M; m++) {
                    for (let c = 0; c < layer.inC; c++) {
                        for (let oh = 0; oh < layer.outH; oh++) {
                            for (let ow = 0; ow < layer.outW; ow++) {
                                let start_h = oh * layer.s;
                                let start_w = ow * layer.s;
                                let count = 0;
                                for (let ph = 0; ph < layer.p; ph++) {
                                    for (let pw = 0; pw < layer.p; pw++) {
                                        let ih = start_h + ph;
                                        let iw = start_w + pw;
                                        if (ih < layer.inH && iw < layer.inW) { count++; }
                                    }
                                }
                                if (count > 0) {
                                    let outIdx = m * layer.outputSize + c * (layer.outH * layer.outW) + oh * layer.outW + ow;
                                    let avgGrad = currentDz[outIdx] / count;
                                    for (let ph = 0; ph < layer.p; ph++) {
                                        for (let pw = 0; pw < layer.p; pw++) {
                                            let ih = start_h + ph;
                                            let iw = start_w + pw;
                                            if (ih < layer.inH && iw < layer.inW) {
                                                let inIdx = m * layer.inputSize + c * (layer.inH * layer.inW) + ih * layer.inW + iw;
                                                nextDz[inIdx] += avgGrad;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            currentDz = nextDz;
        }
    }

    return { loss, accuracy };
}

const updateStatus = (msg) => {
    document.getElementById('statusLog').innerText = msg;
};

const setBoxState = (id, state, text = null, valid = null) => {
    const box = document.getElementById(id);
    if (state === 'running') box.classList.add('running');
    else box.classList.remove('running');

    if (text !== null) {
        const valDiv = box.querySelector('.result-value');
        valDiv.innerText = text;

        if (valid === true) valDiv.style.color = 'var(--success)';
        else if (valid === false) valDiv.style.color = 'var(--error)';
        else valDiv.style.color = '';
    }
};

document.getElementById('runBtn').addEventListener('click', async () => {
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const layersStr = document.getElementById('layerSizes').value;
    const layerSizes = layersStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    const btn = document.getElementById('runBtn');

    if (layerSizes.length < 2) {
        alert("Please define at least an input and output layer (e.g. 784, 10)");
        return;
    }

    btn.disabled = true;
    updateStatus(`Initializing WebGPU Neural Network...`);
    setBoxState('cpu-time', null, '-');
    setBoxState('gpu-time', null, '-');

    try {
        if (!navigator.gpu) throw new Error("WebGPU Not Supported");
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No Adapter");
        const requiredLimits = {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize
        };
        const device = await adapter.requestDevice({ requiredLimits });

        const nn = new WebGPUNeuralNetwork(device);
        await nn.init();

        for (let i = 0; i < layerSizes.length - 1; i++) {
            nn.addLayer(layerSizes[i], layerSizes[i + 1]);
        }

        const inputSize = layerSizes[0];
        const X = new Float32Array(batchSize * inputSize);
        for (let i = 0; i < X.length; i++) X[i] = Math.random();

        // ----- CPU FORWARD PASS -----
        updateStatus(`Running CPU Forward Pass over ${layerSizes.length - 1} layers...`);
        setBoxState('cpu-time', 'running');
        await new Promise(r => setTimeout(r, 50));

        const cpuStart = performance.now();
        const yCPU = cpuNNForward(nn.layers, X, batchSize);
        const cpuTime = performance.now() - cpuStart;
        setBoxState('cpu-time', null, cpuTime.toFixed(1) + ' ms', null);

        // ----- WEBGPU FORWARD PASS -----
        updateStatus(`Running WebGPU Tiled Forward Pass...`);
        setBoxState('gpu-time', 'running');
        await new Promise(r => setTimeout(r, 50));

        // Warmup (no measure)
        let forwardWarmup = await nn.forward(X, batchSize);
        forwardWarmup.activationBuffers.forEach(b => b.destroy());

        const gpuStart = performance.now();
        const { result: yGPU, activationBuffers } = await nn.forward(X, batchSize);
        const gpuTime = performance.now() - gpuStart;
        activationBuffers.forEach(b => b.destroy());

        // Validate accuracy
        let isValid = true;
        for (let i = 0; i < 50; i++) {
            let idx = Math.floor(Math.random() * yCPU.length);
            let diff = Math.abs(yCPU[idx] - yGPU[idx]);
            // Sigmoid deals tightly in 0..1 interval
            if (diff > 5e-3) {
                console.error(`Mismatch at idx ${idx}: CPU=${yCPU[idx]}, GPU=${yGPU[idx]}`);
                isValid = false;
                break;
            }
        }

        setBoxState('gpu-time', null, gpuTime.toFixed(1) + ' ms' + (isValid ? ' ✅' : ' ❌'), isValid);
        updateStatus(`Done! Successfully calculated ${layerSizes.length - 1} Linear + Sigmoid layers sequentially.`);

    } catch (e) {
        updateStatus(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
    }
});
