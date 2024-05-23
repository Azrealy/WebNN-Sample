// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run sd-turbo with WebNN in onnxruntime-web.
//

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

/*
 * get configuration from url
*/
function getConfig() {
    const query = window.location.search.substring(1);

    // Insert model config code below:

    var config = {
        model: "https://huggingface.co/onnxruntime-web-temp/demo/resolve/main/sd-turbo",
        provider: "webnn",
        device: "gpu",
        threads: 2,
        images: 3,
    };

    let vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        let pair = vars[i].split("=");
        if (pair[0] in config) {
            config[pair[0]] = decodeURIComponent(pair[1]);
        } else if (pair[0].length > 0) {
            throw new Error("unknown argument: " + pair[0]);
        }
    }
    config.threads = parseInt(config.threads);
    config.images = parseInt(config.images);
    return config;
}

const config = getConfig();


const models = {
    "unet": {
        url: "unet/model_layernorm.onnx", size: 640,
        sessionOptions: { graphOptimizationLevel: 'disabled' },
    },
    "textEncoder": {
        url: "text_encoder/model_layernorm.onnx", size: 1700,
        sessionOptions: { graphOptimizationLevel: 'disabled' },
    },
    "vaeDecoder": {
        url: "vae_decoder/model.onnx", size: 95,
        sessionOptions: { freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64 } }
    }
}



/*
 * fetch and cache model
 */
async function fetchAndCache(baseUrl, modelPath) {
    const url = `${baseUrl}/${modelPath}`;
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
            log(`${modelPath} (network)`);
        } else {
            log(`${modelPath} (cached)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${modelPath} (network)`);
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

/*
 * load models used in the pipeline
 */
async function loadModels(models) {
    log("Execution provider: " + config.provider);
    const cache = await caches.open("onnx");
    let missing = 0;
    for (const [name, model] of Object.entries(models)) {
        const url = `${config.model}/${model.url}`;
        let cachedResponse = await cache.match(url);
        if (cachedResponse === undefined) {
            missing += model.size;
        }
    }
    if (missing > 0) {
        log(`downloading ${missing} MB from network ... it might take a while`);
    } else {
        log("loading...");
    }
    for (const [name, model] of Object.entries(models)) {
        try {
            const start = performance.now();
            const modelBytes = await fetchAndCache(config.model, model.url);
            const sessionOptions = { ...defaultSessionOptions, ...model.sessionOptions };
            models[name].inferenceSession = await ort.InferenceSession.create(modelBytes, sessionOptions);
            const stop = performance.now();
            log(`${model.url} in ${(stop - start).toFixed(1)}ms`);
        } catch (e) {
            log(`${model.url} failed, ${e}`);
        }
    }
    log("ready.");
}

// Define necessary variables
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let tokenizer;
let loading;
const sigma = 14.6146;
const gamma = 0;
const vaeScalingFactor = 0.18215;
const text = document.getElementById("user-input");

text.value = "Castle surrounded by water and nature";


async function generateImage() {
    try {
        document.getElementById('status').innerText = "generating ...";

        if (tokenizer === undefined) {
            tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
            tokenizer.pad_token_id = 0;
        }
        let canvases = [];
        await loading;

        for (let j = 0; j < config.images; j++) {
            const div = document.getElementById(`img_div_${j}`);
            div.style.opacity = 0.5
        }

        const { input_ids } = await tokenizer(text.value, { padding: true, max_length: 77, truncation: true, return_tensor: false });

        // Text encoder
        let start = performance.now();
        const { last_hidden_state } = await models.textEncoder.inferenceSession.run(
            { "input_ids": new ort.Tensor("int32", input_ids, [1, input_ids.length]) }
        );

        let performanceInfo = [`Text encoder: ${(performance.now() - start).toFixed(1)}ms`];

        for (let j = 0; j < config.images; j++) {
            const latentShape = [1, 4, 64, 64];
            let latent = new ort.Tensor(generateRandomNormalLatents(latentShape, sigma), latentShape);
            const latentModelInput = scaleModelInputs(latent);

            // Unet
            start = performance.now();
            let feed = {
                "sample": new ort.Tensor("float16", convertToUint16Array(latentModelInput.data), latentModelInput.dims),
                "timestep": new ort.Tensor("float16", new Uint16Array([float32to16(999)]), [1]),
                "encoder_hidden_states": last_hidden_state,
            };
            let { out_sample } = await models.unet.inferenceSession.run(feed);
            performanceInfo.push(`Unet: ${(performance.now() - start).toFixed(1)}ms`);

            // Denoising scheduler
            const newLatents = step(new ort.Tensor("float32", convertToFloat32Array(out_sample.data), out_sample.dims), latent);

            // VAE decoder
            start = performance.now();
            const { sample } = await models.vaeDecoder.inferenceSession.run({ "latent_sample": newLatents });
            performanceInfo.push(`VAE decoder: ${(performance.now() - start).toFixed(1)}ms`);
            drawImage(sample, j);
            log(performanceInfo.join(", "));
            performanceInfo = [];
        }
        // this is a gpu-buffer we own, so we need to dispose it
        last_hidden_state.dispose();
        log("done");
    } catch (e) {
        log(e);
    }
}

/**
 * draw an image from tensor
 * @param {ort.Tensor} tensor
 * @param {number} imageNumber
*/
function drawImage(tensor, imageNumber) {
    let pix = tensor.data;
    for (var i = 0; i < pix.length; i++) {
        let x = pix[i];
        x = x / 2 + 0.5
        if (x < 0.0) x = 0.0;
        if (x > 1.0) x = 1.0;
        pix[i] = x;
    }
    const imageData = tensor.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
    const canvas = document.getElementById(`img_canvas_${imageNumber}`);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    const div = document.getElementById(`img_div_${imageNumber}`);
    div.style.opacity = 1.0;
}


// Below are all the helper functions for running the Stable Diffusion website

// Event listener for Ctrl + Enter or CMD + Enter
document.getElementById('user-input').addEventListener('keydown', function (e) {
    if (e.ctrlKey && e.key === 'Enter') {
        generateImage();
    }
});
document.getElementById('send-button').addEventListener('click', function (e) {
    generateImage()
});

// Define default session options that are handy for large models.
const defaultSessionOptions = {
    executionProviders: [config.provider],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: {
        session: {
            disable_prepacking: "1",
            use_device_allocator_for_initializers: "1",
            use_ort_model_bytes_directly: "1",
            use_ort_model_bytes_for_initializers: "1"
        }
    },
};

/*
 * initialize latents with random noise. By walking through this latent space and interpolating between different 
   latent representations of images, the model is able to generate a sequence of intermediate images which show a 
   smooth transition between the original images. 
 */
function generateRandomNormalLatents(shape, noiseSigma) {
    function randomNormal() {
        // Use the Box-Muller transform
        let u = Math.random();
        let v = Math.random();
        let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z;
    }
    let size = 1;
    shape.forEach(element => {
        size *= element;
    });

    let data = new Float32Array(size);
    // Loop over the shape dimensions
    for (let i = 0; i < size; i++) {
        data[i] = randomNormal() * noiseSigma;
    }
    return data;
}

/*
 * scale the latents
*/
function scaleModelInputs(tensor) {
    const dInput = tensor.data;
    const dOutput = new Float32Array(dInput.length);

    const divi = (sigma ** 2 + 1) ** 0.5;
    for (let i = 0; i < dInput.length; i++) {
        dOutput[i] = dInput[i] / divi;
    }
    return new ort.Tensor(dOutput, tensor.dims);
}

/*
 * EulerA step: The Euler method, is a numerical technique for approximating solutions to ordinary differential equations (ODEs) with a given initial value. 
 */
function step(modelOutput, sample) {
    const dOutput = new Float32Array(modelOutput.data.length);
    const previousSample = new ort.Tensor(dOutput, modelOutput.dims);
    const sigmaHat = sigma * (gamma + 1);

    for (let i = 0; i < modelOutput.data.length; i++) {
        const predictedOriginalSample = sample.data[i] - sigmaHat * modelOutput.data[i];
        const derivative = (sample.data[i] - predictedOriginalSample) / sigmaHat;
        const dt = 0 - sigmaHat;
        dOutput[i] = (sample.data[i] + derivative * dt) / vaeScalingFactor;
    }
    return previousSample;
}

// Ensure that the GPU supports fp16 which is needed for WebNN support.
async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

document.addEventListener("DOMContentLoaded", () => {
    hasFp16().then((fp16) => {
        if (fp16) {
            loading = loadModels(models);
        } else {
            log("Your GPU or Browser doesn't support webgpu/f16");
        }
    });
});

// ref: http://stackoverflow.com/questions/32633585/how-do-you-convert-to-half-floats-in-javascript
const float32to16 = (function () {

    var floatView = new Float32Array(1);
    var int32View = new Int32Array(floatView.buffer);

    // This method is faster than the OpenEXR implementation (very often
    // used, eg. in Ogre), with the additional benefit of rounding, inspired
    // by James Tursa's half-precision code.
    return function float32to16(val) {

        floatView[0] = val;
        var x = int32View[0];

        var bits = (x >> 16) & 0x8000; // Get the sign
        var m = (x >> 12) & 0x07FF; // Keep one extra bit for rounding
        var e = (x >> 23) & 0xFF; // Using int is faster here

        // If zero, or denormal, or exponent underflows too much for a denormal
        // half, return signed zero.
        if (e < 103) {
            return bits;
        }

        // If NaN, return NaN. If Inf or exponent overflow, return Inf.
        if (e > 142) {
            bits |= 0x7c00;
            // If exponent was 0xFF and one mantissa bit was set, it means NaN,
            // not Inf, so make sure we set one mantissa bit too.
            bits |= ((e == 255) ? 0 : 1) && (x & 0x007FFFFF);
            return bits;
        }

        // If exponent underflows but not too much, return a denormal.
        if (e < 113) {
            m |= 0x0800;
            // Extra rounding may overflow and set mantissa to 0 and exponent
            // to 1, which is OK.
            bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
            return bits;
        }

        bits |= ((e - 112) << 10) | (m >> 1);
        // Extra rounding. An overflow will set mantissa to 0 and increment
        // the exponent, which is OK.
        bits += m & 1;
        return bits;
    };

})();

// This function converts a Float16 stored as the bits of a Uint16 into a Javascript Number.
// Adapted from: https://gist.github.com/martinkallman/5049614
// input is a Uint16 (eg, new Uint16Array([value])[0])

export function float16To32(input) {
    // Create a 32 bit DataView to store the input
    const arr = new ArrayBuffer(4);
    const dv = new DataView(arr);

    // Set the Float16 into the last 16 bits of the dataview
    // So our dataView is [00xx]
    dv.setUint16(2, input, false);

    // Get all 32 bits as a 32 bit integer
    // (JS bitwise operations are performed on 32 bit signed integers)
    const asInt32 = dv.getInt32(0, false);

    // All bits aside from the sign
    let rest = asInt32 & 0x7FFF;
    // Sign bit
    let sign = asInt32 & 0x8000;
    // Exponent bits
    const exponent = asInt32 & 0x7C00;

    // Shift the non-sign bits into place for a 32 bit Float
    rest <<= 13;
    // Shift the sign bit into place for a 32 bit Float
    sign <<= 16;

    // Adjust bias
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
    rest += 0x38000000;
    // Denormals-as-zero
    rest = (exponent === 0 ? 0 : rest);
    // Re-insert sign bit
    rest |= sign;

    // Set the adjusted float32 (stored as int32) back into the dataview
    dv.setInt32(0, rest, false);

    // Get it back out as a float32 (which js will convert to a Number)
    const asFloat32 = dv.getFloat32(0, false);

    return asFloat32;
}

// convert Uint16Array to Float32Array
export function convertToFloat32Array(fp16Array) {
    const fp32Array = new Float32Array(fp16Array.length);
    for (let i = 0; i < fp32Array.length; i++) {
        fp32Array[i] = float16To32(fp16Array[i]);
    }
    return fp32Array;
}

// convert Float32Array to Uint16Array
export function convertToUint16Array(fp32Array) {
    const fp16Array = new Uint16Array(fp32Array.length);
    for (let i = 0; i < fp16Array.length; i++) {
        fp16Array[i] = float32to16(fp32Array[i]);
    }
    return fp16Array;
}

switch (config.provider) {
    case "webgpu":
        if (!("gpu" in navigator)) {
            throw new Error("webgpu is not supported");
        }
        defaultSessionOptions.preferredOutputLocation = { last_hidden_state: "gpu-buffer" };
        break;
    case "webnn":
        if (!("ml" in navigator)) {
            throw new Error("webnn is not supported");
        }
        defaultSessionOptions.executionProviders = [{
            name: "webnn",
            deviceType: config.device,
            powerPreference: 'default'
        }];
        break;
}
