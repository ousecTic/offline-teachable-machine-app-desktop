const STATUS = document.getElementById("status");
const CLASS_NAME_INPUT = document.getElementById("className");
const ADD_CLASS_BUTTON = document.getElementById("addClass");
const CLASS_CONTAINERS = document.getElementById("classContainers");
const TRAIN_BUTTON = document.getElementById("train");
const PREDICT_BUTTON = document.getElementById("predict");
const RESET_BUTTON = document.getElementById("reset");
const PREDICTION_UPLOAD = document.getElementById("predictionUpload");
const PREDICTION_PREVIEW = document.getElementById("predictionPreview");
const PREDICTION_RESULT = document.getElementById("prediction-result");
const TRAINING_STATUS = document.getElementById("training-status");

const TRAIN_BUTTON_TEXT = TRAIN_BUTTON.innerText;

const MAX_IMAGE_DIMENSION = 2048;
const MOBILE_NET_INPUT_WIDTH = 128;
const MOBILE_NET_INPUT_HEIGHT = 128;
const EPOCHS = 10;
const CLASS_NAMES = [];

let mobilenet = undefined;
let model = tf.sequential();
let trainingData = {};
let isTraining = false;
let isTrained = false;

// Disable buttons initially
TRAIN_BUTTON.disabled = true;
PREDICT_BUTTON.disabled = true;

ADD_CLASS_BUTTON.addEventListener("click", addClass);
TRAIN_BUTTON.addEventListener("click", trainModel);
PREDICT_BUTTON.addEventListener("click", predict);
RESET_BUTTON.addEventListener("click", reset);
PREDICTION_UPLOAD.addEventListener("change", updatePredictionPreview);

async function resizeImageIfNeeded(img) {
  let { width, height } = img;

  if (width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
    console.log("Image exceeds maximum dimension. Resizing...");
    if (width > height) {
      height = Math.round((height * MAX_IMAGE_DIMENSION) / width);
      width = MAX_IMAGE_DIMENSION;
    } else {
      width = Math.round((width * MAX_IMAGE_DIMENSION) / height);
      height = MAX_IMAGE_DIMENSION;
    }
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, width, height);
  return canvas;
}

function sanitizeClassName(name) {
  // Replace spaces and non-alphanumeric characters with underscores
  let sanitized = name.replace(/[^a-zA-Z0-9]/g, "_");

  // Ensure the name starts with a letter (for valid ID creation)
  if (!/^[a-zA-Z]/.test(sanitized)) {
    sanitized = "class_" + sanitized;
  }

  return sanitized;
}

async function processImage(imgElement) {
  console.log("Original Image Size:", imgElement.width, "x", imgElement.height);

  // Resize the image if it's too large
  const resizedCanvas = await resizeImageIfNeeded(imgElement);
  console.log(
    "Processed Image Size:",
    resizedCanvas.width,
    "x",
    resizedCanvas.height
  );

  // Resize to MobileNet input size
  const mobileNetCanvas = document.createElement("canvas");
  mobileNetCanvas.width = MOBILE_NET_INPUT_WIDTH;
  mobileNetCanvas.height = MOBILE_NET_INPUT_HEIGHT;
  const mobileNetCtx = mobileNetCanvas.getContext("2d");
  mobileNetCtx.drawImage(
    resizedCanvas,
    0,
    0,
    MOBILE_NET_INPUT_WIDTH,
    MOBILE_NET_INPUT_HEIGHT
  );

  return tf.tidy(() => {
    const imageTensor = tf.browser.fromPixels(mobileNetCanvas);
    console.log("MobileNet Input tensor shape:", imageTensor.shape);
    return imageTensor.toFloat().div(255).expandDims();
  });
}

async function loadMobileNetFeatureModel() {
  const URL = "model/model.json"; // Path to your local model.json file
  try {
    mobilenet = await tf.loadGraphModel(URL);
    STATUS.innerText = "MobileNet v3 loaded successfully!";

    tf.tidy(() => {
      let answer = mobilenet.predict(
        tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
      );
      console.log(answer.shape);
    });
  } catch (error) {
    console.error("Error loading MobileNet model:", error);
    STATUS.innerText =
      "Failed to load MobileNet model. Please check your connection and try again.";
  }
}

loadMobileNetFeatureModel();

function addClass() {
  let className = CLASS_NAME_INPUT.value.trim();
  if (!className) {
    STATUS.innerText = "Please enter a class name.";
    return;
  }

  const originalClassName = className;
  className = sanitizeClassName(className);

  if (CLASS_NAMES.includes(className)) {
    STATUS.innerText = `Class "${originalClassName}" already exists.`;
    return;
  }

  CLASS_NAMES.push(className);
  const classContainer = document.createElement("div");
  classContainer.className = "class-container";
  classContainer.innerHTML = `
    <div class="class-header">
      <h3>${originalClassName}</h3>
      <button class="remove-class-btn" data-class="${className}">Remove Class</button>
    </div>
    <input type="file" id="upload-${className}" accept="image/*" multiple style="display:none;">
    <label for="upload-${className}" class="file-upload-btn">Add Images</label>
    <div class="image-grid" id="grid-${className}"></div>
  `;
  CLASS_CONTAINERS.appendChild(classContainer);

  const uploadInput = classContainer.querySelector(`#upload-${className}`);
  uploadInput.addEventListener("change", (event) =>
    handleImageUpload(event, className)
  );

  const removeButton = classContainer.querySelector(".remove-class-btn");
  removeButton.addEventListener("click", () => removeClass(className));

  trainingData[className] = [];
  CLASS_NAME_INPUT.value = "";
  updateUIState();

  if (originalClassName !== className) {
    STATUS.innerText = `Class "${originalClassName}" added as "${className}".`;
  } else {
    STATUS.innerText = `Class "${className}" added successfully.`;
  }
}

function removeClass(className) {
  // Remove from CLASS_NAMES array
  const index = CLASS_NAMES.indexOf(className);
  if (index > -1) {
    CLASS_NAMES.splice(index, 1);
  }

  // Remove from trainingData object
  delete trainingData[className];

  // Remove the class container from the DOM
  const classContainer = document.querySelector(
    `.class-container:has(#upload-${className})`
  );
  if (classContainer) {
    classContainer.remove();
  }

  // Update UI state
  updateUIState();

  // Reset the model since we've changed the classes
  model = null;
  isTrained = false;

  STATUS.innerText = `Class "${className}" removed successfully.`;

  // If all classes are removed, reset everything
  if (CLASS_NAMES.length === 0) {
    reset();
  }
}

function removeImage(event) {
  const btn = event.target;
  const className = btn.dataset.class;
  const index = parseInt(btn.dataset.index);

  // Remove the image data from trainingData
  trainingData[className].splice(index, 1);

  // Remove the image element from the DOM
  btn.closest(".image-item").remove();

  // Update the indices of the remaining remove buttons
  const imageGrid = document.getElementById(`grid-${className}`);
  imageGrid.querySelectorAll(".remove-image-btn").forEach((button, idx) => {
    button.dataset.index = idx;
  });

  // Reset the model since we've changed the training data
  model = null;
  isTrained = false;

  STATUS.innerText = `Removed image from class "${className}"`;
  updateUIState();
}

async function handleImageUpload(event, className) {
  const files = event.target.files;
  const imageGrid = document.getElementById(`grid-${className}`);
  const totalFiles = files.length;
  let successfulUploads = 0;

  STATUS.innerText = `Processing ${totalFiles} images for class ${className}...`;

  for (let i = 0; i < totalFiles; i++) {
    const file = files[i];
    try {
      const blobUrl = URL.createObjectURL(file);
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = blobUrl;
      });

      const tensor = await processImage(img);
      const features = mobilenet.predict(tensor);
      trainingData[className].push(features);
      tensor.dispose();

      const imageItem = document.createElement("div");
      imageItem.className = "image-item";
      imageItem.innerHTML = `
        <img src="${blobUrl}" alt="Class image">
        <button class="remove-image-btn" data-class="${className}" data-index="${
        trainingData[className].length - 1
      }">×</button>
      `;
      imageGrid.appendChild(imageItem);

      const removeBtn = imageItem.querySelector(".remove-image-btn");
      removeBtn.addEventListener("click", removeImage);

      successfulUploads++;
    } catch (error) {
      console.error(`Error processing image ${i + 1}:`, error);
    }

    if (i % 5 === 0 || i === totalFiles - 1) {
      STATUS.innerText = `Processed ${
        i + 1
      } of ${totalFiles} images for class ${className}...`;
    }
  }

  STATUS.innerText = `Added ${successfulUploads} out of ${totalFiles} images to class ${className}`;
  updateUIState();
}

async function trainModel() {
  if (isTraining) return;
  isTraining = true;
  updateUIState();

  // Update train button to show loading state
  TRAIN_BUTTON.innerHTML = '<span class="spinner"></span> Training...';

  const numClasses = CLASS_NAMES.length;
  if (numClasses < 2) {
    STATUS.innerText = "Please add at least two classes before training.";
    isTraining = false;
    updateUIState();
    return;
  }

  model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [1001], units: 128, activation: "relu" })
  );
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: numClasses === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const xs = [];
  const ys = [];

  Object.keys(trainingData).forEach((className, index) => {
    const classData = trainingData[className];
    xs.push(...classData);
    ys.push(...Array(classData.length).fill(index));
  });

  if (xs.length === 0) {
    STATUS.innerText =
      "No training data available. Please add images to classes.";
    isTraining = false;
    updateUIState();
    return;
  }

  const xDataset = tf.data.array(xs).map((x) => x.reshape([1001]));
  const yDataset = tf.data
    .array(ys)
    .map((label) => tf.oneHot(label, numClasses));

  const xyDataset = tf.data
    .zip({ xs: xDataset, ys: yDataset })
    .shuffle(100)
    .batch(32);

  try {
    await model.fitDataset(xyDataset, {
      epochs: EPOCHS,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const lossValue = logs.loss ? logs.loss.toFixed(5) : "N/A";
          const accuracyValue = logs.acc
            ? (logs.acc * 100).toFixed(2) + "%"
            : "N/A";
          console.log(
            `Epoch ${
              epoch + 1
            } of ${EPOCHS} completed. Loss: ${lossValue}, Accuracy: ${accuracyValue}`
          );
          TRAINING_STATUS.innerText = `Training: ${
            epoch + 1
          }/${EPOCHS}. Loss: ${lossValue}, Accuracy: ${accuracyValue}`;
        },
      },
    });

    STATUS.innerText = "Model trained successfully!";
    isTrained = true;
  } catch (error) {
    console.error("Training error:", error);
    STATUS.innerText = "Error during training. Check console for details.";
  } finally {
    isTraining = false;
    updateUIState();
  }
}

function updatePredictionPreview(event) {
  const file = event.target.files[0];
  if (file) {
    const imgElement = document.createElement("img");
    imgElement.src = URL.createObjectURL(file);
    imgElement.onload = () => URL.revokeObjectURL(imgElement.src);
    PREDICTION_PREVIEW.innerHTML = "";
    PREDICTION_PREVIEW.appendChild(imgElement);
  }
}

async function predict() {
  if (!model || !isTrained) {
    STATUS.innerText = "Please train the model first.";
    return;
  }

  const img = PREDICTION_PREVIEW.querySelector("img");
  if (!img) {
    STATUS.innerText = "Please upload an image to classify.";
    return;
  }

  try {
    if (!img.complete) {
      await new Promise((resolve) => (img.onload = resolve));
    }

    const tensor = await processImage(img);

    console.log("Final tensor shape:", tensor.shape);
    const features = mobilenet.predict(tensor);
    console.log("MobileNet features shape:", features.shape);
    const prediction = model.predict(features);
    console.log("Final prediction:", prediction.arraySync());
    const classIndex = prediction.argMax(1).dataSync()[0];
    const confidence = prediction.max().dataSync()[0];

    PREDICTION_RESULT.innerText = `Prediction: ${CLASS_NAMES[classIndex]} (${(
      confidence * 100
    ).toFixed(2)}% confidence)`;

    tensor.dispose();
    features.dispose();
    prediction.dispose();
  } catch (error) {
    console.error("Prediction error:", error);
    STATUS.innerText = "Error during prediction. Check console for details.";
  }
}

function reset() {
  CLASS_NAMES.length = 0;
  CLASS_CONTAINERS.innerHTML = "";
  trainingData = {};
  model = null;
  PREDICTION_PREVIEW.innerHTML = "";
  PREDICTION_RESULT.innerText = "";
  STATUS.innerText = "Model and data reset. You can start over.";
  isTrained = false;
  updateUIState();
}

function updateUIState() {
  ADD_CLASS_BUTTON.disabled = isTraining;
  CLASS_NAME_INPUT.disabled = isTraining;
  TRAIN_BUTTON.disabled = isTraining || CLASS_NAMES.length < 2;
  PREDICT_BUTTON.disabled = !isTrained || CLASS_NAMES.length < 2;
  RESET_BUTTON.disabled = isTraining;
  PREDICTION_UPLOAD.disabled = !isTrained || CLASS_NAMES.length < 2;

  // Update train button text
  TRAIN_BUTTON.innerHTML = isTraining
    ? '<span class="spinner"></span> Training...'
    : TRAIN_BUTTON_TEXT;

  document.querySelectorAll(".file-upload-btn").forEach((button) => {
    button.style.pointerEvents = isTraining ? "none" : "auto";
    button.style.opacity = isTraining ? "0.5" : "1";
  });

  document
    .querySelectorAll(".remove-class-btn, .remove-image-btn")
    .forEach((button) => {
      button.disabled = isTraining;
    });
}

// Initial UI state update
updateUIState();
