# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing import image
import os


IMG_SIZE = (224, 224)
def classify(model, img , diseases):
    
    """
    Classifies a chest X-Ray image into 14 pathology probabilities.
    
    Parameters:
    - model: Loaded Keras DenseNet121 model
    - image: String file path, PIL Image, or NumPy array
    - diseases: List of class names (e.g., DISEASE_CLASSES)
    
    Returns:
    - results: List of (disease_name, probability) tuples sorted descending by confidence
    """

    # 2. Resize to match training input size (224, 224)
    img_resized = img.resize(IMG_SIZE)

    # 3. Convert to NumPy array
    img_array = np.array(img_resized, dtype=np.float32)

    # 4. Apply DenseNet-specific preprocessing (matches parse_function in training)
    img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_array)

    # 5. Add batch dimension -> Shape: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # 6. Predict raw sigmoid probabilities
    predictions = model.predict(img_batch, verbose=0)[0]

    # 7. Pair disease names with probabilities
    results = [(disease, float(prob)) for disease, prob in zip(diseases, predictions)]

    # 8. Sort highest confidence to lowest
    results.sort(key=lambda x: x[1], reverse=True)

    return results

CXR_IS_CLASS_1 = True

def predict_if_cxr(img_input, model):
    # 1. Handle file path string vs PIL Image object
    if isinstance(img_input, (str, os.PathLike)):
        img = Image.open(img_input)
    else:
        img = img_input  # Already a PIL Image

    # 2. Resize and ensure 3-channel RGB format
    img = img.convert('RGB').resize((224, 224))

    # 3. Raw [0, 255] float array — preprocessing now happens inside the model
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # 4. Run prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    # 5. Map prediction back to "is this a CXR" using the verified class index
    is_class_1 = prediction >= 0.5
    is_cxr = is_class_1 if CXR_IS_CLASS_1 else not is_class_1
    confidence = (prediction if is_class_1 else 1 - prediction) * 100

    if is_cxr:
        print(f"Result: This IS a Chest X-Ray ({confidence:.2f}% confidence)")
    else:
        print(f"Result: This is NOT a Chest X-Ray ({confidence:.2f}% confidence)")

    return is_cxr