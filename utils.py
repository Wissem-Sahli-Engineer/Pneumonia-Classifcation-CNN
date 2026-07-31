# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from PIL import Image


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

