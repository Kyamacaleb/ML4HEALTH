import numpy as np
import tf as tf
import keras
import tensorflow as tf
from keras import layers
from keras.applications.imagenet_utils import preprocess_input
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.metrics import average_precision_score, accuracy_score

# Load and preprocess the image dataset
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\bigman\\Desktop\\train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    'C:\\Users\\bigman\\Desktop\\test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Build the ResNet50 model
num_classes = 4

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model to prevent changing their weights during training
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for multi-class classification
x = layers.GlobalAveragePooling2D()(base_model.output)
output_layer = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)

# Compile and train the model with fine-tuning
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set the number of initial epochs and fine-tuning epochs
initial_epochs = 60
fine_tune_epochs = 30
total_epochs = initial_epochs + fine_tune_epochs

# Train the model with fine-tuning (unfreeze some layers for fine-tuning)
model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=initial_epochs,  # Start fine-tuning after the initial epochs
)

# Step 4: Evaluate the model
predictions = model.predict(test_generator)
y_true = test_generator.classes  # Get the ground-truth labels (not one-hot encoded)

# Convert one-hot encoded predictions to continuous-valued scores
y_pred = np.argmax(predictions, axis=1)
y_scores = np.max(predictions, axis=1)

# Reshape the y_scores array to a 2D array
y_scores = np.reshape(y_scores, (-1, 1))

accuracy = accuracy_score(y_true, y_pred)
mAP = average_precision_score(np.eye(num_classes)[y_true], y_scores, average='macro')

print("Accuracy:", accuracy)
print("Mean Average Precision (mAP):", mAP)