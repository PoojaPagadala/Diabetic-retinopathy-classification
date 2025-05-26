import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam  # AdamW is now part of tf.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set dataset paths
data_directory = '/kaggle/input/aptos-augmented-images/aptos-augmented-images/aptos-augmented-images'
base_dir = '/kaggle/working/split_dataset'

# Step 1: Split data into train, validation, and test sets
def split_data(data_dir, base_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    os.makedirs(base_dir, exist_ok=True)
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        files = [os.path.join(class_path, f) for f in os.listdir(class_path)]
        np.random.shuffle(files)

        train_size = int(train_ratio * len(files))
        val_size = int(val_ratio * len(files))
        
        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for file in train_files:
            shutil.copy(file, os.path.join(train_dir, class_name))
        for file in val_files:
            shutil.copy(file, os.path.join(val_dir, class_name))
        for file in test_files:
            shutil.copy(file, os.path.join(test_dir, class_name))

split_data(data_directory, base_dir)

# Step 2: Enhanced Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    os.path.join(base_dir, 'validation'),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

test_generator = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Step 3: Define the Hybrid Model with ResNet50 and InceptionV3
def build_hybrid_model_resnet_inception():
    input_layer = Input(shape=(299, 299, 3))

    # ResNet50 Model
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
    resnet_output = GlobalAveragePooling2D()(resnet.output)

    # InceptionV3 Model
    inception = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_layer)
    inception_output = GlobalAveragePooling2D()(inception.output)

    # Combine Outputs using Concatenate
    combined_output = Concatenate()([resnet_output, inception_output])

    # Dense Layers with increased capacity
    dense_layer = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(combined_output)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Activation('relu')(dense_layer)
    dropout_layer = Dropout(0.5)(dense_layer)  # Increased dropout rate to 0.5
    dense_layer = Dense(512, activation='relu')(dropout_layer)  # Additional dense layer
    output_layer = Dense(5, activation='softmax')(dense_layer)  # Adjust number of classes if needed

    model = Model(inputs=input_layer, outputs=output_layer)

    # Unfreeze more layers for fine-tuning
    for layer in resnet.layers[-60:]:  
        layer.trainable = True
    for layer in inception.layers[-60:]:  
        layer.trainable = True

    return model

# Step 4: Compile the Model with AdamW optimizer from tf.keras
hybrid_model_resnet_inception = build_hybrid_model_resnet_inception()
hybrid_model_resnet_inception.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Callbacks
checkpoint = ModelCheckpoint(
    'dr_model_resnet_inception_v3.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,  # Reduced patience for faster adaptation
    min_lr=1e-6,
    verbose=1
)

# Step 6: Class Weights
labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# Step 7: Train the Model
history_resnet_inception = hybrid_model_resnet_inception.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,  # Increased epochs
    callbacks=[checkpoint, early_stopping, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# Step 8: Evaluate on Test Data
test_loss_resnet_inception, test_accuracy_resnet_inception = hybrid_model_resnet_inception.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss_resnet_inception:.4f}, Test Accuracy: {test_accuracy_resnet_inception:.4f}")

# Step 9: Classification Report and Confusion Matrix
predictions_resnet_inception = hybrid_model_resnet_inception.predict(test_generator)
predicted_classes_resnet_inception = np.argmax(predictions_resnet_inception, axis=1)
true_classes_resnet_inception = test_generator.classes

report_resnet_inception = classification_report(true_classes_resnet_inception, predicted_classes_resnet_inception, target_names=test_generator.class_indices.keys())
print(report_resnet_inception)

cm_resnet_inception = confusion_matrix(true_classes_resnet_inception, predicted_classes_resnet_inception)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_resnet_inception, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix for ResNet50 + InceptionV3 (Improved)')
plt.show()

hybrid_model_resnet_inception.save('highest93ri.h5')

# After training the model

# Step 10: Plotting Loss and Accuracy
def plot_metrics(history):
    # Plotting Loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_metrics(history_resnet_inception)

# Additional metrics can be plotted here if needed
