import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks

NUM_CLASSES = 768  # Number of output classes
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
NUM_CHANNELS = 1  # Grayscale frames
input_shape_resnet = (FRAME_HEIGHT, FRAME_WIDTH, NUM_CHANNELS)  # For ResNet
input_shape_bgru = (4, 512)  # Adjusted to match the ResNet output

# Define the ResNet model
def build_resnet(input_shape):
    input_layer = layers.Input(shape=input_shape)

    # Initial convolutional layer
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Residual blocks
    x = residual_block(x, filters=[64, 64, 256], strides=(1, 1))
    x = residual_block(x, filters=[128, 128, 512], strides=(2, 2))
    x = residual_block(x, filters=[256, 256, 1024], strides=(2, 2))
    x = residual_block(x, filters=[512, 512, 2048], strides=(2, 2))

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Reshape to match the input shape of subsequent models
    x = layers.Reshape((4, 512))(x)

    model = models.Model(inputs=input_layer, outputs=x, name='resnet18')
    return model

# Define the residual block
def residual_block(input_layer, filters, strides=(1, 1)):
    filters1, filters2, filters3 = filters
    shortcut = input_layer

    # First convolutional layer
    x = layers.Conv2D(filters1, kernel_size=(1, 1), strides=strides, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolutional layer
    x = layers.Conv2D(filters2, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Third convolutional layer
    x = layers.Conv2D(filters3, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if strides != (1, 1) or shortcut.shape[-1] != filters3:
        shortcut = layers.Conv2D(filters3, kernel_size=(1, 1), strides=strides, padding='same', kernel_regularizer=regularizers.l2(0.01))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)

    return x

# Define the MSTCN model
def build_mstcn(input_shape):
    input_layer = layers.Input(shape=input_shape)

    # 1D convolutional layers with different kernel sizes
    conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
    conv2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(input_layer)
    conv3 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(input_layer)

    # Concatenate the outputs of 1D convolutions
    concat = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # 1D convolutional layer after concatenation
    conv_final = layers.Conv1D(128, kernel_size=1, padding='same', activation='relu')(concat)

    # Global average pooling
    avg_pool = layers.GlobalAveragePooling1D()(conv_final)

    model = models.Model(inputs=input_layer, outputs=avg_pool, name='mstcn')
    return model

# Define the DCTCN model
def build_dctcn(input_shape):
    input_layer = layers.Input(shape=input_shape)

    # 1D convolutional layers with different kernel sizes
    conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
    conv2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(input_layer)
    conv3 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(input_layer)

    # Concatenate the outputs of 1D convolutions
    concat = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # 1D convolutional layer after concatenation
    conv_final = layers.Conv1D(128, kernel_size=1, padding='same', activation='relu')(concat)

    # Global average pooling
    avg_pool = layers.GlobalAveragePooling1D()(conv_final)

    model = models.Model(inputs=input_layer, outputs=avg_pool, name='dctcn')
    return model

# Define the BGRU model
def build_bgru(input_shape):
    input_layer = layers.Input(shape=input_shape)

    # Bidirectional GRU layers
    bgru1 = layers.Bidirectional(layers.GRU(128, return_sequences=True))(input_layer)
    bgru2 = layers.Bidirectional(layers.GRU(128))(bgru1)

    model = models.Model(inputs=input_layer, outputs=bgru2, name='bgru')
    return model

# Define the BiLSTM model
def build_bilstm(input_shape):
    input_layer = layers.Input(shape=input_shape)

    # Bidirectional LSTM layers
    bilstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input_layer)
    bilstm2 = layers.Bidirectional(layers.LSTM(128))(bilstm1)

    model = models.Model(inputs=input_layer, outputs=bilstm2, name='bilstm')
    return model

# Create the models
resnet_model = build_resnet(input_shape_resnet)
mstcn_model = build_mstcn(input_shape_bgru)
dctcn_model = build_dctcn(input_shape_bgru)
bgru_model = build_bgru(input_shape_bgru)
bilstm_model = build_bilstm(input_shape_bgru)

# Connect the output of ResNet-18 to the input of each model
resnet_output = resnet_model.output

mstcn_output = mstcn_model(resnet_output)
dctcn_output = dctcn_model(resnet_output)
bgru_output = bgru_model(resnet_output)
bilstm_output = bilstm_model(resnet_output)

# Concatenate the outputs
ensemble_input = layers.Concatenate()([mstcn_output, dctcn_output, bgru_output, bilstm_output])

# Add a dense layer for final predictions
ensemble_output = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(ensemble_input)

# Create the ensemble model
ensemble_model = models.Model(inputs=resnet_model.input, outputs=ensemble_output)

# Define training parameters
LEARNING_RATE = 0.0003
BATCH_SIZE = 32
EPOCHS = 30

# Define optimizer
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

# Compile the ensemble model
ensemble_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Implement callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

# Convert true labels to one-hot encoded format
all_train_labels_one_hot = tf.one_hot(all_train_labels_final_matched, depth=NUM_CLASSES)
all_val_labels_one_hot = tf.one_hot(all_val_labels_final_matched, depth=NUM_CLASSES)

# Train the ensemble model
history = ensemble_model.fit(
    all_train_frames_final_matched, 
    all_train_labels_one_hot, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    validation_data=(all_val_frames_final_matched, all_val_labels_one_hot),
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the ensemble model
val_loss, val_accuracy = ensemble_model.evaluate(all_val_frames_final_matched, all_val_labels_one_hot)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

# Access training accuracy and training loss from history
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']

print("Training Accuracy:", train_accuracy)
print("Training Loss:", train_loss)

# Calculate average training accuracy and average training loss
avg_train_accuracy = np.mean(train_accuracy)
avg_train_loss = np.mean(train_loss)

print("Average Training Accuracy:", avg_train_accuracy)
print("Average Training Loss:", avg_train_loss)
