import tensorflow as tf

# Dataset Load
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

# Class Names
class_names = train_dataset.class_names
print("Classes:", class_names)

# Normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# CNN Model
model = tf.keras.Sequential([

    tf.keras.Input(shape=(180, 180, 3)),

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        2,
        activation="softmax"
    )
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("CNN Model Created Successfully")

# Train
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15
)

print("Training Completed")

# Evaluate
loss, accuracy = model.evaluate(validation_dataset)

print("Validation Accuracy:", accuracy)

# Save Model
model.save("car_bike_model.keras")

print("Model Saved Successfully")