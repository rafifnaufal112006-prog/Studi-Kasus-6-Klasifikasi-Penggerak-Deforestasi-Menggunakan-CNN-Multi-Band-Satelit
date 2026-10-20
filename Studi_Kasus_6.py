from tensorflow.keras import layers, models


def create_deforestation_driver_cnn(input_shape=(64, 64, 4), num_classes=5):
    """
    Arsitektur CNN untuk klasifikasi penggerak deforestasi 
    berdasarkan citra multi-band (misal: Sentinel-1 & Sentinel-2).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
       
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax') 
    ])
    
    return model

input_shape = (64, 64, 4)
num_classes = 5
model = create_deforestation_driver_cnn(input_shape, num_classes)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()