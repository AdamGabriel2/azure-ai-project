import tensorflow as tf
from tensorflow.keras import layers, models
from azureml.core import Run, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_and_prepare_data():
    run = Run.get_context()
    workspace = run.experiment.workspace
    dataset = Dataset.get_by_name(workspace, name='your_custom_dataset_name')

    data = dataset.to_pandas_dataframe()
    images = data['image'].apply(tf.image.decode_image)
    labels = data['label'].values

    # Adicionando Data Augmentation
    augmented_images = []
    for image in images:
        # Aplicar técnicas de aumento de dados aqui
        # Exemplo: rotação, espelhamento, zoom, etc.
        augmented_images.append(image)

    # Convertendo para tensores TensorFlow
    train_images, test_images, train_labels, test_labels = train_test_split(
        augmented_images, labels, test_size=0.2, random_state=42
    )

    return train_images, test_images, train_labels, test_labels

def create_and_train_model(train_images, train_labels):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=10)
    return model

def evaluate_model(model, test_images, test_labels):
    # Avaliação do modelo
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

def register_experiment(model, test_images, test_labels):
    run = Run.get_context()
    predictions = model.predict(test_images)
    accuracy = accuracy_score(test_labels, tf.argmax(predictions, axis=1))

    run.log('accuracy', accuracy)
    run.log_model('custom_model', model)

def main():
    train_images, test_images, train_labels, test_labels = load_and_prepare_data()
    model = create_and_train_model(train_images, train_labels)
    evaluate_model(model, test_images, test_labels)
    register_experiment(model, test_images, test_labels)

if __name__ == '__main__':
    main()
