import tensorflow as tf
import pandas as pd
from azureml.core import Run, Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    # Carregamento do conjunto de teste
    # ...

def evaluate_model(model, test_images, test_labels):
    # Avaliação do modelo
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Previsões do modelo
    predictions = model.predict(test_images)
    predicted_labels = tf.argmax(predictions, axis=1)

    # Matriz de Confusão
    cm = confusion_matrix(test_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)

    # Relatório de Classificação
    class_names = [str(i) for i in range(10)]  # Altere conforme necessário
    report = classification_report(test_labels, predicted_labels, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # Plotar Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def register_experiment(model, test_images, test_labels):
    run = Run.get_context()

    # Registro do experimento no Azure Machine Learning
    run.log_model('custom_model', model)

def main():
    # Carrega e prepara o conjunto de dados de teste
    test_images, test_labels = load_and_prepare_data()

    # Recupera o modelo treinado
    model_path = Model.get_model_path('custom_model')
    model = tf.keras.models.load_model(model_path)

    # Avalia o modelo
    evaluate_model(model, test_images, test_labels)
    register_experiment(model, test_images, test_labels)

if __name__ == '__main__':
    main()
