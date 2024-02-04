import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class SequentialModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def load_dataset(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def calculate_metrics(self):
        predictions = self.model.predict(self.x_test)
        y_pred = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        classification_report_str = classification_report(self.y_test, y_pred)
        confusion_matrix_arr = confusion_matrix(self.y_test, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report_str,
            'confusion_matrix': confusion_matrix_arr
        }

    def evaluate_model(self):
        self.load_model()
        metrics = self.calculate_metrics()
        return metrics

model_path = '/content/sequential_model_saved.h5'
x_train, y_train, x_test, y_test = load_cifar10_data()
sequential_model_loader = SequentialModelLoader(model_path)
sequential_model_loader.load_dataset(x_test, y_test)
metrics = sequential_model_loader.evaluate_model()

print("Accuracy:", metrics['accuracy'])
print("Precision:", metrics['precision'])
print("Recall:", metrics['recall'])
print("F1 Score:", metrics['f1_score'])
print(metrics['classification_report'])
print(metrics['confusion_matrix'])