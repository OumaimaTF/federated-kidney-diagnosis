import os 
import time
import numpy as np
import flwr as fl
import tensorflow as tf
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from utils import load_and_preprocess_data, build_mlp, evaluate_tf_model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, csv_path, pushgateway_address):
        self.client_id = client_id
        self.pushgateway_address = pushgateway_address
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = load_and_preprocess_data(csv_path)
        self.model = build_mlp(input_dim=self.X_train.shape[1])
        
        self.last_logged_round = None

    def get_parameters(self, config):
        # Retourner les poids du modèle sous forme de liste de numpy arrays
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Charger les poids reçus
        if parameters is not None:
            self.model.set_weights(parameters)

        current_round = config.get("server_round", 0)
        start_time = time.time()
        
        # Entraîner le modèle (par exemple 1 epoch par round)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        
        training_time = time.time() - start_time

        acc, f1_macro, f1_weighted, precision, recall, roc = evaluate_tf_model(
            self.model, self.X_test, self.y_test
        )

        self.push_metrics(current_round, acc, f1_macro, f1_weighted, precision, recall, roc, training_time)

        if current_round != self.last_logged_round:
            print(f"[{self.client_id}] Round {current_round} | Acc: {acc:.4f} | F1-Macro: {f1_macro:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f}")
            self.last_logged_round = current_round

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        if parameters is not None:
            self.model.set_weights(parameters)

        acc, f1_macro, f1_weighted, precision, recall, roc = evaluate_tf_model(
            self.model, self.X_test, self.y_test
        )
    
        # Retourner toutes les métriques au serveur
        return float(1 - acc), len(self.X_test), {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision": float(precision),
            "recall": float(recall),
            "roc_auc": float(roc)
        }



    def push_metrics(self, current_round, acc, f1_macro, f1_weighted, precision, recall, roc, training_time):
        registry = CollectorRegistry()

        g_acc = Gauge('fl_client_accuracy', 'Accuracy', ['client_id', 'round'], registry=registry)
        g_f1_macro = Gauge('fl_client_f1_macro', 'F1 Macro', ['client_id', 'round'], registry=registry)
        g_f1_weighted = Gauge('fl_client_f1_weighted', 'F1 Weighted', ['client_id', 'round'], registry=registry)
        g_precision = Gauge('fl_client_precision', 'Precision', ['client_id', 'round'], registry=registry)
        g_recall = Gauge('fl_client_recall', 'Recall', ['client_id', 'round'], registry=registry)
        g_roc_auc = Gauge('fl_client_roc_auc', 'ROC AUC Score', ['client_id', 'round'], registry=registry)
        g_training_time = Gauge('fl_client_training_time', 'Training Time (s)', ['client_id', 'round'], registry=registry)

        # Appliquer zéro-padding sur le numéro de round
        round_str = f"{current_round:02}"

        labels = {'client_id': self.client_id, 'round': round_str}

        g_acc.labels(**labels).set(acc)
        g_f1_macro.labels(**labels).set(f1_macro)
        g_f1_weighted.labels(**labels).set(f1_weighted)
        g_precision.labels(**labels).set(precision)
        g_recall.labels(**labels).set(recall)
        g_roc_auc.labels(**labels).set(roc)
        g_training_time.labels(**labels).set(training_time)

        push_to_gateway(
            self.pushgateway_address,
            job='fl_client',
            grouping_key={'client_id': self.client_id, 'round': round_str},  # ✅ aussi ici
            registry=registry
        )



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    client_id = "client2"
    pushgateway_address = "pushgateway:9091"
    csv_path = os.path.join(base_dir, "..", "dataset", f"{client_id}_data.csv")

    client = FlowerClient(client_id, csv_path, pushgateway_address)
    fl.client.start_client(server_address="server:8080", client=client)