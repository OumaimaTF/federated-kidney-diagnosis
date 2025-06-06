import time
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes
from typing import List, Tuple
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from flwr.server import ServerConfig


class CustomStrategy(FedAvg):
    def __init__(self, *args, pushgateway_address="pushgateway:9091", **kwargs):
        super().__init__(*args, **kwargs)
        self.pushgateway_address = pushgateway_address

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time = time.time()
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instructions:
            fit_ins.config["server_round"] = server_round
        return instructions

    def aggregate_fit(self, rnd, results, failures):
        round_duration = time.time() - self.round_start_time
        self.push_metrics(rnd, round_duration, len(results))
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> float:
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        start_time = time.time()

        for client, eval_res in results:
            num_examples = eval_res.num_examples
            loss = eval_res.loss
            accuracy = eval_res.metrics.get("accuracy", 0.0)

            total_loss += loss * num_examples
            total_accuracy += accuracy * num_examples
            total_samples += num_examples

        eval_duration = time.time() - start_time

        # Moyennes pondérées
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0

        self.push_eval_metrics(rnd, avg_loss, avg_accuracy, eval_duration)

        return super().aggregate_evaluate(rnd, results, failures)

    def push_metrics(self, round_num, round_duration, num_clients):
        registry = CollectorRegistry()
        g_round = Gauge('fl_rounds_total', 'Total number of training rounds', registry=registry)
        g_clients = Gauge('fl_connected_clients', 'Number of connected clients', registry=registry)
        g_duration = Gauge('fl_round_duration_seconds', 'Training round duration (s)', registry=registry)

        g_round.set(round_num)
        g_clients.set(num_clients)
        g_duration.set(round_duration)

        push_to_gateway(self.pushgateway_address, job="fl_server", registry=registry)

    def push_eval_metrics(self, round_num, loss, accuracy, eval_time):
        registry = CollectorRegistry()
        g_loss = Gauge('fl_loss', 'Global aggregated loss', registry=registry)
        g_acc = Gauge('fl_accuracy', 'Global aggregated accuracy', registry=registry)
        g_eval_time = Gauge('fl_evaluation_duration_seconds', 'Evaluation time in seconds', registry=registry)

        g_loss.set(loss)
        g_acc.set(accuracy)
        g_eval_time.set(eval_time)

        push_to_gateway(self.pushgateway_address, job="fl_server", grouping_key={'round': f"{round_num:02}"}, registry=registry)


if __name__ == "__main__":
    strategy = CustomStrategy(pushgateway_address="pushgateway:9091")
    config = ServerConfig(num_rounds=30)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=config,
    )
