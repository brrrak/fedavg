from __future__ import annotations

import random
import logging
import copy
from functools import reduce
from typing import List, Optional, Dict

import torch
from torch import optim


class Node:
    def __init__(self,
                 node_id,
                 model=None,
                 clients: Optional[List[Node]] = None,
                 train_loader=None,
                 test_loader=None,
                 criterion=None,
                 config: Optional[Dict] = None):
        self.node_id = node_id
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
            self.cuda = True
        else:
            self.cuda = False
        self.clients = clients if clients else []
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = criterion
        self.config = config
        if self.config:
            self.config["client_n"] = len(clients)

    def train(self, local_updates=1):
        if not self.train_loader or not self.model or local_updates < 1:
            return

        for epoch in range(local_updates):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if epoch == local_updates - 1:
                logging.info("Client {} Avg Loss: {:.4f}".format(
                    self.node_id,
                    running_loss / len(self.train_loader.dataset)
                ))

    def test(self):
        if not self.test_loader or not self.model:
            return
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images)
                _, predictions = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        # print(f'Test Accuracy: {(100 * correct / total):.2f} %')

        return 100 * correct / total

    def select_clients(self, n):
        """
        Randomly selects n clients to participate in this round.
        :return: List[int]
        """
        client_n = len(self.clients)
        if n < 1:  # used for selecting a percentage of clients
            n = int(n * client_n)
        n = min(n, client_n)
        client_indexes = random.sample(range(client_n), n)

        return client_indexes

    def add_client(self, client: Node):
        self.clients.append(client)
        if self.config:
            try:
                self.config["client_n"] += 1
            except KeyError:
                self.config["client_n"] = 1

    def get_model(self):
        return copy.deepcopy(self.model)

    def load_model(self, model):
        self.model.load_state_dict(model.state_dict())

    def send_model(self, client: Node):
        client.load_model(self.model)

    def aggregate(self, client_models):
        """
        Aggregates the updated client models and updates the global model.
        :param client_models: List of updated client models for the round
        :return: None
        """
        global_state = self.model.state_dict()
        client_states = [model.state_dict() for model in client_models]
        client_n = len(client_states)

        for layer in global_state:
            client_layers = [state[layer] for state in client_states]
            global_state[layer] = reduce(
                lambda a, b: a + b, client_layers
            ) / client_n

        self.model.load_state_dict(global_state)

    def round(self, round_i=1):
        """
        Runs one round of FedAvg.
        :param round_i: Round count for logging
        :return: None
        """
        logging.info("=========================================")
        logging.info("Round {} Starts.".format(round_i))
        client_idx = self.select_clients(self.config["participant_n"])
        logging.info("Selected Clients: {}".format(client_idx))
        updated_models = []
        for i in client_idx:
            self.send_model(self.clients[i])
            self.clients[i].train()
            updated_models.append(self.clients[i].get_model())

        self.aggregate(updated_models)
        acc = self.test()
        logging.info("Round {} Test Accuracy: {:.2f} %".format(round_i, acc))
