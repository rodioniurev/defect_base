from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass
