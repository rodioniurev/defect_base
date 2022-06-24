from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract class for operations with model
    """
    def __init__(self, config):
        self.config = config
        self.model = None

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception('Build the model first')
        print('Saving model...')
        self.model.save_weights(checkpoint_path)
        print('Model saved')

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception('Build the model first')
        print(f'Loading model checkpoint {checkpoint_path}\n')
        self.model.save_weights(checkpoint_path)
        print('Model saved')

    @abstractmethod
    def build_model(self):
        pass

