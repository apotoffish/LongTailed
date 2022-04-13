import yaml


class Parameters:
    def __init__(self, path=r"settings.yml"):
        self.__path = path
        self.params = {}

    def load(self):
        with open(self.__path, 'rb') as f:
            y = yaml.safe_load(f)

        self.params['data_path'] = y['default']['data_path']
        self.params['batch_size'] = y['default']['batch_size']
        self.params['isShuffled'] = y['default']['isShuffled']
        self.params['class_num'] = y['default']['class_num']
        self.params['learning_rate'] = y['default']['learning_rate']
        self.params['momentum'] = y['default']['momentum']
        self.params['step_size'] = y['default']['step_size']
        self.params['gamma'] = y['default']['gamma']
        self.params['epochs'] = y['default']['epochs']
        self.params['time_elapsed'] = y['default']['time_elapsed']

        if y['default']['criterion'] == 'CrossEntropyLoss':
            self.params['criterion'] = y['default']['criterion']
        elif y['default']['criterion'] == 'TripletLoss':
            self.params['criterion'] = y['default']['criterion']
            self.params['triplet_margin'] = y['TripletLoss']['margin']
            self.params['embedding_size'] = y['TripletLoss']['embedding_size']
        elif y['default']['criterion'] == 'NpairLoss':
            self.params['criterion'] = y['default']['criterion']
            self.params['l2_regression'] = y['NpairLoss']['l2_regression']
            self.params['embedding_size'] = y['NpairLoss']['embedding_size']
            #self.params['triplet_margin'] = y['TripletLoss']['margin']