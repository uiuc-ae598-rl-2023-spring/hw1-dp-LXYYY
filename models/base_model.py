import abc

# define an abc models class
class BaseModel(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.build_model()

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def step(self, obs):
        pass


class PolicyIteration(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        pass

    def step(self, obs):
        return self.model.step(obs)