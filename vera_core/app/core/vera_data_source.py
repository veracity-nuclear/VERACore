import h5py
from abc import ABC, abstractmethod
class VeraDataSource(ABC):

    @property
    @abstractmethod
    def core(self):
        pass

    @property
    @abstractmethod
    def states(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    @abstractmethod
    def active_state(self):
        pass

    @property
    @abstractmethod
    def active_state_index(self):
        pass

    @active_state_index.setter
    @abstractmethod
    def active_state_index(self, index):
        pass
    
    @abstractmethod
    def array(self, array_name):
        pass
