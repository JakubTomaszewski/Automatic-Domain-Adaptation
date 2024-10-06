from abc import ABC, abstractmethod


class BaseGenerationModel(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def generate_images(self, *args, **kwargs) -> list[str]:
        pass
