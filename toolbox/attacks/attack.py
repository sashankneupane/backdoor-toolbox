from abc import ABC, abstractmethod
from typing import Any, Optional

class Attack(ABC):

    def __init__(
        self,
        model,
        trainset: Optional[Any] = None,
        valset: Optional[Any] = None,
        testset: Optional[Any] = None
    ) -> None:
        
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.testset = testset