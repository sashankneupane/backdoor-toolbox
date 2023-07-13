


from .attack import Attack
from ..poisons import HTBAPoison


class HTBAAttack(Attack):

    def __init__(
            self,
            device,
            classifier,
            trainset,
            testset,
            batch_size,
            target_class,
            seed=0
    ) -> None:
        
        super().__init__(device, classifier, trainset, testset, batch_size, target_class, seed)

    

    def attack(self):
        pass

    def evaluate_attack(self):
        return super().evaluate_attack()