class Defense:

    def __init__(self, model, trainset, testset, batch_size, epochs, lr, criterion, optimizer):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer

