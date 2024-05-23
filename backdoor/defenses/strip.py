import torch
from sklearn import metrics
from tqdm import tqdm

class STRIP:
    def __init__(self, model, dataset, transform, num_perturbations=100, entropy_threshold=0.5):
        self.model = model
        self.dataset = dataset
        self.transform = transform
        self.num_perturbations = num_perturbations
        self.entropy_threshold = entropy_threshold

    def entropy(self, probs):
        return -torch.sum(probs * torch.log2(probs + 1e-8), dim=1)

    def detect(self):
        self.model.eval()
        y_true = []
        y_score = []

        for img, label in tqdm(self.dataset):
            img = img.unsqueeze(0).cuda()
            perturbed_preds = []

            for _ in range(self.num_perturbations):
                perturbation = torch.rand_like(img).cuda()
                perturbed_img = torch.clamp(img + perturbation, 0, 1)
                output = self.model(perturbed_img)
                perturbed_preds.append(torch.nn.functional.softmax(output, dim=1))

            perturbed_preds = torch.cat(perturbed_preds, dim=0)
            avg_entropy = self.entropy(perturbed_preds).mean().item()

            y_true.append(label)
            y_score.append(avg_entropy)

        y_true = torch.tensor(y_true)
        y_score = torch.tensor(y_score)
        y_pred = (y_score < self.entropy_threshold).int()

        fpr, tpr, thresholds = metrics.roc_curve(y_true.cpu(), y_score.cpu())
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true.cpu(), y_pred.cpu()).ravel()

        print(f"TPR: {tp / (tp + fn) * 100:.2f}%")
        print(f"FPR: {fp / (tn + fp) * 100:.2f}%")
        print(f"AUC: {auc:.4f}")