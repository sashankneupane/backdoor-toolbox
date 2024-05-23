import torch

from torchvision import transforms


class ScaleUP:
    
    def __init__(self, model, scaling_factors, normalizer, denormalizer, device='cuda'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.scaling_factors = scaling_factors
        self.normalizer = normalizer
        self.denormalizer = denormalizer

    def calculate_spc(self, dataloader):
        
        self.model.eval()
        spc_scores = torch.tensor([]).to(self.device)

        with torch.no_grad():

            for batch in dataloader:

                inputs = batch[0].to(self.device)
                
                # MLaaS scenario - the model only retuns the predictions
                original_preds = self.model(inputs)

                consistent_counts = torch.zeros_like(original_preds)

                for scale in self.scaling_factors:
                    # scaled_inputs = torch.clamp(inputs * scale, 0, 1)
                    # scale the denormalized inputs and normalize them again
                    scaled_inputs = self.normalizer(self.denormalizer(inputs) * scale)
                    
                    scaled_preds = self.model(scaled_inputs)

                    consistent_counts += (original_preds == scaled_preds).int()

                batch_spc_scores = consistent_counts.float() / len(self.scaling_factors)
                spc_scores = torch.cat((spc_scores, batch_spc_scores))
        
        return spc_scores
    

    # write a function that updates mean_best_spc and std_best_spc
    def update_benign_spc(self, benign_loader):
        spc_scores = self.calculate_spc(benign_loader)
        self.mean_benign_spc = spc_scores.mean()
        self.std_benign_spc = spc_scores.std()


    def normalize_spc(self, spc_scores):
        if not hasattr(self, 'mean_benign_spc') or not hasattr(self, 'std_benign_spc'):
            raise ValueError('Please run update_benign_spc first')
        return (spc_scores - self.mean_benign_spc) / self.std_benign_spc
    
    
    def detect_backdoor(self, dataloader, threshold=0.7):
        
        spc_scores = self.calculate_spc(dataloader)

        if hasattr(self, 'mean_benign_spc') and hasattr(self, 'std_benign_spc'):
            spc_scores = self.normalize_spc(spc_scores)

        return spc_scores > threshold