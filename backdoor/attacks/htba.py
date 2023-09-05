import os
import time
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from .attack import Attack
from ..poisons import HTBAPoison


class PoisonGenerationDataset(data.Dataset):
    def __init__(self, data_root, path_to_txt_file, transform):
        self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.file_list[idx])
        img = Image.open(image_path).convert('RGB')
        # target = self.file_list[idx].split()[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, image_path

    def __len__(self):
        return len(self.file_list)
    


class HTBAAttack(Attack):

    def __init__(
            self,
            device,
            classifier,
            trainset,
            testset,
            batch_size,
            target_class,
            eps,
            seed=0
    ) -> None:
        
        super().__init__(device, classifier, trainset, testset, batch_size, target_class, seed)
    

    def attack(self, model, epoch, patch_size, eps, lr, trigger_id, num_source, data_root):

        since = time.time()

        # TRIGGER PARAMETERS
        trans_image = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        ])
        trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                                            transforms.ToTensor(),
                                            ])

        # PERTURBATION PARAMETERS
        eps1 = eps/255.0
        lr1 = lr

        trigger = Image.open('data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
        trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)


        # Use source wnid list
        if num_source==1:
            self.logger.info("Using single source for this experiment.")
        else:
            self.logger.info("Using multiple source for this experiment.")



        dataset_target = PoisonGenerationDataset(data_root + "/train", 1, trans_image)
        dataset_source = PoisonGenerationDataset(data_root + "/train", 2, trans_image)

        # SOURCE AND TARGET DATALOADERS
        train_loader_target = torch.utils.data.DataLoader(dataset_target, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)

        train_loader_source = torch.utils.data.DataLoader(dataset_source, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)


        self.logger.info("Number of target images:{}".format(len(dataset_target)))
        self.logger.info("Number of source images:{}".format(len(dataset_source)))

        # USE ITERATORS ON DATALOADERS TO HAVE DISTINCT PAIRING EACH TIME
        iter_target = iter(train_loader_target)
        iter_source = iter(train_loader_source)

        num_poisoned = 0
        for i in range(len(train_loader_target)):

            # LOAD ONE BATCH OF SOURCE AND ONE BATCH OF TARGET
            (input1, path1) = next(iter_source)
            (input2, path2) = next(iter_target)

            img_ctr = 0

            input1 = input1.cuda(gpu)
            input2 = input2.cuda(gpu)
            pert = nn.Parameter(torch.zeros_like(input2, requires_grad=True).cuda(gpu))

            for z in range(input1.size(0)):
                if not rand_loc:
                    start_x = 224-patch_size-5
                    start_y = 224-patch_size-5
                else:
                    start_x = random.randint(0, 224-patch_size-1)
                    start_y = random.randint(0, 224-patch_size-1)

                # PASTE TRIGGER ON SOURCE IMAGES
                input1[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger

            output1, feat1 = model(input1)
            feat1 = feat1.detach().clone()

            for k in range(input1.size(0)):
                img_ctr = img_ctr+1
                # input2_pert = (pert[k].clone().cpu())

                fname =  '/' + 'badnet_' + str(os.path.basename(path1[k])).split('.')[0] + '_' + 'epoch_' + str(epoch).zfill(2)\
                        + str(img_ctr).zfill(5)+'.png'

                num_poisoned +=1

            for j in range(num_iter):
                # lr1 = adjust_learning_rate(lr, j)

                output2, feat2 = model(input2+pert)

                # FIND CLOSEST PAIR WITHOUT REPLACEMENT
                feat11 = feat1.clone()
                dist = torch.cdist(feat1, feat2)
                for _ in range(feat2.size(0)):
                    dist_min_index = (dist == torch.min(dist)).nonzero().squeeze()
                    feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
                    dist[dist_min_index[0], dist_min_index[1]] = 1e5

                loss1 = ((feat1-feat2)**2).sum(dim=1)
                loss = loss1.sum()

                losses.update(loss.item(), input1.size(0))

                loss.backward()

                pert = pert- lr1*pert.grad
                pert = torch.clamp(pert, -eps1, eps1).detach_()

                pert = pert + input2

                pert = pert.clamp(0, 1)

                if j%100 == 0:
                    self.logger.info("Epoch: {:2d} | i: {} | iter: {:5d} | LR: {:2.4f} | Loss Val: {:5.3f} | Loss Avg: {:5.3f}"
                                .format(epoch, i, j, lr1, losses.val, losses.avg))

                if loss1.max().item() < 10 or j == (num_iter-1):
                    for k in range(input2.size(0)):
                        img_ctr = img_ctr+1
                        input2_pert = (pert[k].clone().cpu())

                        fname = saveDir_poison + '/' + 'loss_' + str(int(loss1[k].item())).zfill(5) + '_' + 'epoch_' + \
                                str(epoch).zfill(2) + '_' + str(os.path.basename(path2[k])).split('.')[0] + '_' + \
                                str(os.path.basename(path1[k])).split('.')[0] + '_kk_' + str(img_ctr).zfill(5)+'.png'

                        save_image(input2_pert, fname)
                        num_poisoned +=1

                    break

                pert = pert - input2
                pert.requires_grad = True

        time_elapsed = time.time() - since
        self.logger.info('Training complete one epoch in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 

    def evaluate_attack(self):
        return super().evaluate_attack()