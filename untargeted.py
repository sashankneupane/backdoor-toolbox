from backdoor.poisons import UntargetedPoison, VOCTransform

transform = VOCTransform()

dataset = UntargetedPoison(
    root='data',
    image_set='train',
    download=False,
    transforms=transform
)

dataset.poison_dataset(
    poison_ratio=1,
    trigger_img='trigger_10',
    trigger_size=25,
    random_loc=False,
    per_image=1
)

img, labels = dataset[0]
print(img.shape, labels)