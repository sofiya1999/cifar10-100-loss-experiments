from multiprocessing import freeze_support
import timm
import torch
import torchvision
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import utils

cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)


print(cifar100_test[0])

model = timm.create_model('resnet18', pretrained=True)
model.eval()
print(model.pretrained_cfg)
top_k = 5
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

for (image, label) in cifar100_test:
    x = transform(image).unsqueeze(0)
    out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, top_k)
    #predictions = [
    #    {"label": labels[i], "score": v.item()}
    #    for i, v in zip(indices, values)
    #]
    #print(predictions)

#out = model(cifar100_test[0])