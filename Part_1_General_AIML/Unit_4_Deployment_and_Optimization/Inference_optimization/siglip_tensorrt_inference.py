import numpy as np
import torch
from PIL import Image
from torch2trt import TRTModule
from transformers import SiglipProcessor

processor = SiglipProcessor.from_pretrained('google/siglip2-base-patch16-224', model_max_length=64)

# first step is to record some constants used in SigLIP models. These are not exported to TensorRT as they do not participate in any computation, e.g. they are not model weights
# model = SiglipModel.from_pretrained('google/siglip2-base-patch16-224', torch_dtype=torch.float16).cuda().eval()
# print(model.logit_scale.exp().item(), model.logit_bias.item())  # record this down then we declare as constant below. 112.4375 -16.765625

logit_scale_exp = torch.tensor([112.4375], device='cuda', dtype=torch.float16, requires_grad=False)
logit_bias = torch.tensor([-16.765625], device='cuda', dtype=torch.float16, requires_grad=False)

# load both parts
vision_trt = TRTModule()
vision_trt.load_state_dict(torch.load('vision_trt.pth'))
text_trt = TRTModule()
text_trt.load_state_dict(torch.load('text_trt.pth'))

image = Image.open("cat.jpg")
image = np.asarray(image)
image = torch.tensor(image, dtype=torch.float16, device='cuda').permute(2, 0, 1)

candidate_labels = ['This is a photo of a cat.', 'This is a photo of a dog.']
feats = processor(images=[image], text=candidate_labels, padding="max_length", return_tensors='pt').to('cuda')

vision_input = feats['pixel_values'].type(torch.float16)
text_input = feats['input_ids']  # tokenized already
image_feat = vision_trt(vision_input)['pooler_output']
text_feat = text_trt(text_input)['pooler_output']
image_feat /= image_feat.norm(p=2, dim=-1, keepdim=True)
text_feat /= text_feat.norm(p=2, dim=-1, keepdim=True)
similarity_score = image_feat @ text_feat.T * logit_scale_exp + logit_bias  # sigmoid activation is not needed here as we are just ranking the scores
similarity_score = similarity_score.squeeze(0).cpu().numpy()  # remove the extra dimension

result = [
    {"score": score, "label": candidate_label}
    for score, candidate_label in zip(similarity_score, candidate_labels)
]
print(result)  # [{'score': -2.04, 'label': 'This is a photo of a cat.'}, {'score': -9.94, 'label': 'This is a photo of a dog.'}]. The picture is indeed a cat.
