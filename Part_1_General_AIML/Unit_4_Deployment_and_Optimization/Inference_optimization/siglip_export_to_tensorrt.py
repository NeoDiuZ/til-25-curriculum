import torch
from torch2trt import torch2trt  # note that you need to use this version of torch2trt https://github.com/aliencaocao/torch2trt/tree/patch-1 for this example to work, due to a bug with the original torch2trt. Install by pip install git+https://github.com/aliencaocao/torch2trt@patch-1
from transformers import SiglipModel

model = SiglipModel.from_pretrained('google/siglip2-base-patch16-224', torch_dtype=torch.float16).cuda().eval()

# we cannot export together as their input and output are all different, and they are not interconnected in the compute graph
text_model = model.text_model
vision_model = model.vision_model

dummy = torch.ones(1, 3, 224, 224, dtype=torch.float16, device='cuda')
# shape is in [bs, channel, height, width]. The line below assumes we are working with minium of BS=1 and maximum of BS=20, with optimal (most common) shape of BS=10. Input size is fixed at 224x224 as with most image models.
model_trt = torch2trt(vision_model, [dummy], fp16_mode=True, min_shapes=[(1, 3, 224, 224)], opt_shapes=[(10, 3, 224, 224)], max_shapes=[(20, 3, 224, 224)], use_onnx=True)
# we want the embedding only so that is the pooler_output branch
y = vision_model(dummy).pooler_output
y_trt = model_trt(dummy)['pooler_output']
torch.save(model_trt.state_dict(), 'vision_trt.pth')
print('Vision model exported. atol:', torch.max(torch.abs(y - y_trt)))

dummy = torch.ones(1, 64, dtype=torch.long, device='cuda')
# shape is in [bs, seq_len]. The line below assumes we are working with minium of BS=1 and maximum of BS=4, with optimal (most common) shape of BS=2. Sequence length means the maximum length of text tokens we expect is 64. This means any preprocessing must pad to 64 and truncate if longer than 64 tokens.
model_trt = torch2trt(text_model, [dummy], fp16_mode=True, min_shapes=[(1, 64)], opt_shapes=[(2, 64)], max_shapes=[(4, 64)], use_onnx=True)
y = text_model(dummy).pooler_output
y_trt = model_trt(dummy)['pooler_output']
torch.save(model_trt.state_dict(), 'text_trt.pth')
print('Text model exported. atol:', torch.max(torch.abs(y - y_trt)))
