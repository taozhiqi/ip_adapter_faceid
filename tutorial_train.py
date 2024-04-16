import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import cv2
import torch
import numpy as np
from PIL import Image
from id_loss import RecogNetWrapper
import torch.nn.functional as F
from torchvision import transforms
from skimage import transform as trans
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path="", lmk_root_path='', text_root_path =''):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.lmk_root_path = lmk_root_path
        self.text_root_path = text_root_path
        # self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]
        self.data = []
        for item in [os.path.join(self.image_root_path, x) for x in os.listdir(self.image_root_path)]:
            txt_path = item.replace(self.image_root_path, self.lmk_root_path).replace('.jpg', '.txt')
            if os.path.exists(txt_path):
                self.data.append(item)
        self.lmk_index = [74, 77, 46, 84, 90]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
    
    def get_trans_matrix(self, kpt_5):
        tform = trans.SimilarityTransform()
        src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]],dtype=np.float32)
        tform.estimate(kpt_5, src)
        M = tform.params
        if np.linalg.det(M) == 0:
            M = np.eye(3)

        return M[0:2, :].astype(np.float32)

        
    def __getitem__(self, idx):
        # item = self.data[idx] 
        # text = item["text"]
        # image_file = item["image_file"]
        img_file_path = self.data[idx]
        img_name, posfix = os.path.splitext(os.path.basename(img_file_path))
        text_file_path = os.path.join(self.text_root_path, img_name+'.txt')
        lmk_file_path = os.path.join(self.lmk_root_path, img_name + '.txt')
        ### load text
        text_content = []
        with open(text_file_path, 'r') as tf:
            text_content = tf.readlines()
            text_content = [x.strip('\n') for x in text_content]    

        # read image
        raw_image = Image.open(img_file_path)
        text = random.choice(text_content) if text_content else ''
        ### load landmarks, 图像size为1024，关键点需要除以2
        lmk = np.loadtxt(lmk_file_path)[self.lmk_index]/2
        M = torch.from_numpy(self.get_trans_matrix(lmk))
        # ### 测试抠图的结果
        # image = cv2.imread(img_file_path)
        # image = cv2.resize(image, (512, 512))
        # test_image = cv2.warpAffine(image,M, (112, 112))
        # cv2.imwrite('test.jpg', test_image)
        # ### end of test

        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            'M': M
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    Ms = torch.stack([example["M"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        'M':Ms
    }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        # import pdb
        # pdb.set_trace()
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

### 定义获取x0的函数
def pred_x0_from_noise(noise_scheduler, sample, pred_noise, timesteps):
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(sample.device)
    timesteps = timesteps.to(sample.device)
    t = timesteps
    alpha_prod_t = noise_scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = noise_scheduler.alphas_cumprod[torch.clamp(t - 1, min=0)]
    beta_prod_t = (1 - alpha_prod_t).view(timesteps.shape[0], 1, 1, 1)
    beta_prod_t_prev = (1 - alpha_prod_t_prev).view(timesteps.shape[0], 1, 1, 1)
    pred_original_sample = (sample - (beta_prod_t ** (0.5)) * pred_noise) / (alpha_prod_t.view(timesteps.shape[0], 1, 1, 1) ** (0.5))
    pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    return pred_original_sample
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    ### 这个位置先直接定义加载模型位置，后面再进行修改和规整
    args.pretrained_model_name_or_path = '/home/work/tubb/AniPortrait/pretrained_model/stable-diffusion-v1-5'
    args.image_encoder_path = '/home/work/tubb/AniPortrait/pretrained_model/image_encoder'
    pretrained_id_path = './pretrained/backbone.pth'
    data_root = '/home/work/tubb/CelebAMask-HQ/CelebA-HQ-img'
    lmk_root = '/home/work/tubb/CelebAMask-HQ/CelebA-HQ-lmk'
    text_root = '/home/work/tubb/CelebAMask-HQ/celeba-caption'
    save_image_dir = './debug_images'
    ### 
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    ### faceid_model loss
    id_wraper = RecogNetWrapper(pretrained_id_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    id_wraper.net.requires_grad_(False)
    
    #ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    id_wraper.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(tokenizer=tokenizer, size=args.resolution, image_root_path=data_root, lmk_root_path=lmk_root, text_root_path =text_root)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
            
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)
                ### 这个位置通过noise pred 计算x0
                latent_pred = pred_x0_from_noise(noise_scheduler, noisy_latents, noise_pred, timesteps)
                latent_decode_image = vae.decode((1/vae.config.scaling_factor) *latent_pred.to(accelerator.device, dtype=weight_dtype)).sample
                    

                # latent_decode_image = (latent_decode_image+1)/2
                # latent_decode_image = latent_decode_image * 255
                # latent_decode_image = latent_decode_image[0].cpu().numpy().transpose(1, 2, 0)
                # cv2.imwrite('test.jpg', latent_decode_image[:, :, ::-1])
                # cv2.imwrite('test1.jpg', batch['images'][0].cpu().numpy().transpose(1, 2, 0)[:,:,::-1]*255.)
                # import pdb
                # pdb.set_trace()
                id_loss = id_wraper.get_id_loss(batch['images'], batch['M'], latent_decode_image, batch['M'])
                ### end of add 
        
                l1loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = l1loss + id_loss * 0.01
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                ### 输出loss
                avg_l1loss = accelerator.gather(l1loss.repeat(args.train_batch_size)).mean().item()
                avg_idloss = accelerator.gather(id_loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if global_step % 1:
                    debug_image = (latent_decode_image.copy()+1)/2
                    debug_image = debug_image * 255
                    nums_images = debug_image.shape[0]
                    save_debug_dir = os.path.join(save_image_dir, str(global_step))
                    os.makedirs(save_debug_dir, exist_ok=True)
                    for i in range(nums_images):
                        img_content = debug_image[i].cpu().numpy()
                        cv2.imwrite(os.path.join(save_debug_dir, '{}.jpg'.format(i)), img_content[:,:,::-1])

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}, l1loss: {}, idloss : {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss, avg_l1loss, avg_idloss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
    ## test dataset
    # data_root = '/home/work/tubb/CelebAMask-HQ/CelebA-HQ-img'
    # lmk_root = '/home/work/tubb/CelebAMask-HQ/CelebA-HQ-lmk'
    # text_root = '/home/work/tubb/CelebAMask-HQ/celeba-caption'
    # pretrained_model_path = '/home/work/tubb/AniPortrait/pretrained_model/stable-diffusion-v1-5'
    # tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    # data = MyDataset(tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=data_root, lmk_root_path=lmk_root, text_root_path =text_root)
    # data.__getitem__(0)
