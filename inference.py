from diffusers import StableDiffusionControlNetPipelineBDM, ControlNetModelBDM, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import torch
import argparse
import os
from PIL import Image
import json
import numpy as np
import re
from diffusers.utils import logging

from safetensors.torch import load_file
from collections import defaultdict

from transformers import pipeline
from transformers import ChineseCLIPModel,ChineseCLIPProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

def get_prompts_with_weights(tokenizer, prompt, max_length):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights

def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights

def get_unweighted_text_embeddings(
    text_encoder,
    text_input,
    chunk_length,
    no_boseos_middle = True,
    cn=False
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    if not cn:
        max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
        if max_embeddings_multiples > 1:
            text_embeddings = []
            for i in range(max_embeddings_multiples):
                # extract the i-th chunk
                text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

                # cover the head and the tail by the starting and the ending tokens
                text_input_chunk[:, 0] = text_input[0, 0]
                text_input_chunk[:, -1] = text_input[0, -1]
                text_embedding = text_encoder(text_input_chunk)[0]

                if no_boseos_middle:
                    if i == 0:
                        # discard the ending token
                        text_embedding = text_embedding[:, :-1]
                    elif i == max_embeddings_multiples - 1:
                        # discard the starting token
                        text_embedding = text_embedding[:, 1:]
                    else:
                        # discard both starting and ending tokens
                        text_embedding = text_embedding[:, 1:-1]

                text_embeddings.append(text_embedding)
            text_embeddings = torch.concat(text_embeddings, axis=1)
        else:
            text_embeddings = text_encoder(text_input)[0]
        return text_embeddings
    else:
        bos=101
        eos=102
        pad=0
        max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
        if max_embeddings_multiples > 1:
            text_embeddings = []
            for i in range(max_embeddings_multiples):
                # extract the i-th chunk
                text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

                # cover the head and the tail by the starting and the ending tokens
                text_input_chunk[:, 0] = bos
                text_input_chunk[:, -1] = pad
                pad_index=torch.argmax((text_input_chunk==pad).float(),dim=1)
                text_input_chunk[torch.arange(pad_index.shape[0]),pad_index]=eos
                # text_input_chunk[:, -1] = text_input[0, -1]
                text_embedding = text_encoder(text_input_chunk)[0]

                if no_boseos_middle:
                    if i == 0:
                        # discard the ending token
                        text_embedding = text_embedding[:, :-1]
                    elif i == max_embeddings_multiples - 1:
                        # discard the starting token
                        text_embedding = text_embedding[:, 1:]
                    else:
                        # discard both starting and ending tokens
                        text_embedding = text_embedding[:, 1:-1]

                text_embeddings.append(text_embedding)
            text_embeddings = torch.concat(text_embeddings, axis=1)
        else:
            text_input[:, 0] = bos
            text_input[:, -1] = pad
            pad_index=torch.argmax((text_input==pad).float(),dim=1)
            text_input[torch.arange(pad_index.shape[0]),pad_index]=eos
            text_embeddings = text_encoder(text_input)[0]
        return text_embeddings

def get_weighted_text_embeddings(
    tokenizer,
    text_encoder,
    prompt,
    uncond_prompt = None,
    model_max_length=77,
    max_embeddings_multiples = 5,
    no_boseos_middle = False,
    skip_parsing  = False,
    skip_weighting  = False,
    cn=False
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
    Args:
        pipe (`StableDiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]
#     import pdb; pdb.set_trace()
    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(tokenizer, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [
            token[1:-1] for token in tokenizer(prompt, max_length=max_length, truncation=True).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1]
                for token in tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    if not cn:
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = getattr(tokenizer, "pad_token_id", eos)
    else:
        bos = tokenizer.cls_token_id
        pad = tokenizer.pad_token_id
        eos = pad
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=text_encoder.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=text_encoder.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        text_encoder,
        prompt_tokens,
        model_max_length,
        no_boseos_middle=no_boseos_middle,
        cn=cn
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=text_encoder.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            text_encoder,
            uncond_tokens,
            model_max_length,
            no_boseos_middle=no_boseos_middle,
            cn=cn
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=text_encoder.device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None

def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

common_prompt_en='Best quality, masterpiece, ultra high res, photorealistic,Ultra realistic illustration,hyperrealistic,8k'
common_prompt_cn='高质量,杰作,超高分辨率,照片质感,非常真实,8k'
common_negative_prompt_en='(low quality:1.3), (worst quality:1.3),poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face,Facial blurring,a large crowd, many people,advertising, information, news, watermark, text, username, signature,out of frame, low res, error, cropped, worst quality, low quality, artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'
common_negative_prompt_cn='最差质量,低质量,低分辨率,伪影,丑陋,脸崩坏,变形,模糊,重复,病态,残缺,多余的手指,变异的手,画得不清楚的手,毁容、比例过大、四肢畸形、手臂缺失、腿部缺失、手臂多余、腿部多余、手指融合、手指过多、脖子过长'

BDM_tokenizer = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14").tokenizer
BDM_text_encoder = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14").text_model

parser=argparse.ArgumentParser()
parser.add_argument('file_in')
parser.add_argument('dir_out')
parser.add_argument('pid',type=int)
parser.add_argument('tol',type=int)
parser.add_argument('ckpt')
args=parser.parse_args()

file_in=args.file_in
dir_out=args.dir_out
pid=args.pid
tol=args.tol
ckpt=args.ckpt

controlnet_path = ckpt+'/controlnet'
controlnet = ControlNetModelBDM.from_pretrained(controlnet_path, torch_dtype=torch.float16)

use_controlnet_sd=False
config_path='config'
diffusers_load_config={}
diffusers_load_config['config_files'] = {
    'v1': f'{config_path}/v1-inference.yaml',
    'v2': f'{config_path}/v2-inference-768-v.yaml',
    'xl': f'{config_path}/sd_xl_base.yaml',
    'xl_refiner': f'{config_path}/sd_xl_refiner.yaml',
}
diffusers_load_config['use_safetensors'] = True
pipe = StableDiffusionControlNetPipelineBDM.from_single_file(
    'model/realisticVisionV60B1_v51VAE.safetensors', controlnet=controlnet, torch_dtype=torch.float16,safety_checker=None,**diffusers_load_config
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')

output_path=dir_out
os.makedirs(output_path,exist_ok=True)

json_path = file_in
with open(json_path, encoding='utf-8') as f:
    json_data = json.load(f)

sub_json_data = []
cnt = 0
for dot in json_data:
    cnt += 1
    if cnt % tol != pid:
        continue
    sub_json_data.append(dot)
content=sub_json_data
content=np.array(content)

batch_size=30
num_images_per_prompt=1

for prompt_idx in range(0,len(content),batch_size):
    name=content[prompt_idx:prompt_idx+batch_size,0].tolist()
    prompt=content[prompt_idx:prompt_idx+batch_size,1].tolist()
    if common_prompt_cn!='':
        for iii in range(len(prompt)):
            prompt[iii]+='， '+common_prompt_cn

    prompt_fanyi=content[prompt_idx:prompt_idx+batch_size,2].tolist()
    if common_prompt_en!='':
        for iii in range(len(prompt_fanyi)):
            prompt_fanyi[iii]+=', '+common_prompt_en
    
    if len(prompt)!=batch_size:
        batch_size=len(prompt)
    generator = torch.manual_seed(3546724423)

    encoder_hidden_states_val,n_encoder_hidden_states_val=get_weighted_text_embeddings(pipe.tokenizer,pipe.text_encoder,prompt_fanyi,[common_negative_prompt_en]*batch_size,model_max_length=77,no_boseos_middle=True)
    BDM_encoder_hidden_states_val,n_BDM_encoder_hidden_states_val=get_weighted_text_embeddings(BDM_tokenizer,BDM_text_encoder,prompt,[common_negative_prompt_cn]*batch_size,model_max_length=52,skip_weighting=True,no_boseos_middle=True,cn=True)
    
    image = pipe(
        num_inference_steps=50,generator=generator,image=Image.open('conditioning_image_1.png'),num_images_per_prompt=num_images_per_prompt,\
        prompt_embeds=[encoder_hidden_states_val,\
                        n_encoder_hidden_states_val,\
                        BDM_encoder_hidden_states_val,
                        n_BDM_encoder_hidden_states_val],controlnet_conditioning_scale=1.0,guidance_scale=7.5
    ).images

    prompt_=list(map(lambda x:x.replace(' ','_').replace('\t','_').replace('：','_').replace('，','_').replace('。','_').replace(':','_').replace('"','_').replace('/','_'),prompt))

    for idx,i in enumerate(image):
        i.save(f"{output_path}/{name[idx//num_images_per_prompt]}_{prompt_[idx//num_images_per_prompt][:30]}_{idx%num_images_per_prompt}.png")
