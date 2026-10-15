# SPDX-License-Identifier: Apache-2.0
# Adapted LLaMA-Factory processing; see licenses/LLaMA-Factory.txt.
"""Batched Qwen input construction used for the CHSZ RAICL evaluations.

Adapted from the benchmark evaluator and LLaMA-Factory HuggingfaceEngine.
The chat template and multimodal plugin preserve the model-specific image tokens.
"""
import torch
from transformers import GenerationConfig
from llamafactory.extras.constants import IMAGE_PLACEHOLDER


def process_one(eng, messages, images, input_kwargs):
    """Replicates llamafactory HuggingfaceEngine._process_args exactly for a
    single (messages, images) pair, so the batch input is byte-identical to the
    single-stream path (e14_eval.py -> ChatModel.chat). Returns
    (gen_kwargs, prompt_length)."""
    videos = None
    audios = None
    mm_input_dict = {"images": [], "videos": [], "audios": [],
                     "imglens": [0], "vidlens": [0], "audlens": [0]}
    if images is not None:
        mm_input_dict.update({"images": images, "imglens": [len(images)]})
        if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
            messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]

    if videos is not None:
        mm_input_dict.update({"videos": videos, "vidlens": [len(videos)]})
        if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
            messages[0]["content"] = IMAGE_PLACEHOLDER * len(videos) + messages[0]["content"]

    if audios is not None:
        mm_input_dict.update({"audios": audios, "audlens": [len(audios)]})
        if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
            messages[0]["content"] = IMAGE_PLACEHOLDER * len(audios) + messages[0]["content"]

    processor = eng.processor
    tokenizer = eng.tokenizer
    template = eng.template
    model = eng.model

    messages = template.mm_plugin.process_messages(
        messages, mm_input_dict["images"], mm_input_dict["videos"],
        mm_input_dict["audios"], processor)
    paired_messages = messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, None, None)
    prompt_ids, _ = template.mm_plugin.process_token_ids(
        prompt_ids, None, mm_input_dict["images"], mm_input_dict["videos"],
        mm_input_dict["audios"], tokenizer, processor)
    prompt_length = len(prompt_ids)
    inputs = torch.tensor([prompt_ids], device=model.device)
    attention_mask = torch.ones_like(inputs, dtype=torch.long)

    generating_args = dict(eng.generating_args)
    generating_args.update(
        dict(
            do_sample=input_kwargs.pop("do_sample", None) if "do_sample" in input_kwargs else generating_args["do_sample"],
            temperature=input_kwargs.pop("temperature", None) if "temperature" in input_kwargs else generating_args["temperature"],
            top_p=input_kwargs.pop("top_p", None) if "top_p" in input_kwargs else generating_args["top_p"],
            top_k=input_kwargs.pop("top_k", None) if "top_k" in input_kwargs else generating_args["top_k"],
            num_return_sequences=input_kwargs.pop("num_return_sequences", 1),
            repetition_penalty=input_kwargs.pop("repetition_penalty", None)
            if "repetition_penalty" in input_kwargs else generating_args["repetition_penalty"],
            length_penalty=input_kwargs.pop("length_penalty", None)
            if "length_penalty" in input_kwargs else generating_args["length_penalty"],
            skip_special_tokens=input_kwargs.pop("skip_special_tokens", None)
            if "skip_special_tokens" in input_kwargs else generating_args["skip_special_tokens"],
            eos_token_id=template.get_stop_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    )

    if isinstance(generating_args["num_return_sequences"], int) and generating_args["num_return_sequences"] > 1:
        generating_args["do_sample"] = True
        generating_args["temperature"] = generating_args["temperature"] or 1.0

    if not generating_args["temperature"]:
        generating_args["do_sample"] = False

    if not generating_args["do_sample"]:
        generating_args.pop("temperature", None)
        generating_args.pop("top_p", None)

    max_length = input_kwargs.pop("max_length", None)
    max_new_tokens = input_kwargs.pop("max_new_tokens", None)
    if max_length:
        generating_args.pop("max_new_tokens", None)
        generating_args["max_length"] = max_length
    if max_new_tokens:
        generating_args.pop("max_length", None)
        generating_args["max_new_tokens"] = max_new_tokens

    gen_kwargs = dict(
        inputs=inputs,
        attention_mask=attention_mask,
        generation_config=GenerationConfig(**generating_args),
    )

    mm_inputs = template.mm_plugin.get_mm_inputs(
        **mm_input_dict, batch_ids=[prompt_ids], processor=processor)
    for key, value in mm_inputs.items():
        if isinstance(value, list) and isinstance(value[0], torch.Tensor):
            value = torch.stack(value)
        elif (isinstance(value, list) and isinstance(value[0], list)
              and isinstance(value[0][0], torch.Tensor)):
            value = torch.stack([torch.stack(v) for v in value])
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if torch.is_floating_point(value):
            value = value.to(model.dtype)
        if key == "second_per_grid_ts":
            gen_kwargs[key] = value.tolist()
        else:
            gen_kwargs[key] = value.to(model.device)

    return gen_kwargs, prompt_length


def assemble_batch(gen_list, tokenizer):
    """Left-pad input_ids/attention_mask to the longest trial in the batch and
    concatenate vision tensors across trials (trial order preserved; the model
    distributes vision features to <image_pad> tokens in global order)."""
    ids = [g["inputs"][0] for g in gen_list]
    max_len = max(len(x) for x in ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = torch.full((len(ids), max_len), pad_id, dtype=ids[0].dtype, device=ids[0].device)
    mask = torch.zeros((len(ids), max_len), dtype=torch.long, device=ids[0].device)
    for i, x in enumerate(ids):
        off = max_len - len(x)
        padded[i, off:] = x
        mask[i, off:] = 1
    bkw = {"input_ids": padded, "attention_mask": mask,
           "generation_config": gen_list[0]["generation_config"]}
    for key in ("pixel_values", "image_grid_thw", "image_sizes"):
        vals = [g[key] for g in gen_list if key in g]
        if not vals:
            continue
        vals = [v.unsqueeze(0) if v.ndim == 3 else v for v in vals]  # 1-image case: [3,H,W] -> [1,3,H,W]
        bkw[key] = torch.cat(vals, dim=0)
    return bkw
