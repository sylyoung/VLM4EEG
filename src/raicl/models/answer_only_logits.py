"""Compute the supervised Qwen loss only at labeled token positions.

The single-example training loss and gradients match full-position cross-entropy.
Generation and evaluation continue to use the stock forward method.
"""
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5

STOCK_FORWARD = qwen3_5.Qwen3_5ForConditionalGeneration.forward
_calls = [0]


def answer_only_forward(self, *args, labels=None, logits_to_keep=0, **kwargs):
    if labels is None or not self.training:
        return STOCK_FORWARD(self, *args, labels=labels, logits_to_keep=logits_to_keep, **kwargs)
    if not (logits_to_keep is None or (isinstance(logits_to_keep, int) and logits_to_keep == 0)):
        raise SystemExit(f"FATAL: answer-only logits got logits_to_keep={logits_to_keep!r} in training")
    # Checked for one example per step only (every config that uses this has per_device_train_batch_size 1); a
    # padded batch of two could not even be collated by llamafactory 0.9.5 for Qwen3.5 in the check.
    if labels.shape[0] != 1:
        raise SystemExit(f"FATAL: answer-only logits is checked for batch size 1, got {labels.shape[0]}")
    # shift[0, p] is the label of token p + 1, the token position p predicts (ForCausalLMLoss's own shift).
    shift = F.pad(labels, (0, 1), value=-100)[..., 1:]
    keep = (shift[0] != -100).nonzero().squeeze(-1)
    out = STOCK_FORWARD(self, *args, labels=None, logits_to_keep=keep, **kwargs)
    if not isinstance(out, qwen3_5.Qwen3_5CausalLMOutputWithPast):
        raise SystemExit(f"FATAL: answer-only logits expected a Qwen3_5CausalLMOutputWithPast, got {type(out)}")
    vocab = self.config.text_config.vocab_size
    loss = F.cross_entropy(out.logits.float().reshape(-1, vocab), shift[:, keep].reshape(-1).to(out.logits.device),
                           ignore_index=-100, reduction="mean")
    _calls[0] += 1
    if _calls[0] == 1:
        print(f"answer-only-logits active: kept {keep.numel()} of {labels.shape[-1]} positions, "
              f"loss {loss.item():.6f}", flush=True)
    # A new output, built as the stock forward builds it: setting .loss on an output whose loss was None would not
    # add it to the output's keys, and the trainer reads outputs["loss"].
    return qwen3_5.Qwen3_5CausalLMOutputWithPast(
        loss=loss, logits=out.logits, past_key_values=out.past_key_values, hidden_states=out.hidden_states,
        attentions=out.attentions, rope_deltas=out.rope_deltas)


def install():
    qwen3_5.Qwen3_5ForConditionalGeneration.forward = answer_only_forward
    print("answer-only-logits installed on Qwen3_5ForConditionalGeneration.forward", flush=True)


def uninstall():
    qwen3_5.Qwen3_5ForConditionalGeneration.forward = STOCK_FORWARD
