import sys, torch
sys.path.insert(0, '/app/zdf/minimind')
print('[1] torch:', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/app/zdf/minimind/model')
print('[2] tokenizer OK | vocab:', tok.vocab_size, '| bos:', tok.bos_token, tok.bos_token_id, '| eos:', tok.eos_token, tok.eos_token_id, '| pad:', tok.pad_token, tok.pad_token_id)

# chat template 渲染(含 tools / open_thinking 扩展参数)
msgs = [{"role": "user", "content": "你好"}]
plain = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print('[3] chat template OK >>>', repr(plain[:120]))
think = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, open_thinking=True)
print('[4] open_thinking template >>>', repr(think[:120]))
tools_msgs = [{"role": "system", "content": "x", "tools": '[{"type":"function","function":{"name":"get_weather","description":"查询天气","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]'}, {"role": "user", "content": "北京天气"}]
tool_prompt = tok.apply_chat_template([dict(m) for m in tools_msgs], tokenize=False, add_generation_prompt=True, tools=None)
print('[5] tools template OK >>>', repr(tool_prompt[:150]))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
cfg = MiniMindConfig()
model = MiniMindForCausalLM(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f'[6] model init OK | params: {n_params/1e6:.2f}M | layers: {cfg.num_hidden_layers} | hidden: {cfg.hidden_size}')

ids = tok('[BOS]你好，介绍一下你自己。[EOS]', add_special_tokens=False).input_ids
x = torch.tensor([ids])
with torch.no_grad():
    out = model(x, labels=x)
print(f'[7] forward OK | logits: {tuple(out.logits.shape)} | loss: {out.loss.item():.4f} | aux_loss: {out.aux_loss}')

with torch.no_grad():
    gen = model.generate(x, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print('[8] generate OK >>>', repr(tok.decode(gen[0][len(x[0]):], skip_special_tokens=True))[:100])

# MoE 结构检查
cfg_moe = MiniMindConfig(use_moe=True)
m_moe = MiniMindForCausalLM(cfg_moe)
with torch.no_grad():
    out_m = m_moe(x, labels=x)
n_moe = sum(p.numel() for p in m_moe.parameters())
print(f'[9] MoE init OK | params: {n_moe/1e6:.2f}M | loss: {out_m.loss.item():.4f} | aux_loss: {out_m.aux_loss.item():.6f}')
print('ALL SMOKE TESTS PASSED')
