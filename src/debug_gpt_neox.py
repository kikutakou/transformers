import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from logzero import logger


model_name = 'cyberagent/gpt-neox-1b-japanese'

logger.info(f'loading tokenizer {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name)

load_args = {}
load_args.update(torch_dtype=torch.float16)

logger.info(f'loading model {model_name}')
logger.info(f'load_args {load_args}')
model = AutoModelForCausalLM.from_pretrained(model_name, **load_args)

#exit()
logger.info(f'sending to gpu')
model.cuda()

inputs = tokenizer('今日は', return_tensors='pt')
logger.info(f'inputs = {inputs}')

logger.info(f'send inputs to gpu')
inputs = dict(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda())


logger.info(f'forward')
outputs = model.generate(**inputs)

logger.info(f'decode')
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

logger.info(decoded)

logger.info(f'done')
