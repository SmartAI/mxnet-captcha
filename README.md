# mxnet-captcha

This is project is based on [xlvector's mxnet-ocr](https://github.com/xlvector/learning-dl/tree/master/mxnet/ocr), but modified the multi-label
network construct method, and write a more concense inference code.

# Pay Attention To
1. Don't define `label` as a input symbol
2. Set `allow_missing=True` when set params to model in predict stage

# Changelog 
1. Updated 2017-3-3 pytorch version [torch-captcha](https://github.com/SmartAILM/mxnet-captcha/blob/torch/torch_captcha.ipynb)
