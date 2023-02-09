# **Welcome!**

Basically, this repo is driven by AI models. So the author won't add as many nodes as possible to cover all ONNX's ops.
It's **RECOMMENDED** that you contribute in these situations:  
1. Some bugs need to be fixed;
2. New public models support;
3. Code optimization;
4. New features;
5. Complete TODOs.

It's **NOT RECOMMENDED** that you contribute if your code can't be tested by public models. You should keep it in your private branch,  
even if this op is a popular one. Because there will be a potential risk of merging untested codes.  
There are some suggestions to avoid sensitive data leaking to the public:
1. Use onnx.helper to build a similar ONNX model;
2. Use onnx_tool to save a model without any weight;
3. Use onnx_tool to get a subgraph model with your interested part only;

# Steps for contributing
1. Introducing your changes;
2. How to test your changes: how to download the ONNX model or how to create a same model.
