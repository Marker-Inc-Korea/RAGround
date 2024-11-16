# Custom Module

In AutoRAG, we already support more than 50 modules for optimize RAG process.
However there are always requirements for custom modules.
So this part we will tell you how to add custom modules to AutoRAG.

## Step 1: Understand the structure of a Module Class

When you go to the each module code, you can find each module is a class.
The class contains like below.

1. `__init__` method: Initialize the module. It will trigger once, so you have to load model, load GPU memory, connect server, I/O process in here.
2. `__del__` method: When the class destory, you have to clean up whole cache and GPU memory.
3. `pure` method:
