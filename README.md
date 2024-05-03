# Research Seminar
As part of the research seminar "Forschungsprojekt Data Mining im Marketing mit R und Python" by Prof. Dr. Daniel Baier we try to add more exercises and new content whenever we can.

In the python files you will find some introductory scripts for huggingface transformers for sentiment analysis. In the [CUDA_GUIDE](./CUDA_GUIDE.md) you will find instructions to install WSL, CUDA, and tensorflow with GPU support on Windows11 for tensorflow version above 2.10.

## Topics
### Few-Shot Learning
Few-Shot learning has very clear advantages in business and science: whenever little labeled data is available and that data is hard or expensive to label we are interesetd in the best results using the least data possible.

There are generally two major approaches for text few-shot classification.

[SetFit](./uebung13.py) and [ChatGPT](./uebung14.py). To get started with them consider your programming environment and maybe have a look at the [CUDA_GUIDE](./CUDA_GUIDE.md). First things first: start by installing the [torch_reqs.txt](./torch_reqs.txt) with ```pip install -r torch_reqs.txt```. This installs all dependencies for torch alongside gpu requirements if you have an nvidia card!

Then install all other requirements.