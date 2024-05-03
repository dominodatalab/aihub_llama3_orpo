# Fine tuning Llama3-8b using ORPO

## About this project
This project shows how to fine tune and align Llama3-8b using supervised fine tuning and alignment in one training step as opposed to have one step for fine tuning and another for reward adjustment and alignment. We use [ORPO](https://arxiv.org/pdf/2403.07691) that's available in the `trl` library from Huggingface. In order to fine tune the model, we will load it in 4bit, train a LoRA adapter and merge it back to the base `llama3-8b` model.

The assets available in this project are:

*llama3_sft.ipynb* - This notebook includes all the code necessary to download the dataset, model files and to run the ORPO training for llama3-8b.

*llama3_sft_eval.ipynb* - This notebook uses Eleuther AI's evaluation harness and benchmark [framework](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) to evaluate the base llama3-8b model and the fine tuned ORPO llama3-8b model

The detailed results of the `eqbench` and `truthfulqa` benchmarks are in the `benchmark_results` folder. Please note that we only used 1000 samples from the dataset to tune the model and the results are only for demonstration purposes only. 



## License
This template is licensed under Apache 2.0 and contains the following components: 
* lm-evaluation-harness [MIT License](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/LICENSE.md)
* accelerate [Apache License 2.0](https://github.com/huggingface/accelerate/blob/main/LICENSE)
* bitsandbytes [MIT License](https://github.com/TimDettmers/bitsandbytes/blob/main/LICENSE)
* transformers [Apache License 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)
* peft [Apache License 2.0](https://github.com/huggingface/peft/blob/main/LICENSE)
* datasets [Apache License 2.0](https://github.com/huggingface/datasets/blob/main/LICENSE)
* trl [Apache License 2.0](https://github.com/huggingface/trl/blob/main/LICENSE)


## Set up instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present. Please ensure the "Automatically make compatible with Domino" checkbox is selected while creating the environment.

### Environment Requirements

**Environment Base**

***base image :*** `nvcr.io/nvidia/pytorch:23.10-py3`

***Dockerfile instructions***
```
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness.git /lm-evaluation-harness && \
    cd /lm-evaluation-harness && \
    pip install -e . && \
    pip install -q -U transformers==4.40.1 datasets==2.19.0 accelerate==0.29.3 peft==0.10.0 trl==0.8.6 bitsandbytes==0.43.1 mlflow==2.12.1

```
***Pluggable Workspace Tools** 
```
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: ["/opt/domino/bin/jupyterlab-start.sh"]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
 title: "vscode"
 iconUrl: "/assets/images/workspace-logos/vscode.svg"
 start: [ "/opt/domino/bin/vscode-start.sh" ]
 httpProxy:
    port: 8888
    requireSubdomain: false
```
Please change the value in `start` according to your Domino version. This repo has been tested with Domino 5.10.0 and 5.11.0 .

### Hardware Requirements

This project will run on any Nvidia GPU with >=24GB of VRAM. Also ensure that the `Workspace and Jobs Volume Size` setting of the workspace is set to 100GB as the model files and datasets can get large.

