# DeepModelTrainer

## NOTICE: Development ongoing!

Update Ongoing:

- [ ] 编写 BaseModel 类，继承 nn.Module 并实现一些通用功能，包括：
  - [ ] 初始化参数
  - [ ] 统计参数量
    - [x] case1: 无参数共享
    - [ ] case2：有参数共享
  - [ ] 打印数据流
    - [x] case1: 单一输入
    - [ ] case2：多输入
- [x] 规范 model 的三级结构：基础模块、单一网络、整体模型
  - [x] 所有模块和模型都应该继承 BaseModel 类
  - [x] 单一网络在组成整体网络时，需要经过封装，使其可以通过 args 实例化

TODOS:

- [ ] Convert post-training logs, evaluations, and other functions into hook functions for unified management
- [x] Disgard the `sample` mode in `Trainer` class and use `eval` mode and hooks to sample
- [ ] Rewrite README.md

[English](/README.md) | [简体中文](/readme/README_zh_CN.md)

DMT is a framework for training deep learning models. It is designed to be modular and extensible, allowing for easy experimentation with different models, datasets, and training methods. DMT consists of four main components:

- configs:  YAML files to configure your experiments
- datasets: a sub package provides dataset interfaces
- models:   a sub package provides model interfaces
- trainers: a sub package provides container object to manage training and evaluating processes

## Dependencies

DMT depends on the following libraries:

- [PyTorch](https://pytorch.org/)
- [PyYAML](https://pyyaml.org/)
- [easydict](https://github.com/makinacorpus/easydict)
- [🤗 Accelerate](https://github.com/huggingface/accelerate)

## Usage

1. Prepare the datasets and models. See [datasets&models](#datasets--models) for details.
2. Prepare your specific trainer. See [trainers](#trainers) for details.
3. Write config files. See [configs](#configs) for details.
4. Launch the experiments. See [scripts](#scripts) for details.

## configs

YAML is a human-friendly data serialization language for all programming languages, which makes it a powerful tool for mapping configuration files to Python objects. We use

- YAML files to record the experiment configurations
- `EasyDict` to access the items in a (nested) dict as attributes

More details and examples can be found in [configs.md](readme/configs.md).

## datasets & models

`datasets` directory is a sub package providing dataset classes inherited from `torch.utils.data.Dataset`. The dataset classes are instantiated by only one `args` or `configs` argument, and are used to load data from files and transform them into tensors.

`models` directory is a sub package providing model classes inherited from `torch.nn.Module`. Similarly, the model classes are also instantiated by only one `args` or `configs` argument, and are used to build the model architecture.

More details and examples can be found in [datasets&models.md](readme/datasets&models.md).

## trainers

`trainers` directory is a sub package providing container object to manage training, evaluating and sampling processes.  

## scripts
