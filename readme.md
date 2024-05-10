# 数据高效的多模态强化学习量化投资策略

**Contributors**: [Wenhao Yan](https://github.com/TaciturnMute)

:bell: If you have any suggestions or notice something I missed, please don't hesitate to let me know. You can directly email **San Shui** (wenhao19990616@gmail.com), or post an issue on this repo.


## 项目介绍

本项目为**三水**的硕士毕业论文代码。主要使用DRL技术来求解量化投资问题。项目使用DDPG模型来构造多模态投资策略，同时使用了表征学习(State Representation Learning)、MCTS和集成等技术。项目主要应用于量化投资的Trading和Portfolio任务。


## 项目结构

项目包含三个文件夹：data/models/notebook

### data

包含项目所需的数据，以及处理数据的代码。csv文件夹和figure文件夹分别为原始csv数据以及根据csv数据生成的图片。

### models

项目的模型代码。baselines文件夹包含DRL模型和基于统计方法的基准模型。agent.py为项目模型的智能体代码部分。

### notebook

包含notebook运行教程。generate_figure.ipynb为生成图片代码。

portfolio和trading开头的文件对应两种任务。

## 
世界上有形形色色的各种生活，而它们都互不关联。如果无法理解，不要妄想去打扰。   ----完美的日子