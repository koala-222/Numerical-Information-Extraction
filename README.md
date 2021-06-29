## NIE (Numerical Information Extraction) 数值信息抽取

### 任务定义
作为信息抽取任务的分支，数值信息抽取旨在抽取文本中的数值信息，包括数值、单位、修饰词、属性、对象等。
类似于事件抽取任务，设定数值抽取任务。以数值作为触发词，并对应不同类型。每个数值对应若干论元。

类型定义如下：

1. 度量：在某属性上测度的取值结果。"a [350] MW power station"
2. 比值："an unemployment rate of over [30%]"
3. 指代："these [two] took part in the Trojan War"
4. 序数："its [first] publication in London"
5. 计数：对计数对象按某个单位计数的结果。"a 450 seat theatre"

论元定义如下：
![](https://raw.githubusercontent.com/Hao-Kailong/blog-image/master/NRE/%E6%95%B0%E5%80%BC%E8%AE%BA%E5%85%83.png)

### 环境配置
> tensorflow==1.15.0
>
> tensorflow-estimator==1.15.1
>
> python==3.6
>
> CUDA==10.1
>
> cuDNN==7605
>
> stanfordcorenlp==4.2.2

### 运行命令
#### 训练
~~~bash
Under code/JointLearningModel:

python run_highly_joint_with_lstm_crf.py --num_train_epochs 10
~~~
#### 预测
~~~bash
Under code:

python infer_pipeline.py --saved_checkpoint ../saved_model --predict_dir ../standard_dataset/infer
~~~

### 目录结构
* code: 源代码
  * Bert: Google Bert
  * JointLearningModel: Neural models
    * run_highly_joint_with_lstm_crf.py
  * NumberTrigger: 触发词规则模块
  * Utils
  * infer_pipeline.py: 推断脚本
* results
* saved_model: 保存的训练好的模型
* standard_dataset
  * split_files: 数据集
  * infer: 待推断的文本文件夹
* stanford-corenlp-4.2.2
* wwm_cased_L-24_H-1024_A-16

