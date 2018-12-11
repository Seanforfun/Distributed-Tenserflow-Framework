# Distributed-Tenserflow-Framework

This is a distributed Framework used for distributed training and evaluation. I am still working on this Framework.

Other than the tutorials in Tensorflow/Model, I wrote this framework in the way that EE in Java, removing the strong coupling between the project and the system. I used a lot of reflaction to remove the initialization of instance(Just like IOC), and create a lot of function handler for user's convinience so that can create their own aspects and cutpoint has already previoded in the framework by hard coding, that's because I didn't find aspect in python, and readlly don't want to spend to much time on creating proxy XP.

Some instruction will come soon once I finished testing the framework in different occasions.

### Link
1. [Source code of this framework](https://github.com/Seanforfun/Distributed-Tensorflow-Framework)
2. [My explaination to this framework: Tensorflow| 集群式处理思想](https://seanforfun.github.io/deeplearning/2018/12/03/DistributeTensorflowFramework.html)
