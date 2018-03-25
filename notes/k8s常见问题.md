安装K8S可以按照官方指南分别安装`minikube`和`kubectl`来进行安装，但是在我的安装过程中，有如下注意事项：

1. 运行`minikube start`之前设置好`MINIKUBE_HOME`环境变量，例如`MINIKUBE_HOME=C:\Users\liuxiong\bin\minikube_home`，这样`minikube`将会把虚拟机相关的设置和缓存都存放在这里，如果你后面需要进行特别的设置，可以找到你设置的这个目录进行必要的设置来自定义`minikube`虚拟机。对于使用`VirtualBox`作为虚拟机来说，`docker`和`minikube`都会创建对应的虚拟机，其中`docker`创建了名为`default`的虚拟机，而`minikube`将会创建`minikube`的虚拟机。

2. 由于`kubernetes`需要从Google镜像`gcr.io`上pull镜像，但是这个域名在国内被屏蔽了，解决方案可以参考https://github.com/fission/fission/issues/76，其实就是先运行'minikube delete'删掉已有`minikube`虚拟机，然后运行`minikube start --extra-config=kubelet.PodInfraContainerImage=kubernetes/pause`从docker hub获取上述的k8s镜像。

3. 国内的docker拉取速度很慢，可以使用阿里云提供的镜像服务，方法可以参考https://yq.aliyun.com/articles/29941，其实就是首先使用`minikube ssh`登录`minikube`虚拟机，然后编辑`/etc/docker/daemon.json`（没有则新建此文件)，加上如下片段:

   ```js
   {
       "registry-mirrors": ["<your accelerate address>"]
   }
   ```

   然后运行`/etc/init.d/docker restart`重启docker服务。但是对于`minikube`来说比较麻烦，每次`minikube start`的时候都会将镜像文件进行覆盖重写，于是我们的`daemon.json`就会丢失，每次都需要设置。解决方法是找到`$(MINIKUBE_HOME)/.minikube/machines/minikube/config.json`，在`RegistryMirror`这一项填上你的加速地址。

   ```js
   {
       "ConfigVersion": 3,
       "DriverName": "virtualbox",
       "HostOptions": {
           "EngineOptions": {
               "RegistryMirror": ["<accelerate address>"]
           }
       }
   }
   ```

   然后使用`minikube start`重启服务即可。