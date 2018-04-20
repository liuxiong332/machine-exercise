安装K8S可以按照官方指南分别安装`minikube`和`kubectl`来进行安装，但是在我的安装过程中，有如下注意事项：

### 修改machine文件夹（可选）

运行`minikube start`之前设置好`MINIKUBE_HOME`环境变量，默认是`C:\users\<user>\.minikube`，例如`MINIKUBE_HOME=C:\Users\<myuser>\bin\minikube_home`，这样`minikube`将会把虚拟机相关的设置和缓存都存放在这里，如果你后面需要进行特别的设置，可以找到你设置的这个目录进行必要的设置来自定义`minikube`虚拟机。对于使用`VirtualBox`作为虚拟机来说，`docker`和`minikube`都会创建对应的虚拟机，其中`docker`创建了名为`default`的虚拟机，而`minikube`将会创建`minikube`的虚拟机。



### 拉取Google镜像

当我们直接通过`kubectl create pod --image=nginx`来创建pod时，会出现`ContainerCreating`这个状态，然后使用`minikube logs`会看到类似于`gcr.io/pause-amd64`无法pull的错误，这就是由于`gcr.io`这个域名在国内被屏蔽的原因，绕过的方法如下：

1. 由于`kubernetes`需要从Google镜像`gcr.io`上pull镜像，但是这个域名在国内被屏蔽了，解决方案可以参考https://github.com/fission/fission/issues/76，其实就是先运行'minikube delete'删掉已有`minikube`虚拟机，然后运行`minikube start --extra-config=kubelet.PodInfraContainerImage=kubernetes/pause`从docker hub获取上述的k8s镜像。
2. 通过Docker Hub或者阿里云docker源下载，具体方案如下链接https://github.com/chunchill/minikube-startup-china，拉取之后直接重命名。

### Pull镜像速度慢

官方的pull地址在国内的访问地址很慢，可以使用阿里云提供的镜像服务，方法可以参考https://yq.aliyun.com/articles/29941，其实就是首先使用`minikube ssh`登录`minikube`虚拟机，然后编辑`/etc/docker/daemon.json`（没有则新建此文件)，加上如下片段:

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

### DNS无法解析

有时候我们设置了阿里云镜像，但是发现还是会拉取失败，很大情况下是DNS解析错误，可以通过`minikube ssh`登陆到服务器上，然后运行`nslookup mirror.aliyuncs.com`来查看是否能够解析，如果不能解析，那么我们可以根据如下方法来修改:

1.修改虚拟机的`/etc/docker/daemon.json:`，参考https://development.robinwinslow.uk/2016/06/23/fix-docker-networking-dns/。

```js
{
    "dns": ["114.114.114.114","114.114.115.115","8.8.8.8","8.8.4.4"]
}
```



2.对于docker toolbox来说，可以通过编辑`~/.minikube/machines/minikube/config.json`，将你的DNS服务器添加到`HostOptions/EngineOptions/Dns`，然后重启docker即可。可以参考https://stackoverflow.com/questions/34296230/how-to-change-default-docker-machines-dns-settings。

```js 
{  
   "HostOptions": {
        "Driver": "",
        "Memory": 0,
        "Disk": 0,
        "EngineOptions": {
            "ArbitraryFlags": [],
            "Dns": ["192.168.99.1","8.8.8.8","8.8.4.4"], <-- set it here
            "GraphDir": ""
        }
}
```

