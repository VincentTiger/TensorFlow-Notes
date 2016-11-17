## TensorFlow运行机制 ##

TensorFlow是一个适用于机器学习的python包，底层基于C++实现，并使用cuda进行gpu加速。一个TF最基本的元素是tensor和operator。tensor是程序存储于内存中的数据，operator是对数据进行的操作。模型训练的过程就是将输入数据转化为tensor,在tensor上执行各种operator的过程。若干tensor和operator组成Graph。Graph通过组装tensor和operator定义了模型训练中的所有计算。Graph只是定义了模型训练的操作流程，最终在Session中真正执行Graph定义的操作。

### Graph ###

**图** 定义了模型的结构和训练、评估的流程。保留数据入口（placeholder、TextReader等），使用Saver作为模型保存的出口。

不考虑在线Server时，TF的程序分为两部分：

- 组装Graph：定义模型结构、训练算法、评估方法、参数等，生成一个完整的模型训练、评估流程。
- 运行Graph：将数据接入Graph，运行Graph中预定义的计算，训练、评估模型，输出训练信息和评估指标。

一个Graph一般由若干的tensor和operator组成。

### tensor ###

tensor就是存储在内存中的数据，可以是数值型标量、数值型多维数组（numpy.ndarray 张量）和字符串。TF中有几种特殊用途的tensor：

- placeholder：图中的占位符，数据入口。在feed dict输入模式中，数据通过placeholder输入到图中。Session不会保存placeholder的取值。
- Variable：模型中的参数，一般是训练的目标，Session中会保存几个snapshot。
- constant：常量，可以理解为图的控制参数。Session中不可变。
- 其他tensor或直接或间接由这三种tensor通过operator变换产生。

TF中tensor可以是dense tensor也可以是sparse tensor。Tensor（dense tensor）保存一个多维数组。SparseTensor由三个子tensor的indeces（shape=[K, ndim]的tensor），values（shape=[K]的tensor），shape（shape=[ndim]的tensor），其中K为原tensor中非零元的个数，ndim为原tensor的维度。在feed dict输入模式中，Tensor可以用ndarray或list进行feed，SparseTensor用SparseTensorValue（由indices, values, shape三个ndarray或list打包而成）进行feed。

### operator ###

operator是对tensor进行的计算，输入和输出都为一个或多个tensor。除了输入和输出之外，可以有额外控制参数。operator可以有cpu和gpu两种实现，gpu实现是可选的。

定义模型时常用的operator：

- TextReader等文件读取operator
- 矩阵乘法等线性代数运算
- sigmoid，softmax
- 卷积
- pooling
- reduce
- 等等

模型训练与评估常用的operator：

- gradients
- apply_gradients
- streaming_auc
- 等等

**optimizer** 封装了一些基于梯度下降的优化算法，例如随机梯度下降（GradientDescent），自适应步长的梯度下降（AdamGradientDescent）等，它由一系列operator组成。

### Session ###

组装好Graph之后，启动一个Session，在Session中将数据feed给Graph，运行Graph中的operator，并返回fetcher（可以是任意tensor）指定的结果。

Session中存储了Graph和所有参数的状态，并定期生成snapshot，通过Saver保存为checkpoint。在一个新的Session（例如在预估的时候）中可以从checkpoint恢复保存的状态。

### Summary ###

组装Graph的过程中可以添加Summary，它指明了在session运行中对哪些tensor进行监控。在Session运行的时候，通过SummaryWriter定期输出预定义的Summary到日志中。Summary自动包含了Graph的结构。

Summary可以被TensorBoard解析，展示在Web页面中。

### device ###

TF会获取系统可用的device列表，包括cpu和gpu，Graph中的operators会分配到各个device上。

在没有为tensor和operator指定设备的情况下，TF会优先使用第一块显卡gpu0进行存储和计算。如果operator没有gpu版本的实现，相关的计算会默认放在cpu上。如果不想用默认行为，可以用with tf.device为作用域内的operator指定运行的设备。

### 多进程TF ###

TF可以采用MPMD的方式进行多进程程序设计。每个进程运行不同的代码使用不同的数据，通过grpc进行通信。但是数据在进程间是共享的，对进行负载均衡调度。

**TODO**：现在每个进程的数据路径需要人为分配，将来会改为哈希分配

由于更新Variable的operator只有cpu实现，所以一般会采用一台机器的cpu作为parameter server，用于在每轮迭代中更新Variable。而woker运行在gpu上，用于计算梯度。

可以采用不同的方式更新Variable，一般有以下几种：

- 异步更新：每计算出一个minibatch的梯度，就将梯度发送到PS的更新队列，并获取新的Variable取值。在PS上Variable按照队列顺序进行更新。
- 同步更新：所有进程计算梯度后发送到PS，PS将所有进程发回的梯度求和后用于梯度更新，然后向所有进程发送新的Variable取值。同步更新会在进程同步上消耗更多的时间。

在大数据量的前提下，以上两种方式在效果上没有明显差别。


### 单机多卡 ###

单机多卡有两种实现方式

- multi-process：在cpu上启动ps进程，为每个gpu启动一个worker进程，每次计算一个minibatch，通过上面异步更新或者同步更新的方式进行训练。
- multi-tower：只启动一个，每个gpu每次获取一个minibatch，计算出梯度之后，将所有的梯度在cpu中求和之后用于梯度更新。

### 多机多卡 ###

上面单机多卡的第一种方式可以简单扩展到分布式计算。现在框架的实现使用的就是这种方式，和异步更新。但是多卡多进程方式推广到分布式之后，网络传输数据量较大。

**TODO**：为了降低网络传输和同步时间，可以将这两种方式结合使用。在每台机器上运行一个进程，使用multi-tower方式计算并合并梯度，然后后发送到PS的更新队列，多台机器之间采用异步更新的方式。这种方式将在下一版实现。