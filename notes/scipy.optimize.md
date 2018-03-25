*scipy.optimize.minimize*通过设置一些参数和算法名字来计算目标值使得评估函数最小。

```js
scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)[source]¶
```

其中`fun`是评估函数，它的形式是`f(x, *args)`，`x`是要计算的目标值，`args`是通过后面的`args`参数传入的参数值。

`x0`是目标值的初始值

`method`是算法的名字，基本有`BFGS`, `Newton-CG`等等

`jac`则是梯度函数，用于计算评估函数的梯度值。