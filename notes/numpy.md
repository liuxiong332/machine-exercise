numpy中索引有如下几种，切片索引，bool索引，数组索引

其中，切片索引将会返回其原对象的视图，而bool型索引和数组索引将会返回其copy值

由于索引可能返回是视图，也有可能是副本，因此于是对于下面代码

```python
def do_something(df):
   foo = df[['bar', 'baz']]  # Is foo a view? A copy? Nobody knows!
   # ... many lines here ...
   foo['quux'] = value       # We don't know whether this will modify df or not!
   return foo
```

numpy无法确定是否修改将会修改df，因此将会爆出`SettingWithCopy`的警告。