def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):#将layer中匹配name的层替换成 new_module
    levels = name.split(".")#先对传入的name进行分割
    if len(levels) > 1:#如果传入的name是多层嵌套的e.g. BloomBlock.mlp.gelu_impl 就一层一层进入
        mod_ = layer
        for l_idx in range(len(levels) - 1):#循环一直到倒数第二层 接上例，直到mlp这一层，下面具体看循环体
            if levels[l_idx].isdigit():#如果当前这一层是一个数字，比如传入的name是 block.3.2
                mod_ = mod_[int(levels[l_idx])] #就直接让mod_等于对应的层
            else:#如果当前这一层分割后不是数字 e.g. ‘mlp’
                mod_ = getattr(mod_, levels[l_idx])#通过getattr来讲取到的值赋给mod_ ;这个循环的意义就是一层一层的去向内取值直到倒数第二层
        setattr(mod_, levels[-1], new_module)#取到倒数第二层之后，将此层对应的levels[-1]变为new_module 例如将gelu_impl变为ScaledActivation
    else:#如果说传入的name是单层的，直接进行替换操作
        setattr(layer, name, new_module)


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x
