## 常见的 OPC UA 节点层级
```TEXT
Root (ns=0;i=84)
 ├─ Objects (ns=0;i=85)                    ← 标准文件夹（命名空间0）
 │   ├─ Server (ns=0;i=2252)               ← 标准系统对象
 │   ├─ SimpleDevice (ns=2;i=1)            ← 您的设备（命名空间2）
 │   │   ├─ Temperature (ns=2;i=2)         ← 变量
 │   │   ├─ Pressure (ns=2;i=3)            ← 变量
 │   │   └─ df (ns=2;i=4)                  ← 变量
 │   └─  OtherDevice (ns=3;i=5)             ← 其他设备（命名空间3）
 │        └─ Temp  (ns=2;i=6) 
 ├─ Types (ns=0;i=86)
 └─ Views (ns=0;i=87)
 ```

 ``` python

# 1. 首先注册所有需要的命名空间
self.idx = self.server.register_namespace("http://simple-opcua-server")  # 假设返回2
self.idx_other = self.server.register_namespace("http://other-device")   # 假设返回3

# 2. 然后获取对象节点
objects = self.server.get_objects_node()

# 3. 最后使用注册的命名空间创建对象
# 主命名空间的设备
self.device = objects.add_object(self.idx, "SimpleDevice") # 系统分配节点ID，比如: ns=2;i=1

# 创建第一个变量：
self.temperature = self.device.add_variable(self.idx, "Temperature", 25.0) # 系统分配节点ID，比如: ns=2;i=2

# 创建第二个变量：
self.pressure = self.device.add_variable(self.idx, "Pressure", 100.0) # 系统分配节点ID，比如: ns=2;i=3


# 4. 其他命名空间的设备
self.other_device = objects.add_object(self.idx_other, "OtherDevice")  # ns=3;i=1
self.other_temp = self.other_device.add_variable(self.idx_other, "Temp", 30.0)  # ns=3;i=2
# 以此类推...
 ```

- ns（NamespaceIndex）：在代码里我们通过 self.idx = self.server.register_namespace(...) 获得自己的命名空间索引（比如 2），然后在 add_object、add_variable 时把它作为第一个参数传入，表示这些节点属于 自定义命名空间。
- i（Identifier）：每个节点在同一命名空间内的唯一标识，通常由库自动递增分配。
Note

Notes:   
- Object 本质上是一个 Folder，因此它可以像文件系统的目录一样，随意在下面 添加任意数量的子 Object、Variable、Method 等。只要把每个 Device 当作子 Object（或子 Folder）放进去即可。


## `opcua库`  
###  `self.temperature = self.device.add_variable(self.idx, "Temperature", 25.0)`    

| Syntax          |                                                                          Description                                                                           |
| :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| `self.device`   |                                 已经是一个 **Object Node**（代表一台设备），比如 `device = plant.add_object(idx, "Device1")`。                                 |
| `add_variable`  |                                          OPC UA Server 提供的 API，用来在 **当前对象** 下创建一个 **Variable Node**。                                          |
| `self.idx`      |                                                   指定变量所属的 **命名空间索引**，确保 NodeId 在全局唯一。                                                    |
| `"Temperature"` |                                        变量的 **BrowseName**（浏览时显示的名称），在客户端里看到的就是 “Temperature”。                                         |
| `25.0`          |                                                       变量的 **初始值**，这里是 `float` 类型, 自动推导。                                                       |
| 返回值          | `add_variable` 返回的是 **Variable Node 对象**（`ua.Node` 实例），我们把它保存到 `self.temperature` 成员变量里，后面可以直接使用（比如更新值、设置访问权限）。 |

- Address Space:
```text
  Objects
 └─ Plant
      └─ Device1
           ├─ Temperature (Variable, Float, Value=25.0)
           ├─ Pressure    (Variable, Float, Value=100.0)
           └─ Status      (Variable, Boolean, Value=True)
```

### Browse_path & node_path
- Browse_path
```python
["0:Objects",        # 命名空间0中的Objects文件夹
 "2:SimpleDevice",   # 命名空间2中的SimpleDevice对象  
 "2:Temperature"]    # 命名空间2中的Temperature变量
 ```
- node_path
```python
node_path = "ns=2;i=1"           # 命名空间2，数字标识符1
 ```