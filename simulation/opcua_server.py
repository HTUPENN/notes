# opcua_server.py
'''
Simulation via localhost
'''
from opcua import Server  # opc ua server
import time
import logging

import pandas as pd
import json

# basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OPCUA_Server")


class SimpleOPCUAServer:
    def __init__(self, endpoint="opc.tcp://localhost:4840"):
        '''
        init OPC UA server

        @args: 
            endpoints(str): server endpoints, localhost4840
            for ethernet: use ip
        '''
        self.server = Server()
        self.endpoint = endpoint
        self.setup_server()

    def setup_server(self):
        """设置基础服务器"""
        # 设置服务器端点
        self.server.set_endpoint(self.endpoint)

        # 设置服务器名称
        self.server.set_server_name("Simple OPC UA Server")

        # 注册自定义命名空间，返回命名空间索引
        # 命名空间用于区分不同供应商或应用的节点
        # 规范推荐使用 可解析的 URL（不一定要真的可访问，只要全局唯一即可）
        # 返回一个 整数索引 （NamespaceIndex），在本次服务器实例的生命周期里保持不变
        self.idx = self.server.register_namespace("http://simple-opcua-server")

        # 获取对象节点
        objects = self.server.get_objects_node()

        # 创建设备对象
        self.device = objects.add_object(self.idx, "SimpleDevice")

        # 创建几个基础变量
        self.temperature = self.device.add_variable(
            self.idx, "Temperature", 25.0)
        self.pressure = self.device.add_variable(self.idx, "Pressure", 100.0)
        self.status = self.device.add_variable(self.idx, "Status", True)

        # 创建df 变量, 实际为str对象
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        self.df = self.device.add_variable(
            self.idx, "df", df.to_json())  # df.to_json() : str

        # 设置变量为可写（客户端可以修改）
        self.temperature.set_writable()
        self.pressure.set_writable()
        self.status.set_writable()
        self.df.set_writable()

        logger.info("服务器设置完成")

    def start(self):
        """启动服务器"""
        try:

            # 启动服务器
            self.server.start()

            # logging info
            logger.info(f"OPC UA服务器已启动在 {self.endpoint}")

            # need hardcode 可以变量
            logger.info("可用变量:")
            logger.info(
                f"  - Temperature (当前值: {self.temperature.get_value()})")
            logger.info(f"  - Pressure (当前值: {self.pressure.get_value()})")
            logger.info(f"  - Status (当前值: {self.status.get_value()})")

            logger.info(f"  - Status (当前值: {self.df.get_value()})")
            # 在服务器启动后打印命名空间索引
            print(f"命名空间索引: {self.idx}")
            # 保持服务器运行
            print("按 Ctrl+C 停止服务器")
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("正在停止服务器...")
        except Exception as e:
            logger.error(f"服务器错误: {e}")
        finally:
            self.server.stop()
            logger.info("服务器已停止")


if __name__ == "__main__":
    server = SimpleOPCUAServer()
    server.start()
