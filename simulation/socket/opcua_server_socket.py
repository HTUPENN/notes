# simple_opcua_server_network.py
from opcua import Server
import time
import logging
import socket

# 设置基础日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OPCUA_Server")


class SimpleOPCUAServer:
    def __init__(self, endpoint=None):
        self.server = Server()

        # 获取本机IP地址
        if endpoint is None:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            endpoint = f"opc.tcp://{local_ip}:4840"

        self.endpoint = endpoint
        self.setup_server()

    def setup_server(self):
        """设置基础服务器"""
        # 设置服务器端点
        self.server.set_endpoint(self.endpoint)

        # 设置服务器名称
        self.server.set_server_name("Network OPC UA Server")

        # 创建命名空间
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

        # 设置变量为可写（客户端可以修改）
        self.temperature.set_writable()
        self.pressure.set_writable()
        self.status.set_writable()

        logger.info("服务器设置完成")

    def start(self):
        """启动服务器"""
        try:
            self.server.start()
            logger.info(f"OPC UA服务器已启动在 {self.endpoint}")
            logger.info("可用变量:")
            logger.info(
                f"  - Temperature (当前值: {self.temperature.get_value()})")
            logger.info(f"  - Pressure (当前值: {self.pressure.get_value()})")
            logger.info(f"  - Status (当前值: {self.status.get_value()})")

            # 显示网络信息
            self.show_network_info()

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

    def show_network_info(self):
        """显示网络连接信息"""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            print("\n" + "="*50)
            print("网络连接信息:")
            print(f"服务器主机名: {hostname}")
            print(f"服务器本地IP: {local_ip}")
            print(f"OPC UA端点: {self.endpoint}")
            print("\n客户端连接方式:")
            print(f"使用端点: {self.endpoint}")
            print("="*50 + "\n")

        except Exception as e:
            logger.error(f"获取网络信息失败: {e}")


def get_network_info():
    """获取网络信息"""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        # 获取所有网络接口的IP地址
        import netifaces
        interfaces = netifaces.interfaces()

        print("可用的网络接口:")
        for interface in interfaces:
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip_addr = addr_info['addr']
                    if ip_addr != '127.0.0.1':
                        print(f"  {interface}: {ip_addr}")

        return local_ip
    except ImportError:
        print("安装 netifaces 库以查看详细网络信息: pip install netifaces")
        return socket.gethostbyname(hostname)


if __name__ == "__main__":
    print("=== OPC UA 网络服务器 ===")

    # 显示网络信息
    server_ip = get_network_info()

    # 询问是否使用特定IP
    choice = input(f"使用自动检测的IP ({server_ip})? (y/n): ").strip().lower()
    if choice == 'n':
        custom_ip = input("请输入服务器IP地址: ").strip()
        endpoint = f"opc.tcp://{custom_ip}:4840"
        server = SimpleOPCUAServer(endpoint)
    else:
        server = SimpleOPCUAServer()

    server.start()
