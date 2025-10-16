# simple_opcua_client_fixed.py
from opcua import Client
import time
import logging

# 设置基础日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OPCUA_Client")


class SimpleOPCUAClient:
    def __init__(self, endpoint="opc.tcp://localhost:4840"):
        self.client = Client(endpoint)
        self.endpoint = endpoint
        self.connected = False

    def connect(self):
        """连接到服务器"""
        try:
            self.client.connect()
            self.connected = True
            logger.info(f"已连接到服务器: {self.endpoint}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def get_node_by_path(self, node_path):
        """通过路径获取节点 - 修复版本"""
        try:
            # 直接使用完整节点ID路径
            nodes = self.client.get_nodes(node_path)
            if nodes:
                return nodes[0]
            return None
        except Exception as e:
            logger.error(f"获取节点失败 {node_path}: {e}")
            return None

    def get_node_by_browse(self, browse_path):
        """通过浏览路径获取节点"""
        try:
            # 使用浏览路径
            root = self.client.get_root_node()
            node = root.get_child(browse_path)
            return node
        except Exception as e:
            logger.error(f"浏览节点失败 {browse_path}: {e}")
            return None

    def read_value(self, node_path=None, browse_path=None):
        """读取变量值"""
        try:
            if browse_path:
                node = self.get_node_by_browse(browse_path)
            else:
                node = self.get_node_by_path(node_path)

            if node:
                value = node.get_value()
                display_path = browse_path if browse_path else node_path
                logger.info(f"读取 {display_path}: {value}")
                return value
            return None
        except Exception as e:
            logger.error(f"读取失败: {e}")
            return None

    def write_value(self, value, node_path=None, browse_path=None):
        """写入变量值"""
        try:
            if browse_path:
                node = self.get_node_by_browse(browse_path)
            else:
                node = self.get_node_by_path(node_path)

            if node:
                node.set_value(value)
                display_path = browse_path if browse_path else node_path
                logger.info(f"写入 {display_path}: {value}")
                return True
            return False
        except Exception as e:
            logger.error(f"写入失败: {e}")
            return False

    def list_all_variables(self):
        """列出所有变量 - 修复版本"""
        try:
            # 直接使用已知的节点ID
            variables = []

            # 尝试读取我们知道应该存在的变量
            known_vars = [
                ("ns=2;i=2", "Temperature"),
                ("ns=2;i=3", "Pressure"),
                ("ns=2;i=4", "Status")
            ]

            for node_id, name in known_vars:
                try:
                    node = self.client.get_node(node_id)
                    value = node.get_value()
                    variables.append((name, value))
                except:
                    continue

            # 如果上面的方法不行，尝试浏览
            if not variables:
                root = self.client.get_root_node()
                objects = root.get_child(["0:Objects"])

                # 尝试找到我们的设备
                try:
                    device = objects.get_child(["2:SimpleDevice"])
                    children = device.get_children()

                    for child in children:
                        try:
                            name = child.get_display_name().Text
                            value = child.get_value()
                            variables.append((name, value))
                        except:
                            continue
                except Exception as e:
                    logger.warning(f"浏览设备失败: {e}")

            return variables

        except Exception as e:
            logger.error(f"列出变量失败: {e}")
            return []

    def get_server_info(self):
        """获取服务器信息"""
        try:
            # 获取命名空间数组
            namespaces = self.client.get_namespace_array()
            logger.info(f"服务器命名空间: {namespaces}")

            # 获取根节点
            root = self.client.get_root_node()
            logger.info(f"根节点: {root}")

            return True
        except Exception as e:
            logger.error(f"获取服务器信息失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            logger.info("已断开连接")


def demo_basic_operations():
    """演示基础操作 - 修复版本"""
    client = SimpleOPCUAClient()

    if not client.connect():
        return

    try:
        # 首先获取服务器信息
        client.get_server_info()

        # 演示使用浏览路径读取操作
        print("\n=== 读取操作演示 ===")
        client.read_value(
            browse_path=["0:Objects", "2:SimpleDevice", "2:Temperature"])
        client.read_value(
            browse_path=["0:Objects", "2:SimpleDevice", "2:Pressure"])
        client.read_value(
            browse_path=["0:Objects", "2:SimpleDevice", "2:Status"])

        # 演示写入操作
        print("\n=== 写入操作演示 ===")
        client.write_value(30.5, browse_path=[
                           "0:Objects", "2:SimpleDevice", "2:Temperature"])
        client.write_value(105.3, browse_path=[
                           "0:Objects", "2:SimpleDevice", "2:Pressure"])
        client.write_value(False, browse_path=[
                           "0:Objects", "2:SimpleDevice", "2:Status"])

        # 验证写入结果
        print("\n=== 验证写入结果 ===")
        client.read_value(
            browse_path=["0:Objects", "2:SimpleDevice", "2:Temperature"])
        client.read_value(
            browse_path=["0:Objects", "2:SimpleDevice", "2:Pressure"])
        client.read_value(
            browse_path=["0:Objects", "2:SimpleDevice", "2:Status"])

        # 列出所有变量
        print("\n=== 所有可用变量 ===")
        variables = client.list_all_variables()
        for name, value in variables:
            print(f"  {name}: {value}")

    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
    finally:
        client.disconnect()


def interactive_mode():
    """交互式模式 - 修复版本"""
    client = SimpleOPCUAClient()

    if not client.connect():
        return

    try:
        print("\n=== OPC UA 客户端交互模式 ===")
        print("命令:")
        print("  read [变量名]   - 读取变量值")
        print("  write [变量名] [值] - 写入变量值")
        print("  list           - 列出所有变量")
        print("  info           - 显示服务器信息")
        print("  quit           - 退出")

        while True:
            command = input("\n请输入命令: ").strip().split()

            if not command:
                continue

            if command[0] == "read" and len(command) == 2:
                # 读取变量
                browse_path = ["0:Objects",
                               "2:SimpleDevice", f"2:{command[1]}"]
                client.read_value(browse_path=browse_path)

            elif command[0] == "write" and len(command) == 3:
                # 写入变量
                browse_path = ["0:Objects",
                               "2:SimpleDevice", f"2:{command[1]}"]
                # 尝试转换为合适的类型
                try:
                    value = float(command[2])
                except ValueError:
                    try:
                        value = int(command[2])
                    except ValueError:
                        if command[2].lower() in ['true', 'false']:
                            value = command[2].lower() == 'true'
                        else:
                            value = command[2]

                client.write_value(value, browse_path=browse_path)

            elif command[0] == "list":
                # 列出所有变量
                variables = client.list_all_variables()
                print("\n所有变量:")
                for name, value in variables:
                    print(f"  {name}: {value}")

            elif command[0] == "info":
                # 显示服务器信息
                client.get_server_info()

            elif command[0] == "quit":
                break

            else:
                print("无效命令，请重新输入")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        logger.error(f"交互模式出错: {e}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    print("选择模式:")
    print("1 - 基础操作演示")
    print("2 - 交互模式")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "1":
        demo_basic_operations()
    elif choice == "2":
        interactive_mode()
    else:
        print("无效选择")
