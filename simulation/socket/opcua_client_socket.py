# simple_opcua_client_network.py
from opcua import Client
import logging

# 设置基础日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OPCUA_Client")


class SimpleOPCUAClient:
    def __init__(self, endpoint=None):
        if endpoint is None:
            # 默认连接本地服务器，但在网络模式下应该指定远程服务器地址
            endpoint = "opc.tcp://localhost:4840"

        self.client = Client(endpoint)
        self.endpoint = endpoint
        self.connected = False

    def connect(self):
        """连接到服务器"""
        try:
            # 设置连接超时
            self.client.set_timeout(10)
            self.client.connect()
            self.connected = True
            logger.info(f"已连接到服务器: {self.endpoint}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def read_value(self, browse_path):
        """读取变量值"""
        try:
            node = self.get_node_by_browse(browse_path)
            if node:
                value = node.get_value()
                logger.info(f"读取 {browse_path[-1]}: {value}")
                return value
            return None
        except Exception as e:
            logger.error(f"读取失败: {e}")
            return None

    def write_value(self, value, browse_path):
        """写入变量值"""
        try:
            node = self.get_node_by_browse(browse_path)
            if node:
                node.set_value(value)
                logger.info(f"写入 {browse_path[-1]}: {value}")
                return True
            return False
        except Exception as e:
            logger.error(f"写入失败: {e}")
            return False

    def get_node_by_browse(self, browse_path):
        """通过浏览路径获取节点"""
        try:
            root = self.client.get_root_node()
            node = root.get_child(browse_path)
            return node
        except Exception as e:
            logger.error(f"浏览节点失败 {browse_path}: {e}")
            return None

    def list_all_variables(self):
        """列出所有变量"""
        try:
            variables = []
            root = self.client.get_root_node()
            objects = root.get_child(["0:Objects"])
            device = objects.get_child(["2:SimpleDevice"])

            children = device.get_children()
            for child in children:
                try:
                    name = child.get_display_name().Text
                    value = child.get_value()
                    variables.append((name, value))
                except:
                    continue

            return variables
        except Exception as e:
            logger.error(f"列出变量失败: {e}")
            return []

    def disconnect(self):
        """断开连接"""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            logger.info("已断开连接")


def interactive_mode(server_address):
    """交互式模式"""
    endpoint = f"opc.tcp://{server_address}:4840"
    client = SimpleOPCUAClient(endpoint)

    if not client.connect():
        return

    try:
        print(f"\n=== OPC UA 客户端 (连接到 {server_address}) ===")
        print("命令:")
        print("  read [变量名]   - 读取变量值")
        print("  write [变量名] [值] - 写入变量值")
        print("  list           - 列出所有变量")
        print("  quit           - 退出")

        while True:
            command = input("\n请输入命令: ").strip().split()

            if not command:
                continue

            if command[0] == "read" and len(command) == 2:
                browse_path = ["0:Objects",
                               "2:SimpleDevice", f"2:{command[1]}"]
                client.read_value(browse_path)

            elif command[0] == "write" and len(command) == 3:
                browse_path = ["0:Objects",
                               "2:SimpleDevice", f"2:{command[1]}"]
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

                client.write_value(value, browse_path)

            elif command[0] == "list":
                variables = client.list_all_variables()
                print("\n所有变量:")
                for name, value in variables:
                    print(f"  {name}: {value}")

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


def test_connection(server_address):
    """测试连接"""
    endpoint = f"opc.tcp://{server_address}:4840"
    client = SimpleOPCUAClient(endpoint)

    print(f"正在测试连接到 {endpoint}...")

    if client.connect():
        print("✅ 连接成功!")

        # 测试读取
        browse_path = ["0:Objects", "2:SimpleDevice", "2:Temperature"]
        value = client.read_value(browse_path)
        if value is not None:
            print(f"✅ 数据读取成功: Temperature = {value}")
        else:
            print("❌ 数据读取失败")

        client.disconnect()
        return True
    else:
        print("❌ 连接失败")
        return False


if __name__ == "__main__":
    print("=== OPC UA 网络客户端 ===")

    # 获取服务器地址
    server_address = input("请输入OPC UA服务器IP地址: ").strip()

    # 先测试连接
    if test_connection(server_address):
        # 进入交互模式
        interactive_mode(server_address)
    else:
        print("无法连接到服务器，请检查:")
        print("1. 服务器IP地址是否正确")
        print("2. 服务器是否正在运行")
        print("3. 网络连接是否正常")
        print("4. 防火墙是否允许4840端口")
