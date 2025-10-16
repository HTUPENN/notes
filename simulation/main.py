from opcua_client1 import SimpleOPCUAClient
from opcua_server import SimpleOPCUAServer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OPCUA_Client")


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
