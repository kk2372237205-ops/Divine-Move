# 2026年2月2日 V1.1.0
# 2026年2月11日 V1.2.0
# 2026年2月12日 V1.3.0
# 2026年4月21日 V1.4.0 增加调参模式

from machine import Pin, PWM, UART
import time
import utime
import math

# 每圈多少步,ID1旋转轴、ID2大臂、ID3小臂
STEPS_PER_REVOLUTION_ID1 = 64000
STEPS_PER_REVOLUTION_ID2 = 64000
STEPS_PER_REVOLUTION_ID3 = 64000

# RS485方向控制引脚
rs485_dir = Pin(22, Pin.OUT)
rs485_dir.value(0)  # 默认设置为接收模式


p0 = Pin(5, Pin.OUT)    # 吸泵
p1 = Pin(23, Pin.OUT)   # 电磁阀
p1.value(0)

p3 = PWM(Pin(13), freq=50)      # 取子机构
p3.duty_u16(6000)

# 上位机通信，可能需要反转
uart1 = UART(1, baudrate=9600, tx=33, rx=25)
# 控制RS485驱动板6
uart2 = UART(2, baudrate=115200, tx=17, rx=16, bits=8, parity=0, stop=1)

relative_position = [0, 150, 60]

# 驱动板CRC校验
def calculate_crc(data, offset, cnt):
    wCrc = 0x0000FFFF
    CRC = [0, 0]
    for i in range(cnt):
        wCrc ^= (data[i + offset] & 0xFF)
        for j in range(8):
            if (wCrc & 0x00000001) == 1:
                wCrc >>= 1
                wCrc ^= 0x0000A001  # 异或多项式
            else:
                wCrc >>= 1
    CRC[1] = (wCrc & 0x0000FF00) >> 8  # 低字节
    CRC[0] = wCrc & 0x000000FF         # 高字节
    return CRC

# 接收上位机串口数据
def receive_data():
    if uart1.any():
        try:
            data = uart1.readline().decode('utf-8').strip()
            print(f"收到: {data}")
            return data
        except Exception as e:
            print(f"接收解码错误: {e}")
    return None



def rs485_send(data):
    """RS485发送数据"""
    rs485_dir.value(1)  # 设置为发送模式
    utime.sleep_ms(1)  # 短暂延时确保模式切换
    uart2.write(data)
    # 等待发送完成
    while not uart2.txdone():  # 等待硬件发完最后一个 stop bit
        pass
    rs485_dir.value(0)

def rs485_receive(timeout=50):
    """接收一帧完整报文，返回 bytes 或 None"""
    buf = bytearray()
    last = utime.ticks_ms()
    while utime.ticks_diff(utime.ticks_ms(), last) < timeout:   # 3 ms 静默 ≈ 7 T
        if uart2.any():
            buf.extend(uart2.read())      # 一次性把当前 FIFO 全拿走
            last = utime.ticks_ms()       # 只要有新字节就刷新计时
        utime.sleep_us(100)                 # 0.1s 轮询一次，省电
    return bytes(buf) if buf else None

# 测试Modbus驱动板通信
def init_driver_board():
    try:
        num = 0
        for i in range(1, 4):
            # 测试通信 - 循环读取电机的状态
            test_cmd = [i, 0x03, 0x00, 0x00, 0x00, 0x01]
            crc = calculate_crc(test_cmd, 0, len(test_cmd))
            full_cmd = test_cmd + crc
            
            # 发送命令
            rs485_send(bytes(full_cmd))
            # 等待并读取响应
            response = rs485_receive(50)
            
            if response and len(response) >= 6:
                print(f"ID:{i}驱动板通信正常, 响应: {response.hex()}")
                num += 1
            else:
                print(f"ID:{i}驱动板通信失败，无响应: {response.hex()}")
    except Exception as e:
        print(f"驱动板初始化错误: {e}")
        return False
    if num == 3:
        wait_for_completion(1, 5000)
        wait_for_completion(2, 5000)
        wait_for_completion(3, 5000)
        return True

# 读取3个电机当前步数
def read_current_step():
    try:
        for i in range(1, 4):
            # 读取电机的实际步数
            test_cmd = [i, 0x03, 0x00, 0x01, 0x00, 0x02]
            crc = calculate_crc(test_cmd, 0, len(test_cmd))
            full_cmd = test_cmd + crc
            # 发送命令
            rs485_send(bytes(full_cmd))
            # 等待并读取响应
            response = rs485_receive(50)
            
            if response and len(response) >= 7:
                print(f"ID:{i}实际步数为: {response.hex()}")
            else:
                print(f"ID:{i}驱动板通信失败，无响应")
    except Exception as e:
        print(f"读取步数错误: {e}")
        return False

def read_register(address, register):
    try:
        # 读取驱动板寄存器
        test_cmd = [address, 0x03, 00, register, 0x00, 0x01]
        crc = calculate_crc(test_cmd, 0, len(test_cmd))
        full_cmd = test_cmd + crc
        # 发送命令
        rs485_send(bytes(full_cmd))
        # 等待并读取响应
        response = rs485_receive(50)
        
        if response and len(response) >= 7:
            print(f"ID{address}读取寄存器值: {response.hex()}")
        else:
            print(f"ID{address}读取寄存器失败，无响应")
    except Exception as e:
        print(f"读取寄存器错误: {e}")
        return False

# 当前用来设置复位速度为50.0
def set_register(address, register):
    try:
        # 设置驱动板寄存器
        test_cmd = [address, 0x10, 0x00, register, 0x00, 0x01, 0x02, 0x01, 0xF4]
        crc = calculate_crc(test_cmd, 0, len(test_cmd))
        full_cmd = test_cmd + crc
        # 发送命令
        rs485_send(bytes(full_cmd))
        # 等待并读取响应
        response = rs485_receive(50)
        
        if response and len(response) >= 7:
            print(f"ID{address}设置寄存器值成功: {response.hex()}")
        else:
            print(f"ID{address}设置寄存器失败，无响应")
    except Exception as e:
        print(f"设置寄存器错误: {e}")
        return False

def send_advanced_position_command(motor_addr, steps, speed_rpm, accel_time, tolerance):
    """发送定点模式-高级命令"""
    try:
        # 转换为字节
        # 需要手动处理字节序和符号
        def int_to_bytes_signed(value, length):
            if value < 0:
                # 负数：使用补码表示
                value = (1 << (length * 8)) + value
            return list(value.to_bytes(length, 'big'))
        
        steps_bytes = int_to_bytes_signed(steps, 4)
        speed_value = int(speed_rpm * 10)  # 速度 × 10
        speed_bytes = int_to_bytes_signed(speed_value, 4)
        accel_bytes = list(accel_time.to_bytes(2, 'big'))
        tolerance_bytes = list(tolerance.to_bytes(2, 'big'))
        
        # 构造命令 - 定点模式高级版
        data = [
            motor_addr,
            0x10,           # 功能码: 写多个寄存器
            0x00, 0x10,     # 起始地址: 0x0010 (定点模式)
            0x00, 0x06,     # 寄存器数量: 6
            0x0C,           # 字节数: 12
        ] + steps_bytes + speed_bytes + accel_bytes + tolerance_bytes
        # 计算CRC
        crc = calculate_crc(data, 0, len(data))
        command = data + crc
        
        # 发送命令
        rs485_send(bytes(command))
        # 读取响应
        response = rs485_receive(50)
        
        if response and len(response) >= 5:
            print(f"speed:{speed_bytes}")
            return True
        else:
            print(f"电机{motor_addr}定点运动无响应或响应异常,{response.hex()}")
            return False
            
    except Exception as e:
        print(f"发送命令到电机{motor_addr}失败: {e}")
        return False

def send_forward_command(motor_addr, steps, speed_rpm, accel_time, tolerance):
    """发送前进模式-高级命令"""
    try:
        # 转换为字节
        def int_to_bytes_signed(value, length):
            if value < 0:
                # 负数：使用补码表示
                value = (1 << (length * 8)) + value
            return list(value.to_bytes(length, 'big'))
        
        # 步数
        steps_bytes = int_to_bytes_signed(steps, 4)
        # 目标速度，speed_rpm为50比较好
        speed_value = int(speed_rpm * 10)  # 速度 × 10
        speed_bytes = int_to_bytes_signed(speed_value, 4)
        # 加速时间
        accel_bytes = list(accel_time.to_bytes(2, 'big'))
        # 容忍误差，默认100步
        tolerance_bytes = list(tolerance.to_bytes(2, 'big'))
        
        # 构造命令 - 定点模式高级版
        data = [
            motor_addr,
            0x10,           # 功能码: 写多个寄存器
            0x00, 0x40,     # 起始地址: 0x0040 (前进模式)
            0x00, 0x06,     # 寄存器数量: 6
            0x0C,           # 字节数: 12
        ] + steps_bytes + speed_bytes + accel_bytes + tolerance_bytes
        # 计算CRC
        crc = calculate_crc(data, 0, len(data))
        command = data + crc
        
        # 发送命令
        rs485_send(bytes(command))
        # 读取响应
        response = rs485_receive(50)
        
        if response and len(response) >= 5:
            return True
        else:
            print(f"电机{motor_addr}无响应或响应异常,{response.hex()}")
            return False
            
    except Exception as e:
        print(f"发送命令到电机{motor_addr}失败: {e}")
        return False

# 等待电机运动完成
def wait_for_completion(addr, timeout=5000):
    """
    等待 addr 号电机运动完成。
    返回 True 表示运动完成或达到停止条件（含反限位停），False 表示异常或超时。
    针对状态 0x0000 需要连续两次确认，状态 0x0004 直接视为完成。
    """
    start_time = utime.ticks_ms()
    zero_count = 0          # 连续读到 0x0000 的次数

    while utime.ticks_diff(utime.ticks_ms(), start_time) < timeout:
        # 1. 构造读取状态寄存器 0x00 的命令
        status_cmd = [addr, 0x03, 0x00, 0x00, 0x00, 0x01]
        crc = calculate_crc(status_cmd, 0, len(status_cmd))
        full_cmd = status_cmd + crc

        # 打印发送报文
        print(f"[TX] {bytes(full_cmd).hex().upper()}")
        rs485_send(bytes(full_cmd))

        # 2. 接收响应
        response = rs485_receive(50)

        # 打印接收原始报文
        if response:
            print(f"[RX] {response.hex().upper()}")
        else:
            print("[RX] None (timeout)")
            utime.sleep_ms(200)
            continue

        # 3. 从接收数据中提取有效 Modbus 帧（7字节）
        valid_frame = None
        if response and len(response) >= 7:
            # 搜索第一个以 addr 开头、功能码 0x03 的 7 字节片段
            for i in range(len(response) - 6):
                if response[i] == addr and response[i+1] == 0x03:
                    valid_frame = response[i:i+7]
                    break

        if valid_frame is None:
            print(f"[WARN] 未找到有效的响应帧 (addr={addr})")
            utime.sleep_ms(200)
            continue

        # 4. 解析状态值（大端序）
        status = (valid_frame[3] << 8) | valid_frame[4]
        print(f"[STATUS] 0x{status:04X}")

        # 5. 状态处理
        if status == 0x0000:
            # 待机/到达：需要连续两次确认
            zero_count += 1
            if zero_count >= 2:
                print(f"电机{addr}运动完成（连续确认2次状态0x0000）")
                return True
            else:
                print(f"电机{addr}状态0x0000第一次确认，等待第二次...")
                # 继续等待下一次轮询

        elif status == 0x0004:
            # 反限位停：视为完成（常用于归零）
            print(f"电机{addr}反限位停，运动完成")
            return True

        elif status == 0x0001:
            # 正在运行，继续等待（重置连续0计数）
            zero_count = 0
            # 继续循环

        elif status in (0x0002, 0x0003):
            # 异常停止（碰撞停、正限位停）
            stop_reason = {0x0002: "碰撞停", 0x0003: "正限位停"}
            print(f"电机{addr}异常停止: {stop_reason.get(status, '未知')}")
            return False

        else:
            # 未知状态
            print(f"电机{addr}未知状态: 0x{status:04X}")
            zero_count = 0

        # 等待下次轮询
        utime.sleep_ms(200)

    # 超时
    print(f"电机{addr}等待超时 ({timeout}ms)")
    return False

# 设置上电复位
def ResetTo():
    try:
        for addr in range(1, 4):
            # 构造命令 - 定点模式高级版
            data = [
                addr,
                0x10,              # 功能码: 写多个寄存器
                0x00, 0xEC,     # 地址: 复位寄存器
                0x00, 0x02,     # 寄存器个数: 2
                0x04,              # 字节数:4
                0x00, 0x00, 0xFA, 0x00   # 反向64000步找限位
            ] 
            # 计算CRC
            crc = calculate_crc(data, 0, len(data))
            command = data + crc
            rs485_send(bytes(command))
            # 读取响应
            response = rs485_receive(20)
            
            if response and len(response) >= 5:
                print(f"电机{addr}上电复位响应正常: {response.hex()}")
            else:
                print(f"电机{addr}上电复位无响应或响应异常,{response.hex()}")
    except Exception as e:
        print(f"发送上电复位命令到电机{addr}失败: {e}")

def set_zero_point(address):
    # 置零
    test_cmd = [address, 0x10, 0x00, 0x01, 0x00, 0x02, 0x04, 0x00, 0x00, 0x00, 0x00]
    crc = calculate_crc(test_cmd, 0, len(test_cmd))
    full_cmd = test_cmd + crc
    print(' '.join(f'{b:02X}' for b in full_cmd))
    # 发送命令
    rs485_send(bytes(full_cmd))
    # 等待并读取响应
    response = rs485_receive(50)
        
    if response and len(response) >= 7:
        print(f"ID{address}置零成功: {response.hex()}")
    else:
        print(f"ID{address}置零失败，无响应")

# 设置手动复位
def UserReset():
    try:
        success_count = 0
        # 电机1（基座旋转）
        if send_forward_command(1, -64000, 50, 250, 100):
            success_count += 1
        else:
            print("电机1控制失败")
        
        # 电机2（大臂）
        if send_forward_command(2, -64000, 50, 250, 100):
            success_count += 1
        else:
            print("电机2控制失败")
        
        # 电机3（小臂）
        if send_forward_command(3, -64000, 50, 250, 100):
            success_count += 1
        else:
            print("电机3控制失败")
        
        wait_for_completion(2, 5000)
        set_zero_point(2)
        wait_for_completion(3, 5000)
        set_zero_point(3)
        wait_for_completion(1, 5000)
        set_zero_point(1)
        
    except Exception as e:
        print(f"复位失败: {e}")

# 逆运动学求解，将XYZ坐标转换成角度
def ik3(x, y, z):
    # 机械臂参数
    h_c = 50    # 末端执行器高度补偿
    l_c = 65    # 末端执行器长度补偿
    l1 = 200    # 大臂长度
    l2 = 200    # 小臂长度
    h = 140     # 基座高度
    
    z = z + h_c
    a1 = 0
    if y == 0:
        x1 = x - l_c
        y1 = 0
    else:
        x1 = abs(x) - l_c * math.sin((math.atan(abs(x) / abs(y))))
        y1 = abs(y) - l_c * math.cos((math.atan(abs(x) / abs(y))))
    
    if y == 0:
        a1 = 90
    else:
        a1 = math.degrees(math.atan(y / x)) + 90
    
    if z < h:
        length = (((x1 ** 2 + y1 ** 2) ** 0.5) ** 2 + (h - z) ** 2) ** 0.5
        a21 = math.degrees(math.atan(((x1 ** 2 + y1 ** 2) ** 0.5) / (h - z))) 
        a22 = math.degrees(math.acos((l1 ** 2 + length ** 2 - l2 ** 2) / (2 * l1 * length)))
        a2 = a21 + a22
        a3 = math.degrees(math.acos((l1 ** 2 + l2 ** 2 - length ** 2) / (2 * l1 * l2)))
    elif z == h:
        length = ((x1 ** 2 + y1 ** 2) ** 0.5)
        a2 = math.degrees(math.acos((l1 ** 2 + length ** 2 - l2 ** 2) / (2 * l1 * length))) + 90
        a3 = math.degrees(math.acos((l1 ** 2 + l2 ** 2 - length ** 2) / (2 * l1 * l2)))
    else:
        length = (((x1 ** 2 + y1 ** 2) ** 0.5) ** 2 + (z - h) ** 2) ** 0.5
        a21 = math.degrees(math.atan(((z - h) / (x1 ** 2 + y1 ** 2) ** 0.5))) 
        a22 = math.degrees(math.acos((l1 ** 2 + length ** 2 - l2 ** 2) / (2 * l1 * length)))
        a2 = a21 + a22 + 90
        a3 = math.degrees(math.acos((l1 ** 2 + l2 ** 2 - length ** 2) / (2 * l1 * l2)))
    
    r1 = a1 + 1
    r2 = 169 - a2
    r3 = 126 + r2 - a3
    return r1, r2, r3

def MoveTo(x, y, z, speed_rpm=30, accel_time=250, tolerance=100):
    """使用定点模式控制机械臂移动到目标位置"""
    if x < 0:
        print("机械臂不可达")
        return False
    
    # 1. 计算逆运动学角度
    a1, a2, a3 = ik3(x, y, z)
    print(f"目标位置({x},{y},{z}) → 关节角度: a1={a1:.1f}°, a2={a2:.1f}°, a3={a3:.1f}°")

    # 2. 计算每个电机需要的步数
    steps_1 = int(a1 * STEPS_PER_REVOLUTION_ID1 / 360) 
    steps_2 = int(a2 * STEPS_PER_REVOLUTION_ID2 / 360) 
    steps_3 = int(a3 * STEPS_PER_REVOLUTION_ID3 / 360) 
    
    print(f"所需步数: 电机1={steps_1}, 电机2={steps_2}, 电机3={steps_3}")
    
    # 3. 发送运动命令到三个电机
    success_count = 0
    
    # 电机1（基座旋转）
    if send_advanced_position_command(1, steps_1, speed_rpm, accel_time, tolerance):
        success_count += 1
    else:
        print("电机1控制失败")
    time.sleep(1)
    wait_for_completion(1, 5000)

    # 电机3（小臂）
    if send_advanced_position_command(3, steps_3, speed_rpm, accel_time, tolerance):
        success_count += 1
    else:
        print("电机3控制失败")  
    
    # 电机2（大臂）
    if send_advanced_position_command(2, steps_2, speed_rpm, accel_time, tolerance):
        success_count += 1
    else:
        print("电机2控制失败")

    # 判断是否运动完成
    time.sleep(1)
    wait_for_completion(3, 3000)
    wait_for_completion(2, 3000)

def BackTo(x, y, z, speed_rpm=100.0, accel_time=200, tolerance=100):
    """返回安全位置 - 使用Modbus驱动板"""
    if x < 0:
        print("机械臂不可达")
        return False
    
    # 1. 计算逆运动学角度
    a1, a2, a3 = ik3(x, y, z)
    print(f"返回位置({x},{y},{z}) → 关节角度: a1={a1:.1f}°, a2={a2:.1f}°, a3={a3:.1f}°")
    
    # 2. 计算每个电机需要的步数
    steps_1 = int(a1 * STEPS_PER_REVOLUTION_ID1 / 360)
    steps_2 = int(a2 * STEPS_PER_REVOLUTION_ID2 / 360) 
    steps_3 = int(a3 * STEPS_PER_REVOLUTION_ID3 / 360)
    
    # 3. 发送运动命令到三个电机
    success_count = 0
    
    # 先移动大臂（ID2，电机1）
    if send_advanced_position_command(2, steps_2, speed_rpm, accel_time, tolerance):
        success_count += 1
    else:
        print("电机2控制失败")
    
    if send_advanced_position_command(3, steps_3, speed_rpm, accel_time, tolerance):
        success_count += 1
    else:
        print("电机3控制失败")
    wait_for_completion(2, 3000)
    wait_for_completion(3, 3000)
    
    # 最后移动基座（ID1）
    if send_advanced_position_command(1, steps_1, speed_rpm, accel_time, tolerance):
        success_count += 1
    else:
        print("电机1控制失败")
    wait_for_completion(1, 5000)

def chess_retract(x, y, z, speed_rpm, p0, p1, i1, i2, i3, o1, o2, o3):
    """悔棋操作 - 使用Modbus驱动板"""
    print("接收到悔棋指令")
    
    # 移动到悔棋位置（稍微偏移避免碰撞）
    MoveTo(x + 2, y, z - 3, speed_rpm)
    
    # 取子
    p0.value(1)  # 吸泵开启
    utime.sleep(3)
    p0.value(0)  # 吸泵关闭
    utime.sleep(1)
    
    # 先返回到放子点
    BackTo(i1, i2, 100, speed_rpm)
    utime.sleep(1)
    
    # 下降到放子点
    MoveTo(o1, o2, o3 - 10, speed_rpm)
    utime.sleep(1)
    
    # 放子
    p1.value(1)  # 电磁阀开启
    utime.sleep(1)
    p1.value(0)  # 电磁阀关闭
    utime.sleep(0.2)
    
    # 返回待命位置
    BackTo(i1, i2, i3, speed_rpm)

def chess(x, y, z, speed_rpm, p0, p1, p3, i1, i2, i3):
    """正常走棋 - 使用Modbus驱动板"""
    # 取子机构准备
    p3.duty_u16(6000)
    utime.sleep(1)
    p3.duty_u16(3150)    
    utime.sleep(0.2)
    
    # 移动到取子点
    MoveTo(i1, i2, i3, speed_rpm)
    utime.sleep(1)
    
    # 取子
    p0.value(1)  # 吸泵开启
    utime.sleep(1)
    p0.value(0)  # 吸泵关闭
    
    # 抬起到安全高度
    BackTo(i1, i2, 100, speed_rpm)
    
    # 移动到目标位置
    MoveTo(x, y, z, speed_rpm)
    
    # 放子
    p1.value(1)  # 电磁阀开启
    utime.sleep(0.6)
    p1.value(0)  # 电磁阀关闭
    
    # 返回待命位置
    BackTo(i1, i2, 100, speed_rpm)
    # 2026年2月11日，增加执行完成回复
    uart1.write("success")

# ==================== 调参模式函数 ====================

def debug_mode_move():
    """
    调参模式：只执行MoveTo运动，不执行取子/放子等其他逻辑
    接收格式: x/y/z 或 P3/duty_value 或 PUMP/state
    发送格式: "DEBUG_OK" 或 "DEBUG_ERR"
    """
    print("========== 进入调参模式 ==========")
    uart1.write("DEBUG_MODE_ENTER\n")
    
    while True:
        recv_data = receive_data()
        if recv_data is None:
            continue
            
        print(f"调参模式收到: {recv_data}")
        
        # 退出调参模式指令
        if recv_data == "EXIT" or recv_data == "exit":
            print("退出调参模式")
            # 退出前确保气泵和电磁阀关闭
            p0.value(0)
            p1.value(0)
            uart1.write("DEBUG_MODE_EXIT\n")
            return True
            
        # 复位指令
        if recv_data == "RESET" or recv_data == "reset":
            print("调参模式：执行复位")
            UserReset()
            uart1.write("DEBUG_RESET_OK\n")
            continue
        
        # 取子机构控制
        if recv_data.startswith("P3/") or recv_data.startswith("p3/"):
            try:
                duty = int(recv_data.split('/')[1])
                p3.duty_u16(duty)
                print(f"调参模式：p3.duty_u16({duty})")
                uart1.write(f"DEBUG_P3_OK:{duty}\n")
            except Exception as e:
                print(f"p3控制错误: {e}")
                uart1.write(f"DEBUG_ERR_P3:{str(e)}\n")
            continue
        
        # 气泵控制
        if recv_data.startswith("PUMP/") or recv_data.startswith("pump/"):
            try:
                state = int(recv_data.split('/')[1])
                if state == 1:
                    # 吸子：开启气泵
                    p0.value(1)  # 吸泵开启
                    utime.sleep(0.5)
                    p0.value(0)  # 吸泵关闭
                    print("调参模式：气泵吸子")
                    uart1.write("DEBUG_PUMP_ON:吸子\n")
                elif state == 0:
                    # 放子：关闭气泵，开启电磁阀
                    p1.value(1)  # 电磁阀开启
                    utime.sleep(0.6)
                    p1.value(0)  # 电磁阀关闭
                    print("调参模式：气泵放子")
                    uart1.write("DEBUG_PUMP_OFF:放子\n")
                elif state == 2:
                    # 停止：关闭气泵和电磁阀
                    p0.value(0)
                    p1.value(0)
                    print("调参模式：气泵停止 p0=0, p1=0")
                    uart1.write("DEBUG_PUMP_STOP:停止\n")
                else:
                    uart1.write("DEBUG_ERR_PUMP:无效状态\n")
            except Exception as e:
                print(f"气泵控制错误: {e}")
                uart1.write(f"DEBUG_ERR_PUMP:{str(e)}\n")
            continue
        
        # 解析坐标
        try:
            parts = recv_data.split('/')
            if len(parts) >= 3:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                
                print(f"调参MoveTo: x={x}, y={y}, z={z}")
                MoveTo(x, y, z, 40)  # 固定速度40
                uart1.write("DEBUG_OK\n")
                print("调参MoveTo完成")
            else:
                print(f"调参模式：格式错误: {recv_data}")
                uart1.write("DEBUG_ERR_FORMAT\n")
                
        except Exception as e:
            print(f"调参模式处理错误: {e}")
            uart1.write(f"DEBUG_ERR:{str(e)}\n")

# ==================== 主函数 ====================

def main():
    # 初始化RS485驱动板
    """
    if not init_driver_board():
        print("驱动板初始化失败")
        while True:
            time.sleep(1)
    """
    UserReset()

    # 参数设置
    normal_speed = 40    # 正常移动速度
    return_speed = 100    # 返回时使用更快速度
    
    # 位置定义
    i1, i2, i3 = 206, 183, 52 # 取子点xyz轴
    i4 = 100
    # 原208 188 100
    o1, o2, o3 = 400, 168, 67   # 悔棋专用放子点xyz轴
    # 移动到初始位置
    print("user control")
    MoveTo(i1, i2, i4, normal_speed)
    
    while True:
        recv_data = receive_data()
        if recv_data is None:
            continue
        print(recv_data)
        
        # ====== 调参模式入口 ======
        # 收到 "DEBUG" 或 "debug" 进入调参模式
        if recv_data == "DEBUG" or recv_data == "debug":
            debug_mode_move()
            # 调参模式退出后，回到正常主循环
            print("回到正常模式")
            MoveTo(i1, i2, i4, normal_speed)  # 回到初始位置
            continue
        # ==========================
        
        x, y, z = map(float, recv_data.split('/')[:3])
        if x == 0 and y == 0 and z == 0:
            UserReset()
            continue
        retract = recv_data.split('/')[3]
        
        if retract == "True":
            # 悔棋
            chess_retract(x, y, z, normal_speed, p0, p1, i1, i2, i3, o1, o2, o3)
        else:
            chess(x, y, z, normal_speed, p0, p1, p3, i1, i2, i3)

def main_test():
    # 测试板卡上每个功能
    print("1 测试与驱动板通信\n"
          "2 测试 LED 灯\n"
          "3 测试吸泵\n"
          "4 测试电磁阀\n"
          "5 测试驱动机械臂\n"
          "6 测试和上位机通信\n"
          "7 测试取子转盘的功能\n"
          "8 测试运动学逆解驱动(例：300/-20/62)\n"
          "9 配置复位参数\n"
          "10 返回\n"
          "11 前进模式运动\n"
          "12 点位测试\n"
          "13 调参模式\n"
        )
    
    UserReset()
    i1, i2, i3 = 210, 188, 100  # 取子点xyz轴
    
    while True:
        dev = int(input("请输入一个整数进行单个功能测试："))
        if dev == 1:
            init_driver_board()
        elif dev == 2:
            led_light(2)
            time.sleep(2)
            led_light(6)
            time.sleep(2)
            led_light(10)
            time.sleep(2)
            led_light(2)
        elif dev == 3:
            value = int(input("高低电平："))
            p0.value(value)
        elif dev == 4:
            value = int(input("高低电平："))
            p1.value(value)
        elif dev == 5:
            while True:
                a, b = map(int, input("请输入两个整数(1ID，2步数)，用斜杠/分隔：").split('/'))
                if a == 0:
                    break
                send_advanced_position_command(a, b, 30, 250, 100)           
        elif dev == 6:
            while True:
                recv_data = receive_data()
                if recv_data is None:
                    continue
                print(recv_data)
                uart1.write("success")
                debugLed.value(0)
        elif dev == 7:
            while True:
                step = int(input("请输入1个整数，1638‑8192："))
                if step < 1600:
                    break
                p3.duty_u16(step)
        elif dev == 8:
            while True:
                a, b, c = map(float, input("请输入三个距离，用斜杠/分隔：").split('/'))
                if a == 1 and b == 1 and c == 1:
                    MoveTo(i1, i2, i3, 50)
                elif a == 0 and b == 0 and c == 0:
                    UserReset()
                    break
                else:
                    MoveTo(i1, i2, i3, 50)
                    chess(a, b, c - 5, 50, p0, p1, p3, i1, i2, i3)
        elif dev == 9:
            ResetTo()
            set_register(1, 0xE7)
            set_register(2, 0xE7)
            set_register(3, 0xE7)
            set_register(1, 0xEE)
            set_register(2, 0xEE)
            set_register(3, 0xEE)
        elif dev == 10:
            return True
        elif dev == 11:
            steps = int(input("请输入一个整数步数，可正负："))
            send_forward_command(1, steps, 50, 250, 100)
        elif dev == 12:
            while True:
                a, b, c = map(float, input("请输入三个距离，用斜杠/分隔：").split('/'))
                if a == 0 and b == 0 and c == 0:
                    UserReset()
                    MoveTo(i1, i2, i3, 50)
                    break
                else: 
                    MoveTo(a, b, c, 50)
        elif dev == 13:
            print("进入调参模式，通过串口发送坐标进行调试")
            print("格式: x/y/z 或 P3/duty 或 PUMP/state")
            print("发送 EXIT 退出调参模式")
            debug_mode_move()
        else:
            print("重新输入值！！！")

# 取子点位调参函数
def point_adjust():
    UserReset()

    # 参数设置
    normal_speed = 40    # 正常移动速度
    return_speed = 100    # 返回时使用更快速度
    
    # 位置定义
    i1, i2, i3 = 210, 188, 100  # 取子点xyz轴
    # 原208 188 100
    o1, o2, o3 = 400, 168, 67   # 悔棋专用放子点xyz轴
    # 移动到初始位置
    print("user control")
    MoveTo(i1, i2, 100, normal_speed)
    
    p3.duty_u16(3150)    
    utime.sleep(0.2)
    while True:
        recv_data = receive_data()
        if recv_data is None:
            continue
        print(recv_data)
        x, y, z = map(float, recv_data.split('/')[:3])
        if x == 0 and y == 0 and z == 0:
            UserReset()
            MoveTo(i1, i2, 100, normal_speed)
            continue
        elif x == 1 and y == 1 and z == 1:
            print("into Mode1")
            while True:
                recv_data = receive_data()
                if recv_data is None:
                    continue
                print(recv_data)
                xxx, yyy, zzz = map(float, recv_data.split('/')[:3])
                if xxx == 9 and yyy == 9 and zzz == 9:
                    break
                else:
                    chess(xxx, yyy, zzz, normal_speed, p0, p1, p3, i1, i2, i3)
        else:
            # 默认调参模式
            MoveTo(x, y, z, normal_speed)
            i1, i2, i3 = x, y, z

if __name__ == '__main__':
    # main_test()
    main() 
    # point_adjust()
    # UserReset()

