# 2026年3月9日 V1.0.0
from machine import Pin, PWM,UART
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

debugLed = Pin(26, Pin.OUT)
debugLed.value(1)

p0 = Pin(23,Pin.OUT)    #吸泵
p1 = Pin(5,Pin.OUT)   #电磁阀
p1.value(0)

p3 = PWM(Pin(13),freq=50)      #取子机构
p3.duty_u16(6000)

# 上位机通信，可能需要反转
uart1 = UART(1, baudrate=9600, tx=33, rx=25)
# 控制RS485驱动板6
uart2 = UART(2, baudrate=115200, tx=17, rx=16, bits=8, parity=0, stop=1)

relative_position = [0,150,60]

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
        for i in range(1,4):
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
    if(num==3):
        wait_for_completion(1,5000)
        wait_for_completion(2,5000)
        wait_for_completion(3,5000)
        return True
    

# 读取3个电机当前步数
def read_current_step():
    try:
        for i in range(1,4):
            # 读取电机的实际步数
            test_cmd = [i, 0x03, 0x00, 0x01, 0x00, 0x02]
            crc = calculate_crc(test_cmd, 0, len(test_cmd))
            full_cmd = test_cmd + crc
            # print(' '.join(f'{b:02X}' for b in full_cmd))
            # 发送命令
            rs485_send(bytes(full_cmd))
            # 等待并读取响应
            response = rs485_receive(50)
            
            if response and len(response) >=7:
                print(f"ID:{i}实际步数为: {response.hex()}")
            else:
                print(f"ID:{i}驱动板通信失败，无响应")
    except Exception as e:
        print(f"读取步数错误: {e}")
        return False

def read_register(address, register):
    try:
        # 读取驱动板寄存器
        test_cmd = [address,0x03,00,register,0x00, 0x01]
        crc = calculate_crc(test_cmd, 0, len(test_cmd))
        full_cmd = test_cmd + crc
        # print(' '.join(f'{b:02X}' for b in full_cmd))
        # 发送命令
        rs485_send(bytes(full_cmd))
        # 等待并读取响应
        response = rs485_receive(50)
        
        if response and len(response) >=7:
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
        test_cmd = [address,0x10,0x00,register,0x00,0x01,0x02,0x01,0xF4]
        crc = calculate_crc(test_cmd, 0, len(test_cmd))
        full_cmd = test_cmd + crc
        # print(' '.join(f'{b:02X}' for b in full_cmd))
        # 发送命令
        rs485_send(bytes(full_cmd))
        # 等待并读取响应
        response = rs485_receive(50)
        
        if response and len(response) >=7:
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
        # print(' '.join(f'{b:02X}' for b in command))
        
        """
        test_cmd = [motor_addr, 0x10, 0x00, 0x10, 0x00, 0x02, 0x04, 0x00, 0x00, 0x19, 0x00, 0xF9, 0x33]
        crc = calculate_crc(test_cmd, 0, len(test_cmd))
        full_cmd = test_cmd + crc
        """
        
        # 发送命令
        # print(f"电机{motor_addr}命令: 步数={steps}, 速度={speed_rpm}RPM")
        rs485_send(bytes(command))
        # 读取响应
        response = rs485_receive(50)
        
        if response and len(response) >= 5:
            # print(f"电机{motor_addr}响应正常: {response.hex()}")
            print(f"speed:{speed_bytes}")
            return True
        else:
            print(f"电机{motor_addr}定点运动无响应或响应异常,{response.hex()}")
            return False
            
    except Exception as e:
        print(f"发送命令到电机{motor_addr}失败: {e}")
        return False

# 等待电机运动完成
def wait_for_completion(addr, timeout=5000):
    """等待 addr 号电机运动完成"""
    start_time = utime.ticks_ms()

    while utime.ticks_diff(utime.ticks_ms(), start_time) < timeout:
        status_cmd = [addr, 0x03, 0x00, 0x00, 0x00, 0x01]
        crc = calculate_crc(status_cmd, 0, len(status_cmd))
        rs485_send(bytes(status_cmd + crc))
        response = rs485_receive(50)

        # 假设协议：bit0=1 表示“正在运行”
        if response and len(response) >= 7 and response[4] != 1:
            print(f"电机{addr}运动完成")
            return True
        utime.sleep(0.2)                  # 200 ms 后再扫一轮

    print("等待超时")
    return False

def set_zero_point(address):
    # 置零
    # test_cmd = [address,0x10,0x00,0x04,0x00,0x01,0x02,0xAA,0xAA]
    # 新驱动板
    test_cmd = [address,0x10,0x00,0x01,0x00,0x02,0x04,0x00,0x00,0x00,0x00]
    crc = calculate_crc(test_cmd, 0, len(test_cmd))
    full_cmd = test_cmd + crc
    print(' '.join(f'{b:02X}' for b in full_cmd))
    # 发送命令
    rs485_send(bytes(full_cmd))
    # 等待并读取响应
    response = rs485_receive(50)
        
    if response and len(response) >=7:
        print(f"ID{address}置零成功: {response.hex()}")
    else:
        print(f"ID{address}置零失败，无响应")

def send_forward_command(motor_addr, steps, speed_rpm, accel_time, tolerance):
    """发送前进模式-高级命令"""
    try:
        # 转换为字节
        # 需要手动处理字节序和符号
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
        # print(' '.join(f'{b:02X}' for b in command))
        
        # 发送命令
        # print(f"电机{motor_addr}命令: 步数={steps}, 速度={speed_rpm}RPM")
        rs485_send(bytes(command))
        # 读取响应
        response = rs485_receive(50)
        
        if response and len(response) >= 5:
            # print(f"电机{motor_addr}响应正常: {response.hex()}")
            return True
        else:
            print(f"电机{motor_addr}无响应或响应异常,{response.hex()}")
            return False
            
    except Exception as e:
        print(f"发送命令到电机{motor_addr}失败: {e}")
        return False


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
        
        wait_for_completion(2,5000)
        set_zero_point(2)
        wait_for_completion(3,5000)
        set_zero_point(3)
        wait_for_completion(1,5000)
        set_zero_point(1)
        
        
    except Exception as e:
        print(f"复位失败: {e}")

# 逆运动学求解，将XYZ坐标转换成角度
def ik3(x,y,z):
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
        x1 = abs(x) - l_c*math.sin((math.atan(abs(x)/abs(y))))
        y1 = abs(y) - l_c*math.cos((math.atan(abs(x)/abs(y))))
    
    if y == 0:
        a1 = 90
    else : 
        a1 = math.degrees(math.atan(y/x)) +90
    if z < h:
        # print("low")
        length = (((x1**2 + y1**2)**0.5)**2 + (h-z) **2) **0.5
        a21 = math.degrees(math.atan(((x1**2 + y1**2)**0.5)/(h-z))) 
        a22 = math.degrees(math.acos((l1 ** 2  + length **2  - l2 **2)/(2*l1 *length)))
        a2 = a21 + a22
        a3 = math.degrees(math.acos((l1**2 + l2 **2 - length **2)/(2*l1*l2)))
    elif z == h:
        length = ((x1**2 + y1**2)**0.5)
        a2 = math.degrees(math.acos((l1 ** 2  + length **2  - l2 **2)/(2*l1 *length))) +90
        a3 = math.degrees(math.acos((l1**2 + l2 **2 - length **2)/(2*l1*l2)))
    else:
        # print("high")
        length = (((x1**2 + y1**2)**0.5)**2 + (z-h) **2) **0.5
        a21 = math.degrees(math.atan(((z-h)/(x1**2 + y1**2)**0.5))) 
        a22 = math.degrees(math.acos((l1 ** 2  + length **2  - l2 **2)/(2*l1 *length)))
        a2 = a21 + a22 +90
        a3 = math.degrees(math.acos((l1**2 + l2 **2 - length **2)/(2*l1*l2)))
    # print(a1,a2,a3)
    r1 = a1 + 1
    r2 = 169 - a2
    r3 = 126 + r2 - a3
    return r1,r2,r3

def MoveTo(x, y, z, speed_rpm=30, accel_time=250, tolerance=100):
    """使用定点模式控制机械臂移动到目标位置"""
    if x < 0:
        print("机械臂不可达")
        return False
    
    # 1. 计算逆运动学角度
    a1, a2, a3 = ik3(x, y, z)
    # a1 = x
    # a2 = y
    # a3 = z
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
    wait_for_completion(1,5000)

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
    wait_for_completion(3,3000)
    wait_for_completion(2,3000)

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
    
    # print(f"返回步数: 电机0={steps_0}, 电机1={steps_1}, 电机2={steps_2}")
    
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
    
    wait_for_completion(2,3000)
    wait_for_completion(3,3000)
    
    # 最后移动基座（ID1）
    if send_advanced_position_command(1, steps_1, speed_rpm, accel_time, tolerance):
        success_count += 1
    else:
        print("电机1控制失败")  
    
    wait_for_completion(1,5000)
    

def chess(x, y, z, suck, speed_rpm, p0, p1, p3):
    """使用Modbus驱动板"""
    
    if suck == 1:
        MoveTo(x, y, z, speed_rpm)
        utime.sleep(0.5)
        p0.value(1)  # 吸泵开启
        utime.sleep(1)
        p0.value(0)  # 吸泵关闭
        utime.sleep(0.2)
    elif suck == 0:
        # 放子
        BackTo(x, y, 100, speed_rpm)
        utime.sleep(0.5)
        MoveTo(x, y, z, speed_rpm)
        p1.value(1)  # 电磁阀开启
        utime.sleep(0.3)
        p1.value(0)  # 电磁阀关闭
    else:
        # 不动作
        BackTo(x, y, 100, speed_rpm)
        utime.sleep(0.5)
        MoveTo(x, y, z, speed_rpm)
    
    # 2026年2月11日，增加执行完成回复
    uart1.write("success")

def main():
    # 初始化RS485驱动板
    UserReset()

    # 参数设置
    normal_speed = 40    # 正常移动速度
    return_speed = 100    # 返回时使用更快速度
    
    # 位置定义
    i1, i2, i3 = 210, 188, 100  # 取子点xyz轴
    MoveTo(i1, i2, i3, normal_speed)
    
    while True:
        recv_data = receive_data()
        if recv_data is None:
            continue
        print(recv_data)
        x, y, z, suck = map(float, recv_data.split('/')[:4])
        if(x==0 and y==0 and z==0):
            UserReset()
            continue
        chess(x, y, z-5, suck, normal_speed, p0, p1, p3)
            
            
if __name__ == '__main__':
    main()