# ESP32 DEBUG 吸子补丁

目标：让 DEBUG 模式的 `PUMP/1` 立即启动并立即回执，随后持续保持气泵开启，机械臂移动期间继续吸附，直到明确收到 `PUMP/0` 放子。`PUMP/0` 先在气泵仍运行时打开电磁阀主动破坏真空，再停泵并保持泄压约 2 秒，避免棋子继续粘住。完整取放流程保证累计吸附至少 10 秒，但不能在取子点静止等待 10 秒。

设备端 `main.py` 的 `debug_mode_move()` 中需要使用以下 `PUMP/` 分支。项目已提供 VS Code 上传任务，无需使用 Thonny：

```python
        # 末端执行器控制
        if recv_data.startswith("PUMP/") or recv_data.startswith("pump/"):
            try:
                state = int(recv_data.split('/')[1])
                if state == 1:
                    # 吸子：关闭电磁阀，立即开启气泵并回执。
                    # 机械臂移动期间保持吸力，直到 PUMP/0。
                    p1.value(0)
                    p0.value(1)
                    print("调参模式：气泵持续吸子")
                    uart1.write("DEBUG_PUMP_HOLD_ON\n")
                elif state == 0:
                    # 放子：先开阀破坏真空，再停泵，最后保持泄压。
                    p1.value(1)
                    utime.sleep(0.3)
                    p0.value(0)
                    utime.sleep(1.7)
                    p1.value(0)
                    print("调参模式：放子")
                    uart1.write("DEBUG_PUMP_RELEASE_OK\n")
                elif state == 2:
                    # 停止：关闭气泵和电磁阀。
                    p0.value(0)
                    p1.value(0)
                    print("调参模式：末端执行器停止")
                    uart1.write("DEBUG_PUMP_STOP_OK\n")
                elif state == 3:
                    # 诊断：停泵并保持阀门打开，等待人工观察释放效果。
                    p0.value(0)
                    p1.value(1)
                    print("调参模式：阀门保持打开")
                    uart1.write("DEBUG_PUMP_VENT_ON\n")
                elif state == 4:
                    # 诊断结束：关闭阀门，保持气泵关闭。
                    p1.value(0)
                    print("调参模式：阀门关闭")
                    uart1.write("DEBUG_PUMP_VENT_OFF\n")
                else:
                    uart1.write("DEBUG_ERR_PUMP:无效状态\n")
            except Exception as e:
                print(f"末端执行器控制错误: {e}")
                uart1.write(f"DEBUG_ERR_PUMP:{str(e)}\n")
            continue
```

上传步骤：

1. 在电脑端点击“退出调试”，释放 `COM3`。
2. 在 VS Code 中运行任务“上传 ESP32 吸泵补丁并启动”，任务通过 `COM4` 流式改写并启动设备端 `main.py`。
3. 脚本会先移除缓存的 `main` 模块，再导入并启动新文件；等待终端显示 `user control` 和 `ESP32_PUMP_PATCH_AND_START_OK`。
4. 运行任务“验证吸泵 10 秒保持”，确认立即回执、持续保持和最终释放全部通过。

注意：本补丁只改变 DEBUG 模式的 `PUMP/1`、`PUMP/0`、`PUMP/2`，不改变普通 `x/y/z/retract` 落子逻辑。
