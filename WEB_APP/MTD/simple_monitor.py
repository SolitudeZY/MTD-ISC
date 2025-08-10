import os
import time
import subprocess
from datetime import datetime

def simple_network_monitor(duration=60, output_file=None):
    """简单的网络监控函数"""
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'/tmp/network_monitor_{timestamp}.txt'
    
    print(f"开始网络监控，时长: {duration}秒")
    print(f"输出文件: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"网络监控开始\n")
            f.write(f"开始时间: {datetime.now()}\n")
            f.write(f"监控时长: {duration}秒\n")
            f.write("=" * 50 + "\n\n")
            
            start_time = time.time()
            count = 0
            
            while time.time() - start_time < duration:
                try:
                    # 获取网络连接信息
                    result = subprocess.run(['ss', '-tuln'], 
                                          capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        f.write(f"[{current_time}] 网络连接快照 #{count + 1}\n")
                        f.write("-" * 40 + "\n")
                        f.write(result.stdout)
                        f.write("\n" + "=" * 50 + "\n\n")
                        f.flush()  # 立即写入文件
                        
                        count += 1
                        print(f"已采集 {count} 次网络信息")
                    
                    time.sleep(5)  # 每5秒采集一次
                    
                except Exception as e:
                    print(f"采集出错: {e}")
                    continue
            
            f.write(f"\n监控结束时间: {datetime.now()}\n")
            f.write(f"总采集次数: {count}\n")
            
        print(f"监控完成，文件保存在: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"监控失败: {e}")
        return None

if __name__ == "__main__":
    # 直接运行监控
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result_file = simple_network_monitor(duration, output_file)
    if result_file:
        print(f"监控成功完成，文件: {result_file}")
    else:
        print("监控失败")