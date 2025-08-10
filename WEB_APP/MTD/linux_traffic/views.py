import os
import subprocess
import threading
import time
import json
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 改进的全局状态管理
capture_lock = threading.Lock()
capture_info = {
    'is_running': False,
    'start_time': None,
    'duration': 0,
    'process': None,
    'should_stop': False,  # 新增：停止标志
    'actual_start_time': None  # 新增：实际开始时间
}

def get_linux_network_interfaces():
    """获取Linux网络接口列表"""
    try:
        result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True, timeout=10)
        interfaces = []
        for line in result.stdout.split('\n'):
            if ': ' in line and 'state' in line.lower():
                interface = line.split(':')[1].strip().split('@')[0]
                if interface != 'lo':  # 排除回环接口
                    interfaces.append(interface)
        return interfaces if interfaces else ['eth0', 'ens33', 'enp0s3']
    except Exception as e:
        logger.error(f"获取网络接口失败: {e}")
        return ['eth0', 'ens33', 'enp0s3']

def get_available_capture_tool():
    """检查可用的抓包工具，优先使用tcpdump"""
    tools = [
        ('tcpdump', ['/usr/sbin/tcpdump', '/usr/bin/tcpdump']),
        ('tshark', ['/usr/bin/tshark', '/usr/local/bin/tshark'])
    ]
    
    for tool_name, tool_paths in tools:
        for tool_path in tool_paths:
            if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
                return tool_name, tool_path
    
    # 如果找不到，尝试通过which命令查找
    for tool_name in ['tcpdump', 'tshark']:
        try:
            result = subprocess.run(['which', tool_name], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return tool_name, result.stdout.strip()
        except:
            continue
    
    return None, None

def check_sudo_permission():
    """检查sudo权限"""
    import pwd
    try:
        current_uid = os.getuid()
        current_user = pwd.getpwuid(current_uid).pw_name
        logger.info(f"当前Django运行用户: {current_user} (UID: {current_uid})")
        
        if current_user in ['www', 'root'] or current_uid == 0:
            return True
            
        result = subprocess.run(['sudo', '-n', 'true'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"权限检查失败: {e}")
        return False

def linux_traffic_capture(request):
    """Linux流量抓包主页面"""
    interfaces = get_linux_network_interfaces()
    tool, tool_path = get_available_capture_tool()
    
    with capture_lock:
        current_status = capture_info.copy()
    
    context = {
        'interfaces': interfaces,
        'capture_tool': tool,
        'capture_status': current_status,
    }
    
    return render(request, 'linux_traffic_capture.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def start_linux_capture(request):
    """启动Linux流量抓包 - 修复版本"""
    global capture_info
    
    try:
        with capture_lock:
            # 检查是否已在抓包
            if capture_info['is_running']:
                return JsonResponse({
                    'success': False,
                    'message': '抓包任务已在运行中，请先停止当前任务'
                })
            
            # 获取参数
            interface = request.POST.get('interface', 'eth0')
            duration = int(request.POST.get('duration', 300))
            filter_rule = request.POST.get('filter', '').strip()
            
            # 立即设置为运行状态
            capture_info = {
                'is_running': True,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': duration,
                'process': None,
                'should_stop': False,
                'actual_start_time': datetime.now()
            }
        
        # 检查抓包工具
        tool, tool_path = get_available_capture_tool()
        if not tool:
            with capture_lock:
                capture_info['is_running'] = False
            return JsonResponse({
                'success': False,
                'message': '未找到可用的抓包工具（tshark或tcpdump）'
            })
        
        # 检查sudo权限
        if not check_sudo_permission():
            with capture_lock:
                capture_info['is_running'] = False
            return JsonResponse({
                'success': False,
                'message': '需要sudo权限进行网络抓包，请配置免密sudo或联系管理员'
            })
        
        # 创建抓包目录
        capture_dir = '/var/log/traffic_captures'
        if not os.path.exists(capture_dir):
            try:
                os.makedirs(capture_dir, mode=0o755)
            except:
                capture_dir = os.path.expanduser('~/traffic_captures')
                os.makedirs(capture_dir, exist_ok=True)
        
        # 检查目录写入权限
        if not os.access(capture_dir, os.W_OK):
            with capture_lock:
                capture_info['is_running'] = False
            return JsonResponse({
                'success': False,
                'message': f'无法写入抓包目录: {capture_dir}'
            })
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'traffic_capture_{timestamp}.pcap'
        filepath = os.path.join(capture_dir, filename)
        
        # 构建抓包命令
        if tool == 'tshark':
            cmd = ['sudo', 'tshark', '-i', interface, '-w', filepath]
            if filter_rule:
                cmd.extend(['-f', filter_rule])
            cmd.extend(['-a', f'duration:{duration}'])
        else:  # tcpdump
            cmd = ['sudo', 'tcpdump', '-i', interface, '-w', filepath]
            if filter_rule:
                cmd.append(filter_rule)
        
        def run_capture_fixed(command, duration):
            """修复的抓包线程函数 - 保持状态稳定"""
            global capture_info
            process = None
            
            try:
                logger.info(f"启动抓包命令: {' '.join(command)}")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # 更新进程信息
                with capture_lock:
                    if capture_info['is_running']:  # 确保状态仍然有效
                        capture_info['process'] = process
                
                # 监控抓包进程 - 关键修复：不主动重置状态
                start_time = time.time()
                while time.time() - start_time < duration:
                    # 检查是否被手动停止
                    with capture_lock:
                        if capture_info['should_stop']:
                            logger.info("检测到停止信号，终止抓包")
                            break
                    
                    # 检查进程是否异常退出
                    if process.poll() is not None:
                        logger.warning("抓包进程异常退出")
                        break
                    
                    time.sleep(1)
                
                # 正常结束或被停止
                logger.info("抓包时间到达或被停止，准备结束")
                
            except Exception as e:
                logger.error(f"抓包异常: {e}")
            finally:
                # 确保进程被终止
                if process and process.poll() is None:
                    try:
                        process.terminate()
                        time.sleep(1)
                        if process.poll() is None:
                            process.kill()
                    except:
                        pass
                
                # 重置状态 - 关键修复：只在真正结束时重置
                with capture_lock:
                    logger.info("抓包线程结束，重置状态")
                    capture_info = {
                        'is_running': False,
                        'start_time': None,
                        'duration': 0,
                        'process': None,
                        'should_stop': False,
                        'actual_start_time': None
                    }
        
        # 启动抓包线程
        capture_thread = threading.Thread(
            target=run_capture_fixed, 
            args=(cmd, duration),
            daemon=True
        )
        capture_thread.start()
        
        return JsonResponse({
            'success': True,
            'message': f'流量抓包已启动！工具: {tool}, 接口: {interface}, 时长: {duration}秒{f", 过滤: {filter_rule}" if filter_rule else ""}'
        })
        
    except Exception as e:
        logger.error(f"启动抓包失败: {e}")
        with capture_lock:
            capture_info = {
                'is_running': False,
                'start_time': None,
                'duration': 0,
                'process': None,
                'should_stop': False,
                'actual_start_time': None
            }
        return JsonResponse({
            'success': False,
            'message': f'启动抓包失败: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def stop_linux_capture(request):
    """停止Linux流量抓包 - 修复版本"""
    global capture_info
    
    try:
        with capture_lock:
            if not capture_info['is_running']:
                return JsonResponse({
                    'success': False,
                    'message': '当前没有正在运行的抓包任务'
                })
            
            # 设置停止标志，让线程自然结束
            capture_info['should_stop'] = True
            process = capture_info.get('process')
        
        # 终止进程
        if process and process.poll() is None:
            try:
                logger.info("手动停止抓包进程...")
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
                    time.sleep(1)
                logger.info("抓包进程已停止")
            except Exception as e:
                logger.error(f"停止进程失败: {e}")
        
        return JsonResponse({
            'success': True,
            'message': '抓包停止信号已发送'
        })
        
    except Exception as e:
        logger.error(f"停止抓包失败: {e}")
        return JsonResponse({
            'success': False,
            'message': f'停止抓包失败: {str(e)}'
        })

def linux_capture_status(request):
    """获取Linux抓包状态 - 修复版本"""
    global capture_info
    
    try:
        with capture_lock:
            current_status = capture_info.copy()
        
        # 计算运行时间 - 关键修复：使用实际开始时间
        running_time = '00:00:00'
        if current_status['is_running'] and current_status.get('actual_start_time'):
            try:
                running_seconds = int((datetime.now() - current_status['actual_start_time']).total_seconds())
                hours = running_seconds // 3600
                minutes = (running_seconds % 3600) // 60
                seconds = running_seconds % 60
                running_time = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
            except:
                pass
        
        # 获取文件列表
        capture_dirs = ['/var/log/traffic_captures', os.path.expanduser('~/traffic_captures')]
        files = []
        
        for capture_dir in capture_dirs:
            if os.path.exists(capture_dir):
                try:
                    for filename in os.listdir(capture_dir):
                        if filename.endswith('.pcap'):
                            filepath = os.path.join(capture_dir, filename)
                            if os.path.isfile(filepath):
                                stat = os.stat(filepath)
                                files.append({
                                    'name': filename,
                                    'size': f'{stat.st_size / 1024:.1f} KB' if stat.st_size < 1024*1024 else f'{stat.st_size / (1024*1024):.1f} MB',
                                    'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                                })
                except PermissionError:
                    continue
        
        # 按创建时间排序（最新的在前）
        files.sort(key=lambda x: x['created'], reverse=True)
        
        # 返回稳定的状态数据
        response_data = {
            'is_running': current_status['is_running'],
            'start_time': current_status['start_time'],
            'end_time': None,
            'running_time': running_time,
            'packet_count': 0,  # 简化：不计算包数量
            'capture_file': None,
            'interface': None,
            'duration': current_status.get('duration', 30),
            'filter': None,
            'capture_tool': None,
            'files': files
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        return JsonResponse({
            'success': False,
            'message': f'获取状态失败: {str(e)}',
            'is_running': False,
            'start_time': None,
            'end_time': None,
            'running_time': '00:00:00',
            'packet_count': 0,
            'files': [],
            'duration': 30
        })

@csrf_exempt
@require_http_methods(["POST"])
def delete_linux_capture(request, filename):
    """删除Linux抓包文件"""
    try:
        # 安全检查文件名
        if not filename.endswith('.pcap') or '/' in filename or '..' in filename:
            return JsonResponse({
                'success': False,
                'message': '无效的文件名'
            })
        
        # 查找并删除文件
        capture_dirs = ['/var/log/traffic_captures', os.path.expanduser('~/traffic_captures')]
        file_deleted = False
        
        for capture_dir in capture_dirs:
            filepath = os.path.join(capture_dir, filename)
            if os.path.exists(filepath) and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                    file_deleted = True
                    logger.info(f"已删除文件: {filepath}")
                    break
                except PermissionError:
                    return JsonResponse({
                        'success': False,
                        'message': '没有权限删除该文件'
                    })
                except Exception as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'删除文件失败: {str(e)}'
                    })
        
        if not file_deleted:
            return JsonResponse({
                'success': False,
                'message': '文件不存在'
            })
        
        return JsonResponse({
            'success': True,
            'message': f'文件 {filename} 已成功删除'
        })
        
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        return JsonResponse({
            'success': False,
            'message': f'删除文件失败: {str(e)}'
        })

def download_linux_capture(request, filename):
    """下载Linux抓包文件"""
    try:
        # 安全检查文件名
        if not filename.endswith('.pcap') or '/' in filename or '..' in filename:
            raise Http404("无效的文件名")
        
        # 查找文件
        capture_dirs = ['/var/log/traffic_captures', os.path.expanduser('~/traffic_captures')]
        filepath = None
        
        for capture_dir in capture_dirs:
            potential_path = os.path.join(capture_dir, filename)
            if os.path.exists(potential_path) and os.path.isfile(potential_path):
                filepath = potential_path
                break
        
        if not filepath:
            raise Http404("文件不存在")
        
        if os.path.getsize(filepath) == 0:
            raise Http404("文件为空")
        
        # 返回文件
        with open(filepath, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/vnd.tcpdump.pcap')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            response['Content-Length'] = os.path.getsize(filepath)
            return response
            
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        raise Http404("下载失败")


@csrf_exempt
@require_http_methods(["POST"])
def delete_all_linux_captures(request):
    """删除所有Linux抓包文件"""
    try:
        capture_dirs = ['/var/log/traffic_captures', os.path.expanduser('~/traffic_captures')]
        deleted_count = 0
        total_files = 0
        
        for capture_dir in capture_dirs:
            if os.path.exists(capture_dir):
                try:
                    for filename in os.listdir(capture_dir):
                        if filename.endswith('.pcap'):
                            total_files += 1
                            filepath = os.path.join(capture_dir, filename)
                            if os.path.isfile(filepath):
                                try:
                                    os.remove(filepath)
                                    deleted_count += 1
                                    logger.info(f"已删除文件: {filepath}")
                                except PermissionError:
                                    logger.warning(f"没有权限删除文件: {filepath}")
                                except Exception as e:
                                    logger.error(f"删除文件失败 {filepath}: {e}")
                except PermissionError:
                    logger.warning(f"没有权限访问目录: {capture_dir}")
                    continue
        
        if total_files == 0:
            return JsonResponse({
                'success': True,
                'message': '没有找到需要删除的文件'
            })
        
        if deleted_count == total_files:
            return JsonResponse({
                'success': True,
                'message': f'成功删除所有 {deleted_count} 个文件'
            })
        else:
            return JsonResponse({
                'success': True,
                'message': f'删除了 {deleted_count}/{total_files} 个文件，部分文件可能因权限问题无法删除'
            })
        
    except Exception as e:
        logger.error(f"删除所有文件失败: {e}")
        return JsonResponse({
            'success': False,
            'message': f'删除所有文件失败: {str(e)}'
        })