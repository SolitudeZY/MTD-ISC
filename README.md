[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/SolitudeZY/MTD-ISC)
# SafeIDS: 基于数据增强和集成学习的网络入侵检测系统项目说明
# 目前本项目已经部署在阿里云服务器上，欢迎大家来玩，同时**也请不要删除模型、数据集和检测记录等数据**，这只是一个本科生作品，没有完善的网络安全防护机制，也不是靶场
地址：www.detection.xin
截图：
<img width="1397" height="973" alt="image" src="https://github.com/user-attachments/assets/72abc09b-0ebb-4b8e-8650-dbe3c4952d4a" />
<img width="2525" height="1334" alt="image" src="https://github.com/user-attachments/assets/c2745719-d06a-426a-a3d8-f437d7a271d8" />
## 2025.9.19 加了个小彩蛋
点击主页中的小飞机会出现项目的网址：
<img width="1499" height="726" alt="image" src="https://github.com/user-attachments/assets/60a08d0f-787e-4c2c-a8d6-55cac484a5a0" />

## 2025.8.27 发现漏洞
观察到当项目部署在linux服务器上时，由于tcpdump程序存在退出异常问题，导致其抓到的pcap文件包无法被系统自动清除，可能导致服务器磁盘爆满的情况，需要手动在控制台输入`sudo pkill tcpdump`来停止抓包进程，然后将抓包的文件（存放于/var/log/traffic_captures/）进行删除，即可成功释放空间
## 项目概述
MTD(Malicious Traffic Detection) 是一个基于深度学习和集成学习的网络入侵检测系统。该系统提供恶意流量检测、数据增强、模型管理、实时流量监控等功能。
反馈邮箱： q2205773452@163.com 如发送邮箱请说明来意
## 技术架构
- 后端框架 : Django 3.2.12
- 数据库 : MySQL 8.0+
- 前端 : Bootstrap 5.3.5 + ECharts + Three.js
- 机器学习 : 支持CNN、RNN、LSTM、TCN、BiLSTM、BiTCN等模型
- 数据增强 : 基于改进扩散模型的数据增强技术
## 快速开始
### 环境准备
1. Python 3.9+
2. MySQL 8.0+
3. Git
### 安装步骤

```
# 1. 克隆项目
git clone https://github.com/SolitudeZY/
MTD-ISC.git
cd MTD-ISC

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```
### 数据库配置
1. 创建数据库 :
```
CREATE DATABASE MTD_DMSE CHARACTER SET 
utf8mb4;
```
2. 修改配置 : 编辑 WEB_APP/APP_core/settings.py
```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'MTD_DMSE',
        'USER': '你的用户名',
        'PASSWORD': '你的密码',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```
3. 导入数据 : 使用Navicat等工具导入 mtd-dmse_mysql_data.sql
4. 执行迁移 :

```
cd WEB_APP
python manage.py makemigrations
python manage.py migrate
```
### 启动项目

```
python manage.py runserver
```
访问: http://127.0.0.1:8000

### 额外说明
由于本项目依赖于一些其他资源文件（如数据增强模块），请自行创建相应的同名文件在`WEB_APP\media` 文件夹中。
1. 在`media\source`中，请确保`ema_0.9999_017000.pt`、`samples.npz`、`Improved_diffusion_module.zip`等文件存在，这是保证数据增强模块正常运行的必须文件；
2. 在`\media\models`文件夹中，请**使用本系统来上传文件并确保文件存在**，Django会自动检查文件是否存在，若没有则会报错；
3. 在`media\datasets`也请**确保使用本系统上传数据集文件**（可以是同名文件，只要存在即可），不然Django会报错
4. `media\avatars`是存放各个用户头像的文件夹
5. 模型介绍用的PDF和滚动播放的图片在`MTD-ISC/WEB_APP/MTD/static/assets/`下，如有需要请修改对应代码和图片
6. 数据库若使用本仓库提供的SQL文件可能导致模型管理和数据集管理模块报错（也可能一登录就报错），因为你的本地文件中没有我上文提到的一些数据集和模型的真实文件
7. 模型的检测效果是随机数，可以在views.py中修改相关代码
## 项目结构
```
MTD-ISC/
├── WEB_APP/                    # Django项目根
目录
│   ├── APP_core/              # 项目配置
│   │   ├── settings.py        # 主配置文件
│   │   └── urls.py           # 主路由
│   ├── MTD/                   # 主应用
│   │   ├── detection_module/  # 检测算法模块
│   │   ├── Improved_diffusion_module/  # 数据增强模块
│   │   ├── linux_traffic/     # Linux流量捕获
│   │   ├── static/           # 静态文件
│   │   ├── templates/        # HTML模板
│   │   ├── models.py         # 数据模型
│   │   ├── views.py          # 视图函数
│   │   └── urls.py           # 应用路由
│   └── media/                # 上传文件存储
├── requirements.txt          # 依赖包列表
├── mtd-dmse_mysql_data.sql  # 数据库初始化脚本
└── README.md                # 项目说明
```
## 核心功能模块

### 1. 用户管理
- 用户注册、登录、个人信息管理
- 基于Django内置认证系统
### 2. 模型管理
- 支持多种深度学习模型上传和管理
- 模型性能评估和版本控制
- 文件位置: MTD/detection_module/
### 3. 数据集管理
- 支持PCAP、CSV、图像等多种格式
- 数据集上传、编辑、删除功能
- 数据统计和可视化
### 4. 恶意流量检测
- 实时流量分析和威胁检测
- 多模型集成检测
- 检测结果可视化和报告生成
### 5. 数据增强
- 基于改进扩散模型的数据增强
- 支持自定义训练参数
- 文件位置: MTD/Improved_diffusion_module/
### 6. 流量捕获
- Windows: 基于WinPcap/Npcap
- Linux: 基于tcpdump
- 实时流量监控和存储
### 7. 可视化分析
- ECharts图表展示
- 3D态势感知界面
- 实时数据更新
## 开发指南
### 添加新功能
1. 创建新视图 : 在 views.py 中添加视图函数
2. 配置路由 : 在 urls.py 中添加URL映射
3. 创建模板 : 在 templates/ 中添加HTML文件
4. 添加静态文件 : 在 static/ 中添加CSS/JS文件
### 数据库操作
1. 修改模型 : 编辑 models.py
2. 生成迁移 : python manage.py makemigrations
3. 执行迁移 : python manage.py migrate
### 调试技巧
1.  开启调试模式 : settings.py 中设置 DEBUG = True
2.  查看日志 : 检查 logs/ 目录下的日志文件
3.  使用Django shell : python manage.py shell

## 常见问题
### 数据库连接失败
- 检查MySQL服务是否启动
- 验证数据库配置信息
- 确认用户权限
### 静态文件加载失败
- 运行 python manage.py collectstatic
- 检查 STATIC_URL 和 STATIC_ROOT 配置
### 模型加载错误
- 检查模型文件格式和路径
- 验证文件权限
- 查看错误日志
### 流量捕获失败
- Windows: 安装WinPcap或Npcap
- Linux: 确认tcpdump权限
- 检查网络接口配置
## 部署说明
### 开发环境
- 使用Django内置服务器
- 设置 DEBUG = True
### 生产环境
- 使用Nginx + uWSGI
- 设置 DEBUG = False
- 配置HTTPS和安全设置
- 参考 docker-compose.yml 和 nginx.conf
## 技术支持
- GitHub Issues: https://github.com/SolitudeZY/MTD-ISC/issues
- 项目文档: 查看各模块的README文件
- 代码注释: 关键函数都有详细注释
## 许可证
本项目采用MIT许可证，详见LICENSE文件。
## 项目截图
<img width="2085" height="3290" alt="image" src="https://github.com/user-attachments/assets/02fcc813-fca9-42b3-9efe-1ab44177ae15" />
<img width="2096" height="3800" alt="image" src="https://github.com/user-attachments/assets/2df9000f-524f-4643-8d52-935da3a0c474" />
<img width="2066" height="2581" alt="image" src="https://github.com/user-attachments/assets/ba40e9b2-5ef3-4919-80e2-84bc97838c2a" />
<img width="2067" height="2572" alt="image" src="https://github.com/user-attachments/assets/c76bb849-8452-4044-b9b3-f8a2ca983afe" />
