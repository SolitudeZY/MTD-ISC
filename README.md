# SafeIDF: 基于数据增强和集成学习的网络入侵检测系统项目说明
## 1. 运行项目
我们在github上上传了项目代码，地址为：https://github.com/SolitudeZY/MTD-ISC
### 安装依赖项--首先确保终端位于项目根目录下（例如 MTD-ISC)
我们使用的系统为Windows系统，请按照如下指引安装：
```bash
pip install -r requirements.txt
```
然后使用以下命令启动项目：
```bash
python .\WEB_APP\manage.py runserver
```
如果报错可能是因为没有调整到正确的目录，可以先使用以下命令：
```bash
 cd .\MTD-ISC
```
然后再：
```bash
python .\WEB_APP\manage.py runserver
```

## 2. 导入数据库
- 我们使用的是MySQL（数据库版本号为：8.0.32，编码模式为utf8mb4）数据库，请先安装MySQL并创建名为MTD_DMSE的数据库，然后修改WEB_APP/APP_core/settings.py中的数据库配置项。
- 需要修改的代码部分如下：
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'MTD_DMSE',  # 数据库名称，把这里改为你的数据库名称(我们建议您手动新建名为MTD_DMSE的数据库)
        'USER': 'zhangyang',    # 数据库用户名，把这里改为你的数据库用户名（比如root）
        'PASSWORD': '123456',  #  数据库密码，把这里改为你的数据库用户名对应的密码
        'HOST': 'localhost',  # 或者你的数据库主机地址
        'PORT': '3306',  # MySQL默认端口
    }
}
```
- 我们使用Navicat 数据库管理工具进行数据库导入,您可以在项目根目录找到mtd_dmse.sql文件，使用Navicat打开该文件并运行，即可完成数据库导入。
- 以下是导入文件的的部分参数
```
 Navicat Premium Data Transfer

 Source Server         : link_by_zy
 Source Server Type    : MySQL
 Source Server Version : 80032 (8.0.32)
 Source Host           : localhost:3306
 Source Schema         : mtd_dmse

 Target Server Type    : MySQL
 Target Server Version : 80032 (8.0.32)
 File Encoding         : 65001
```

## 3. 运行迁移
```bash
cd .\WEB_APP
python manage.py makemigrations
python manage.py migrate
```

## 项目内容介绍

-  [WEB_APP](WEB_APP)/APP_CORE : 存放项目配置文件（settings.py，urls.py，wsgi.py）
 -  [WEB_APP](WEB_APP)/MTD : 恶意流量检测平台应用，template存放的是各种模版文件(html)，static存放的是各种静态文件(css,js,img)，views存放的是各种视图函数，models存放的是各种模型，urls存放的是各种url
 -  [WEB_APP](WEB_APP)/media: 存放各种媒体文件和平台上传的模型、数据集文件 

## 额外说明
- 本系统的数据增强模块位于WEB_APP/MTD/Improved_diffusion_module
- 本项目的模型检测模块位于WEB_APP/MTD/detection_module


## 项目一览
```
Malicious_Detection_Platform/
├── WEB_APP/  # Django项目根目录
│   ├── APP_core/  # 项目核心配置目录
│   │   ├── settings.py  # 项目全局配置（数据库、静态文件路径、安全设置等）
│   │   ├── urls.py      # 项目URL路由总入口
│   │   ├── wsgi.py      # WSGI服务器配置
│   │   └── __init__.py  # 初始化文件
│   │
│   ├── MTD/  # 流量检测平台主应用
│   │   ├── migrations/  # 数据库迁移文件（如0001_initial.py）
│   │   ├── static/  # 静态资源目录
│   │   │   ├── echarts-2.27/  # ECharts可视化库（图表模板、示例）
│   │   │   │   ├── src/        # ECharts源代码
│   │   │   │   ├── doc/        # 文档和示例页面
│   │   │   │   └── example/    # 可视化示例（如动态图表、地图）
│   │   │   │
│   │   │   ├── bootstrap-5.3.5-dist/  # Bootstrap前端框架
│   │   │   ├── feather-icons/  # 简洁图标库（CSS/JS）
│   │   │   └── ...  # 其他静态资源（CSS/JS/图片）
│   │   │
│   │   ├── templates/  # HTML模板文件（如登录页、检测结果展示页）
│   │   │   └── ...     # 各视图对应的HTML文件
│   │   │
│   │   ├── models.py  # 数据库模型定义（如用户表UserInfo）
│   │   ├── views.py   # 视图函数（处理请求与业务逻辑）
│   │   ├── urls.py    # 应用内URL路由配置
│   │   └── apps.py    # 应用配置类
│   │
│   ├── staticfiles/  # 通过collectstatic收集的静态文件（生产环境使用）
│   │
│   ├── manage.py  # Django项目启动入口
│   │
│   └── Readme.md  # 项目说明文档（运行、测试、迁移指令）
│
├── requirements.txt  # 项目依赖包列表（如Django、decouple等）
│
└── ...  # 其他辅助文件（如日志配置、测试用例）
```