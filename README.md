# 集成数据增强的恶意流量检测平台应用的项目说明
## 1. 运行项目

### 安装依赖项
```bash
pip install -r requirements.txt
```
然后使用以下命令启动项目：
```bash
python .\WEB_APP\manage.py runserver
```
如果报错可能是因为没有调整到正确的目录，可以先使用以下命令：
```bash
 cd .\WEB_APP
```
然后再：
```bash
python manage.py runserver
```

## 2. 运行测试
```bash
python manage.py test
```

## 3. 运行迁移
```bash
cd .\WEB_APP
python manage.py makemigrations
python manage.py migrate
```

## 项目内容介绍
 -  APP_CORE : 这里存放的是管理整个项目的一些配置文件，比如：settings.py，urls.py，wsgi.py
 -  MTD : 恶意流量检测平台应用，template存放的是各种模版文件(html)，static存放的是各种静态文件(css,js,img)，views存放的是各种视图函数，models存放的是各种模型，urls存放的是各种url
 -  media: 存放各种媒体文件和平台上传的模型、数据集文件 

## 额外说明
- 本项目的数据增强模块依赖于improved-diffusion文件夹中的readme.md，请先阅读该文件并安装相关依赖
- 数据增强模块目录在WEB_APP/MTD/Improved_diffusion_module/中 （[Improved_diffusion_module](WEB_APP%2FMTD%2FImproved_diffusion_module)）


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
│   ├── MTD/  # 恶意流量检测平台主应用
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