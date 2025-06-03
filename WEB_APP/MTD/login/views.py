# WEB_APP/MTD/login/views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import RegisterForm
from django.contrib import messages


def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()  # 保存用户数据
            login(request, user)  # 自动登录
            messages.success(request, f'欢迎 {user.username}！您的账户已创建成功。')
            return redirect('home')  # 重定向到主页
        else:
            # 如果表单无效，显示错误信息
            messages.error(request, '注册失败，请检查下面的错误。')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})


def custom_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # 登录成功后重定向到主页
        else:
            return render(request, 'login.html', {'error': '无效的凭据，用户名或密码错误'})
    else:
        return render(request, 'login.html')


@login_required
def logout_view(request):
    logout(request)
    # 登出就删除所有session
    request.session.flush()
    return redirect('/login')
