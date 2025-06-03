document.addEventListener("DOMContentLoaded", function() {
    // 初始化Feather图标
    if (typeof feather !== 'undefined') {
        feather.replace();
    }

    // 侧边栏折叠/展开
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebarOverlay = document.getElementById('sidebarOverlay');

    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            document.body.classList.toggle('sidebar-collapsed');

            // 在移动设备上显示/隐藏侧边栏
            if (window.innerWidth < 992) {
                const sidebar = document.querySelector('.sidebar');
                sidebar.classList.toggle('show');
            }
        });
    }

    // 点击遮罩层关闭侧边栏
    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', function() {
            const sidebar = document.querySelector('.sidebar');
            sidebar.classList.remove('show');
        });
    }

    // 下拉菜单交互
    const dropdownToggles = document.querySelectorAll('.sidebar-dropdown-toggle');
    dropdownToggles.forEach(toggle => {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            const parent = this.closest('.sidebar-nav-item');

            // 关闭其他打开的下拉菜单
            if (!parent.classList.contains('open')) {
                document.querySelectorAll('.sidebar-nav-item.open').forEach(item => {
                    if (item !== parent) {
                        item.classList.remove('open');
                    }
                });
            }

            parent.classList.toggle('open');
        });
    });

    // 记住侧边栏状态
    const sidebarState = localStorage.getItem('sidebarState');
    if (sidebarState === 'collapsed') {
        document.body.classList.add('sidebar-collapsed');
    }

    // 保存侧边栏状态
    document.body.addEventListener('click', function(e) {
        if (e.target.closest('#sidebarToggle')) {
            const isCollapsed = document.body.classList.contains('sidebar-collapsed');
            localStorage.setItem('sidebarState', isCollapsed ? 'collapsed' : 'expanded');
        }
    });

    // 响应式处理
    function handleResize() {
        if (window.innerWidth < 992) {
            document.querySelector('.sidebar').classList.remove('show');
            document.querySelector('.sidebar-toggle').style.display = 'flex';
        } else {
            document.querySelector('.sidebar-toggle').style.display = 'flex';
        }
    }

    // 初始化时执行一次
    handleResize();

    // 窗口大小改变时执行
    window.addEventListener('resize', handleResize);

    // 设置活动项目
    const currentPath = window.location.pathname;
    document.querySelectorAll('.sidebar-nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.closest('.sidebar-nav-item').classList.add('active');

            // 如果活动项在下拉菜单中，展开父菜单
            const dropdownItem = link.closest('.sidebar-dropdown-item');
            if (dropdownItem) {
                dropdownItem.closest('.sidebar-nav-item').classList.add('open', 'active');
            }
        }
    });

    // 主题切换功能（如果需要）
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('light-theme');
            const theme = document.body.classList.contains('light-theme') ? 'light' : 'dark';
            localStorage.setItem('theme', theme);
        });

        // 加载保存的主题
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
        }
    }
});