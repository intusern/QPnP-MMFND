// ========== 主应用 ==========
new Vue({
    el: '#app',
    data() {
        return {
            currentUser: null,
            activeMenu: 'dashboard',
            stats: {
                total_detections: 0,
                real_news: 0,
                fake_news: 0,
                avg_confidence: 0,
                fake_rate: 0
            },
            trendData: [],
            modelFlow: { stages: [] },
            quantumComponents: { components: [] },
            systemLogs: [],
            loading: true
        }
    },
    methods: {
        async handleLogin(loginData) {
            localStorage.setItem('access_token', loginData.access_token);
            this.currentUser = loginData.user;
            
            // 加载初始数据
            await this.loadDashboardData();
        },

        async loadDashboardData() {
            const token = localStorage.getItem('access_token');
            
            try {
                // 加载统计数据
                const statsResponse = await api.getDashboardStats(token);
                if (statsResponse.data.code === 200) {
                    this.stats = statsResponse.data.data;
                }

                // 加载趋势数据
                const trendResponse = await api.getDetectionTrend(token);
                if (trendResponse.data.code === 200) {
                    this.trendData = trendResponse.data.data;
                }

                // 加载模型流程
                const flowResponse = await api.getModelFlow(token);
                if (flowResponse.data.code === 200) {
                    this.modelFlow = flowResponse.data.data;
                }

                // 加载量子组件
                const componentsResponse = await api.getQuantumComponents(token);
                if (componentsResponse.data.code === 200) {
                    this.quantumComponents = componentsResponse.data.data;
                }

                // 加载日志
                const logsResponse = await api.getLogs(token);
                if (logsResponse.data.code === 200) {
                    this.systemLogs = logsResponse.data.data;
                }
            } catch (error) {
                this.$message.error('加载数据失败');
                console.error(error);
            }
        },

        handleDetection(data) {
            // 重新加载数据看板
            this.loadDashboardData();
            this.activeMenu = 'dashboard';
        },

        handleLogout() {
            this.$confirm('确认退出登录?', '提示', {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning'
            }).then(() => {
                localStorage.removeItem('access_token');
                this.currentUser = null;
                this.$message.success('已退出登录');
            }).catch(() => {});
        }
    },
    mounted() {
        // 检查是否已登录
        const token = localStorage.getItem('access_token');
        if (token) {
            api.getCurrentUser(token)
                .then(response => {
                    if (response.data.code === 200) {
                        this.currentUser = response.data.data;
                        this.loadDashboardData();
                    } else {
                        localStorage.removeItem('access_token');
                    }
                })
                .catch(error => {
                    console.error('获取用户信息失败:', error);
                });
        }

        // 检查后端连接
        api.healthCheck()
            .then(() => {
                console.log('✅ 后端服务已连接');
            })
            .catch(() => {
                this.$message.warning('⚠️ 后端服务未连接，请确保Flask应用正在运行');
            });

        // 定期刷新数据（每30秒）
        setInterval(() => {
            if (this.currentUser && this.activeMenu === 'dashboard') {
                this.loadDashboardData();
            }
        }, 30000);
    }
});
