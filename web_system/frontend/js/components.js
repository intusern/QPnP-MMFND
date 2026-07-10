// ========== 登录组件 ==========
Vue.component('login', {
    template: `
        <div class="login-wrapper">
            <div class="login-box">
                <h1>🔬 QMMFND 系统登录</h1>
                <p class="login-subtitle">量子启发多模态虚假新闻检测系统</p>
                
                <div class="login-form">
                    <div class="form-group">
                        <label>账户类型</label>
                        <select v-model="accountType" class="form-control">
                            <option value="login">登录</option>
                            <option value="register">注册</option>
                        </select>
                    </div>

                    <div v-if="accountType === 'register'" class="form-group">
                        <label>选择角色</label>
                        <select v-model="registerForm.role" class="form-control">
                            <option value="operator">操作员</option>
                            <option value="analyst">分析师</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>用户名</label>
                        <input v-model="form.username" type="text" class="form-control" placeholder="请输入用户名" />
                    </div>

                    <div class="form-group">
                        <label>密码</label>
                        <input v-model="form.password" type="password" class="form-control" placeholder="请输入密码" />
                    </div>

                    <div v-if="accountType === 'register'" class="form-group">
                        <label>邮箱</label>
                        <input v-model="registerForm.email" type="email" class="form-control" placeholder="请输入邮箱" />
                    </div>

                    <div v-if="errorMessage" class="error-message">{{ errorMessage }}</div>

                    <button 
                        @click="handleSubmit" 
                        :disabled="loading"
                        class="submit-btn"
                    >
                        {{ loading ? '处理中...' : (accountType === 'login' ? '登录' : '注册') }}
                    </button>
                </div>

                <div class="demo-users">
                    <h4>📝 演示账户</h4>
                    <p><strong>管理员：</strong> admin / 123456</p>
                    <p><strong>操作员：</strong> operator / 123456</p>
                    <p><strong>分析师：</strong> analyst / 123456</p>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            accountType: 'login',
            form: {
                username: '',
                password: ''
            },
            registerForm: {
                email: '',
                role: 'operator'
            },
            loading: false,
            errorMessage: ''
        }
    },
    methods: {
        handleSubmit() {
            this.errorMessage = '';

            if (!this.form.username || !this.form.password) {
                this.errorMessage = '用户名和密码不能为空';
                return;
            }

            this.loading = true;

            if (this.accountType === 'login') {
                api.login(this.form.username, this.form.password)
                    .then(response => {
                        if (response.data.code === 200) {
                            this.$emit('login', response.data.data);
                        } else {
                            this.errorMessage = response.data.message;
                        }
                    })
                    .catch(error => {
                        this.errorMessage = '登录失败，请检查服务器连接';
                        console.error(error);
                    })
                    .finally(() => {
                        this.loading = false;
                    });
            } else {
                if (!this.registerForm.email) {
                    this.errorMessage = '邮箱不能为空';
                    this.loading = false;
                    return;
                }

                api.register(this.form.username, this.form.password, this.registerForm.email, this.registerForm.role)
                    .then(response => {
                        if (response.data.code === 200) {
                            this.errorMessage = '注册成功，请登录';
                            this.accountType = 'login';
                            this.registerForm.email = '';
                        } else {
                            this.errorMessage = response.data.message;
                        }
                    })
                    .catch(error => {
                        // 增强的错误处理
                        if (error.response) {
                            // 服务器返回的错误
                            const errorData = error.response.data;
                            this.errorMessage = errorData.message || `服务器错误: ${error.response.status}`;
                        } else if (error.request) {
                            // 请求已发出但没有响应
                            this.errorMessage = '无法连接到服务器，请确保后端已启动（python start.py）';
                        } else {
                            // 其他错误
                            this.errorMessage = `注册失败: ${error.message}`;
                        }
                        console.error('[Register Error]', error);
                    })
                    .finally(() => {
                        this.loading = false;
                    });
            }
        }
    }
});


// ========== 数据看板组件 ==========
Vue.component('dashboard', {
    template: `
        <div class="dashboard">
            <h2>📊 系统数据看板</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">总检测数</div>
                    <div class="stat-value">{{ stats.total_detections }}</div>
                    <div class="stat-icon">📈</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">真实新闻</div>
                    <div class="stat-value">{{ stats.real_news }}</div>
                    <div class="stat-icon">✅</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">虚假新闻</div>
                    <div class="stat-value">{{ stats.fake_news }}</div>
                    <div class="stat-icon">⚠️</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">虚假率</div>
                    <div class="stat-value">{{ stats.fake_rate }}%</div>
                    <div class="stat-icon">🎯</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">平均置信度</div>
                    <div class="stat-value">{{ (stats.avg_confidence * 100).toFixed(2) }}%</div>
                    <div class="stat-icon">🔐</div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-box">
                    <h3>检测趋势分析</h3>
                    <div id="trendChart" style="width: 100%; height: 300px;"></div>
                </div>
            </div>

            <div class="quick-actions">
                <h3>快捷操作</h3>
                <button class="action-btn">📥 导入数据</button>
                <button class="action-btn">📊 生成报告</button>
                <button class="action-btn">⚙️ 系统设置</button>
                <button class="action-btn">💾 数据备份</button>
            </div>
        </div>
    `,
    props: ['stats', 'trend'],
    mounted() {
        this.$nextTick(() => {
            this.renderChart();
        });
    },
    watch: {
        trend() {
            this.renderChart();
        }
    },
    methods: {
        renderChart() {
            const chartDom = document.getElementById('trendChart');
            if (!chartDom) return;

            const myChart = echarts.init(chartDom);
            
            const dates = this.trend.map(d => d.date);
            const realData = this.trend.map(d => d.real_count);
            const fakeData = this.trend.map(d => d.fake_count);

            const option = {
                color: ['#67C23A', '#F56C6C'],
                tooltip: {
                    trigger: 'axis'
                },
                legend: {
                    data: ['真实新闻', '虚假新闻']
                },
                xAxis: {
                    type: 'category',
                    data: dates,
                    axisLabel: {
                        interval: Math.floor(dates.length / 6)
                    }
                },
                yAxis: {
                    type: 'value'
                },
                series: [
                    {
                        name: '真实新闻',
                        data: realData,
                        type: 'line',
                        smooth: true
                    },
                    {
                        name: '虚假新闻',
                        data: fakeData,
                        type: 'line',
                        smooth: true
                    }
                ]
            };

            myChart.setOption(option);
            window.addEventListener('resize', () => myChart.resize());
        }
    }
});


// ========== 新闻检测组件 ==========
Vue.component('detection', {
    template: `
        <div class="detection">
            <h2>🔍 新闻真假检测</h2>
            
            <div class="detection-form">
                <div class="form-group">
                    <label>新闻标题 (可选)</label>
                    <input v-model="newsTitle" type="text" class="form-control" placeholder="请输入新闻标题" />
                </div>

                <div class="form-group">
                    <label>新闻内容 / URL / 链接粘贴</label>
                    <textarea 
                        v-model="newsContent" 
                        class="form-control" 
                        placeholder="请输入新闻内容或粘贴URL..."
                        rows="10"
                    ></textarea>
                </div>

                <div class="form-actions">
                    <button @click="analyzeNews" :disabled="!newsContent || analyzing" class="detect-btn">
                        {{ analyzing ? '🔄 检测中...' : '🚀 一键智能检测' }}
                    </button>
                    <button @click="clearForm" class="clear-btn">🗑️ 清空</button>
                </div>

                <div v-if="errorMessage" class="error-message">{{ errorMessage }}</div>
            </div>

            <div v-if="detectionResult" class="detection-result">
                <h3>🎯 检测结果</h3>
                
                <div :class="['result-box', detectionResult.prediction === 'fake' ? 'fake' : 'real']">
                    <div class="result-icon">
                        {{ detectionResult.prediction === 'fake' ? '⚠️ 虚假新闻' : '✅ 真实新闻' }}
                    </div>
                    
                    <div class="confidence-section">
                        <h4>可信度评分</h4>
                        <div class="confidence-bar">
                            <div 
                                class="confidence-fill"
                                :style="{ 
                                    width: (detectionResult.confidence * 100) + '%',
                                    backgroundColor: detectionResult.confidence > 0.7 ? '#67C23A' : '#F56C6C'
                                }"
                            ></div>
                        </div>
                        <div class="confidence-value">{{ (detectionResult.confidence * 100).toFixed(2) }}%</div>
                    </div>

                    <div class="result-details">
                        <p><strong>检测ID:</strong> #{{ detectionResult.detection_id }}</p>
                        <p><strong>模型版本:</strong> {{ detectionResult.model_version }}</p>
                        <p><strong>处理时间:</strong> {{ detectionResult.processing_time }}s</p>
                        <p><strong>检测时间:</strong> {{ new Date(detectionResult.timestamp).toLocaleString('zh-CN') }}</p>
                    </div>

                    <div class="feature-visualization">
                        <h4>特征可视化</h4>
                        <div class="feature-chart" id="featureChart"></div>
                    </div>
                </div>
            </div>

            <div class="detection-history">
                <h3>📋 最近检测记录</h3>
                <div v-if="history.length" class="history-table">
                    <table>
                        <thead>
                            <tr>
                                <th>标题</th>
                                <th>结果</th>
                                <th>置信度</th>
                                <th>时间</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="item in history" :key="item.id">
                                <td>{{ item.news_title || '未命名' }}</td>
                                <td :class="'result-' + item.prediction">
                                    {{ item.prediction === 'fake' ? '虚假' : '真实' }}
                                </td>
                                <td>{{ (item.confidence * 100).toFixed(2) }}%</td>
                                <td>{{ new Date(item.created_at).toLocaleString('zh-CN') }}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div v-else class="empty-state">暂无检测记录</div>
            </div>
        </div>
    `,
    data() {
        return {
            newsTitle: '',
            newsContent: '',
            analyzing: false,
            errorMessage: '',
            detectionResult: null,
            history: []
        }
    },
    methods: {
        analyzeNews() {
            if (!this.newsContent.trim()) {
                this.$message.error('请输入新闻内容');
                return;
            }

            this.analyzing = true;
            this.errorMessage = '';

            const token = localStorage.getItem('access_token');
            api.analyzeNews(token, this.newsContent, this.newsTitle)
                .then(response => {
                    if (response.data.code === 200) {
                        this.detectionResult = response.data.data;
                        this.$nextTick(() => {
                            this.renderFeatureChart();
                        });
                        this.$message.success('检测完成');
                    } else {
                        this.errorMessage = response.data.message;
                    }
                })
                .catch(error => {
                    this.errorMessage = '检测失败，请重试';
                    console.error(error);
                })
                .finally(() => {
                    this.analyzing = false;
                });
        },

        clearForm() {
            this.newsTitle = '';
            this.newsContent = '';
            this.detectionResult = null;
            this.errorMessage = '';
        },

        renderFeatureChart() {
            const chartDom = document.getElementById('featureChart');
            if (!chartDom) return;

            const myChart = echarts.init(chartDom);
            
            const option = {
                radar: {
                    indicator: [
                        { name: '文本特征', max: 100 },
                        { name: '图像特征', max: 100 },
                        { name: '语义相关性', max: 100 },
                        { name: '信息新鲜度', max: 100 },
                        { name: '来源可信度', max: 100 }
                    ]
                },
                series: [{
                    name: '特征分析',
                    value: [85, 78, 92, 75, 88],
                    areaStyle: {}
                }]
            };

            myChart.setOption(option);
        }
    },
    mounted() {
        // 加载检测历史
        const token = localStorage.getItem('access_token');
        api.getDetectionHistory(token)
            .then(response => {
                if (response.data.code === 200) {
                    this.history = response.data.data.slice(0, 5);
                }
            })
            .catch(error => console.error(error));
    }
});


// ========== 可视化组件 ==========
Vue.component('visualization', {
    template: `
        <div class="visualization">
            <h2>📈 量子多模态模型可视化</h2>

            <div class="pipeline-visualization">
                <h3>QMMFND 处理流程</h3>
                <div class="pipeline">
                    <div v-for="(stage, index) in flow.stages" :key="stage.id" class="pipeline-stage">
                        <div class="stage-box">
                            <div class="stage-number">{{ index + 1 }}</div>
                            <h4>{{ stage.name }}</h4>
                            <p>{{ stage.description }}</p>
                            <div class="stage-model">{{ stage.model }}</div>
                            <div class="stage-output">输出: {{ stage.output_dim }}</div>
                        </div>
                        <div v-if="index < flow.stages.length - 1" class="arrow">→</div>
                    </div>
                </div>
            </div>

            <div class="quantum-modules">
                <h3>⚛️ 量子核心模块</h3>
                <div class="modules-grid">
                    <div v-for="component in components.components" :key="component.name" class="module-card">
                        <h4>{{ component.name }}</h4>
                        <p>{{ component.description }}</p>
                        <div class="equation">📐 {{ component.equation }}</div>
                        <div v-if="Object.keys(component.params).length" class="params">
                            <strong>参数:</strong>
                            <span v-for="(v, k) in component.params" :key="k">{{ k }}: {{ v }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="detailed-flow">
                <h3>详细流程说明</h3>
                <div class="flow-explanation">
                    <div class="flow-section">
                        <h4>📝 第1-4步: 多模态特征提取</h4>
                        <ul>
                            <li>BERT编码: 文本→768维向量</li>
                            <li>MAE编码: 图像→768维向量</li>
                            <li>CLIP融合: 图文对齐→512维向量</li>
                        </ul>
                    </div>

                    <div class="flow-section">
                        <h4>⚛️ 第5-6步: 量子编码与纠缠</h4>
                        <ul>
                            <li>波函数编码: ψ = A·e^(iθ) (复值)</li>
                            <li>纠缠对齐: 建模图文关联</li>
                            <li>保持相位信息和语义特征</li>
                        </ul>
                    </div>

                    <div class="flow-section">
                        <h4>🔀 第7-8步: 融合与分类</h4>
                        <ul>
                            <li>multiDomain PLE: 多域专家混合</li>
                            <li>分类器: 真/假新闻判别</li>
                            <li>输出: 标签+置信度</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="architecture-chart">
                <h3>系统架构图</h3>
                <div id="architectureChart" style="width: 100%; height: 400px;"></div>
            </div>
        </div>
    `,
    props: ['flow', 'components'],
    mounted() {
        this.$nextTick(() => {
            this.renderArchitectureChart();
        });
    },
    methods: {
        renderArchitectureChart() {
            const chartDom = document.getElementById('architectureChart');
            if (!chartDom) return;

            const myChart = echarts.init(chartDom);

            const option = {
                graph: {
                    layout: 'force',
                    roam: true,
                    label: {
                        position: 'right'
                    },
                    draggable: true,
                    focusNodeAdjacency: true,
                    animation: false,
                    edges: [
                        { source: 0, target: 1 },
                        { source: 0, target: 2 },
                        { source: 1, target: 3 },
                        { source: 2, target: 3 },
                        { source: 3, target: 4 },
                        { source: 4, target: 5 },
                        { source: 5, target: 6 },
                        { source: 6, target: 7 }
                    ],
                    data: [
                        { name: '文本输入', id: 0, value: 1 },
                        { name: 'BERT编码', id: 1, value: 2 },
                        { name: 'CLIP融合', id: 2, value: 2 },
                        { name: '量子编码', id: 3, value: 3 },
                        { name: '纠缠对齐', id: 4, value: 3 },
                        { name: 'PLE融合', id: 5, value: 2 },
                        { name: '分类器', id: 6, value: 2 },
                        { name: '结果输出', id: 7, value: 1 }
                    ],
                    itemStyle: {
                        color: '#67C23A'
                    },
                    lineStyle: {
                        width: 2,
                        curveness: 0.3
                    }
                },
                series: [{
                    type: 'graph',
                    data: [
                        { name: '文本输入', id: 0 },
                        { name: 'BERT编码', id: 1 },
                        { name: 'CLIP融合', id: 2 },
                        { name: '量子编码', id: 3 },
                        { name: '纠缠对齐', id: 4 },
                        { name: 'PLE融合', id: 5 },
                        { name: '分类器', id: 6 },
                        { name: '结果输出', id: 7 }
                    ],
                    links: [
                        { source: 0, target: 1 },
                        { source: 0, target: 2 },
                        { source: 1, target: 3 },
                        { source: 2, target: 3 },
                        { source: 3, target: 4 },
                        { source: 4, target: 5 },
                        { source: 5, target: 6 },
                        { source: 6, target: 7 }
                    ],
                    roam: true,
                    label: {
                        show: true
                    },
                    lineStyle: {
                        width: 2,
                        curveness: 0.3
                    }
                }]
            };

            myChart.setOption(option);
            window.addEventListener('resize', () => myChart.resize());
        }
    }
});


// ========== 日志组件 ==========
Vue.component('logs', {
    template: `
        <div class="logs">
            <h2>📋 系统操作日志</h2>
            
            <div v-if="logs.length" class="logs-table">
                <table>
                    <thead>
                        <tr>
                            <th>操作</th>
                            <th>描述</th>
                            <th>时间</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="log in logs" :key="log.id">
                            <td>{{ getActionLabel(log.action) }}</td>
                            <td>{{ log.description }}</td>
                            <td>{{ new Date(log.created_at).toLocaleString('zh-CN') }}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div v-else class="empty-state">暂无日志记录</div>
        </div>
    `,
    props: ['logs'],
    methods: {
        getActionLabel(action) {
            const labels = {
                'LOGIN': '登录',
                'LOGOUT': '退出',
                'DETECTION': '新闻检测',
                'EXPORT': '导出数据'
            };
            return labels[action] || action;
        }
    }
});
