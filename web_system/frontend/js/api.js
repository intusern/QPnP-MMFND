// ========== API 模块 ==========
const API_BASE_URL = 'http://localhost:5000/api';

// 添加详细的错误日志
axios.interceptors.response.use(
    response => response,
    error => {
        if (error.response) {
            // 服务器响应了错误状态码
            console.error('[API Error]', {
                status: error.response.status,
                url: error.response.config.url,
                method: error.response.config.method,
                data: error.response.data,
                headers: error.response.headers
            });
        } else if (error.request) {
            // 请求已发出但没有收到响应
            console.error('[Network Error] 无法连接到服务器:', error.request);
        } else {
            // 其他错误
            console.error('[Error]', error.message);
        }
        return Promise.reject(error);
    }
);

const api = {
    // 认证相关
    register(username, password, email, role = 'operator') {
        console.log('[Register] 发送注册请求:', { username, email, role });
        return axios.post(`${API_BASE_URL}/auth/register`, {
            username,
            password,
            email,
            role
        }).then(response => {
            console.log('[Register] 成功:', response.data);
            return response;
        }).catch(error => {
            console.error('[Register] 失败:', error.message);
            throw error;
        });
    },

    login(username, password) {
        console.log('[Login] 发送登录请求:', { username });
        return axios.post(`${API_BASE_URL}/auth/login`, {
            username,
            password
        }).then(response => {
            console.log('[Login] 成功:', response.data);
            return response;
        }).catch(error => {
            console.error('[Login] 失败:', error.message);
            throw error;
        });
    },

    getCurrentUser(token) {
        return axios.get(`${API_BASE_URL}/auth/me`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
    },

    // 数据看板相关
    getDashboardStats(token) {
        return axios.get(`${API_BASE_URL}/dashboard/stats`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
    },

    getDetectionTrend(token, days = 30) {
        return axios.get(`${API_BASE_URL}/dashboard/trend?days=${days}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
    },

    // 新闻检测相关
    analyzeNews(token, content, title = '') {
        return axios.post(`${API_BASE_URL}/detection/analyze`, {
            content,
            title
        }, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
    },

    getDetectionHistory(token, page = 1, pageSize = 20) {
        return axios.get(`${API_BASE_URL}/detection/history?page=${page}&page_size=${pageSize}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
    },

    // 可视化相关
    getModelFlow(token) {
        return axios.get(`${API_BASE_URL}/visualization/model-flow`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
    },

    getQuantumComponents(token) {
        return axios.get(`${API_BASE_URL}/visualization/quantum-components`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
    },

    // 日志相关
    getLogs(token, page = 1, pageSize = 20) {
        return axios.get(`${API_BASE_URL}/logs?page=${page}&page_size=${pageSize}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
    },

    // 健康检查
    healthCheck() {
        return axios.get(`${API_BASE_URL}/health`);
    }
};
