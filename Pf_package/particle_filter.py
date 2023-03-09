import numpy as np

from change_detection.HH_package.hodgkinhuxley import HodgkinHuxley


class ParticleFilter(HodgkinHuxley):
    """
    n期先予測の実行.
    """
    def __init__(self, V_abnormal, predict_count, n_particle=100, dt=0.05, C_m=1.0, g_Na=120.0, g_K=36.0, E_Na=50.0,
                 E_K=-77.0, E_L=-54.387):
        self.V_abnormal = V_abnormal  # 異常時データ
        self.predict_count = predict_count  # n期先予測のn
        self.n_particle = n_particle
        super().__init__(dt, C_m, g_Na, g_K, E_Na, E_K, E_L)

        self.T = len(self.V_abnormal)  # 時系列データ数
        # 粒子フィルタ用 (T * n_particle)
        self.V_p = np.zeros((self.T + 1, self.n_particle))
        self.m_p = np.zeros((self.T + 1, self.n_particle))
        self.h_p = np.zeros((self.T + 1, self.n_particle))
        self.n_p = np.zeros((self.T + 1, self.n_particle))
        # 粒子の10期先予測
        self.V_p_predict = np.zeros((self.T - self.predict_count, self.n_particle))
        # 粒子フィルタ.リサンプリング用
        self.V_p_resampled = np.zeros((self.T + 1, self.n_particle))
        self.m_p_resampled = np.zeros((self.T + 1, self.n_particle))
        self.h_p_resampled = np.zeros((self.T + 1, self.n_particle))
        self.n_p_resampled = np.zeros((self.T + 1, self.n_particle))
        # 尤度(test data との誤差)
        self.w_V = np.zeros((self.T, self.n_particle))
        # 尤度を正規化したもの
        self.w_V_normed = np.zeros((self.T, self.n_particle))
        # 平均値
        self.V_average_predict = np.zeros((self.T - self.predict_count))
        self.V_average = np.zeros((self.T))

    def norm_likelihood(self, y, x):
        """
        尤度を計算.
        Args: y: 観測値 ,x:一期先予測した粒子
        Returns: exp(-(x-y)^2)
        """
        return np.exp(-(x - y) ** 2)

    def F_inv(self, w_cumsum, idx, u):
        """
        乱数を生成し, その時の粒子の番号を返す.

        Args: w_cumsum(array): 正規化した尤度の累積和, idx(array):[0,~,99](n=100)のarray, u(float):0~1の乱数）

        Returns: k+1: 選択された粒子の番号
        """
        if not np.any(w_cumsum < u):  # 乱数uがw_cumsumのどの値よりも小さいとき0を返す
            return 0
        k = np.max(idx[w_cumsum < u])  # uがwより大きいもので最大のものを返す
        return k + 1

    def resampling(self, weights):
        """
        リサンプリングを行う.

        Args: weights(array): 正規化した尤度の配列

        Returns: k_list: リサンプリングされて選択された粒子の番号の配列
        """
        w_cumsum = np.cumsum(weights)  # 正規化した重みの累積和をとる
        idx = np.asanyarray(range(self.n_particle))  # -> [0,1,2,,,98,99] (n=100)
        k_list = np.zeros(self.n_particle, dtype=np.int32)  # サンプリングしたｋのリスト格納

        # 乱数を粒子数つくり、関数を用いてk_listに値を格納
        for i, u in enumerate(np.random.random_sample(self.n_particle)):
            k = self.F_inv(w_cumsum, idx, u)
            k_list[i] = k
        return k_list

    def simulate(self, gL_particle=0.3):
        """
        粒子フィルタの実行

        Args: gL_particle: 粒子フィルタを実行するときのg_Lの値
        """
        # 初期値設定
        initial_V_p = np.random.normal(-65.0, 0.5, self.n_particle)
        initial_m_p = np.random.normal(0.03, 0.005, self.n_particle)
        initial_h_p = np.random.normal(0.6, 0.01, self.n_particle)
        initial_n_p = np.random.normal(0.32, 0.01, self.n_particle)
        self.V_p[0] = initial_V_p
        self.m_p[0] = initial_m_p
        self.h_p[0] = initial_h_p
        self.n_p[0] = initial_n_p
        self.V_p_resampled[0] = initial_V_p
        self.m_p_resampled[0] = initial_m_p
        self.h_p_resampled[0] = initial_h_p
        self.n_p_resampled[0] = initial_n_p

        # 外部電流
        I_inj = np.zeros(self.T)
        I_inj[:] = 20

        for t in range(self.T):
            for i in range(self.n_particle):
                # HHモデルのステップに入力する値を設定
                self.m = self.m_p_resampled[t, i]
                self.h = self.h_p_resampled[t, i]
                self.n = self.n_p_resampled[t, i]
                self.V = self.V_p_resampled[t, i]
                # step関数を実行して, t → t+1へ遷移（一期先予測）
                noise = np.random.normal(0, 0.1)  # ノイズあり
                noise_m = np.random.normal(0, 0.001)
                result = self.step(I_inj[t], gL_particle)
                self.m_p[t + 1, i] = result[0] + noise_m
                self.h_p[t + 1, i] = result[1] + noise_m
                self.n_p[t + 1, i] = result[2] + noise_m
                self.V_p[t + 1, i] = result[3] + noise
                # 粒子を10期先予測
                if t >= 1 and t <= (self.T - self.predict_count):
                    self.m = self.m_p_resampled[t, i]
                    self.h = self.h_p_resampled[t, i]
                    self.n = self.n_p_resampled[t, i]
                    self.V = self.V_p_resampled[t, i]
                    for _ in range(self.predict_count):
                        result_predict = self.step(I_inj[t], gL_particle)
                        self.V = result_predict[3]
                    self.V_p_predict[t - 1, i] = self.V
                # 尤度を観測データを用いて計算
                self.w_V[t, i] = self.norm_likelihood(self.V_abnormal[t], self.V_p[t + 1, i])
            # 求めた尤度を正規化e
            self.w_V_normed[t] = self.w_V[t] / np.sum(self.w_V[t])
            k_V = self.resampling(self.w_V_normed[t])
            self.V_p_resampled[t + 1] = self.V_p[t + 1, k_V]
            self.m_p_resampled[t + 1] = self.m_p[t + 1, k_V]
            self.h_p_resampled[t + 1] = self.h_p[t + 1, k_V]
            self.n_p_resampled[t + 1] = self.n_p[t + 1, k_V]
            # n期先予測を行った粒子の平均値を計算
            if t == 0:
                self.V_average[t] = np.sum(self.V_p_resampled[t + 1]) / self.n_particle
            else:
                self.V_average[t] = np.sum(self.V_p[t + 1]) / self.n_particle
            if t >= 1 and t <= (self.T - self.predict_count):
                self.V_average_predict[t - 1] = np.sum(self.V_p_predict[t - 1]) / self.n_particle
