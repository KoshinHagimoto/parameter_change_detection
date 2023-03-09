import numpy as np
import matplotlib.pyplot as plt

from HH_package.hodgkinhuxley import HodgkinHuxley
from Pf_package.particle_filter import ParticleFilter

"""
Global variables
"""
steps = 4000 #step数
predict_count = 10 # 10期先予測
sigma = 0.5

def generate_data(gL_abnormal:list):
    """
    テストデータの作成
    :returns V_n: gL一定 V_ab:g_L変化あり
    """
    HH_n = HodgkinHuxley()  # ノイズなしの正常用データ
    HH_ab = HodgkinHuxley()  # g_Lを変化させるデータ
    # 外部からの入力電流
    I_inj = np.zeros(steps)
    I_inj[:] = 20
    # 観測膜電位をそれぞれ作成
    V_n = np.zeros(steps)
    V_ab = np.zeros(steps)
    # それぞれに初期値設定
    V_n[0] = -65.0
    V_ab[0] = -65.0

    # HH.stepを実行し, データを生成
    for i in range(steps - 1):
        result_n = HH_n.step(I_inj[i])
        result_ab = HH_ab.step(I_inj[i], gL_abnormal[i])
        V_n[i+1] = result_n[3]
        V_ab[i+1] = result_ab[3]

    # 観測ノイズを付加
    noise_ab = np.random.normal(0, sigma, (steps - 1,))
    noise_ab = np.insert(noise_ab, 0, 0)
    V_n = V_n + noise_ab
    V_ab = V_ab + noise_ab
    return V_n, V_ab

def show_graph_observation_data(V_n, V_ab, gL_abnormal):
    """
    観測データの可視化
    :param V_n: 通常時
    :param V_ab: パラメータが途中で変化するデータ
    """
    t = np.arange(0, steps)
    plt.figure(figsize=(8, 5))
    plt.plot(t, V_n, label='V(t):normal')
    plt.plot(t, V_ab, c='orange', label='V(t):abnormal')
    plt.xlabel('t[ms]', fontsize=15)
    plt.ylabel('V[mV]', fontsize=15)
    plt.legend(fontsize=15, loc='lower left')
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(t, gL_abnormal, c='orange', label='gL')
    plt.ylim(0, 0.8)
    plt.xlabel('t[ms]', fontsize=15)
    plt.ylabel('gL', fontsize=15)
    plt.legend()
    plt.grid()
    plt.show()

def show_graph_mse(mse, mse_predict):
    t = np.arange(0, steps)
    t_predict = np.arange(0, steps-predict_count)
    """ データを可視化 """
    plt.plot(t, mse, label='mse')
    plt.title('mse')
    plt.xlabel('t')
    plt.ylabel('mse')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(t_predict, mse_predict, label='mse:predict')
    plt.title('mse')
    plt.xlabel('t')
    plt.ylabel('mse')
    plt.grid()
    plt.legend()
    plt.show()

def main():
    """ gL変化のndArrayを作成 """
    gL_abnormal = np.zeros(steps)
    gL_abnormal[:] = 0.3
    gL_abnormal[(steps) // 2:] = 0.5
    """ 観測データを作成 """
    data = generate_data(gL_abnormal)
    V_n = data[0]
    V_ab = data[1]
    show_graph_observation_data(V_n, V_ab, gL_abnormal)
    """ n期先予測を用いたパラメータ変化検出 """
    # n期先予測の実行
    pf = ParticleFilter(V_ab, predict_count)
    pf.simulate()
    V_particle = pf.V_average
    V_predict = pf.V_average_predict
    """ 予測を行った平均値と異常時データの誤差を計算."""
    mse = np.zeros((steps))
    mse_predict = np.zeros((steps - predict_count))
    for i in range(steps):
        mse[i] = (V_particle[i] - V_ab[i]) ** 2
    for i in range(steps - predict_count):
        mse_predict[i] = (V_predict[i] - V_ab[predict_count + i]) ** 2
    """ 誤差を可視化 """
    show_graph_mse(mse, mse_predict)

if __name__ == '__main__':
    main()