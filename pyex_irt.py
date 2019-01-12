# -*- utf-8 -*-
# version 2018
# update 2018-01-20


"""
    IRT可以说是心理测量理论的一次革命，也正是因为IRT理论的存在。
    SAT、ACT、雅思、托业等考试才能做到一年多次考试，其中的玄机在于IRT等值和基于IRT的自适应测验。
    同时，运用IRT的非认知测验（例如人格等），也在处理自比数据和抵抗作假等方面成果卓越。
"""


from __future__ import print_function, division
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import warnings
# import seaborn as sn


def test_Irt2PL():
    data_file = 'f:/mydoc-book/ItemResponseTheory/miami/lsat2.csv'
    score_data = np.loadtxt(data_file, delimiter=",")
    res = Irt2PL(scores=score_data, m_step_method='newton').em()
    print('slop: {0}\nthreshold: {1}'.format(res[0], res[1]))
    return score_data


def test_Grm():
    data_file = 'f:/mydoc-book/ItemResponseTheory/miami/science2.csv'
    score_data = np.loadtxt(data_file, delimiter=',')
    grm = Grm(scores=score_data)
    print('{}'.format(grm.em()))


def test_EAPIrt2PLModel(self):
    # 模拟参数
    a = np.random.uniform(1, 3, 1000)
    b = np.random.normal(0, 1, size=1000)
    z = Irt2PL.z(a, b, 1)
    p = Irt2PL.p(z)
    _score_data = np.random.binomial(1, p, 1000)
    # 计算潜在特质估计值
    eap_result = EAPIrt2PLModel().eap(_score_data, a, b)
    print(round(float(eap_result), 4))


def test_McMc():
    # 样本量和题量
    person_size = 1000
    item_size = 60
    # 模拟参数
    a = np.random.lognormal(0, 1, (1, item_size))
    a[a > 4] = 4
    b = np.random.normal(0, 1, (1, item_size))
    b[b > 4] = 4
    b[b < -4] = -4
    c = np.random.beta(5, 17, (1, item_size))
    c[c < 0] = 0
    c[c > 0.2] = 0.2
    true_theta = np.random.normal(0, 1, (person_size, 1))
    p_val = McMc().logistic(a, b, c, true_theta)
    scores = np.random.binomial(1, p_val)

    # MCMC参数估计
    thetas, slops, thresholds, guesses = McMc().mcmc(7000, score_data=scores)
    est_theta = np.mean(thetas[3000:], axis=0)
    est_slop = np.mean(slops[3000:], axis=0)
    est_threshold = np.mean(thresholds[3000:], axis=0)
    est_guess = np.mean(guesses[3000:], axis=0)

    # 输出估计误差
    print('estimate error is :\n',
          np.mean(np.abs(est_slop - a[0])),
          np.mean(np.abs(est_threshold - b[0])),
          np.mean(np.abs(est_guess - c[:, 0])),
          np.mean(np.abs(est_theta - true_theta[:, 0])))


def irt_response_curve():
    """
    it is a idealization to use norm distribution as response curve
    limit distribution is difficultlly to reach in real data
    """

    x = [v/100 for v in range(-400, 400)]
    y = [st.norm.cdf(v) for v in x]

    plt.plot(x, y)
    plt.xlabel(r'$\theta$')
    plt.ylabel('P')
    plt.show()


class BaseIrt(object):
    """"""
    def __init__(self, scores=None):
        self.scores = scores

    @staticmethod
    def p(zv):
        # 回答正确的概率函数
        et = np.exp(zv)
        pp = et / (1.0 + et)
        return pp

    # 双参数IRT似然函数
    def _lik(self, p_val):
        # 似然函数
        # add efficient small 1e-200 to avoid log(zero)
        scores = self.scores
        log_lik_val = np.dot(np.log(p_val + 1e-200), scores.transpose()) + \
            np.dot(np.log(1 - p_val + 1e-200), (1 - scores).transpose())
        return np.exp(log_lik_val)

    # EM算法的E步算法
    def _e_step(self, p_val, weights):
        # 计算theta的分布人数
        scores = self.scores
        lik_wt = self._lik(p_val) * weights
        # 归一化
        lik_wt_sum = np.sum(lik_wt, axis=0)
        lik_wt_normalizing = lik_wt / lik_wt_sum
        # theta的人数分布
        full_dis = np.sum(lik_wt_normalizing, axis=1)
        # theta下回答正确的人数分布
        right_dis = np.dot(lik_wt_normalizing, scores)
        full_dis.shape = full_dis.shape[0], 1
        # 对数似然值
        print('log likehood value = {}'.format(np.sum(np.log(lik_wt_sum))))
        return full_dis, right_dis


class Irt2PL(BaseIrt):
    """
    由于a 和\theta 均为未知，采用EM算法（当然，也可以用MCMC算法），将\theta 作为缺失数据。
    E步：计算 \theta 下的样本量分布（人数）和答对试题的样本量分布（人数），
    M步：应用极大似然法求解a 和b的值 。
    """
    def __init__(self,
                 init_slop=None,
                 init_threshold=None,
                 max_iter=10000,
                 gp_size=11,
                 m_step_method='newton',
                 tol=1e-5,
                 *args, **kwargs):
        """
        :param init_slop: 斜率初值
        :param init_threshold: 阈值初值
        :param max_iter: EM算法最大迭代次数
        :param gp_size: Gauss–Hermite 积分点数
        :param m_step_method: M步求解方法，缺省为牛顿法('newton'), 可以选择'irls'法（迭代加权最小二乘法）
        :param tol: 精度,缺省为le-5，即10**-5
        """
        super(Irt2PL, self).__init__(*args, **kwargs)
        # 斜率初值
        if init_slop is not None:
            self._init_slop = init_slop
        else:
            self._init_slop = np.ones(self.scores.shape[1])
        # 阈值初值
        if init_threshold is not None:
            self._init_threshold = init_threshold
        else:
            self._init_threshold = np.zeros(self.scores.shape[1])
        self._max_iter = max_iter
        self._tol = tol
        self._m_step_method = '_{0}'.format(m_step_method)
        self.x_nodes, self.x_weights = self.get_gh_point(gp_size)

    @staticmethod
    def z(slop, threshold, theta):
        # z函数
        _z = slop * theta + threshold
        _z[_z > 35] = 35
        _z[_z < -35] = -35
        return _z

    # Gauss–Hermite积分静态方法
    @staticmethod
    def get_gh_point(gp_size):
        x_nodes, x_weights = np.polynomial.hermite.hermgauss(gp_size)
        x_nodes = x_nodes * 2 ** 0.5
        x_nodes.shape = x_nodes.shape[0], 1
        x_weights = x_weights / np.pi ** 0.5
        x_weights.shape = x_weights.shape[0], 1
        return x_nodes, x_weights

    # M步的求解算法很多，此处实现两种：收敛速度快的牛顿迭代（newton）、
    # 稳健见长的迭代加权最小二乘法（irls）。
    # irls是一种收敛速度很快，也很稳健，同时易于实现编程的非线性方程求解算法
    def _irls(self, p_val, full_dis, right_dis, slop, threshold, theta):
        # 所有题目误差列表
        e_list = (right_dis - full_dis * p_val) / full_dis * (p_val * (1 - p_val))
        # 所有题目权重列表
        _w_list = full_dis * p_val * (1 - p_val)
        # z函数列表
        z_list = self.z(slop, threshold, theta)
        # 加上了阈值哑变量的数据
        x_list = np.vstack((threshold, slop))
        # 精度
        delta_list = np.zeros((len(slop), 2))
        for i in range(len(slop)):
            e = e_list[:, i]
            _w = _w_list[:, i]
            w = np.diag(_w ** 0.5)
            wa = np.dot(w, np.hstack((np.ones((self.x_nodes.shape[0], 1)), theta)))
            temp1 = np.dot(wa.transpose(), w)
            temp2 = np.linalg.inv(np.dot(wa.transpose(), wa))
            x0_temp = np.dot(np.dot(temp2, temp1), (z_list[:, i] + e))
            delta_list[i] = x_list[:, i] - x0_temp
            slop[i], threshold[i] = x0_temp[1], x0_temp[0]
        return slop, threshold, delta_list

    # 牛顿迭代也是一种收敛速度很快的算法，但缺点是必须要计算步长，否则可能会不收敛，
    # 我们假设不需要计算步长，事实上步长恒为1的收敛效果还不错。
    # 下面的牛顿迭代中，并没有计算所有参数形成的雅克比矩阵和黑塞矩阵，
    # 因为求稀疏矩阵的逆近乎于求每个小矩阵的逆，事实也是如此，参数估计效果一致。
    @staticmethod
    def _newton(p_val, full_dis, right_dis, slop, threshold, theta):
        # 一阶导数
        dp = right_dis - full_dis * p_val
        # 二阶导数
        ddp = full_dis * p_val * (1 - p_val)
        # jac矩阵和hess矩阵
        jac1 = np.sum(dp, axis=0)
        jac2 = np.sum(dp * theta, axis=0)
        hess11 = -1 * np.sum(ddp, axis=0)
        hess12 = hess21 = -1 * np.sum(ddp * theta, axis=0)
        hess22 = -1 * np.sum(ddp * theta ** 2, axis=0)
        delta_list = np.zeros((len(slop), 2))
        # 把求稀疏矩阵的逆转化成求每个题目的小矩阵的逆
        for i in range(len(slop)):
            jac = np.array([jac1[i], jac2[i]])
            hess = np.array(
                [[hess11[i], hess12[i]],
                 [hess21[i], hess22[i]]]
            )
            delta = np.linalg.solve(hess, jac)
            slop[i], threshold[i] = slop[i] - delta[1], threshold[i] - delta[0]
            delta_list[i] = delta
        return slop, threshold, delta_list

    def _est_item_parameter(self, slop, threshold, theta, p_val):
        full_dis, right_dis = self._e_step(p_val, self.x_weights)
        return self._m_step(p_val, full_dis, right_dis, slop, threshold, theta)

    def _m_step(self, p_val, full_dis, right_dis, slop, threshold, theta):
        # EM算法M步
        m_step_method = getattr(self, self._m_step_method)
        return m_step_method(p_val, full_dis, right_dis, slop, threshold, theta)

    def em(self):
        max_iter = self._max_iter
        tol = self._tol
        slop = self._init_slop
        threshold = self._init_threshold
        for i in range(max_iter):
            zz = self.z(slop, threshold, self.x_nodes)
            p_val = self.p(zz)
            slop, threshold, delta_list = self._est_item_parameter(slop, threshold, self.x_nodes, p_val)
            if np.max(np.abs(delta_list)) < tol:
                print('accuracy is reached at iter time {}'.format(i))
                return slop, threshold
        warnings.warn("no convergence")
        return slop, threshold


class EAPIrt2PLModel(object):
    """
    use EAP to evaluate trait
    EM算法只能估计项目参数，对于特质参数，需要单独估计。
    特质参数估计的方法很多，有极大似然法，加权极大似然法，MAP（岭回归的另一种贝叶斯叫法），EAP。
    EAP（expected a posteriori）算法是唯一不需要迭代的算法，所以它的计算速度是最快的，常用于在线测验的参数估计，
    其理论依据是贝叶斯法则。EAP的公式为
    E(\theta_i) = \theta_i =\frac{\int \theta_i g(\theta)L(\theta_i)d\theta}{\int g(\theta)L(\theta_i)d\theta} ，
    g(\theta) 是概率密度函数，常假设服从正态分布，所以上式的积分可以用Gauss–Hermite积分求解方法计算。
    """

    # def __init__(self, score_array, slop, threshold, model=Irt2PL):
    #     self.x_nodes, self.x_weights = model.get_gh_point(21)
    #     z = model.z(slop, threshold, self.x_nodes)
    #     p = model.p(z)
    #     self.lik_values = np.prod(p ** score_array * (1.0 - p) ** (1 - score_array), axis=1)
    #
    # @property
    # def g(self):
    #     x = self.x_nodes[:, 0]
    #     weight = self.x_weights[:, 0]
    #     return np.sum(x * weight * self.lik_values)
    #
    # @property
    # def h(self):
    #     weight = self.x_weights[:, 0]
    #     return np.sum(weight * self.lik_values)
    #
    # @property
    # def res(self):
    #     return round(self.g / self.h, 3)

    @staticmethod
    def eap(score_array, slop, threshold, model=Irt2PL):
        x_nodes, x_weights = model.get_gh_point(21)
        zz = model.z(slop, threshold, x_nodes)
        pp = model.p(zz)
        lik_values = np.prod(pp ** score_array * (1.0 - pp) ** (1 - score_array), axis=1)
        # get g
        x = x_nodes[:, 0]
        weight = x_weights[:, 0]
        g = np.sum(x * weight * lik_values)
        # get h
        weight = x_weights[:, 0]
        h = np.sum(weight * lik_values)
        # get result
        # r = g / h
        return g/h


class Grm(object):

    def __init__(self, scores=None, init_slop=None, init_threshold=None, max_iter=1000, tol=1e-5, gp_size=11):
        # 试题最大反应计算
        max_score = int(np.max(scores))
        min_score = int(np.min(scores))
        self._rep_len = max_score - min_score + 1
        self.scores = {}
        for i in range(scores.shape[1]):
            temp_scores = np.zeros((scores.shape[0], self._rep_len))
            for j in range(self._rep_len):
                temp_scores[:, j][scores[:, i] == min_score + j] = 1
            self.scores[i] = temp_scores
        # 题量
        self.item_size = scores.shape[1]
        if init_slop is not None:
            self._init_slop = init_slop
        else:
            self._init_slop = np.ones(scores.shape[1])
        if init_threshold is not None:
            self._init_thresholds = init_threshold
        else:
            self._init_thresholds = np.zeros((scores.shape[1], self._rep_len - 1))
            for i in range(scores.shape[1]):
                self._init_thresholds[i] = np.arange(self._rep_len / 2 - 1, -self._rep_len / 2, -1)
        self._max_iter = max_iter
        self._tol = tol
        self.x_nodes, self.x_weights = self.get_gh_point(gp_size)

    @staticmethod
    def get_gh_point(gp_size):
        x_nodes, x_weights = np.polynomial.hermite.hermgauss(gp_size)
        x_nodes = x_nodes * 2 ** 0.5
        x_nodes.shape = x_nodes.shape[0], 1
        x_weights = x_weights / np.pi ** 0.5
        x_weights.shape = x_weights.shape[0], 1
        return x_nodes, x_weights

    @staticmethod
    def p(z):
        # 回答为某一反应的概率函数
        p_val_dt = {}
        for key in z.keys():
            e = np.exp(z[key])
            p = e / (1.0 + e)
            p_val_dt[key] = p
        return p_val_dt

    @staticmethod
    def z(slop, thresholds, theta):
        # z函数
        z_val = {}
        temp = slop * theta
        for i, threshold in enumerate(thresholds):
            z_val[i] = temp[:, i][:, np.newaxis] + threshold
        return z_val

    def _lik(self, p_val_dt):
        loglik_val = 0
        rep_len = self._rep_len
        scores = self.scores
        for i in range(self.item_size):
            for j in range(rep_len):
                p_pre = 1 if j == 0 else p_val_dt[i][:, j - 1]
                p = 0 if j == rep_len - 1 else p_val_dt[i][:, j]
                loglik_val += np.dot(np.log(p_pre - p + 1e-200)[:, np.newaxis], scores[i][:,  j][np.newaxis])
        return np.exp(loglik_val)

    def _e_step(self, p_val_dt, weights):
        # E步计算theta的分布人数
        scores = self.scores
        lik_wt = self._lik(p_val_dt) * weights
        # 归一化
        lik_wt_sum = np.sum(lik_wt, axis=0)
        _temp = lik_wt / lik_wt_sum
        # theta的人数分布
        full_dis = np.sum(_temp, axis=1)
        # theta下回答的人数分布
        right_dis_dt = {}
        for i in range(self.item_size):
            right_dis_dt[i] = np.dot(_temp, scores[i])
        # full_dis.shape = full_dis.shape[0], 1
        # 对数似然值
        print(np.sum(np.log(lik_wt_sum)))
        return full_dis, right_dis_dt

    @staticmethod
    def _pq(p_val):
        return p_val * (1 - p_val)

    @staticmethod
    def _item_jac(p_val, pq_val, right_dis, len_threshold, rep_len, theta):
        # 雅克比矩阵
        dloglik_val = np.zeros(len_threshold + 1)
        _theta = theta[:, 0]
        for i in range(rep_len):
            p_pre, pq_pre = (1, 0) if i == 0 else (p_val[:, i - 1], pq_val[:, i - 1])
            p, pq = (0, 0) if i == rep_len - 1 else (p_val[:, i], pq_val[:, i])
            temp1 = _theta * right_dis[:, i] * (1 - p_pre - p)
            dloglik_val[-1] += np.sum(temp1)
            if i < rep_len - 1:
                temp2 = right_dis[:, i] * pq / (p - p_pre + 1e-200)
                dloglik_val[i] += np.sum(temp2)
            if i > 0:
                temp3 = right_dis[:, i] * pq_pre / (p_pre - p + 1e-200)
                dloglik_val[i - 1] += np.sum(temp3)
        return dloglik_val

    @staticmethod
    def _item_hess(p_val, pq_val, full_dis, len_threshold, rep_len, theta):
        # 黑塞矩阵
        ddloglik_val = np.zeros((len_threshold + 1, len_threshold + 1))
        _theta = theta[:, 0]
        for i in range(rep_len):
            p_pre, dp_pre = (1, 0) if i == 0 else (p_val[:, i - 1], pq_val[:, i - 1])
            p, dp = (0, 0) if i == rep_len - 1 else (p_val[:, i], pq_val[:, i])
            if i < rep_len - 1:
                temp1 = full_dis * _theta * dp * (dp_pre - dp) / (p_pre - p + 1e-200)
                ddloglik_val[len_threshold:, i] += np.sum(temp1)
                temp2 = full_dis * dp ** 2 / (p_pre - p + 1e-200)
                ddloglik_val[i, i] += -np.sum(temp2)
            if i > 0:
                temp3 = full_dis * _theta * dp_pre * (dp - dp_pre) / (p_pre - p + 1e-200)
                ddloglik_val[len_threshold:, i - 1] += np.sum(temp3, axis=0)
                temp4 = full_dis * dp_pre ** 2 / (p_pre - p + 1e-200)
                ddloglik_val[i - 1, i - 1] += -np.sum(temp4)
            if 0 < i < rep_len - 1:
                ddloglik_val[i, i - 1] = np.sum(full_dis * dp * dp_pre / (p_pre - p + 1e-200))
            temp5 = full_dis * _theta ** 2 * (dp_pre - dp) ** 2 / (p - p_pre)
            ddloglik_val[-1, -1] += np.sum(temp5, axis=0)
        ddloglik_val += ddloglik_val.transpose() - np.diag(ddloglik_val.diagonal())
        return ddloglik_val

    def _m_step(self, p_val_dt, full_dis, right_dis_dt, slop, thresholds, theta):
        # M步，牛顿迭代
        rep_len = self._rep_len
        len_threshold = thresholds.shape[1]
        delta_list = np.zeros((self.item_size, len_threshold + 1))
        for i in range(self.item_size):
            p_val = p_val_dt[i]
            pq_val = self._pq(p_val)
            right_dis = right_dis_dt[i]
            jac = self._item_jac(p_val, pq_val, right_dis, len_threshold, rep_len, theta)
            hess = self._item_hess(p_val, pq_val, full_dis, len_threshold, rep_len, theta)
            delta = np.linalg.solve(hess, jac)
            slop[i], thresholds[i] = slop[i] - delta[-1], thresholds[i] - delta[:-1]
            delta_list[i] = delta
        return slop, thresholds, delta_list

    def _est_item_parameter(self, slop, threshold, theta, p_val):
        full_dis, right_dis_dt = self._e_step(p_val, self.x_weights)
        return self._m_step(p_val, full_dis, right_dis_dt, slop, threshold, theta)

    def em(self):
        max_iter = self._max_iter
        tol = self._tol
        slop = self._init_slop
        thresholds = self._init_thresholds
        for i in range(max_iter):
            z = self.z(slop, thresholds, self.x_nodes)
            p_val = self.p(z)
            slop, thresholds, delta_list = self._est_item_parameter(slop, thresholds, self.x_nodes, p_val)
            if np.max(np.abs(delta_list)) < tol:
                print(i)
                return slop, thresholds
        warnings.warn("no convergence")
        return slop, thresholds


class McMc(object):
    """
    我们从贝叶斯的角度来求解IRT参数，即MCMC算法。
    MCMC算法优点是实现简单，容易编程，对初值不敏感，可以同时估计项目参数和潜在变量，缺点是耗时。
    本次对三参数IRT模型进行参数估计： c + (1 - c) * （e^{a * \theta + b}）/（1 + e^（a * \theta + b））
    与双参数模型相比，三参数多了一个 c 参数，这个 c 参数通常称为猜测参数。
    我们采用的MCMC算法是最简单的gibbs抽样。
    定的抽样分布是:
        \theta_{t}\sim N(\theta_{t-1}, 1) ，
        a_{t}\sim N(a_{t-1}, 0.3) ，
        b_{t}\sim N(b_{t-1}, 0.3) ，
        c_{t}\sim unif(c_{t-1}, 0.03) （均匀分布）
    """
    @staticmethod
    def _log_normal(param):
        # 正态分布的概率密度分布的对数
        return param ** 2 * -0.5

    def _log_lognormal(self, param):
        # 对数正态分布的概率密度分布的对数
        return np.log(1.0 / param) + self._log_normal(np.log(param))

    def _param_den(self, slop, threshold, guess):
        # 项目参数联合概率密度
        return self._log_normal(threshold) + self._log_lognormal(slop) + 4 * np.log(guess) + 16 * np.log(1 - guess)

    @staticmethod
    def logistic(slop, threshold, guess, theta):
        # logistic函数
        return guess + (1 - guess) / (1.0 + np.exp(-1 * (slop * theta + threshold)))

    def loglik(self, slop, threshold, guess, theta, scores, axis=1):
        # 对数似然函数
        p = self.logistic(slop, threshold, guess, theta)
        p[p <= 0] = 1e-10
        p[p >= 1] = 1 - 1e-10
        return np.sum(scores * np.log(p) + (1 - scores) * np.log(1 - p), axis=axis)

    def _tran_theta(self, slop, threshold, guess, theta, next_theta, scores):
        # 特质的转移函数
        pi = (self.loglik(slop, threshold, guess, next_theta, scores) +
              self._log_normal(next_theta)[:, 0]) - (self.loglik(slop, threshold, guess, theta, scores) +
                                                     self._log_normal(theta)[:, 0])
        pi = np.exp(pi)
        # 下步可省略
        pi[pi > 1] = 1
        return pi

    def _tran_item_para(self, slop, threshold, guess, next_slop, next_threshold, next_guess, theta, score_data):
        # 项目参数的转移函数
        nxt = self.loglik(next_slop, next_threshold, next_guess, theta, score_data, 0) + \
              self._param_den(next_slop, next_threshold, next_guess)
        now = self.loglik(slop, threshold, guess, theta, score_data, 0) + \
            self._param_den(slop, threshold, guess)
        pi = nxt - now
        pi.shape = pi.shape[1]
        pi = np.exp(pi)
        # 下步可省略
        pi[pi > 1] = 1
        return pi

    def mcmc(self, chain_size, score_data):
        # 样本量
        person_size = score_data.shape[0]
        # 项目量
        item_size = score_data.shape[1]
        # 潜在特质初值
        theta = np.zeros((person_size, 1))
        # 斜率初值
        slop = np.ones((1, item_size))
        # 阈值初值
        threshold = np.zeros((1, item_size))
        # 猜测参数初值
        guess = np.zeros((1, item_size)) + 0.1
        # 参数储存记录
        theta_list = np.zeros((chain_size, len(theta)))
        slop_list = np.zeros((chain_size, item_size))
        threshold_list = np.zeros((chain_size, item_size))
        guess_list = np.zeros((chain_size, item_size))
        bar = progressbar.ProgressBar()
        for i in bar(range(chain_size)):
            next_theta = np.random.normal(theta, 1)
            theta_pi = self._tran_theta(slop, threshold, guess, theta, next_theta, score_data)
            theta_r = np.random.uniform(0, 1, len(theta))
            theta[theta_r <= theta_pi] = next_theta[theta_r <= theta_pi]
            theta_list[i] = theta[:, 0]
            next_slop = np.random.normal(slop, 0.3)
            # 防止数值溢出
            next_slop[next_slop < 0] = 1e-10
            next_threshold = np.random.normal(threshold, 0.3)
            next_guess = np.random.uniform(guess - 0.03, guess + 0.03)
            # 防止数值溢出
            next_guess[next_guess <= 0] = 1e-10
            next_guess[next_guess >= 1] = 1 - 1e-10
            param_pi = self._tran_item_para(slop, threshold, guess, next_slop,
                                            next_threshold, next_guess, theta, score_data)
            param_r = np.random.uniform(0, 1, item_size)
            slop[0][param_r <= param_pi] = next_slop[0][param_r <= param_pi]
            threshold[0][param_r <= param_pi] = next_threshold[0][param_r <= param_pi]
            guess[0][param_r <= param_pi] = next_guess[0][param_r <= param_pi]
            slop_list[i] = slop[0]
            threshold_list[i] = threshold[0]
            guess_list[i] = guess[0]
        return theta_list, slop_list, threshold_list, guess_list
