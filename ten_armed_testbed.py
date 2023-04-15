import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, probs):
        self.arm_size = 10
        self.probs = probs

        self.optimal_actions = np.where(self.probs == self.probs.max())
        self.optimal_actions = self.optimal_actions[0] #最適行動
        # print("probs = "+str(self.probs))
        # print("optimal action = "+str(self.optimal_actions))

    # 行動を所与として、報酬を返す状態遷移関数
    def play_bandit(self, arm): #armは引く腕
        # print("---------------start play_bandit-------------------")
        
        # 実際の報酬は平均真の確率、分散1の正規分布に従う
        tmp = np.random.randn(1, 1) + self.probs[arm]
        tmp = float(tmp[0])
        # print("tmp = "+str(tmp))
        # print("---------------end play_bandit---------------------")
        return tmp


class Agent:
    # とりあえずegreedy前提でepsだけ
    def __init__(self, eps, arm_size, correct_probs, optimal_actions):
        self.epsilon = eps
        self.arm_size = arm_size
        self.correct_prob = correct_probs #真の報酬確率
        
        self.optimal_prob = np.max(correct_probs) #最大報酬確率
        self.nt = np.zeros(self.arm_size) #tstepまでに腕indexを選択肢した回数
        self.Qt = np.zeros(self.arm_size) #tstepのときの、腕indexの価値関数
        self.reward = [] #1行step列　もらった報酬を記録
        self.optimal_actions = optimal_actions
        self.optimal_actions_len = len(optimal_actions)
        self.optimal_action_history = [] #1行step列　最適行動選択率
        self.optimal_action_counter = 0
        self.step_counter = 1

        # print("reward = "+str(self.reward))
        # print("nt = "+str(self.nt))
        # print("Qt = "+str(self.Qt))
        # print("optimal prob = "+str(self.optimal_prob))
        

    # (状態を所与として)行動を返す方策関数,環境に対して行動を生成するやつ
    def e_greedy(self):
        # print("---------------start egreedy-------------------")
        r = np.random.rand()
        # print("random = "+str(r))
        if (r < self.epsilon):
            #確率epsilonでランダムに選ぶ
            # print("random select")
            selected_arm = np.random.randint(0, self.arm_size)
            # print(selected_arm)
        else:
            #greedyに行動
            # print("greedy select")
            # print("Qt = "+str(self.Qt))
            maxs = np.where(self.Qt == self.Qt.max())
            maxs = maxs[0]
            tmp = np.random.randint(0, len(maxs))
            # print("maxs = "+str(maxs))
            # print("tmp = "+str(tmp))
            selected_arm = maxs[tmp]
        
        # print("select = "+str(selected_arm))
        
        # print("---------------end egreedy---------------------")
        return selected_arm

    # 報酬を所与として、価値関数を更新する関数 環境から情報を観測して内部情報を更新するやつ
    def update(self, arm, reward):
        # print("---------------start update-------------------")
        #rewardの更新
        self.reward.append(reward)
        # print("reward list : "+str(self.reward))

        # 最適行動かどうかを判定
        for i in range(self.optimal_actions_len):
            if (arm == self.optimal_actions[i]):
                # print("optimal action!")
                # 回数を更新
                self.optimal_action_counter += 1
                break
                # print("optimal action counter = "+str(self.optimal_action_counter))
        
        # 更新
        self.optimal_action_history.append(self.optimal_action_counter / self.step_counter)
        self.step_counter += 1
        # print("optimal action history = "+str(self.optimal_action_history))
        

        #各腕の惹かれた回数を更新
        self.nt[arm] += 1
        # print("n = "+str(self.nt))

        #観測から求める価値を更新
        self.Qt[arm] = self.Qt[arm] + (reward - self.Qt[arm]) / self.nt[arm] 
        # print("Q = "+str(self.Qt))

        # print("---------------end update---------------------")

class Manegement:

    def __init__(self, eps, pr, probs):
        #（環境の生成・エージェントの生成）
        self.correct_probs = probs
        self.environment = Environment(self.correct_probs)
        self.optimal_actions = self.environment.optimal_actions
        #1000stepを2000シミュレーション
        self.step = 1000
        self.sim = 2000
        # self.step = 10 #debug
        # self.sim = 1 #debug
        self.reward_mean = np.zeros(self.step)
        self.optimal_action_mean = np.zeros(self.step)
        self.eps = eps
        self.arm_size = 10
        self.pr = pr

    def start(self):
        for i in range(self.sim):
            self.agent = Agent(self.eps, self.arm_size, self.correct_probs, self.optimal_actions)

            for i in range(self.step):
                # エージェントによって腕(インデックス)を選択する
                selected_arm = self.agent.e_greedy()

                # 選択された腕を環境に伝え、その腕に関する報酬を返す
                reward = self.environment.play_bandit(selected_arm)
                # print("reward = "+str(reward))

                # 環境から返された報酬をエージェントに伝え、期待値の更新を行う
                self.agent.update(selected_arm, reward)

            # print("end simulation "+str(self.sim))
            # print(self.reward_mean)
            # print(self.agent.reward)
            self.reward_mean = self.reward_mean + np.array(self.agent.reward)
            self.optimal_action_mean = self.optimal_action_mean + np.array(self.agent.optimal_action_history)
            # print("optimal action sum = "+str(self.optimal_action_mean))

        self.reward_mean /= self.sim
        self.optimal_action_mean /= self.sim
        # print("reward mean = "+str(self.reward_mean))
        # print("optimal action mean = " +str(self.optimal_action_mean))

        self.pr.store_result(self.reward_mean, self.optimal_action_mean)

class PlotResult:
    def __init__(self, eps):
        self.eps = eps
        self.eps_len = len(self.eps)
        self.average_rewards = []
        self.optimal_actions = []

    # 各結果を格納しておくところ
    def store_result(self, reward_mean, regret_mean):
        self.average_rewards.append(reward_mean)
        self.optimal_actions.append(regret_mean)

    def plot_average_reward(self):
        plt.xlabel("step")
        plt.ylabel("average reward")
        for i in range(self.eps_len):
            print(self.eps[i])
            plt.plot(self.average_rewards[i], label="epsilon = "+str(self.eps[i]))
        plt.legend()
        plt.ylim(0.00, 1.50)
        plt.show()

    def plot_optimal_action(self):
        plt.xlabel("step")
        plt.ylabel("optimal action")
        for i in range(self.eps_len):
            plt.plot(self.optimal_actions[i], label="epsilon = "+str(self.eps[i]))
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.show()

    

if __name__ == "__main__":
    print("eps = ", end="")
    eps = list(map(float, input().split()))
    eps_len = len(eps)
    pr = PlotResult(eps)

     #各腕の真の報酬確率 平均0,分散1の正規分布で初期化
    probs = np.random.randn(1, 10)
    probs = probs[0]
    print("probs = "+str(probs))
        
    for i in range(eps_len):
        # print(eps[i])
        management = Manegement(eps[i], pr, probs)
        management.start()
    
    # おかしい！原因不明！とりあえず保留
    pr.plot_average_reward()
    pr.plot_optimal_action()