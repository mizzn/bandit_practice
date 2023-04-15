import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, probs):
        self.probs = probs #各腕の真の報酬確率
        # print("probs = "+str(self.probs))

    # 行動を所与として、報酬を返す状態遷移関数
    def play_bandit(self, arm): #armは引く腕
        # print("---------------start play_bandit-------------------")
        r = np.random.rand() #くじを引く
        # print("r = "+str(r))
        if (r < self.probs[arm]):
            #　あたりのとき
            # print("hit")
            # print("---------------end play_bandit---------------------")
            return 1 
        else:
            # print("miss")
            # print("---------------end play_bandit---------------------")
            return 0 



class Agent:
    # とりあえずegreedy前提でepsだけ
    def __init__(self, eps, arm_size, correct_probs):
        self.epsilon = eps
        self.arm_size = arm_size
        self.correct_prob = correct_probs #真の報酬確率
        self.optimal_prob = np.max(correct_probs) #最大報酬確率
        self.nt = np.zeros(self.arm_size) #tstepまでに腕indexを選択肢した回数
        self.Qt = np.zeros(self.arm_size) #tstepのときの、腕indexの価値関数
        self.reward = [] #1行step列　もらった報酬を記録
        self.regret = [] #1行step列　(最大報酬確率 - 選んだ腕の報酬確率)の総和を記録
        self.sum_regret = 0


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

        #regretの更新
        #(最大報酬確率 - 選んだ腕の報酬確率)の総和
        # print(str(self.optimal_prob)+" - "+str(self.correct_prob[arm]))
        self.sum_regret += (self.optimal_prob - self.correct_prob[arm])
        self.regret.append(self.sum_regret)
        # print("regret = "+str(self.sum_regret))
        # print("regret list = "+str(self.regret))

        #各腕の惹かれた回数を更新
        self.nt[arm] += 1
        # print("n = "+str(self.nt))

        #観測から求める価値を更新
        self.Qt[arm] = self.Qt[arm] + (reward - self.Qt[arm]) / self.nt[arm] 
        # print("Q = "+str(self.Qt))

        # print("---------------end update---------------------")

class Manegement:

    def __init__(self, eps, arm_size, probs):
        #（環境の生成・エージェントの生成）
        self.environment = Environment(probs)
        self.correct_probs = self.environment.probs
        #1000stepを100シミュレーション
        self.step = 1000
        self.sim = 100
        # self.step = 5 #debug
        # self.sim = 2 #debug
        self.reward_mean = np.zeros(self.step)
        self.regret_mean = np.zeros(self.step)
        self.eps = eps
        self.arm_size = arm_size

    def start(self):
        for i in range(self.sim):
            self.agent = Agent(self.eps, self.arm_size, self.correct_probs)

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
            self.regret_mean = self.regret_mean + np.array(self.agent.regret)
            # print("reward sum = "+str(self.reward_mean))
            # print("regret sum = "+str(self.regret_mean))

        self.reward_mean /= self.sim
        self.regret_mean /= self.sim
        # print("reward mean = "+str(self.reward_mean))
        # print("regret mean = "+str(self.regret_mean))

        # self.plot_reward(self.reward_mean)
        self.plot_regret(self.regret_mean)


    # def plot_reward(self, result):
    #     plt.xlabel("step")
    #     plt.ylabel("reward")
    #     plt.plot(result)
    #     plt.show()

    def plot_regret(self, result):
        plt.xlabel("step")
        plt.ylabel("regret")
        plt.plot(result, label="epsilon = 0.1")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    print("arm size = ", end="")
    arm_size = int(input())
    print("probably = ", end="")
    prob = list(map(float, input().split()))
    if len(prob) != arm_size:
        print("error!")
        exit(0)
        
    eps = 0.1
    management = Manegement(eps, arm_size, prob)
    management.start()
    