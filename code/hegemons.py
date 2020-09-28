

import math
import random
from copy import copy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class Context:
    def __init__(self, linear_power=True, end_at_tick=100000, US_strat="low", Iran_strat="low", default_strat="low", coop_multiplier=1, 
                conflict_multiplier=1, coop_gain=0.01, war_cost=0.5, war_steal=0.5, power_scale=0.1):
        # params: US strat, Iran strat, default strat, coop multiplier, conflict multiplier, coop gain, war cost, war steal, power scale
        self.linear_power = linear_power                    # Whether to use the logarithmic or the linear power function

        self.coop_gain =coop_gain # Multiplier for the gain from a cooperative interaction
        self.war_cost = war_cost  # Multiplier for the cost of war, representing destruction etc. - subtracted from the victor's war winnings (result will be negative if the winnings are smaller)
        self.war_steal = war_steal            # Proportion of loser's wealth that the victor of a war gets to keep
        self.min_wealth = 0.1            # The cutoff point in the linear power function
        
        self.power_scale = power_scale
        self.max_wealth = 1
        self.end_at_tick = end_at_tick

        if self.linear_power:
            self.power = lambda w: self.power_scale*max(0, w - self.min_wealth)          # A straight line with slope 1, intercepting the wealth axis at MIN_WEALTH
        
        
        self.default_attack_policy      = lambda wealth: random.random()*self.power(wealth) / 3
        self.default_retaliation_policy = lambda intensity,wealth: min(intensity,self.power(wealth))

        self.high_intensity_attack = lambda wealth: ((random.random()+2)/3)*self.power(wealth)
        self.high_intensity_retaliation = lambda intensity,wealth: ((random.random()+2)/3)*self.power(wealth)
        self.low_intensity_attack = lambda wealth: ((random.random())/5)*self.power(wealth)
        self.low_intensity_retaliation = lambda intensity,wealth: min(intensity, ((random.random())/5)*self.power(wealth))

        if US_strat=="low":
            self.US_attack = self.low_intensity_attack
            self.US_retaliation = self.low_intensity_retaliation
        else:
            self.US_attack = self.high_intensity_attack
            self.US_retaliation = self.high_intensity_retaliation
        if Iran_strat=="low":
            self.Iran_attack = self.low_intensity_attack
            self.Iran_retaliation = self.low_intensity_retaliation
        else:
            self.Iran_attack = self.high_intensity_attack
            self.Iran_retaliation = self.high_intensity_retaliation
        """
        # one on one
        self.num_states = 2
        self.neighbours = [[1],[0]]
        self.state_names = ["US", "Iran"]
        self.wealths = [139,15]
        self.defence_agreement_matrix = np.asarray([[0,0],
                                                    [0,0]])
        self.trade_value_matrix = np.asarray([[ 0.00000e+00, 2.00000e+02],
                                            [2.00000e+02, 0.00000e+00]])
        self.conflict_matrix = np.asarray([[ 0., 14.],
                                            [14., 0.]])
        self.attack_policies = np.asarray([[ self.default_attack_policy, self.US_attack],
                                            [self.Iran_attack, self.default_attack_policy]])
        self.retaliation_policies = np.asarray([[self.default_retaliation_policy, self.US_retaliation],
                                            [self.Iran_retaliation, self.default_retaliation_policy]])
        self.cooperation_matrix = (self.defence_agreement_matrix ) + (self.trade_value_matrix / np.max(self.trade_value_matrix))
        """
        # full ensemble
        self.num_states = 9
        self.neighbours = [[1,2,3,4,5,6,7,8],[0,2,3,4,5,6,7,8],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
        self.state_names = ["US", "Iran", "UK", "France", "Germany", "China", "Russia", "Israel", "Saudi"]
        self.wealths = [139,15,15,14,17,218,40,4,14]
        self.defence_agreement_matrix = np.asarray([[ 0.,  0., 23., 17.,  0.,  0.,  0., 24., 14.],
                                            [ 0.,  0.,  0.,  0.,  0., 11.,  7.,  0.,  0.],
                                            [23.,  0.,  0.,  1.,  0.,  0.,  0.,  7., 13.],
                                            [17.,  0.,  1.,  0.,  0.,  0., 17.,  7.,  5.],
                                            [ 0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,  0.],
                                            [ 0., 11.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
                                            [ 0.,  7.,  0., 17., 15., 16.,  0.,  8.,  3.],
                                            [24.,  0.,  7.,  7.,  0.,  0.,  8.,  0.,  0.],
                                            [14.,  0., 13.,  5.,  0.,  0.,  3.,  0.,  0.]])

        self.trade_value_matrix = np.asarray([[ 0.00000e+00, 2.00000e+02, 9.65530e+04, 7.23580e+04, 1.66368e+05, 6.55808e+05, 4.24190e+04, 3.16100e+04, 6.75840e+04],
                                            [ 2.00000e+02, 0.00000e+00, 2.18000e+02, 6.96000e+02, 3.80200e+03, 5.48960e+04, 1.81200e+03, 0.00000e+00, 1.48900e+03],
                                            [ 9.65530e+04, 2.18000e+02, 0.00000e+00, 6.87700e+04, 1.55258e+05, 9.20080e+04, 1.74650e+04, 3.79400e+03, 7.80700e+03],
                                            [ 7.23580e+04, 6.96000e+02, 6.87700e+04, 0.00000e+00, 2.27918e+05, 6.76640e+04, 2.35340e+04, 2.81200e+03, 1.40170e+04],
                                            [ 1.66368e+05, 3.80200e+03, 1.55258e+05, 2.27918e+05, 0.00000e+00, 1.93803e+05, 8.04310e+04, 6.27800e+03, 1.22430e+04],
                                            [ 6.55808e+05, 5.48960e+04, 9.20080e+04, 6.76640e+04, 1.93803e+05, 0.00000e+00, 9.39430e+04, 1.46060e+04, 7.29520e+04],
                                            [ 4.24190e+04, 1.81200e+03, 1.74650e+04, 2.35340e+04, 8.04310e+04, 9.39430e+04, 0.00000e+00, 2.11500e+03, 1.25400e+03],
                                            [ 3.16100e+04, 0.00000e+00, 3.79400e+03, 2.81200e+03, 6.27800e+03, 1.46060e+04, 2.11500e+03, 0.00000e+00, 0.00000e+00],
                                            [ 6.75840e+04, 1.48900e+03, 7.80700e+03, 1.40170e+04, 1.22430e+04, 7.29520e+04, 1.25400e+03, 0.00000e+00, 0.00000e+00]])

        self.conflict_matrix = np.asarray([[ 0., 14., 0., 0., 0., 20., 10., 0., 0.],
                                        [14., 0., 1., 0., 0., 0., 0., 1., 0.],
                                        [ 0., 1., 0., 0., 0., 0., 2., 0., 0.],
                                        [ 0., 0., 0., 0., 0., 0., 1., 1., 0.],
                                        [ 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                        [20., 0., 0., 0., 0., 0., 3., 0., 0.],
                                        [10., 0., 2., 1., 1., 3., 0., 0., 0.],
                                        [ 0., 1., 0., 1., 0., 0., 0., 0., 1.],
                                        [ 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
        
        self.attack_policies = [[self.default_attack_policy, self.US_attack, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy],
                                [self.Iran_attack, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy],
                                [self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy],
                                [self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy],
                                [self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy],
                                [self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy],
                                [self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy],
                                [self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy],
                                [self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy, self.default_attack_policy]]
       
        self.retaliation_policies = [[self.default_retaliation_policy, self.US_retaliation, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy],
                                [self.Iran_retaliation, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy],
                                [self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy],
                                [self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy],
                                [self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy],
                                [self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy],
                                [self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy],
                                [self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy],
                                [self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy, self.default_retaliation_policy]]
        self.cooperation_matrix = (self.defence_agreement_matrix / np.max(self.defence_agreement_matrix)) + (self.trade_value_matrix / np.max(self.trade_value_matrix)) 
        
        
 
        self.cooperation_matrix = self.cooperation_matrix / 2
        np.set_printoptions(suppress=True)
        #print(self.conflict_matrix / 20)
        #print(self.cooperation_matrix)

        self.defenderCost = 0.5          # Cost to the defender of being attacked
        self.retaliationCost = 0.75      # Cost to the defender of being attacked and retaliating
        self.retaliationEffect = 0.5     # Cost to the attacker of being retaliated against

        self.attackTypes = 10
        self.initialValue = 0
        self.intensities = [a + 1 for a in range(self.attackTypes)]

        self.rationality = 0.5
        self.attribution = 0.5

        self.flow_array = np.zeros((4,9))
        
        self.plot_range = 1000
        
            
        
            
        self.states = [State(self, i) for i in range(self.num_states)]
        self.plot_states = copy(self.states)
        
    def tick(self, ax, time):
        random.shuffle(self.states)
        self.max_wealth = 0
        for state in self.states:
            state.record_history()
            self.max_wealth = max(self.max_wealth, state.wealth)
        for state in self.states:
            state.tick() # Each state gets to perform an interaction - each tick they make their moves in a different order
        if not self.linear_power:
            for state in self.states: # Adjust wealth (for logarithmic power function) to counterbalance growth/decline from war costs and cooperate gains
                state.wealth *= self.wealth_growth_multiplier
        
        if ax is not None:
            ax.clear()
            for state in self.plot_states:
                state.plot(ax)
            ax.set_ylim(bottom=0)
            if time < self.plot_range:
                ax.set_xlim(0, self.plot_range)
            else:
                ax.set_xlim(time - self.plot_range, time)
        
    def war(self, a, b):
        """ Returns winner, loser """
        return (a, b) if random.random() < a.power()/(a.power() + b.power()) else (b, a)
    
    def surprise_war(self, a, b):
        """ b is surprised """
        return (a, b) if random.random() < a.power()/(a.power() + b.power()*self.coop_penalty) else (b, a)
        
    def get_model_run(self, start_time, end_time):
        return (np.array([state.history[start_time:end_time] for state in self.plot_states]), self.flow_array)
        
    def get_dwell_time(self):
        """ Tracks how long states spend above the minimum wealth line, and how wealthy they get """
        result = []
        threshold = self.min_wealth if self.linear_power else 10**(-0.5) # For the logarithmic power, consider log_10(wealth) = -0.5 to be the threshold
                                                                         # this is arbitrarily based on looking at the wealth density graph with varying exponents
        for state in self.plot_states:
            wealth = 0 # Highest wealth reached since we rose above the threshold
            time = 0   # Ticks spent above threshold
            for w in state.history:
                if w <= threshold:
                    if time > 5: # Discard small dwell times (which contribute most of the data points but are unreadable on the plot)
                        result.append((time, wealth))
                    time = 0
                    wealth = 0
                else:
                    time += 1
                    wealth = max(wealth, w)
            if time > 5:
                result.append((time, wealth))
        return np.array(result)
        
    def get_log_wealth_heatmap(self, minBucket, bucketSize, numBuckets):
        """ Tracks how much time states spend at different levels of wealth - thus a whole run of the model
            is reduced to a single dimension of data, which can be plotted against some parameter
        """
        result = [0]*numBuckets
        for state in self.plot_states:
            for w in state.history:
                W = math.log10(w)
                if W < minBucket:
                    result[0] += 1
                elif W >= minBucket + bucketSize * numBuckets:
                    result[-1] += 1
                else:
                    bucket = int((W - minBucket) / bucketSize)
                    result[bucket] += 1
        return np.array([result])

class State:
    def __init__(self, context, id):
        self.id = id
        self.context = context
        self.wealth = context.wealths[id]
        self.cooperation = (context.cooperation_matrix[id,:]) / 2
        self.conflict = ((context.conflict_matrix[id,:]) / 20) / 2
        self.conflict = self.conflict * 0.5
        self.neighbours = context.neighbours[id]
        self.history = [self.wealth]
        self.attack_policy = context.attack_policies[id]
        self.retaliation_policy = context.retaliation_policies[id]
        self.attribution = context.attribution
        
    def plot(self, ax):
        ax.plot(self.history)
        ax.legend(["US", "Iran", "UK", "France", "Germany", "China", "Russia", "Israel", "Saudi"])
        ax.set_xlabel("Tick")
        ax.set_ylabel("Wealth")
        
    def choose_move(self, other):
        r = random.random()
        move = "Neutral"
        if r <= self.conflict[other.id]:
            move = "Conflict"
        elif r > (1 - self.cooperation[other.id]):
            move = "Cooperate"
        return move
        
    def power(self):
        """ Power corresponds to the probability of victory in a war.
            A power of 0 causes a State to ignore all wars
        """
        return self.context.power(self.wealth)
    def expectedValue(self, context, intensity, defender):
        pNo = defender.numIgnores[intensity] / (defender.numIgnores[intensity] + defender.numRetaliates[intensity])
        pSuccess = (1 - pNo) * context.attribution
        pFail = (1 - pNo) * (1 - context.attribution)
        return intensity * (pNo + pFail - pSuccess * context.retaliationEffect)

    def attribute(self, intensity):
        if intensity < 1:
            return random.random() < 0.5
        else:
            return True
    
    def tick(self):
        other = random.choice(self.context.states)
        #for state in self.context.states:
        #    if state.id == other:
        #        other = state
        #        break

        aMove = self.choose_move(other)
        if aMove == "Cooperate":
            #print(self.context.state_names[self.id] + " cooperated with " + self.context.state_names[other.id])
            gain = self.context.coop_gain * (self.wealth + other.wealth) / 2
            self.wealth += gain
            other.wealth += gain
        elif self.power() <= 1 or other.power() <= 1:
            pass
        elif aMove == "Neutral":
            pass
        else:
            
            # attacking
            a_intensity = self.attack_policy[other.id](self.wealth)
            self.wealth = self.wealth - self.context.war_cost*a_intensity
            if other.attribute(a_intensity):
                b_intensity = other.retaliation_policy[self.id](a_intensity, other.wealth)
                other.wealth = other.wealth - self.context.war_cost*b_intensity

                p_aWins = a_intensity / (a_intensity + b_intensity)
                if random.random() < p_aWins:
                    # a wins
                    self.wealth = self.wealth + other.wealth*self.context.war_steal*(a_intensity/self.context.power(self.context.max_wealth))
                    other.wealth = other.wealth - other.wealth*self.context.war_steal*(a_intensity/self.context.power(self.context.max_wealth))

                    if other.id in (0,1):
                        self.context.flow_array[other.id*2+1,self.id] += other.wealth*self.context.war_steal*(a_intensity/self.context.power(self.context.max_wealth))
                    if self.id in (0,1):
                        self.context.flow_array[self.id*2,other.id] += other.wealth*self.context.war_steal*(a_intensity/self.context.power(self.context.max_wealth))
                    
                else:
                    # b wins
                    pass
            else:
                # no attribution
                # do attack
                self.wealth = self.wealth + other.wealth*self.context.war_steal*(a_intensity/self.context.power(self.context.max_wealth))
                other.wealth = other.wealth - other.wealth*self.context.war_steal*(a_intensity/self.context.power(self.context.max_wealth))
                if other.id in (0,1):
                    self.context.flow_array[other.id*2+1,self.id] += other.wealth*self.context.war_steal*(a_intensity/self.context.power(self.context.max_wealth))
                if self.id in (0,1):
                    self.context.flow_array[self.id*2,other.id] += other.wealth*self.context.war_steal*(a_intensity/self.context.power(self.context.max_wealth))
                    
                    # flow from other to self
                    
        
        """
        aMove == "Conflict" and bMove == "Conflict":
            winner, loser = self.context.war(self, other)
            winner.wealth += loser.wealth*self.context.war_steal - self.context.war_cost * (self.wealth + other.wealth)
            loser.wealth -= loser.wealth*self.context.war_steal
            print(self.context.state_names[winner.id] + " defeated " + self.context.state_names[loser.id])
        elif aMove == "Conflict":
            winner, loser = self.context.surprise_war(self, other)
            winner.wealth += loser.wealth*self.context.war_steal - self.context.war_cost * (self.wealth + other.wealth)
            loser.wealth -= loser.wealth*self.context.war_steal
            print(self.context.state_names[winner.id] + " defeated " + self.context.state_names[loser.id])
        elif bMove == "Conflict":
            winner, loser = self.context.surprise_war(other, self)
            winner.wealth += loser.wealth*self.context.war_steal - self.context.war_cost * (self.wealth + other.wealth)
            loser.wealth -= loser.wealth*self.context.war_steal
            print(self.context.state_names[winner.id] + " defeated " + self.context.state_names[loser.id])
    """
            
    def record_history(self):
        self.history.append(self.wealth)

def update(time, ax, context):
    context.tick(ax, time)
    return ax
    
def run(**kwargs):
    context = Context(**kwargs)
    for t in range(context.end_at_tick):
        context.tick(None, t)
    return context
            
def animate(**kwargs):
    context = Context(**kwargs)
    fig, ax = plt.gcf(), plt.gca()
    ani = animation.FuncAnimation(fig, update, context.end_at_tick, fargs=(ax, context), interval=5, blit=False, repeat=False)
    plt.show()
    """
    flow_array = context.flow_array
    US_gains = flow_array[0,:]
    US_losses = flow_array[1,:]
    width =0.3
    plt.bar(np.arange(len(US_gains)), US_gains, width=width)
    plt.bar(np.arange(len(US_losses))+ width, US_losses, width=width)
    plt.show()"""
    
    
            
if __name__ == "__main__":
    animate()
    