import math
import random
from copy import copy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class Context:
    def __init__(self, linear_power=True, cooperate_probability=0.5, power_exponent=1.75, end_at_tick=10000):
    
        self.linear_power = linear_power                    # Whether to use the logarithmic or the linear power function
        self.cooperate_probability = cooperate_probability  # Probability (per state) of cooperating in an interaction. Thus, the probability of a particular interaction being cooperative is the square of this value
        self.power_exponent = power_exponent                # The exponent in the logarithmic power function
        
        self.wealth_mod_factor = 0.01   # A general factor that sets the value of war costs, cooperate gains, and growth
        self.coop_gain = self.wealth_mod_factor # Multiplier for the gain from a cooperative interaction
        self.war_cost = self.wealth_mod_factor  # Multiplier for the cost of war, representing destruction etc. - subtracted from the victor's war winnings (result will be negative if the winnings are smaller)
        self.war_steal = 0.1            # Proportion of loser's wealth that the victor of a war gets to keep
        self.coop_penalty = 0.5         # "War preparedness" power penalty for a state that cooperates against an attacker
        self.min_wealth = 1             # The cutoff point in the linear power function
        
        self.end_at_tick = end_at_tick
        self.num_states = 20
        
        self.plot_range = 1000
            
        if self.linear_power:
            self.power = lambda w: max(0, w - self.min_wealth)          # A straight line with slope 1, intercepting the wealth axis at MIN_WEALTH
        else:
            self.power = lambda w: math.log1p(w)**self.power_exponent   # ln(w+1)^1.75
            self.wealth_growth_multiplier = 1.0 / (1 - self.wealth_mod_factor + 2 * self.wealth_mod_factor * self.cooperate_probability * self.cooperate_probability)
            self.wealth_growth_multiplier *= self.wealth_growth_multiplier # (1 / (1 - f + 2fp^2))^2  - calculated to balance out the effects of cooperate gains and war costs
            
        self.states = [State(self) for _ in range(self.num_states)]
        self.plot_states = copy(self.states)
        
    def tick(self, ax, time):
        random.shuffle(self.states)
        for state in self.states:
            state.tick() # Each state gets to perform an interaction - each tick they make their moves in a different order
        if not self.linear_power:
            for state in self.states: # Adjust wealth (for logarithmic power function) to counterbalance growth/decline from war costs and cooperate gains
                state.wealth *= self.wealth_growth_multiplier
        for state in self.states:
            state.record_history()

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
        return np.array([state.history[start_time:end_time] for state in self.plot_states])
        
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
    def __init__(self, context):
        self.context = context
        self.wealth = 1
        self.history = [self.wealth]
        
    def plot(self, ax):
        ax.plot(self.history)
        
    def choose_move(self):
        return random.random() <= self.context.cooperate_probability
        
    def power(self):
        """ Power corresponds to the probability of victory in a war.
            A power of 0 causes a State to ignore all wars
        """
        return self.context.power(self.wealth)
    
    def tick(self):
        other = self
        while other is self:
            other = random.choice(self.context.states)
        
        aCoop = self.choose_move()
        bCoop = other.choose_move()
        
        if aCoop and bCoop:
            gain = self.context.coop_gain * (self.wealth + other.wealth) / 2
            self.wealth += gain
            other.wealth += gain
        elif self.power() == 0 or other.power() == 0:
            pass # If either state has zero power, wars are ignored
        elif aCoop:
            winner, loser = self.context.surprise_war(other, self)
            winner.wealth += loser.wealth*self.context.war_steal - self.context.war_cost * (self.wealth + other.wealth)
            loser.wealth -= loser.wealth*self.context.war_steal
        elif bCoop:
            winner, loser = self.context.surprise_war(self, other)
            winner.wealth += loser.wealth*self.context.war_steal - self.context.war_cost * (self.wealth + other.wealth)
            loser.wealth -= loser.wealth*self.context.war_steal
        else:
            winner, loser = self.context.war(self, other)
            winner.wealth += loser.wealth*self.context.war_steal - self.context.war_cost * (self.wealth + other.wealth)
            loser.wealth -= loser.wealth*self.context.war_steal
            
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
            
if __name__ == "__main__":
    animate()
    