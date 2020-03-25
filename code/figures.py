import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os

from hegemons import run

def figure_1():
    plt.close()
    x1 = np.arange(0.0, 2.0, 0.01)
    y1 = np.exp(x1) - 1
    x2 = np.arange(0.0, 3.0, 0.01)
    y2 = np.log1p(x2)
    x3 = np.arange(0.0, 10.0, 0.01)
    y3 =  list(map(lambda x: x*x if x < 5 else 20*x - x*x - 50, x3))
    x4 = np.arange(0.0, 20.0, 0.01)
    y4 = np.cbrt(x4-3) + np.cbrt(3)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(x1, y1, 'royalblue')
    axes[0, 1].plot(x2, y2, 'royalblue')
    axes[1, 0].plot(x3, y3, 'royalblue')
    axes[1, 1].plot(x4, y4, 'royalblue')
    for ax in axes.flatten():
        ax.tick_params('both', bottom=False, labelbottom=False, left=False, labelleft=False)
    fig.suptitle('Figure 1')
    fig.text(0,0.5,'Power')
    fig.text(0.5,0,'Wealth')
    plt.show()

def figure_2():
    plt.close()
    x = np.arange(0.0, 5.0, 0.01)
    y1 = list(map(lambda x: 0 if x < 1 else (x-1), x))
    ax = plt.gca()
    ax.plot(x, y1, 'royalblue')
    ax.set_title('Figure 2')
    ax.set_ylabel('Power')
    ax.set_xlabel('Wealth')
    ax.tick_params('both', bottom=False, labelbottom=False, left=False, labelleft=False)
    plt.show()

def figure_3_data():
    if not os.path.exists('Model Runs'):
        os.makedirs('Model Runs')

    data = run(end_at_tick=2001, cooperate_probability=0.3).get_model_run(1000, 2001)
    np.savetxt('Model Runs/Linear 0.3 coop.csv', data.T, fmt='%f', delimiter=',')

    data = run(end_at_tick=2001, cooperate_probability=0.5).get_model_run(1000, 2001)
    np.savetxt('Model Runs/Linear default.csv', data.T, fmt='%f', delimiter=',')

    data = run(end_at_tick=2001, cooperate_probability=0.7).get_model_run(1000, 2001)
    np.savetxt('Model Runs/Linear 0.7 coop.csv', data.T, fmt='%f', delimiter=',')

def get_model_run(path):
    return np.loadtxt(path, delimiter=',')

def figure_3():
    plt.close()
    x = range(1000,2001)
    y1 = get_model_run('Model Runs/Linear 0.3 coop.csv')
    y2 = get_model_run('Model Runs/Linear default.csv')
    y3 = get_model_run('Model Runs/Linear 0.7 coop.csv')

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(x, y1)
    axes[0].set_title('Probability of cooperation = 0.3')
    axes[1].plot(x, y2)
    axes[1].set_title('Probability of cooperation = 0.5')
    axes[2].plot(x, y3)
    axes[2].set_title('Probability of cooperation = 0.7')
    for ax in axes:
        ax.set_ylabel('Wealth')
        ax.set_xlabel('Time')
    fig.suptitle('Figure 3')
    plt.show()

def warontherocks_figure_1():
    plt.close()
    x = range(1000,2001)
    y1 = get_model_run('Model Runs/Linear 0.3 coop.csv')
    y2 = get_model_run('Model Runs/Linear default.csv')

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(x, y1)
    axes[0].set_title('Probability of cooperation = 0.3')
    axes[1].plot(x, y2)
    axes[1].set_title('Probability of cooperation = 0.5')
    for ax in axes:
        ax.set_ylabel('Wealth')
        ax.set_xlabel('Time')
        ax.set_ylim(0,30)
    plt.show()

def figure_4_data():
    if not os.path.exists('Dwell Time'):
        os.makedirs('Dwell Time')

    data = run(end_at_tick=100000).get_dwell_time()
    np.savetxt('Dwell Time/Linear.csv', data, fmt=['%d', '%f'], delimiter=',')

def get_dwell_time(path):
    data = np.loadtxt(path, delimiter =',')
    return data[:,0], data[:,1]

def figure_4():
    plt.close()
    x, y = get_dwell_time('Dwell Time/Linear.csv')

    ax = plt.gca()
    ax.scatter(x, y)
    ax.set_xscale('log')
    ax.set_ylabel('Max Wealth')
    ax.set_xlabel('Dwell Time')
    ax.set_title('Figure 4')
    plt.show()

def figure_5_data():
    if not os.path.exists('Dwell Time'):
        os.makedirs('Dwell Time')

    data = np.array([]).reshape(0,3)

    for coop in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        result = run(end_at_tick=100000, cooperate_probability=coop).get_dwell_time()
        coop_values = np.array([[coop]]*len(result))
        result = np.concatenate([coop_values,result], axis=1)
        data = np.concatenate([data,result])

    np.savetxt('Dwell Time/LinearColours.csv', data, fmt=['%.1f', '%d', '%f'], delimiter=',')

def get_dwell_time_colour(path):
    df = pd.read_csv(path, header=None, names=['Coop','Dwell','Wealth'])
    filtered = [(v, df.loc[df['Coop'] == v]) for v in df['Coop'].unique()]
    return [(v, f['Dwell'].values, f['Wealth'].values) for v, f in filtered]

def figure_5():
    plt.close()
    values = get_dwell_time_colour('Dwell Time/LinearColours.csv')

    ax = plt.gca()
    for c,x,y in reversed(values):
        ax.scatter(x, y, label=str(c))
    ax.set_xscale('log')
    ax.set_ylabel('Max Wealth')
    ax.set_xlabel('Dwell Time')
    ax.set_title('Figure 5')
    ax.legend()
    plt.show()

def warontherocks_figure_2():
    plt.close()
    values = get_dwell_time_colour('Dwell Time/LinearColours.csv')

    ax = plt.gca()
    for c,x,y in reversed(values):
        green = (c - values[0][0])/values[-1][0]
        ax.scatter(x, y, label=str(c), c=[(1-green,green,0)], s=9)
    ax.set_xscale('log')
    ax.set_ylabel('Max Wealth')
    ax.set_xlabel('Dwell Time')
    ax.legend()
    plt.show()

def figure_6_data():
    if not os.path.exists('Heatmaps'):
        os.makedirs('Heatmaps')

    data = np.array([]).reshape(0,50)

    for coop in np.arange(0.14, 0.71, 0.01):
        result = run(cooperate_probability=coop).get_log_wealth_heatmap(0, 0.05, 51)[:,1:]
        data = np.concatenate([data,result])

    np.savetxt('Heatmaps/CooperateHeatmap.csv', data, fmt='%d', delimiter=',')

def get_heatmap(path):
    return np.loadtxt(path, delimiter=',').T

def get_modified_rainbow_cmap():
    rainbow = plt.get_cmap('rainbow')
    colors = [(0,'white')] + [(i/100, rainbow(i/100)) for i in range(10,101)]
    return LinearSegmentedColormap.from_list(None, colors)

def figure_6():
    plt.close()
    data = get_heatmap('Heatmaps/CooperateHeatmap.csv')
    data[data == 0] = -data.max()/9 # 1/9th so that the entire first 10% of the colormap is reserved i.e. data = 1 normalises to 0.1
    ax = plt.gca()
    ax.imshow(data, extent=[0.135, 0.705, 0.05, 2.55], aspect='auto', origin='lower', interpolation='bilinear', cmap=get_modified_rainbow_cmap())
    ax.set_ylabel('Wealth Level')
    ax.set_xlabel('Probability of Cooperation')
    ax.set_yticks([1,2])
    ax.set_yticklabels(['10', '100'])
    ax.set_title('Figure 6')
    plt.show()

def figure_7():
    plt.close()
    x = np.arange(0.0, 5.0, 0.01)
    y1 = np.log1p(x)**1.75
    y2 = np.log1p(x)**2.5
    ax = plt.gca()
    ax.plot(x, y1, 'lightcoral', label='k=1.75')
    ax.plot(x, y2, 'royalblue', label='k=2.5')
    ax.set_title('Figure 7')
    ax.set_ylabel('Power')
    ax.set_xlabel('Wealth')
    ax.tick_params('both', bottom=False, labelbottom=False, left=False, labelleft=False)
    plt.legend()
    plt.show()

def figure_8_and_9_data():
    if not os.path.exists('Model Runs'):
        os.makedirs('Model Runs')

    data = run(end_at_tick=2001, cooperate_probability=0.3, linear_power=False, power_exponent=1.75).get_model_run(1000, 2001)
    np.savetxt('Model Runs/Log 0.3 coop.csv', data.T, fmt='%f', delimiter=',')

    data = run(end_at_tick=2001, cooperate_probability=0.5, linear_power=False, power_exponent=1.75).get_model_run(1000, 2001)
    np.savetxt('Model Runs/Log default.csv', data.T, fmt='%f', delimiter=',')

    data = run(end_at_tick=2001, cooperate_probability=0.7, linear_power=False, power_exponent=1.75).get_model_run(1000, 2001)
    np.savetxt('Model Runs/Log 0.7 coop.csv', data.T, fmt='%f', delimiter=',')

    data = run(end_at_tick=2001, cooperate_probability=0.5, linear_power=False, power_exponent=3.0).get_model_run(1000, 2001)
    np.savetxt('Model Runs/Log 3.0 exponent.csv', data.T, fmt='%f', delimiter=',')

    data = run(end_at_tick=2001, cooperate_probability=0.5, linear_power=False, power_exponent=1.0).get_model_run(1000, 2001)
    np.savetxt('Model Runs/Log 1.0 exponent.csv', data.T, fmt='%f', delimiter=',')

def figure_8():
    plt.close()
    x = range(1000,2001)
    y1 = get_model_run('Model Runs/Log 0.3 coop.csv')
    y2 = get_model_run('Model Runs/Log default.csv')
    y3 = get_model_run('Model Runs/Log 0.7 coop.csv')

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(x, y1)
    axes[0].set_title('Cooperate probability = 0.3, Exponent = 1.75')
    axes[1].plot(x, y2)
    axes[1].set_title('Cooperate probability = 0.5, Exponent = 1.75')
    axes[2].plot(x, y3)
    axes[2].set_title('Cooperate probability = 0.7, Exponent = 1.75')
    for ax in axes:
        ax.set_ylabel('Wealth')
        ax.set_xlabel('Time')
    fig.suptitle('Figure 8')
    plt.show()

def figure_9():
    plt.close()
    x = range(1000,2001)
    y1 = get_model_run('Model Runs/Log 3.0 exponent.csv')
    y2 = get_model_run('Model Runs/Log default.csv')
    y3 = get_model_run('Model Runs/Log 1.0 exponent.csv')

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(x, y1)
    axes[0].set_title('Cooperate probability = 0.5, Exponent = 3')
    axes[1].plot(x, y2)
    axes[1].set_title('Cooperate probability = 0.5, Exponent = 1.75')
    axes[2].plot(x, y3)
    axes[2].set_title('Cooperate probability = 0.5, Exponent = 1')
    for ax in axes:
        ax.set_ylabel('Wealth')
        ax.set_xlabel('Time')
    fig.suptitle('Figure 9')
    plt.show()

def figure_10_data():
    if not os.path.exists('Dwell Time'):
        os.makedirs('Dwell Time')

    data = np.array([]).reshape([0,2])

    for _ in range(10):
        data = np.concatenate([data, run(linear_power=False).get_dwell_time()])

    np.savetxt('Dwell Time/Logarithmic.csv', data, fmt=['%d', '%f'], delimiter=',')

def figure_10():
    plt.close()
    x, y = get_dwell_time('Dwell Time/Logarithmic.csv')

    ax = plt.gca()
    ax.scatter(x, y)
    ax.set_xscale('log')
    ax.set_ylabel('Max Wealth')
    ax.set_xlabel('Dwell Time')
    ax.set_title('Figure 10')
    plt.show()

def figure_11_data():
    if not os.path.exists('Dwell Time'):
        os.makedirs('Dwell Time')

    data = np.array([]).reshape(0,3)

    for exponent in [1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]:
        result = run(linear_power=False, power_exponent=exponent).get_dwell_time()
        exponent_values = np.array([[exponent]]*len(result))
        result = np.concatenate([exponent_values,result], axis=1)
        data = np.concatenate([data,result])

    np.savetxt('Dwell Time/LogarithmicColours.csv', data, fmt=['%.2f', '%d', '%f'], delimiter=',')

def figure_11():
    plt.close()
    values = get_dwell_time_colour('Dwell Time/LogarithmicColours.csv')

    ax = plt.gca()
    for c,x,y in values:
        ax.scatter(x, y, label=str(c))
    ax.set_xscale('log')
    ax.set_ylabel('Max Wealth')
    ax.set_xlabel('Dwell Time')
    ax.set_title('Figure 11')
    ax.legend()
    plt.show()

def figure_12_data():
    if not os.path.exists('Heatmaps'):
        os.makedirs('Heatmaps')

    data = np.array([]).reshape(0,700)

    for exponent in np.arange(1.0, 3.01, 0.01):
        print('{:.1%} done'.format((exponent-1)/2))
        result = []
        for _ in range(4):
            result.append(run(linear_power=False, power_exponent=exponent).get_log_wealth_heatmap(-5, 0.01, 700))
        result = np.array(result).mean(axis=0)
        data = np.concatenate([data,result])

    np.savetxt('Heatmaps/ExponentHeatmap.csv', data, fmt='%g', delimiter=',')

def figure_12():
    plt.close()
    data = get_heatmap('Heatmaps/ExponentHeatmap.csv')
    data[data == 0] = -data.max()/9
    ax = plt.gca()
    ax.imshow(data, extent=[0.995, 3.005, -5, 2], aspect='auto', origin='lower', interpolation='bilinear', cmap=get_modified_rainbow_cmap())
    ax.set_ylabel('Wealth Level')
    ax.set_xlabel('Power Exponent')
    ax.set_xlim(3.005, 0.995)
    ax.set_yticks([-4,-3,-2,-1,0,1,2])
    ax.set_yticklabels(['1/10000', '1/1000', '1/100', '1/10', '1', '10', '100'])
    ax.set_title('Figure 12')
    plt.show()

def figure_13():
    plt.close()
    x = np.arange(0.0, 7.0, 0.01)
    y1 = x*0.35
    y2 = np.log1p(x)
    ax = plt.gca()
    ax.plot(x, y1, 'lightcoral')
    ax.plot(x, y2, 'royalblue')
    ax.fill_between(x, y1, y2, where=y2 >= y1, facecolor='mistyrose', interpolate=True)
    ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='skyblue', interpolate=True)
    ax.set_title('Figure 13')
    ax.set_ylim(0,5)
    ax.set_xlim(0,7)
    ax.set_ylabel('Power')
    ax.set_xlabel('Wealth')
    plt.show()

def figure_14():
    plt.close()
    x = np.arange(0.0, 7.0, 0.01)
    y1 = x*0.55
    y2 = np.log1p(x)**1.75
    ax = plt.gca()
    ax.plot(x, y1, 'lightcoral')
    ax.plot(x, y2, 'royalblue')
    ax.fill_between(x, y1, y2, where=y2 >= y1, facecolor='mistyrose', interpolate=True)
    ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='skyblue', interpolate=True)
    ax.set_title('Figure 14')
    ax.set_ylim(0,5)
    ax.set_xlim(0,7)
    ax.set_ylabel('Power')
    ax.set_xlabel('Wealth')
    plt.show()

def figure_15():
    plt.close()
    x = np.arange(0.0, 7.0, 0.01)
    y1 = 0.49*x
    y2 = list(map(lambda x: 0 if x < 1 else (x-1)*0.6, x))
    ax = plt.gca()
    ax.plot(x, y1, 'lightcoral')
    ax.plot(x, y2, 'royalblue')
    ax.fill_between(x, y1, y2, where=y2 >= y1, facecolor='mistyrose', interpolate=True)
    ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='skyblue', interpolate=True)
    ax.set_title('Figure 15')
    ax.set_ylim(0,5)
    ax.set_xlim(0,7)
    ax.set_ylabel('Power')
    ax.set_xlabel('Wealth')
    plt.show()



if __name__ == "__main__":
    '''figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    figure_7()
    figure_8()
    figure_9()
    figure_10()
    figure_11()
    figure_12()
    figure_13()
    figure_14()
    figure_15()'''
    warontherocks_figure_1()
    warontherocks_figure_2()