#WPT software simulator for Q Factor Calculation
import sympy as sp 
from PyLTSpice import SimRunner, SpiceEditor, LTspice,AscEditor, RawRead
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LT,RT = 5e-6,0.1
LR,RR = 5e-6,0.1

f = 25e3
omega = 2*3.14*f

def q_factor(f):
    QT = ((2*3.14*f))*LT/RT
    QR = ((2*3.14*f))*LR/RR
    Q = sp.sqrt(QT*QR)
    return Q

Q= q_factor(f)
print(Q)

q_data=[]
f_data=[]
for i in range(1,100,1):
    q_test = q_factor(f)
    if  q_test <= 110:
        f=f+25e3
        f_data.append(f)
        q_data.append(q_test)

CT_data=[]
CR_data=[]
K_data =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for j in f_data:
    omega = 2*3.14*j
    CT_data.append(1/((omega**2)*(LT)))
    CR_data.append(1/((omega**2)*(LR)))

data_param = {
    'Frequency': f_data,
    'Q':q_data,
    'CT': CT_data,
    'CR': CR_data,
}
df_param = pd.DataFrame(data_param)
print(df_param,len(df_param))

def processing_data(raw_filename, log_filename):
    #This is the function that will process the data from simulations
    print("Handling the simulation data of %s, log file %s" % (raw_filename, log_filename))
    s_split = str(raw_filename).split('_')
    Q_val =s_split[-2]
    s_split_dot = str(s_split).split('.')
    k_val = "0."+s_split_dot[-2]
    log_info = LTSpiceLogReader(log_filename)
    raw_info = RawRead(raw_filename)
    I = raw_info.get_trace("I(RB)")
    V = raw_info.get_trace("V(n004)")
    x = raw_info.get_trace('frequency')
    current = I.get_wave().tolist()
    voltage = V.get_wave().tolist()
    W = [abs(v*i) for i, v in zip(current, voltage)]
    frequency = x.get_wave()
    return current,voltage,W,frequency,k_val, Q_val

print(df_param['Frequency'].max())


asc_file = 'circuit/WPTSIM'
LTC = SimRunner(output_folder='./temp', simulator=LTspice, parallel_sims=len(df_param)-1)

netlist = SpiceEditor(asc_file)  # Open the Spice Model, and creates the .net
netlist.set_component_value('LT', LT)
netlist.set_component_value('LR', LR)
netlist.set_component_value('R1', RT)
netlist.set_component_value('R2', RR)
#netlist.set_component_value('CT', CT)
#netlist.set_component_value('CR', CR)
#netlist.set_parameter("FR", f)

print("Add instructions to netlist\n")

netlist.add_instructions(
    "; Simulation settings",
    ".ac lin 1000 1 "+str(df_param['Frequency'].max()*1.5)
)

#netlist.write_netlist("temp/wptnet.net")

for count, i in enumerate(df_param.values):
    current_F = df_param.loc[count, 'Frequency']
    current_CT= df_param.loc[count, 'CT']
    current_CR= df_param.loc[count, 'CR']
    current_Q = df_param.loc[count, 'Q']
    netlist.set_parameter("CT", current_CT)
    netlist.set_parameter("CR", current_CR)
    netlist.set_parameter("FR", current_F)
    for KV in K_data:
        netlist.set_parameter("KV", KV)
        run_netlist_file = "{}_{}_{}.net".format(netlist.netlist_file.name, int(current_Q),KV)
        LTC.run(netlist, run_filename=run_netlist_file, callback=processing_data)

LTC.wait_completion()

print('Successful/Total Simulations: ' + str(LTC.okSim) + '/' + str(LTC.runno))
print("success")

I_data = []
V_data = []
W_data = []
F_data = []
Kv_data = []
Qv_data = []

for current, voltage, W, frequency, k_val, Q_val in LTC:
    #print("The return of the callback "+str(count)+" function is ", current, voltage, W, frequency)
    I_data.append(current)
    V_data.append(voltage)
    W_data.append(W)
    F_data.append(frequency)
    Kv_data.append(k_val)
    Qv_data.append(int(Q_val))

data = {
    'Current': I_data,
    'Voltage': V_data,
    'Watt': W_data,
    'Frequency': F_data,
    'K':Kv_data,
    'Q':Qv_data
}

df_data = pd.DataFrame(data)
print(df_data)

kode_plot=1
F_max = []
P_max = []
K_max = []
Q_max = []
for k_value in (K_data):
    print(k_value)
    selected_rows = df_data[df_data['K'] == str(k_value)]
    selected_rows = selected_rows.reset_index().sort_values(by=['Q'])
    for count,i in  enumerate(selected_rows.values):
        x=abs(selected_rows.loc[count, 'Frequency'])
        y=selected_rows.loc[count, 'Watt']
        i_max_index = int(np.argmax(y))
        freq_max = x[i_max_index]
        power_max = y[i_max_index]
        label_Q=int(selected_rows.loc[count, 'Q'])
        plt.plot(x,y,label="Q:"+str(label_Q))
        plt.plot(x[i_max_index], y[i_max_index], color='black', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=3)
        #plt.text(x[i_max_index], y[i_max_index], str('P:%.2f' % y[i_max_index]), fontsize=6)
        #plt.text(x[i_max_index], y[i_max_index]-0.05, str('F:%.2f' % x[i_max_index]), fontsize=6)
        plt.title("K:"+str(k_value))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (Watt)")
        F_max.append(freq_max)
        P_max.append(power_max)
        K_max.append(k_value)
        Q_max.append(label_Q)
        #print("Q:",label_Q," Freq:",freq_max," Watt:",power_max, " K:",str(k_value) )
    plt.grid()
    plt.legend()
    plt.show() 

data_max = {
    'P_max': P_max,
    'F_max': F_max,
    'K_max': K_max,
    'Q_max': Q_max
}

df_data_max = pd.DataFrame(data_max)
with pd.ExcelWriter('compare2.xlsx') as writer: 
    df_data_max.to_excel(writer, sheet_name='main') 

PMAX = []
FMAX = []
KMAX = []
QMAX = []


for k_value in (K_data):
    #find_max_k = df_data_max[df_data['K_max'] == str(k_value)]
    selected = df_data_max.loc[df_data_max['K_max'] == k_value].reset_index()
    x = selected['Q_max']
    y = selected['P_max']
    i_max_index = int(np.argmax(y))
    plt.title("K:"+str(k_value))
    plt.xlabel("Q")
    plt.ylabel("Power (Watt)")
    plt.plot(x[i_max_index], y[i_max_index], color='black', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    #plt.text(x[i_max_index], y[i_max_index],str('P:%.2f' %y[i_max_index])+" W")
    maxlabel_P = str('P:%.2f' %y[i_max_index])+" W\n"
    maxlabel_Q = str('Q:'+str(x[i_max_index]))
    maxlabel_F = str('\nF:'+str(int(df_data_max['F_max'].iloc[i_max_index]))+" Hz")
    PMAX.append('P:%.2f' %y[i_max_index])
    QMAX.append(x[i_max_index])
    FMAX.append(int(df_data_max['F_max'].iloc[i_max_index]))
    KMAX.append(k_value)
    print("CT:", df_param['CT'].loc[df_param['Q'].astype(int)==x[i_max_index]])
    print("CR:", df_param['CR'].loc[df_param['Q'].astype(int)==x[i_max_index]])
    plt.text(x[i_max_index], y[i_max_index]-0.5, maxlabel_P+maxlabel_Q+maxlabel_F, fontsize = 16, bbox = dict(facecolor = 'red', alpha = 0.2))
    plt.plot(x,y,".",linestyle="dashdot")
    plt.grid()
    plt.legend()
    plt.show()

maxdata = {
    'PMAX': PMAX,
    'QMAX': QMAX,
    'FMAX': FMAX,
    'KMAX': KMAX
}
maxdatadf = pd.DataFrame(maxdata)

with pd.ExcelWriter('compare3.xlsx') as writer: 
    maxdatadf.to_excel(writer, sheet_name='main') 

with pd.ExcelWriter('compare.xlsx') as writer: 
    df_data.to_excel(writer, sheet_name='main') 
    df_param.to_excel(writer, sheet_name='C Pairs')