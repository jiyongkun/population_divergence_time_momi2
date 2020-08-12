#coding:UTF-8
"""
Author: Yongkun Ji
"""
import argparse
import sys,os
import pickle
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import momi
import time
from scipy import stats
from multiprocessing import Process
import pandas as pd
#export OMP_NUM_THREADS=10

parser = argparse.ArgumentParser( description='momi2')
parser.add_argument('-i', '--input_fold', type=str, required=True, help= 'input fold')
args = parser.parse_args()

def fig_plot(model,group,dir,model_name):
    fig=momi.DemographyPlot(
        model, ["Central",group],
        figsize=(6,8),
        major_yticks=[1e3,3e3,5e3,7e3,9e3,1e4,3e4,5e4,7e4,9e4,1e5,3e5,5e5,7e5,9e5,1e6],
        linthreshy=1e4)
    plt.savefig(dir+"/"+group+"/"+group + "_"+model_name+".svg",format="svg",bbox_inches='tight')
    plt.savefig(dir+"/"+group+"/"+group + "_"+model_name+".png",format="png",bbox_inches='tight')
    plt.close()
def model_optimize(model,total_runs,model_name,group,dir,no_pulse_model=0): #模型最优化函数
    results = []
    model_paramdict = {"log_likelihood":[]} #用于归集各个run结果的似然值及参数的字典
    n_runs = 1
    while n_runs <= total_runs:
        print(group+": Starting run "+ str(n_runs) + " out of " + str(total_runs))
        if no_pulse_model == 0:
            model.set_params(randomize=True)
        else:
            model.set_params(no_pulse_model.get_params(),randomize=True)
        try:
            result = model.optimize(options={"maxiter":400}
                                            #,method="L-BFGS-B"
                                         )
        except Exception as e:
            print(group+': Error:',e)
            traceback.print_exc()
            result.success = False
            continue
        print(result.success)
        if result.success == True:
            results.append(result)
            model_paramdict["log_likelihood"].append(result.log_likelihood)
            for key in result.parameters:
                if key in model_paramdict:
                    model_paramdict[key].append(result.parameters[key])
                else:
                    model_paramdict[key] = [result.parameters[key]]
            n_runs += 1
        else:
            print(group+": Not Converged! " + result.message)
    f = open(dir+"/"+group+"/"+group+ "_" + model_name +"_results.csv","w")
    pd.DataFrame(model_paramdict).to_csv(f)
    f.close()
    #  sort results according to log likelihood, pick the best one
    best_result = sorted(results, key=lambda r: r.log_likelihood, reverse = True)[0]
    return best_result
    
def run_proc(dir,group): #多进程用函数
    print("Start "+group)
    print('Run child process %s (%s)...' % (group, os.getpid()))
    sfs = momi.Sfs.load(dir + "/" +
                        group +
                        "/chroms/sfs.gz")
    no_pulse_model = momi.DemographicModel(N_e=1e5, gen_time=1, muts_per_gen=5.27e-9)
    no_pulse_model.set_data(sfs)
    no_pulse_model.add_size_param("n_Central",
                                    lower=100,
                                    upper=1e7)
    no_pulse_model.add_size_param("n_"+group,
                                    lower=100,
                                    upper=1e7)
    no_pulse_model.add_size_param("n_ancestor",
                                    lower=100,
                                    upper=1e7)
    no_pulse_model.add_leaf("Central", N="n_Central")

    no_pulse_model.add_leaf(group, N="n_"+group)

    no_pulse_model.add_time_param("t_"+group+"_Central", 
                                lower=100,
                                upper=1e7
                              )
    no_pulse_model.move_lineages(group, "Central", t="t_"+group+"_Central",
                                  N="n_ancestor")

    log_likelihoods = []
    print("Start running the no_pulse_model:")
    best_result = model_optimize(model=no_pulse_model,
                                    total_runs=10,
                                    model_name="no_pulse_model",
                                    group=group,
                                    dir=dir)
    best_results = []
    best_results.append(best_result)
    print("Best result:")
    print(best_result)
    no_pulse_model.set_params(best_result.parameters)
    log_likelihoods.append(best_result.log_likelihood)
    fig_plot(no_pulse_model,group,dir,"no_pulse_model")

    ### set add_pulse_model1 ###
    add_pulse_model = no_pulse_model.copy()
    add_pulse_model.add_pulse_param("p_pulse_Central_"+group, upper=1.0)
    add_pulse_model.add_pulse_param("p_pulse_"+group+"_Central", upper=1.0)
    add_pulse_model.add_time_param("t_pulse_Central_"+group, upper_constraints=["t_"+group+"_Central"])
    add_pulse_model.move_lineages("Central", group, t="t_pulse_Central_"+group, p="p_pulse_Central_"+group)
    add_pulse_model.move_lineages(group, "Central", t="t_pulse_Central_"+group, p="p_pulse_"+group+"_Central")
    print("Start running the add_pulse_model1:")
    best_result = model_optimize(model=add_pulse_model,
                                    total_runs=100,
                                    model_name="add_pulse_model1",
                                    group=group,
                                    dir=dir,
                                    no_pulse_model=no_pulse_model)
    best_results.append(best_result)
    add_pulse_model.set_params(best_result.parameters)
    fig_plot(add_pulse_model,group,dir,"add_pulse_model1")
    log_likelihoods.append(best_result.log_likelihood)
    
    ### set add_pulse_model2 ###
    add_pulse_model2 = no_pulse_model.copy()
    add_pulse_model2.add_pulse_param("p_pulse_Central_"+group, upper=1.0)
    add_pulse_model2.add_pulse_param("p_pulse_"+group+"_Central", upper=1.0)
    add_pulse_model2.add_time_param("t_pulse_Central_"+group, upper_constraints=["t_"+group+"_Central"])
    add_pulse_model2.add_size_param("n_central_re",
                                    lower=10,
                                    upper=1e7)
    add_pulse_model2.add_size_param("n_"+group+"_re",
                                    lower=10,
                                    upper=1e7)
    add_pulse_model2.move_lineages("Central", group, t="t_pulse_Central_"+group, p="p_pulse_Central_"+group,
                                    N="n_"+group+"_re")
    add_pulse_model2.move_lineages(group, "Central", t="t_pulse_Central_"+group, p="p_pulse_"+group+"_Central",
                                     N="n_central_re")
    print("Start running the add_pulse_model2:")
    best_result = model_optimize(model=add_pulse_model2,
                                    total_runs=100,
                                    model_name="add_pulse_model2",
                                    group=group,
                                    dir=dir,
                                    no_pulse_model=no_pulse_model)
    best_results.append(best_result)
    add_pulse_model2.set_params(best_result.parameters)
    fig_plot(add_pulse_model2,group,dir,"add_pulse_model2")
    log_likelihoods.append(best_result.log_likelihood)
   
   #保存变量到dump文件
    f=open(dir+"/"+group+"/"+group+"_best_results.dump","wb")
    pickle.dump(best_results, f)
    f.close()
    

    LR = 2*(log_likelihoods[2]-log_likelihoods[1])
    p_value = stats.chi2.sf(LR,2)
    lr_dict = {"no_pulse_model":log_likelihoods[0],
            "add_pulse_model1":log_likelihoods[1],
            "add_pulse_model2":log_likelihoods[2],
            "LR":LR,
            "p":p_value}
    f = open(dir+"/"+group+"/"+group+"_LR.csv","w")
    pd.DataFrame(lr_dict,index=[0]).to_csv(f,index=False)
    f.close()
    print(' child process %s (%s) done.' % (group, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    groups=["Central",
            #"Malay",
            #"Hainan","Bomi",
            #"Northeast"
            #,"Aba","Qinghai","Taiwan",
            "Kash_Paki"
            ]
    p_obj = [] #Store sub processes instance
    for group in groups[1:]:
        p = Process(target=run_proc, 
                        args=(args.input_fold,group))
        print('Child process will start.')
        p.start()
        p_obj.append(p)
        time.sleep(10)
    for p in p_obj:
        p.join() #waiting for all the sub processes
    print("All done!")
