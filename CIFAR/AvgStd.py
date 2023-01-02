
import os
import pandas as pd
import numpy as np
import re


def _get_one_table(log_path):
    print('log_path', log_path)    
    f = open(log_path, 'r')
    lines = f.readlines()
    i = 0
    for one_line in lines:
        if 'Table' in one_line:
            # print('Table:', lines[i:i+33])
            TableText = lines[i:i+32]
            arch = TableText[2].split('_Tra')[0]
            break
        i = i+1
    return TableText, arch

def _get_table_from_csv(log_path):
    # print('log_path:', log_path)   
    df = pd.read_csv(log_path) 
    # print('df', df)
    arch = df['Model'][1]
    return df, arch

def _get_acc(file):
    f = open(data_path+file, 'r')
    line = f.readlines()[-4:]
    test_acc = line[0].strip('\n').split('Accuracy: ')[1]
    train_acc = line[1].strip('\n').split('Accuracy: ')[1]
    test_noise_acc = line[2].strip('\n').split('Accuracy_pertur: ')[1]
    train_noise_acc = line[3].strip('\n').split('Accuracy_pertur: ')[1]
    return float(test_acc), float(train_acc), float(test_noise_acc), float(train_noise_acc)


def get_coefficient(file_key):
    split = file_key.split('_')
    model = split[0]
    a0 = split[2]
    a1 = split[3]
    a2 = split[4]
    b0 = split[5]
    noise = split[-1]
    return model, float(a0), float(a1), float(a2), float(b0), float(noise)


def _get_train_test_detail(model, a0, a1, a2, b0, noise, root, stab, files):

    test_accs = []
    train_accs = []
    test_noise_accs = []
    train_noise_accs = []

    for file in files:
        test_acc, train_acc, test_noise_acc, train_noise_acc = _get_acc(file)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        test_noise_accs.append(test_noise_acc)
        train_noise_accs.append(train_noise_acc)

    test_mean, test_std = np.mean(test_accs), np.std(test_accs, ddof=1)
    train_mean, train_std = np.mean(train_accs), np.std(train_accs, ddof=1)
    test_noise_mean, test_noise_std = np.mean(
        test_noise_accs), np.std(test_noise_accs, ddof=1)
    train_noise_mean, train_noise_std = np.mean(
        train_noise_accs), np.std(train_noise_accs, ddof=1)

    return [model, a0, a1, a2, b0, noise, root, stab,
            round(test_mean, 2), round(test_std, 2),
            round(test_noise_mean, 2),
            round(test_noise_std, 2),
            round(train_mean, 2), round(train_std, 2),
            round(train_noise_mean, 2), round(train_noise_std, 2)]


def file2dic(data_path):
    file_dic = {}
    for file_name in os.listdir(data_path):
        # if '.py' in file_name or '.csv' in file_name:
        #     continue
        if not 'log.txt' in file_name:
            continue        
        file_key = file_name.split('_tag', 1)[0]
        if not file_dic.get(file_key):
            file_dic[file_key] = [file_name]
        else:
            file_dic.get(file_key).append(file_name)
    return file_dic


def get_root_stab(a0, a1, a2, b0):
    file_path = '/media3/clm/HighOrderNetStructure/robust_dychf/known.xlsx'
    df = pd.DataFrame(pd.read_excel(file_path, sheet_name='c100'))
    for index, row in df.iterrows():
        if a0 == float(row['a_0']) and a1 == float(row['a_1']) and a2 == float(row['a_2']) and b0 == float(row['b_0']):
            stability = row['Stability']
            root1 = round(float(row['root1']), 2)
            root2 = round(float(row['root2']), 2)
            root3 = round(float(row['root3']), 2)
            return str(root1)+', '+str(root2)+', '+str(root3), stability

    return None, None

def get_log_path(file_dir):   
    File = 'Not found'
    dirs = 'Not found'
    d = []
    for root, dirs, files in os.walk(file_dir):  
        root = root
        dirs = dirs
        files = files
        if 'exp' in dirs:
            print('dirs', dirs)
            d = d.append([dirs])
    for f in files:
        if '.csv' in f:
            File = f
    
    return File, d

def get_csv_path(file_dir):   
    ff = []
    dirs = 'Not found'
    dd = []
    for root, dirs, files in os.walk(file_dir):  
        root = root
        dirs = dirs
        files = files
        print('root, dirs, files', root, dirs, files)
        
        # for d in dirs:        
        #     if 'exp' in dirs:
        #         dd.append(d)
        # for f in files:
        #     if '.csv' in files:
        #         ff.append(f)
    
    return ff, dd

def traverse_dir_files(root_dir, ext=None, is_sorted=True):

    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  
                continue
            if ext:  
                if name.endswith(tuple(ext)) and 'Ablation' in name:
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    if not names_list:  
        return paths_list, names_list
    # if is_sorted:
    #     paths_list, names_list = sort_two_list(paths_list, names_list)
    return paths_list, names_list

def not_empty(s):
    return s and s.strip()

    # print(list(c))
if __name__ == '__main__':

    data_path = '/media/bdc/clm/OverThreeOrders/CIFAR/runs/cifar10/ZeroSNet/'#'/media/bdc/clm/OverThreeOrders/CIFAR/runs/mnist/ZeroSNet/'
    # '/media/bdc/clm/OverThreeOrders/CIFAR/runs/cifar100/ZeroSNet/'
    # '/media/bdc/clm/OverThreeOrders/CIFAR/runs/cifar100/ZeroSNet/'
    # '/media/bdc/clm/OverThreeOrders/CIFAR/runs/cifar100/ZeroSNet/WithRobSGD_noLS_adv/'
    # '/media/bdc/clm/OverThreeOrders/CIFAR/runs/adv_train_runseps0.031/cifar100/ZeroSNet/WithRobSGD_noLS_adv/'
    
    # "/media/bdc/clm/OverThreeOrders/CIFAR/runs/cifar10/ZeroSNet/WithRobSGD_noLS_adv/"
    # 'media_HDD_1/lab415/clm/OverThreeOrders/OverThreeOrders/CIFAR/runs/cifar10/ZeroSNet/'    
    type = 'C10_NoAdvFromCsv_AblasCQ3090'
    NoiseType = [
    'PGD',
    'FGM',
    'PGD',
    'FGM',
    'PGD',
    'FGM',
    'PGD',
    'FGM',
    'const',
    'const',
    'const',
    'const',
    'const',
    'const',
    'const',
    'const',
    'const',
    'randn',
    'randn',
    'randn',
    'randn',
    'randn',
    'rand',
    'rand',
    'rand',
    'rand',
    'rand',
    'rand',
    'rand',
    'rand',
    'rand',
    'rand',
    ]
    # file_dic = file2dic(data_path)
    data_arr = []
    first = 1
    paths_list, names_list = traverse_dir_files(data_path, ext='csv')
    print(' paths_list',  paths_list)
    for path in paths_list:
        if not 'ablation' in path:
            continue
        log_path = os.path.split(path)[0]+'/log.txt'
        print('log_path', log_path)
        f = open(log_path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if 'flops, params, trainable_params' in line:
                # print('line.split(' ')', line.split(' '))
                flops = line.split(' ')[-1]
                params = line.split(' ')[-2]
                trainable_params = line.split(' ')[-3]
                print('flops, params, trainable_params', flops, params, trainable_params)
        # test_acc = line[0].split('Accuracy: ')[1]        
        print('log_path', log_path)
        
        TableText, arch = _get_table_from_csv(path)
        
        ablation = path.split('ablation')[1].split('20220')[0]
        TableText.insert(1,'Ablation',ablation)
        TableText.insert(TableText.shape[1], '# FLOPs', float(flops))
        TableText.insert(TableText.shape[1], '# Params', float(params))
        TableText.insert(TableText.shape[1], '# Trainable Params', float(trainable_params))
        if first == 1:
                data_arr = TableText
        else:
            data_arr = pd.merge(data_arr, TableText, how = 'outer')
        first = 0     
    # print(data_arr)
    df = pd.DataFrame(data_arr)#, columns=['Model', "Step", "Order", "...", 'Noise 
    # csv_name, dirs = get_csv_path(data_path)
    # print('csv_name, dirs:',csv_name, dirs)
    df['Order'] = df['Order'].map(lambda s: 'Order = '+str(s))
    df['Step'] = df['Step'].map(lambda s: str(s)+' step')
    df['Ablation'] = df['Ablation'].replace(['mnistConvStride2Learn',
                                             'mnistConvStride2ResLikeShare',
                                             'mnistAllEle',
                                             'mnistOri',
                                             'mnistConvStride2Share',
                                             'mnistConvStride2ResLike'],['Conv',
                                                                         'BN-ReLU-conv, sharing',
                                                                         'Element transfering',
                                                                         'Pool-conv, Dirac',
                                                                         'Conv, sharing',
                                                                         'BN-ReLU-conv'])
    
    # .map({'mnistConvStride2ResLikeShare': 'BN-ReLU-conv, sharing'})
    # df['Ablation'] = df['Ablation'].map({'mnistAllEle': 'All elements'})
    # df['Ablation'] = df['Ablation'].map({'mnistOri': 'Pool-conv'})
    # df['Ablation'] = df['Ablation'].map({'mnistConvStride2Share': 'Conv, sharing'})
    # df['Ablation'] = df['Ablation'].map({'mnistConvStride2ResLike': 'BN-ReLU-conv'})

    
    # df.loc[df['Order'] == 1]['Order'] = '1st Order'
    
    print('df', df )
    
    save_path = data_path+'csv/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_excel(save_path+type+'.xlsx')   
            
    # for file_name in os.listdir(data_path):
    #     # csv_name, dirs = get_csv_path(log_path)

    #     file_key = file_name.split('#2022')[0]
    #     log_path = data_path+file_name
    #     # if 'adv_train' in file_key:
    #     #     continue
    #     if not '/csv' in log_path:
            
    #         # print('log_path:', log_path)
    #         # csv_name, dirs = get_log_path(log_path)
        
    #         # print('csv_name, dirs',csv_name, dirs)
    #         if '.csv' in csv_name and 'exp' in dirs:
    #             TableText, arch = _get_table_from_csv(log_path+'/'+csv_name)
    #             print('arch', arch)
    #             # print('TableText', TableText)

    #             # for i in range(len(TableText)):
    #             #     tmp_arrs = TableText[i].split(' ')
    #             #     tmp_arrs = list(filter(not_empty,tmp_arrs))[1:]
    #             #     for j in range(len(tmp_arrs)):
    #             #         tmp_arrs[j] = tmp_arrs[j].strip()

    #             #     tmp_arrs.insert( 4, NoiseType[i])
    #             #     while '' in tmp_arrs:
    #             #         tmp_arrs.remove('')
    #             if first == 1:
    #                  data_arr = TableText
    #             else:
    #                 data_arr = pd.merge(data_arr, TableText, how = 'outer')
    #             first = 0
    # print(data_arr)
    # df = pd.DataFrame(data_arr)#, columns=['Model', "Step", "Order", "...", 'Noise Type', 'Noise Value', 'Train Acc.', 'Test Acc.'])


     





    # for file_name in os.listdir(data_path):
    #     # if '.py' in file_name or '.csv' in file_name:
    #     #     continue
    #     # if not 'log.txt' in file_name:
    #     #     continue  
    #     file_key = file_name.split('#2022')[0]
    #     log_path = data_path+file_name
    #     if 'adv_train' in file_key:
    #         continue
    #     log_exist = get_log_path(log_path)
    #     if 'log.txt' in log_exist:
    #         TableText, arch = _get_one_table(log_path+'/log.txt')
    #         for i in range(len(TableText)):
    #             tmp_arrs = TableText[i].split(' ')
    #             tmp_arrs = list(filter(not_empty,tmp_arrs))[1:]
    #             for j in range(len(tmp_arrs)):
    #                 tmp_arrs[j] = tmp_arrs[j].strip()

    #             tmp_arrs.insert( 4, NoiseType[i])
    #             while '' in tmp_arrs:
    #                 tmp_arrs.remove('')
    #             data_arr.append(tmp_arrs)
    # print(data_arr)
    # df = pd.DataFrame(data_arr)#, columns=['Model', "Step", "Order", "...", 'Noise Type', 'Noise Value', 'Train Acc.', 'Test Acc.'])
    # save_path = data_path+'csv/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # df.to_csv(save_path+type+'.csv')
        # print(file_key)
        # for same_name in os.listdir(data_path):
        #     search_key = same_name.split('#2022')[0]
            # if file_key == search_key:
                # print('file_name: ', file_name)
                # print('same_name: ', same_name)
        # tmp_arr        
        # data_arr.append(tmp_arr)

    # isTest = False

    # noise_type = ['randn', 'rand', 'const']
    # if isTest:
    #     data_root = "/media3/clm/HighOrderNetStructure/robust_dychf/test/"
    # else:
    #     data_root = "/media3/clm/HighOrderNetStructure/robust_dychf/allmodel/"
    # for type in noise_type:
    #     data_path = data_root + 'results_'+type+'/'
    #     file_dic = file2dic(data_path)
    #     data_arr = []
    #     for file_key in file_dic.keys():

    #         model, a0, a1, a2, b0, noise = get_coefficient(file_key)
    #         if a0 == 0 or a1 == 0 or a2 == 0 or b0 == 0:
    #             continue

    #         root, stab = get_root_stab(a0, a1, a2, b0)
    #         if root is None:
    #             continue

    #         files = file_dic.get(file_key)
    #         tmp_arr = _get_train_test_detail(
    #             model, a0, a1, a2, b0, noise, root, stab, files)
    #         data_arr.append(tmp_arr)

    #     df = pd.DataFrame(data_arr, columns=['Model', 'a0', 'a1', 'a2', 'b0', 'noise',
    #                                          'Moduli of roots', 'Stab.',
    #                                          'Test mean', 'Test STD',
    #                                          'Test noise mean', 'Test noise STD',
    #                                          'Train mean', 'Train STD',
    #                                          'Train noise mean', 'Train noise STD'])

    #     df = df.sort_values(by=['a0', 'a1', 'a2', 'b0', 'noise'])

    #     print(df)
    #     save_path = data_root+'csv/'
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)

    #     df.to_csv(save_path+type+'.csv')
