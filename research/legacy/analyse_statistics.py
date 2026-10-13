from matplotlib import lines
import matplotlib.pyplot as plt
import json
import os, sys
import numpy as np
import pandas as pd

from typing import List, Dict, Any

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def plot_statistics_category(categories:List[str]):
    dir_stats_files = 'outputs/statistics/' # 
    files_dir = os.listdir(dir_stats_files) # 
    
    files_of_interest = []
    for category in categories:
        files_of_interest_cat = []
        for file in files_dir:
            if file.startswith(category) and file.endswith('.json'):
                files_of_interest_cat.append(file)
        files_of_interest.append(files_of_interest_cat)
    
    for i in range(len(categories)):
        data_table = None         # 
        headers_table = []      # 
        category = categories[i]
        files_of_interest_cat = files_of_interest[i]
        
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        n_epochs = 0
        diff_colors = plt.get_cmap('tab10')
        
        for i_f, file in enumerate(files_of_interest_cat):
            file_path = os.path.join(dir_stats_files, file)
            data = load_json(file_path)
            
            hp_value = file.split('_')[-1]
            hp_value = hp_value.replace('.json', '')
            
            train_mean = data['train_mean'] # 
            train_std = data['train_std']   # 
            val_mean = data['val_mean']     # 
            val_std = data['val_std']       # 
            
            train_mean_lu = data['train_mean_lu']   #
            train_std_lu = data['train_std_lu']     #
            val_mean_lu = data['val_mean_lu']       # 
            val_std_lu = data['val_std_lu']         # 
            
            # :x8
            data = np.array([train_mean, train_std, val_mean, val_std, train_mean_lu, train_std_lu, val_mean_lu, val_std_lu]).T
            if data_table is None:
                data_table = data
            else:
                data_table = np.concatenate((data_table, data), axis=1)
            headers_table += [
                f'[{i_f}] Train (mean)', 
                f'[{i_f}] Train (std)', 
                f'[i_{i_f}f] Val (mean)', 
                f'[{i_f}] Val (std)', 
                f'[{i_f}] Train (lu, mean)', 
                f'[{i_f}] Train (lu, std)', 
                f'[{i_f}] Val (lu, mean)', 
                f'[{i_f}] Val (lu, std)'
            ]
            
            if n_epochs == 0:
                n_epochs = len(train_mean)
                epochs = range(n_epochs)
            
            # ax[0].plot(epochs, train_mean, label=f'Train {hp_value}', color=diff_colors(i_f))
            # # ax[0].fill_between(
            # #     epochs, 
            # #     [m - s for m, s in zip(train_mean, train_std)], 
            # #     [m + s for m, s in zip(train_mean, train_std)], 
            # #     color='blue', alpha=0.2
            # # )
            # ax[0].plot(epochs, val_mean, label=f'Val {hp_value}', color=diff_colors(i_f), linestyle='--')
            # ax[0].fill_between(
            #     epochs, 
            #     [m - s for m, s in zip(val_mean, val_std)], 
            #     [m + s for m, s in zip(val_mean, val_std)], 
            #     color=diff_colors(i_f), alpha=0.2
            # )
            
            # Save last 10 epochs
            ax[1].plot(epochs[-10:], train_mean[-10:], label=f'Train {hp_value}', color=diff_colors(i_f))
            # ax[1].fill_between(
            #     epochs[-5:], 
            #     [m - s for m, s in zip(train_mean[-5:], train_std[-5:])], 
            #     [m + s for m, s in zip(train_mean[-5:], train_std[-5:])], 
            #     color='blue', alpha=0.2
            # )
            ax[1].plot(epochs[-10:], val_mean[-10:], label=f'Val {hp_value}', color=diff_colors(i_f), linestyle='--')
            ax[1].fill_between(
                epochs[-10:], 
                [m - s for m, s in zip(val_mean[-10:], val_std[-10:])], 
                [m + s for m, s in zip(val_mean[-10:], val_std[-10:])], 
                color=diff_colors(i_f), alpha=0.2
            )
            
            # Save last 10 epochs            
            ax[0].plot(epochs[-10:], val_mean_lu[-10:], label=f'Val {hp_value}', color=diff_colors(i_f), linestyle='--')
            ax[0].fill_between(
                epochs[-10:], 
                [m - s for m, s in zip(val_mean_lu[-10:], val_std_lu[-10:])], 
                [m + s for m, s in zip(val_mean_lu[-10:], val_std_lu[-10:])], 
                color=diff_colors(i_f), alpha=0.2
            )
        
        pd.DataFrame(data_table).to_excel(f'outputs/statistics/{category}.xlsx', header=headers_table)
            
        # set the title and labels
        ax[0].set_title(f'{category} Statistics')
        ax[0].set(
            xlabel='Epochs',
            ylabel='Accuracy'
        )
        ax[0].legend()
        ax[0].grid()
        
        ax[1].set_title(f'Last 10 Epochs of {category} Statistics')
        ax[1].set(
            xlabel='Epochs',
            ylabel='Accuracy'
        )
        ax[1].legend()
        ax[1].grid()
        
        # save the figure
        fig.savefig(f'outputs/statistics/{category}.png')
        plt.close(fig)
        
        print(f'Plot saved for {category} statistics.')
    
if __name__ == "__main__":
    categories = ['n_unfolds'] # ['eps', 'initializer', 'lr', 'nunfolds]
    plot_statistics_category(categories)