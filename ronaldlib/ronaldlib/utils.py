import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
import os

def shorten_column_names(df, max_length=10):
    '''this function shortens long column names of the provide df -- used for visualization purposes'''
    df_temp = df.copy()
    df_temp.columns = [col[:max_length - 3] + '...' if len(col) > max_length else col for col in df_temp.columns]
    return df_temp

def load_data(filename, resave_as_pickle=True, save_xlsx_preview=False):
    '''load the specified datafile and return it as a pandas dataframe'''
    # add .csv if filename has no extension
    if '.' not in filename:
        filename += '.csv'
    # try load data from pickle file; if it does not exist, try load from .csv file; if that also fails, show an error
    loading_start_time = time.time() 
    try:
        # load pickle file
        print(f"loading data from pickle file...")
        df = pd.read_pickle(filename.replace('.csv','.pkl'))
        print(f"data loaded (took {time.time() - loading_start_time:.1f} seconds)")
    except FileNotFoundError:
        try:
            print("  -> no pickle file found")
            print("loading data from csv file instead (can take a while)...")
            df = pd.read_csv(filename)
            print(f"  -> data loaded (took {time.time() - loading_start_time:.1f} seconds)")
            # resave as .pkl file - this loads around 7 times faster on my machine            
            if resave_as_pickle:
                print(f"  -> resaving it as .pkl file (for faster loading in the future)")
                df.to_pickle(filename.replace('.csv','.pkl'))
            # save a preview of the data in an excel file, with just 100 rows 
            if save_xlsx_preview:
                print(f"  -> saving a preview of the data as {filename.replace('.csv','_preview.xlsx')}")
                df.head(100).to_excel(filename.replace('.csv','_preview.xlsx'), index=False)
        except FileNotFoundError:
            print(f"\nERROR: data '{filename}' does not exist.\nCheck if the file exists and if Jupyter is in the correct working directory")
    return df    

def show_column_info(df, show_centrality_and_dispersion=False):
    # build and show a summary of the data (shorten long column names to avoid wrapping in the visualization)
    info_df = pd.DataFrame(index=shorten_column_names(df, 14).columns)
    info_df['COLUMN'] = info_df.index
    info_df['TYPE'] = df.dtypes.values
    info_df['VALID CNT'] = df.count().values
    info_df['MISSING CNT'] = df.isnull().sum().values
    info_df['UNIQUE CNT'] = df.nunique().values
    info_df = info_df.sort_values(['TYPE', 'UNIQUE CNT'])
    info_df = info_df.reset_index(drop=True)
    print(info_df)
    # show central tendency measures
    if show_centrality_and_dispersion:
        print("\n### Measures of centrality and dispersion ###")
        print(shorten_column_names(df, 14).describe().transpose())

def create_column_histograms(df, hist_tail_cut=5):
    '''this function prints a summary of the provided df'''
    # histograms
    num_cols = df.select_dtypes(include=['number']).columns
    fig_n_cols = 3
    fig_n_rows = int(np.ceil(len(num_cols) / fig_n_cols))
    fig, axs = plt.subplots(fig_n_rows, fig_n_cols, figsize=(15, 30))
    axs = axs.ravel()
    for i, col in enumerate(num_cols):
        lower = np.percentile(df[col].dropna(), hist_tail_cut/2)
        upper = np.percentile(df[col].dropna(), 100-hist_tail_cut/2)
        filtered_data = df[(df[col] >= lower) & (df[col] <= upper)][col]
        n_unique_values = filtered_data.nunique()
        if np.issubdtype(filtered_data.dtype, np.integer) or n_unique_values <= 10:  
            value_counts = filtered_data.value_counts(normalize=True).sort_index()
            bar_width = 0.8 / len(value_counts)  
            axs[i].bar(value_counts.index.astype(str), value_counts, width=bar_width, align='center')
        else:
            bins = 30
            sns.histplot(filtered_data, bins=bins, ax=axs[i])
        if n_unique_values > 10:            
            axs[i].set_title(f'central {100-hist_tail_cut}% of {col}')
        else:
            axs[i].set_title(f'{col}')
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()

def plot_target_var_split_by_feature_level(df, target_col, max_levels=10, ylims=(None, None)):
    '''this function plots the target variable split by each level of each feature'''
    # count number of cols to plot
    cnt = 0
    for col in df.columns:
        if df[col].nunique() <= max_levels and col != target_col:
            cnt += 1
    # initialize the figure
    fig_n_cols = 3
    fig_n_rows = int(np.ceil(cnt / fig_n_cols))
    fig, axs = plt.subplots(fig_n_rows, fig_n_cols, figsize=(15, 30))
    axs = axs.ravel()
    i = 0 
    for col in df.columns:
        if df[col].nunique() <= max_levels and col != target_col:
            grouped = df.groupby(col)[target_col].agg(['mean', 'count'])
            grouped['std_error'] = np.sqrt(grouped['mean'] * (1 - grouped['mean']) / grouped['count'])
            
            # Set the bar positions
            x_positions = np.arange(len(grouped.index))
            axs[i].bar(x_positions, grouped['mean'], yerr=grouped['std_error'], 
                       capsize=4, align='center')
            axs[i].set_title(f'{target_col} split by {col}')
            axs[i].set_xlabel(f'{col}')
            axs[i].set_ylabel(f'{target_col}')
            axs[i].set_xticks(x_positions)  # setting xticks to unique values
            axs[i].set_xticklabels(grouped.index.astype(str))
            if ylims[0] is not None:
                axs[i].set_ylim(ylims)
            i += 1
    for j in range(i, 18):  
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()

def create_column_boxplot(df, min_levels=10, separate_panels=False):
    df_selected = df.loc[:, df.nunique() >= min_levels]
    if separate_panels:
        fig, axs = plt.subplots(len(df_selected.columns), 1, figsize=(8, 4*len(df_selected.columns)))
        for i, col in enumerate(df_selected.columns):
            sns.boxplot(x=df_selected[col], ax=axs[i])
            axs[i].set_title(f'boxplot of {col}')
    else:
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        sns.boxplot(data=df_selected, ax=axs)
        axs.set_title('boxplots of all columns')
    plt.tight_layout()
    plt.show()


def create_corr_plot(df, target_col=None, threshold=0.1):
    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    correlation_matrix = shorten_column_names(df, 20).corr()
    # if we have a target column, we only show correlations with the target column that are above the threshold
    if target_col is not None:
        low_corr = correlation_matrix[np.abs(correlation_matrix[target_col]) < threshold].index
        correlation_matrix.drop(low_corr, axis=0, inplace=True)
        correlation_matrix.drop(low_corr, axis=1, inplace=True)
    if correlation_matrix.isnull().all().all():
        print("All correlations with with target column are below the threshold, nothing to plot.")
    else:
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, ax=axs[0], annot_kws={"size": 18})
        axs[0].set_xticklabels(axs[0].get_xticklabels(), size=18, rotation=90)
        axs[0].set_yticklabels(axs[0].get_yticklabels(), size=18, rotation=0)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        axs[0].set_title("Spearman correlations")
        axs[1].text(0.5, 0.5, "observations", ha='left', va='center', wrap=True)
        axs[1].axis('off')  
        plt.tight_layout()
        plt.show()    

def set_openai_api_key(key=None):
    '''this function sets the OPENAI_API_KEY environment variable'''
    import os
    if key is None:
        # load from C:/my_openai_api_key.txt
        try:
            with open('C:/my_openai_api_key.txt', 'r') as f:
                key = f.read()
        except FileNotFoundError:
            print("ERROR: could not find C:/my_openai_api_key.txt")
    else:
        os.environ['OPENAI_API_KEY'] = key
        print("OpenAI API key set successfully")