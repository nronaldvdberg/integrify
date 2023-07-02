from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
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

def show_column_info(df, show_centrality_and_dispersion=False, string_examples=True):
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
    if string_examples:
        # for string columns, show the top-5 most frequent string, to give an impression of the column contents
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            top_counts = df[col].value_counts().nlargest(5)
            print(f"\nTop 5 strings in column '{col}':")
            for i, (the_string, the_count) in enumerate(top_counts.items(), start=1):
                clipped_string = (the_string[:60] + '...') if len(the_string) > 60 else the_string
                print(f"{i}. {clipped_string} (count={the_count})")

def create_column_histograms(df, hist_tail_cut=5):
    '''this function prints a summary of the provided df'''
    num_cols = df.select_dtypes(include=['number']).columns
    fig_n_cols = 3
    fig_n_rows = int(np.ceil(len(num_cols) / fig_n_cols))
    fig, axs = plt.subplots(fig_n_rows, fig_n_cols, figsize=(15, fig_n_rows*5))
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
    fig, axs = plt.subplots(fig_n_rows, fig_n_cols, figsize=(15, fig_n_rows*5))
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
    # remove empty subplots (hide axes)
    for j in range(i, 18):  
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()
    
def plot_target_split_by_categorical(df, target):
    categorical_cols = [col for col in df.columns if df[col].nunique() <= 10 and col != target]
    cnt = len(categorical_cols)
    
    # Initialize the figure
    fig_n_cols = 3
    fig_n_rows = int(np.ceil(cnt / fig_n_cols))
    fig, axs = plt.subplots(fig_n_rows, fig_n_cols, figsize=(15, fig_n_rows*5))
    axs = axs.ravel()
    
    for i, col in enumerate(categorical_cols):
        grouped = df.groupby(col)[target].agg(['mean', 'sem'])
        
        # Set the bar positions
        x_positions = np.arange(len(grouped.index))
        axs[i].bar(x_positions, grouped['mean'], yerr=grouped['sem'], capsize=4, align='center')
        axs[i].set_title(f'{target} split by {col}')
        axs[i].set_xlabel(f'{col}')
        axs[i].set_ylabel(f'{target}')
        axs[i].set_xticks(x_positions)  # setting xticks to unique values
        axs[i].set_xticklabels(grouped.index.astype(str))

    # Remove empty subplots (hide axes)
    for j in range(i+1, fig_n_rows*fig_n_cols):
        fig.delaxes(axs[j])
        
    plt.tight_layout()
    plt.show()    
    
def create_column_boxplot(df, min_levels=10, separate_panels=False):
    df_selected = df.loc[:, df.nunique() >= min_levels]
    if separate_panels:
        fig, axs = plt.subplots(len(df_selected.columns), 1, figsize=(4, 3*len(df_selected.columns)))
        for i, col in enumerate(df_selected.columns):
            sns.boxplot(x=df_selected[col], ax=axs[i])
            axs[i].set_title(f'boxplot of {col}')
    else:
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        sns.boxplot(data=df_selected, ax=axs)
        axs.set_title('boxplots of all columns')
    plt.tight_layout()
    plt.show()

def create_corr_plot(df, target_col=None, threshold=0.1, show_full=True, take_abs=False):
    fsize = 10
    fig, ax = plt.subplots(figsize=(9, 7))
    correlation_matrix = shorten_column_names(df, 20).corr(numeric_only=True)
    if take_abs:
        correlation_matrix = correlation_matrix.abs()
    # if we have a target column, we only show correlations with the target column that are above the threshold
    if target_col is not None:
        low_corr = correlation_matrix[np.abs(correlation_matrix[target_col]) < threshold].index
        correlation_matrix.drop(low_corr, axis=0, inplace=True)
        correlation_matrix.drop(low_corr, axis=1, inplace=True)
    if correlation_matrix.isnull().all().all():
        print("All correlations with the target column are below the threshold, nothing to plot.")
    else:
        if show_full:
            heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": fsize})
        else:
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, ax=ax, annot_kws={"size": fsize})
        ax.set_xticklabels(ax.get_xticklabels(), size=fsize, rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), size=fsize, rotation=0)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fsize)
        if take_abs:
            ax.set_title("Spearman correlations (abs)")
        else:
            ax.set_title("Spearman correlations")
        plt.tight_layout()
        plt.show()

def perform_binary_classification_analysis(self, target_column, model_names = ['LogisticRegression'], metric = 'accuracy', cv_folds = 5, n_iter = 20, verbose_level = 1, show_warnings = False, show_plots = True):
    '''Perform classification analysis'''
    if not show_warnings:
        warnings.filterwarnings('ignore')
    # first check that the target column is binary
    if len(self.df[target_column].unique()) != 2:
        print(f"ERROR: target column {target_column} is not binary")
        return
    # next, check that the models in model_names are valid
    valid_models = ["LogisticRegression", "RandomForest", "GradientBoosting", "KNeighbors", "SVC"]
    for model_name in model_names:
        if model_name not in valid_models:
            print(f"ERROR: model {model_name} is not valid")
            return
    # next, check that the metric is valid
    valid_metrics = ["accuracy", "precision", "recall", "f1"]
    if metric not in valid_metrics:
        print(f"ERROR: metric {metric} is not valid")
        return
    # split into X and y
    X = self.df.drop([target_column], axis=1)
    y = self.df[target_column]
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # specify column transformer (leave binary columns untouched, scale continuous features, one-hot-encode non-numerical features)
    binary_cols = [col for col in X_train.columns if X_train[col].nunique() == 2]
    continuous_cols = [col for col in X_train.columns if col not in binary_cols]
    non_numerical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_cols),
            ('passthrough', 'passthrough', binary_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), non_numerical_cols)
        ])
    # define the model and parameter spaces
    for model_name in model_names:
        if model_name == 'LogisticRegression':
            model = LogisticRegression()
            param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000], 'classifier__penalty': ['l1', 'l2']}
        elif model_name == 'RandomForest':
            model = RandomForestClassifier()
            param_grid = {'classifier__n_estimators': [100, 200, 300, 400, 500], 'classifier__max_features': ['auto', 'sqrt', 'log2']}
        elif model_name == 'GradientBoosting':
            model = GradientBoostingClassifier()
            param_grid = {'classifier__n_estimators': [100, 200, 300, 400, 500], 'classifier__learning_rate': [0.001, 0.01, 0.1, 1, 10]}
        elif model_name == 'KNeighbors':
            model = KNeighborsClassifier()
            param_grid = {'classifier__n_neighbors': [3, 5, 7, 9, 11], 'classifier__weights': ['uniform', 'distance']}
        elif model_name == 'SVC':                
            model = SVC()
            param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000], 'classifier__kernel': ['linear', 'rbf']}
        # create a pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        # perform randomized search
        if verbose_level > 0:
            print(f"Performing randomized search for {model_name}...")
        search = RandomizedSearchCV(pipeline, param_grid, cv=cv_folds, scoring=metric, n_iter=n_iter, verbose=verbose_level)
        search.fit(X_train, y_train)
        # print the results
        if verbose_level > 0:
            print(f"Best parameter (CV score={search.best_score_:.3f}):")
            print(search.best_params_)
            # print the test score
            print(f"Test score: {search.score(X_test, y_test):.3f}")
        # print the classification report
        y_pred = search.predict(X_test)
        if verbose_level > 0:
            print(classification_report(y_test, y_pred))
            # print the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix: \n{cm}")
        if verbose_level > 0:
            print("\n") 
        # plot ROC curve
        if show_plots:
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                RocCurveDisplay.from_estimator(search, X_test, y_test, ax=ax)
                ax.plot([0, 1], [0, 1], linestyle='-', lw=2, color='k', alpha=.8)
                ax.set_title(f"ROC curve for {model_name}")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")                    
                plt.show()
            except:
                print("ERROR: could not plot ROC curve")
    # reset warnings
    warnings.filterwarnings('default')


