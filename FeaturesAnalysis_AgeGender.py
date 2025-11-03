import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from matplotlib.lines import Line2D

#Reading csv file containing features information
df_features = pd.read_csv("path/to/features.csv")
df_patient_info = pd.read_csv("path/to/patient_info.csv")
df_features['KID'] = df_features['KID'].str.replace('_abdomen_selected', '')# unifiying the IDs between features and patient's dataframes

#Create the IDs list based on the fetaures file IDs list 
list_IDs=[]
list_IDs=df_features['KID']


#Select only the needed column in the patient info file 
df_patient_info= df_patient_info[['Anonymized_IDs','AGE', 'SEX','Image_item']]

# Convert the 'Anonymized_IDs' column to string and apply zfill
df_patient_info['Anonymized_IDs'] = df_patient_info['Anonymized_IDs'].astype(str).str.zfill(8)

#Rename all the cases in the patient info dataframe to match the form in the features dataframe
prefix_2 = 'OSU^A00^'
df_patient_info['Anonymized_IDs'] = df_patient_info['Anonymized_IDs'].apply(lambda x: prefix_2 + str(x))

# Merging patient info with features dataframe
df_patient_info.rename(columns={'Anonymized_IDs': 'KID'}, inplace=True)
df_features_gender_age= pd.merge(df_features,df_patient_info,how='left', on='KID')

# Calculate the number of cases, females, and males
num_cases = len(df_features_gender_age)
num_f= (df_features_gender_age['SEX'] == 'F').sum()
num_m= (df_features_gender_age['SEX'] == 'M').sum()

df_patient_info['age_group'] = (df_patient_info['AGE'] / 10).apply(np.floor)

# Adjust the age groups
df_patient_info['adjusted_age_group'] = df_patient_info['age_group'].apply(lambda x: 0 if x in [0, 1, 2] else x - 2)

age_group_mapping = {
    0: '00->29', 1: '30->39', 2: '40->49', 3: '50->59',
    4: '60->69', 5: '70->79', 6: '80->89', 7: '90->99'
}

# Apply the mapping to adjust age groups
df_patient_info['adjusted_age_group'] = df_patient_info['adjusted_age_group'].map(age_group_mapping)

# Merge the DataFrames on 'KID'
df_features_gender_age = pd.merge(df_features_gender_age, df_patient_info[['KID', 'adjusted_age_group']], how='left', on='KID')
age_group_order = ['00->29', '30->39','40->49', '50->59', '60->69', '70->79', '80->89'] #ignoring  '90->99' in the plots

# Path to the combined CSV file
combined_csv_path = 'path/to/uncertainty.csv'

# Read the combined CSV file
data = pd.read_csv(combined_csv_path)

# Calculate the 95th percentile of the average uncertainty
percentile_95 = data['Average'].quantile(0.95)

# Filter the DataFrame to select rows with average uncertainty >= 95th percentile
selected_cases = data[data['Average'] >= percentile_95]


case_ids_to_drop = selected_cases['ID'].tolist()

# Remove the suffix "_abdomen_selected" and clean case IDs by stripping whitespace and converting to a consistent case
case_ids_to_drop = [case_id.replace('_abdomen_selected', '').strip() for case_id in case_ids_to_drop]


# Check if case IDs in 'KID' column match those in case_ids_to_drop
matching_ids = df_features_gender_age[df_features_gender_age['KID'].isin(case_ids_to_drop)]

# Drop rows that contain case IDs from the list
df_features_gender_age = df_features_gender_age[~df_features_gender_age['KID'].isin(case_ids_to_drop)]

df_features_gender_age = df_features_gender_age[df_features_gender_age['adjusted_age_group'] != '90->99'] 

rc('font', family='Times New Roman')

# Define significance threshold
alpha = 0.05/(5*7) #5 muscles*7 age groups
#Second group
Structures_short_list = [
              'Rectus abdominis',
              'Psoas major',
              'Erector spinae', 
              'Quadratus lumborum',  
              'Latissimus Dorsi']

# Set up the figure with subplots
fig, axs = plt.subplots(nrows=4, ncols=len(Structures_short_list), figsize=(8 * len(Structures_short_list), 20), sharex=True)

# Features to plot and their corresponding units
features = ['mean_HU', 'volume', 'lean_mass', 'fat_ratio']
features_lbl = ['Mean HU', 'Volume', 'Lean mass', 'Fat ratio']

units = ['[HU]', '[CC]', '[g]', '[%]']

# Mean HU scale limits
min_mean_HU = -100
max_mean_HU = 100

# Create a boxplot for each feature of each structure and perform the t-test
for col, structure in enumerate(Structures_short_list):
    for row, (feature, featl_lbl, unit) in enumerate(zip(features, features_lbl, units)):
        feature_name = f"{structure}_{feature}"
        if feature_name in df_features_gender_age.columns:  # Check if the feature column exists
            sns.boxplot(x="adjusted_age_group", 
                        y=feature_name, 
                        data=df_features_gender_age, 
                        hue='SEX',  # Assuming 'SEX' is the gender column
                        order=age_group_order,
                        ax=axs[row, col],
                        showmeans=True,
                        palette='pastel')
            axs[row, col].set_title(f"{structure}",fontsize=18)
            axs[row, col].set_xlabel('Age groups', fontsize=18)
            axs[row, col].grid(True)
            axs[row, col].legend(loc='upper left', bbox_to_anchor=(1, 1))  
            axs[row, col].tick_params(axis='x', labelsize=18) 
            axs[row, col].tick_params(axis='y', labelsize=18)
            if feature == 'mean_HU':  
                axs[row, col].set_ylim(min_mean_HU, max_mean_HU)  # Set the y-axis limits for mean HU plots
            axs[row, col].set_ylabel(f"{featl_lbl} {unit}", fontsize=18)
            
            # Perform staistical test for each age group
            age_groups = df_features_gender_age['adjusted_age_group'].unique()
            for age_group in age_groups:
                #print(age_group)
                group_data = df_features_gender_age[df_features_gender_age['adjusted_age_group'] == age_group]
                females = group_data[group_data['性別'] == 'F'][feature_name]
                #print(females)
                males = group_data[group_data['性別'] == 'M'][feature_name]
                #print(males)
                #Calculate the p-value
                stat, p_value = mannwhitneyu(females, males)
                
                # Print p-value on the plot
                if p_value < alpha:
                    axs[row, col].text(age_group, axs[row, col].get_ylim()[1] * 0.9, '*', ha='center', fontsize=18, color='red')
                else:
                    axs[row, col].text(age_group, axs[row, col].get_ylim()[1] * 0.9, 'n.s', ha='center', fontsize=18, color='black')
            axs[row, col].tick_params(axis='x', rotation=30)


# Add custom legend with number of cases, females, and males
fig.text(1.05, 0.5,f"Number of cases:     {num_cases}\nNumber of females: {num_f}\nNumber of males:    {num_m}",
         fontsize=14, bbox=dict(edgecolor='none', facecolor='white', alpha=0.5), verticalalignment='center', horizontalalignment='right') 


# Add custom legend with number of females and males within each age group 
custom_legend_text = """Age groups
(00->29): F= 45, M= 29
(30->39): F= 39, M= 17
(40->49): F= 67, M= 32
(50->59): F= 119, M= 41
(60->69): F= 160, M= 90
(70->79): F= 157, M= 101
(80->89): F= 34, M= 19
"""

# Add the custom legend text to the plot area
fig.text(1, 0.4, custom_legend_text, fontsize=16, 
         bbox=dict(edgecolor='none', facecolor='white', alpha=0.5),
         verticalalignment='center', horizontalalignment='left')
         
plt.tight_layout()
plt.savefig('path/to/features/plot.png', bbox_inches='tight')
plt.show()