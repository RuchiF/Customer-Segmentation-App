# Importing Required Libraries

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler , LabelEncoder
import plotly.express as px
from sklearn.decomposition import PCA
import seaborn as sns


# Storing data in dataframe

df=pd.read_csv(r"Customer_Final.csv")
# This will help us to plot graphs (its just raw_df (after data preprocessing)+ cluster column)

# It contains the final data along with ClusterID and Country name

df.set_index('CustomerID', inplace=True)

raw_df=pd.read_csv(r"Customer_input.csv")
# raw_df contains unscaled data , with Country as categorical column

temp_df=pd.read_csv(r"Customer_input.csv")
# temp_df contains unscaled data , with Country as categorical column


# We will modify raw_df later thus we have created temp_df



# processing data
numericCols=['Amount' ,'Recency' ,'num_transactions']
scaler=StandardScaler()
raw_df[numericCols]=scaler.fit_transform(raw_df[numericCols])
label=LabelEncoder()
raw_df['Country']=label.fit_transform(raw_df['Country'])

# Data for plotting Elbow Curve (Taken from the notebook)
inertia=[17460.714226498654, 12763.049140714247, 9424.962624227266, 6790.131901281124, 5777.870857173226]
n_clusters_values=range(3,8)


# Data for PCA
pca_df=temp_df.drop('Country', axis='columns')
pca_df.set_index('CustomerID' , inplace=True)
pca_scaler=StandardScaler()
pca_df[numericCols]=pca_scaler.fit_transform(pca_df[numericCols])

# PCA
pca=PCA(n_components=2)
pca_data=pca.fit_transform(pca_df)

# PCA Model
pca_model = KMeans(n_clusters=3 , n_init=10 , max_iter=50)
pca_model.fit(pca_data)
pca_model = KMeans(n_clusters=3 , n_init=10 , max_iter=50)
pca_model.fit(pca_data)
pca_labels=pca_model.labels_
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
pca_df['Cluster'] = pca_labels

# Countries list for giving optiions to user
countries=['Norway', 'Iceland', 'Finland', 'Italy', 'United Kingdom',
       'Bahrain', 'Spain', 'Portugal', 'Austria', 'Belgium',
       'Unspecified', 'Cyprus', 'Denmark', 'Switzerland', 'Australia',
       'France', 'Germany', 'RSA', 'Greece', 'Sweden', 'Israel', 'USA',
       'Saudi Arabia', 'Poland', 'United Arab Emirates', 'Japan',
       'Netherlands', 'Lebanon', 'Brazil', 'Czech Republic', 'EIRE',
       'Channel Islands', 'European Community', 'Lithuania', 'Canada',
       'Malta']


# Adding few styles
st.markdown("""<style>
            p{
            font-size:25px;
            }
            
            </style>""" , unsafe_allow_html=True)

st.sidebar.title("Navigation")
page=st.sidebar.radio("Go to" ,['Home' , 'EDA' ,'Predict', 'Predict Using PCA Data'])




# Home Page

if page=='Home':
    st.image('segmentation.jpg')
    st.markdown('# Customer Segmentation App' , unsafe_allow_html=True)
    st.write("""
            ## Overview:
            <p>In this project, we dive into the world of customer behavior by classifying them 
            based on their buying habits using K-Means Clustering. The goal is to uncover hidden 
            patterns in how people shop, which can be a game-changer for businesses. With 
            these insights, companies can better understand their customers, 
            craft personalized marketing strategies, and optimize their approach to 
            make every interaction more meaningful.
             You can download the dataset from 
            <a href="https://www.kaggle.com/datasets/tunguz/online-retail" target="_blank">here</a>, for complete details, you can refer to my 
            <a href="https://github.com/RuchiF/Customer-Segmentation-App">Github repository here</a>. 
            </p> """ , unsafe_allow_html=True)
    st.write('---')
    st.write(' ## ➡ Description of the Dataset', unsafe_allow_html=True)
    st.write("""
        **Note** - Although the original dataset contains some other features, we have extracted some useful information from it
        and transformed features. 
    """, unsafe_allow_html=True)
    st.write('**Amount** - Total amount spent by the customer', unsafe_allow_html=True)
    st.write('**num_transactions** - Total number of transactions made by the customer', unsafe_allow_html=True)
    st.write('**Recency** - Number of days since the last transaction was made', unsafe_allow_html=True)
    st.write('**Country** - Refers to the country to which the customer belongs', unsafe_allow_html=True)


    st.write(raw_df)

    st.write(' ## ➡ Approach', unsafe_allow_html=True)
    st.image('clustering.jpg',width=400)
    st.write("""We will be a using an unsupervised learning algorithm K-Means Clustering
             to group our customers based on their behaviour patterns.""")
    st.write('## How does this work ?')
    st.write('• Randomly initialize k data points as the centroid of clusters.')
    st.write('• Assign each data point to their nearest cluster. ')
    st.write('• Update clusters based on mean of data points in each cluster. ')
    st.write('• Repeat these steps until centroids stop changing.')

    st.write('## Steps Followed :-')
    st.write('• Imported required libraries.')
    st.write('• Manipulated features as per the need.')
    st.write('• Explored Data through various visualization techniques and plots available in matplotlib ,seaborn and plotly.')
    st.write('• Removed few outliers.')
    st.write('• Standardized data and did label encoding for Country column.')
    st.write('• Used elbow method and Silhouette score to find the best no. of clusters')
    st.write('• Visualized relationship between clusters and features')



# EDA Page

if page=='EDA':
    chart_select=st.selectbox(
    label='Select the type of Chart',
    options=['Scatterplots' ,'Displot' , 'Boxplot' ,'Elbow Curve' , 'PCA']
)
    num_cols=['Amount' ,'num_transactions' ,'Country','Recency']
    df['ClusterID'] = df['ClusterID'].astype(str)
    custom_colors = {
            '0': '#1F9E89',
            '1': '#440154',
            '2': '#B6DE2B'
        }
    if chart_select=='Scatterplots':
        st.subheader('Scatterplot Settings')
        try:
            x_values=st.selectbox('X axis' , options=num_cols)
            y_values=st.selectbox('Y axis' , options=num_cols)
            plot=px.scatter(data_frame=df ,x=x_values , y=y_values , color='ClusterID', 
            color_discrete_map=custom_colors)
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select=='Displot':
        st.subheader('Scatterplot Settings')
        try:
            x_values=st.selectbox('X axis' , options=num_cols)
            # plot = px.histogram(df, x=x_values, title=f"Histogram and KDE for {x_values}" )
            plot=sns.displot(df ,x=x_values , kde=True , color='g')
            st.pyplot(plot)
        except Exception as e:
            print(e)
    if chart_select=='Boxplot':
        fig,ax=plt.subplots(figsize=(10, 6))
        sns.set_style('darkgrid')
        x=df.drop('Country' , axis='columns')
        sns.boxplot(data=x ,color='skyblue', ax=ax)
        plt.title('Box Plot for different attributes')
        plt.ylabel('Value')
        plt.xlabel('Features')
        st.pyplot(fig)
        plt.close(fig)
    if chart_select=='Elbow Curve':
        plot=px.line(x=n_clusters_values , y=inertia)
        plot.update_layout(title='Elbow Curve', xaxis_title='Number of Clusters', yaxis_title='Sum of Squared Distances')
        st.write(plot)
    if chart_select=='PCA':
        plot=px.scatter(pca_df ,x='PC1' , y='PC2' , color='Cluster' , color_continuous_scale='viridis')
        plot.update_layout(title='PCA of Data with K-Means Clusters' ,xaxis_title='Principal Component 1' , yaxis_title='Principal Component 2')
        st.write(plot)


    
# Predict Page

if page=='Predict':
    def user_input_features():
        AMOUNT=st.slider('Amount' ,float(temp_df.Amount.min()) ,float(temp_df.Amount.max()) ,float(temp_df.Amount.mean()))
        RECENCY=st.slider('Recency' ,float(temp_df.Recency.min()) ,float(temp_df.Recency.max()) ,float(temp_df.Recency.mean()))
        COUNTRY=st.selectbox('Country' , options=countries)
        NUMBER_OF_TRANSACTIONS=st.slider('No of Transactions' ,float(temp_df.num_transactions.min()) ,float(temp_df.num_transactions.max()) ,float(temp_df.num_transactions.mean()))
        data={
            'Country':COUNTRY,
            'num_transactions':NUMBER_OF_TRANSACTIONS,
            'Amount':AMOUNT,
            'Recency':RECENCY,
        }
        features=pd.DataFrame(data ,index=[0])
        return features

    st.header('Specify Input Parameters')

    user_df=user_input_features()
    st.write(user_df)
    user_df[numericCols]=scaler.transform(user_df[numericCols])
    user_df['Country']=label.transform(user_df['Country'])
    # st.write(user_df)



    raw_df.drop('CustomerID' , axis='columns' ,inplace=True)

    model=KMeans(n_clusters=3 , max_iter=50 , n_init=10)
    model.fit(raw_df)

    if st.button('Predict'):
        user_label=model.predict(user_df)
        st.write(f'This person belongs to Cluster {user_label[0]}')
       

# Prediction After performing PCA on data
if page=='Predict Using PCA Data':
    def user_input_features():
        AMOUNT=st.slider('Amount' ,float(temp_df.Amount.min()) ,float(temp_df.Amount.max()) ,float(temp_df.Amount.mean()))
        RECENCY=st.slider('Recency' ,float(temp_df.Recency.min()) ,float(temp_df.Recency.max()) ,float(temp_df.Recency.mean()))
        NUMBER_OF_TRANSACTIONS=st.slider('No of Transactions' ,float(temp_df.num_transactions.min()) ,float(temp_df.num_transactions.max()) ,float(temp_df.num_transactions.mean()))
        data={
            'num_transactions':NUMBER_OF_TRANSACTIONS,
            'Amount':AMOUNT,
            'Recency':RECENCY,
        }
        features=pd.DataFrame(data ,index=[0])
        return features

    st.header('Specify Input Parameters')

    user_df=user_input_features()
    st.write(user_df)

    user_df[numericCols]=pca_scaler.transform(user_df[numericCols])
    pca_user_df=pca.transform(user_df)

    if st.button('Predict'):
        pca_user_label=pca_model.predict(pca_user_df)

        st.write(f'This person belongs to Cluster {pca_user_label[0]}')

    
    
    






