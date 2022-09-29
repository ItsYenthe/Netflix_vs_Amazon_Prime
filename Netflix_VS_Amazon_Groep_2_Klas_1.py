#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.figure_factory as ff
os.getcwd()


# In[2]:


#datasets inladen
netflix_basis = pd.read_csv('NetflixOriginals.csv')
netflix_extra = pd.read_csv('netflix_original_movie_data.csv')
amazon_prime_basis = pd.read_csv('amazon prime movies.csv')


# In[3]:


st.title("Netflix_vs_Amazon")


# In[4]:


# 2 datasets samenvoegen
netflix_merge = pd.merge(netflix_basis, netflix_extra, on = 'Title')

# Kijken hoeveel NA's er zijn
netflix_merge.isna().sum()

# Kolommen selecteren
netflix_kolommen = netflix_merge[['Title', 'Genre', 'Premiere', 'Runtime', 'IMDB Score', 'Language_x', 'Country']]

# Duplicates verwijderen
netflix = netflix_kolommen.drop_duplicates()

amazon_prime = amazon_prime_basis.drop_duplicates()


# In[5]:


#'IMDb Rating' wordt al float getrasformeert 
amazon_prime['IMDb Rating'] = pd.to_numeric(amazon_prime['IMDb Rating'], errors='coerce')
amazon_prime = amazon_prime.replace(np.nan, 0, regex=True)


# In[6]:


#dus filter de IMDB gelijk aan 0 
amazon_prime_zonder_IMDB_0 = amazon_prime[amazon_prime['IMDb Rating'] != 0]


# In[12]:


#Alles in streamlit

########################################### FIGUUR ########################################################
# Add histogram data
x1 = amazon_prime_zonder_IMDB_0['IMDb Rating']
x2 = netflix_merge['IMDB Score']

# Group data together
hist_data = [x1, x2]
group_labels = ['Amazon', 'Netflix']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(title_text='IMDB Score van netflix en amazon prime', xaxis_title = 'IMDB-score', yaxis_title = 'Kans dichtheid')

st.plotly_chart(fig)

############################################## FIGUUR 1 ###################################################

# Plotten van percentage films per taal met dropdownbox voor amazon, netflix en beide
new_amazon = pd.DataFrame(amazon_prime['Language'].value_counts())
new_amazon['Totaal'] = new_amazon['Language'].sum()
new_amazon['Percentage'] = new_amazon['Language']/new_amazon['Totaal']

#Dataframe maken voor de Netflix Language
new_netflix = pd.DataFrame(netflix_merge['Language_x'].value_counts())
new_netflix['Totaal'] = new_netflix['Language_x'].sum()
new_netflix['Percentage'] = new_netflix['Language_x']/new_netflix['Totaal']

fig1 = go.Figure()

dropdown_buttons = [
    {'label': 'Amazon', 'method':'update',
    'args':[{'visible':[True, False]}, {'title':'Amazon'}]},
    {'label': 'Netflix', 'method':'update',
    'args':[{'visible':[False, True]}, {'title':'Netflix'}]},
    {'label': 'Beide', 'method':'update',
    'args':[{'visible':[True, True]}, {'title':'Beide'}]}
]


fig1.add_trace(go.Bar(x = new_amazon.index, y = new_amazon['Percentage'], name = 'Amazon Prime films/series'))
fig1.add_trace(go.Bar(x = new_netflix.index, y = new_netflix['Percentage'], name = 'Netflix films/series'))
fig1.update_layout({'updatemenus':[{'type': 'dropdown',
                                 'x':1.18, 'y':0.5,
                                 'showactive':True,
                                 'active': 0,
                                 'buttons':dropdown_buttons}]},
                  title_text = 'Aantal films/series die beschikbaar zijn via amazon of netflix, verdeeld per taal', yaxis_tickformat="2%",
                 )
fig1.update_xaxes(title_text = 'Taal')
fig1.update_yaxes(title_text = 'Percentage films')

st.plotly_chart(fig1, use_container_width = True)

############################################# FIGUUR 2 ########################################################

# Selecteren naar de data waarbij het 1 van die 5 genres is
netflix_genre = netflix.loc[(netflix['Genre'] == 'Documentary') | (netflix['Genre'] == 'Drama')| (netflix['Genre'] == 'Comedy') |
            (netflix['Genre'] == 'Thriller') | (netflix['Genre'] == 'Romantic comedy')]

# Plotten van de IMDB Scores per genre
# Figuur van Yenthe
my_scale = ['rgb(0, 171, 169)', 'rgb(0, 138, 0)', 'rgb(96, 169, 23)', 'rgb(164, 196, 0)', 'rgb(227, 200, 0)']
fig2 = px.box(data_frame = netflix_genre, x = 'Genre', y = 'IMDB Score', color = 'Genre',
             color_discrete_sequence= my_scale, title = 'IMDB Scores per genre')

my_buttons = [{'label':'Boxplot', 'method':'update', 'args': [{'type': 'box'}]},
              {'label':'Violin', 'method':'update', 'args': [{'type': 'violin'}]}]

fig2.update_layout({'updatemenus':[{'type': 'buttons', 'direction': 'down', 'x': 1.13, 'y': 0.5, 'showactive': True,
                                  'active': 0, 'buttons': my_buttons}]})

st.plotly_chart(fig2, use_container_width = True)

############################################# FIGUUR 3 ########################################################

# Nieuwe kolom in netflix met runtime per groep
bins= [0,51,101,151,251]
labels = ['0-50','51-100','101-150','151-250']
netflix['Runtime_group'] = pd.cut(netflix['Runtime'], bins=bins, labels=labels, right=False)

# Visualisatie van runtime van de (Netflix)films tegen de IMDB scores
# Figuur van Yenthe
fig3 = go.Figure()

for runtime in ['0-50', '51-100', '101-150', '151-250']:
    df = netflix[netflix['Runtime_group'] == runtime]
    fig3.add_trace(go.Box(x = df['Runtime_group'], y = df['IMDB Score'], name = runtime))
    
sliders = [{'steps':[
    {'label':'all', 'method': 'update', 'args': [{'visible': [True, True, True, True]}, {'title': 'Runtime tegen de IMDB Score'}]},
    {'label':'0-50 minuten', 'method': 'update', 'args': [{'visible': [True, False, False, False]}, {'title': 'Runtime van 0-50 minuten tegen de IMDB Score'}]},
    {'label':'51-100 minuten', 'method': 'update', 'args': [{'visible': [False, True, False, False]}, {'title': 'Runtime van 51-100 minuten tegen de IMDB Score'}]},
    {'label':'101-150 minuten', 'method': 'update', 'args': [{'visible': [False, False, True, False]}, {'title': 'Runtime van 101-150 minuten tegen de IMDB Score'}]},
    {'label':'151-250 minuten', 'method': 'update', 'args': [{'visible': [False, False, False, True]}, {'title': 'Runtime van 151-250 minuten tegen de IMDB Score'}]},
]}]

fig3.update_xaxes(title_text = 'Runtime (minuten)')
fig3.update_yaxes(title_text = 'IMDB Score')
fig3.update_layout({'sliders': sliders})


st.plotly_chart(fig3, use_container_width = True)

############################################# FIGUUR 4 ########################################################

# Selecteren naar de data waarbij het 1 van die 5 genres is
netflix_country = netflix.loc[(netflix['Country'] == 'United states') | (netflix['Country'] == 'India')| (netflix['Country'] == 'United Kingdom') |
            (netflix['Country'] == 'Spain') | (netflix['Country'] == 'France')]

# Plotten van de IMDB Score per land
# Figuur van Yenthe
fig4 = go.Figure()

dropdown_buttons3 = [
    {'label': 'All', 'method':'update',
    'args':[{'visible':[True, True, True, True]}, {'title':'IMDB Score van alle landen'}]},
    {'label': 'United States', 'method':'update',
    'args':[{'visible':[True, False, False, False, False]}, {'title':'IMDB Score van United States'}]},
    {'label': 'India', 'method':'update',
    'args':[{'visible':[False, True, False, False, False]}, {'title':'IMDB Score van India'}]},
    {'label': 'United Kindom', 'method':'update',
    'args':[{'visible':[False, False, True, False, False]}, {'title':'IMDB Score van United Kingdom'}]},
    {'label': 'Spain', 'method':'update',
    'args':[{'visible':[False, False, False, True, False]}, {'title':'IMDB Score van Spanje'}]},
    {'label': 'France', 'method':'update',
    'args':[{'visible':[False, False, False, False, True]}, {'title':'IMDB Score van Frankrijk'}]}
]

for country in ['United States', 'India', 'United Kingdom', 'Spain', 'France']:
    df = netflix[netflix['Country'] == country]
    fig4.add_trace(go.Box(x = df['Country'], y = df['IMDB Score'], name = country))

fig4.update_layout({'updatemenus':[{'type': 'dropdown',
                                 'x':1.21, 'y':0.5,
                                 'showactive':True,
                                 'active': 0,
                                 'buttons':dropdown_buttons3}]},
                  title_text = 'IMDB Score per land'
                 )
fig4.update_xaxes(title_text = 'Country')
fig4.update_yaxes(title_text = 'IMDB Score')

st.plotly_chart(fig4, use_container_width = True)

############################################# FIGUUR 5 ########################################################

# Nieuwe kolom met gemiddelde IMDB Score per taal en gemiddelde IMDB Score per runtime
netflix['Gemiddelde IMDB Score Per Taal'] = netflix.groupby('Language_x')['IMDB Score'].transform('mean')
netflix['Gemiddelde IMDB Score Per Runtime'] = netflix.groupby('Runtime')['IMDB Score'].transform('mean')

# Als gegeven IMDB Score van taal hoger is als gemiddeld, voeg 1 toe
for lab, row in netflix.iterrows():
    if row["IMDB Score"] > row["Gemiddelde IMDB Score Per Taal"]:
        netflix.loc[lab, "Onder/Boven gemiddelde taal"] = 1
    else:
        netflix.loc[lab, "Onder/Boven gemiddelde taal"] = 0
        
# Als gegeven IMDB Score van runtime hoger is als gemiddeld, voeg 1 toe
for lab, row in netflix.iterrows():
    if row["IMDB Score"] > row["Gemiddelde IMDB Score Per Runtime"]:
        netflix.loc[lab, "Onder/Boven gemiddelde runtime"] = 1
    else:
        netflix.loc[lab, "Onder/Boven gemiddelde runtime"] = 0

# Aantal keer dat de IMDB van de film hoger is als de gemiddelde IMDB van de taal/runtime
# Figuur Yenthe
fig5 = go.Figure()

dropdown_buttons4 = [
    {'label': 'Taal', 'method':'update',
    'args':[{'visible':[True, False]}, {'title':'Aantal keer dat de IMDB Score boven de gemiddelde IMDB van de taal zit'}]},
    {'label': 'Runtime', 'method':'update',
    'args':[{'visible':[False, True]}, {'title':'Aantal keer dat de IMDB Score boven de gemiddelde IMDB van de runtime zit'}]},
]

fig5.add_trace(go.Bar(x = netflix['Language_x'], y = netflix['Onder/Boven gemiddelde taal'], name = 'Boven gemiddelde taal'))
fig5.add_trace(go.Bar(x = netflix['Runtime_group'], y = netflix['Onder/Boven gemiddelde runtime'], name = 'Boven gemiddelde runtime'))
fig5.update_layout({'updatemenus':[{'type': 'dropdown',
                                 'x':1.18, 'y':0.5,
                                 'showactive':True,
                                 'active': 0,
                                 'buttons':dropdown_buttons4}]},
                  title_text = 'Aantal keer dat de IMDB van de film hoger is als het gemiddelde van taal/runtime ',
                 )
fig5.update_xaxes(title_text = 'Taal/Runtime (minuten)')
fig5.update_yaxes(title_text = 'Aantal keer boven gemiddelde')


st.plotly_chart(fig5, use_container_width = True)

############################################# FIGUUR 6 ########################################################

#importeren van de datasets
netflix_6 = pd.read_csv("netflix_original_movie_data.csv")
amazon_6 = pd.read_csv("amazon prime movies.csv")
netflix_6 = pd.read_csv("NetflixOriginals.csv", encoding = "ISO-8859-1", engine = "python")

#NA values in kolom IMDb droppen
amazon_6.dropna(subset=['IMDb Rating', 'Year of Release'],inplace= True)

#Kolommen IMDb en Year of Release omzetten van strings naar integers
res = [eval(i) for i in amazon_6['Year of Release']]
amazon_6['Year of Release'] = res
res2 = [eval(i) for i in amazon_6['IMDb Rating']]
amazon_6['IMDb Rating'] = res2

#Groepen maken van Year of Release
bins= [0,1990,2000,2009,2015,2018,2030]
labels = ['t/m 1989','1990-1999','2000-2008','2009-2014','2015-2017','2017+']
amazon_6['release_group'] = pd.cut(amazon_6['Year of Release'], bins=bins, labels=labels)

#De Lineplot: De legenda dient als checkbox, hierin kunnen de talen aan en uit worden gezet door erop te klikken
fig6 = px.line(
    data_frame=amazon_6.groupby(['release_group', 'Language']).mean().reset_index(),
    x="release_group",
    y="IMDb Rating",
    color='Language',
    title = 'IMDb scores van films door de jaren heen per taal'
)
fig6.update_yaxes(range=[0, 10], title_text = 'IMDb score')
fig6.update_xaxes(title_text = 'Jaar van uitgave')
fig6.update_layout(legend_title_text='Talen')

st.plotly_chart(fig6)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




