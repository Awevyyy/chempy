from streamlit_option_menu import option_menu
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import plotly.express as px
import datetime as dt
from time import sleep
import streamlit as st
from scipy import stats
import io
import re

mypath = ""
data = pd.read_csv(mypath + "chemistrycasestudy.csv")
st.set_page_config(layout="wide")


yesno = {1: "Yes", 0: "No"}
data["is_saturated"] = np.where(data["compound"].str.contains("an"), "Yes", "No")
data["is_chloro"] = np.where(data["compound"].str.contains("chloro"), "Yes", "No")
data["is_amino"] = np.where(data["compound"].str.contains("amin"), "Yes", "No")
data["is_OH"] = np.where(data["compound"].str.contains("hydro") | data["compound"].str.contains("Hydro"), "Yes", "No")
table = data["compound"].str.contains("meth") | data["compound"].str.contains("eth") | data["compound"].str.contains("but")  | data["compound"].str.contains("pro") | data["compound"].str.contains("pen") | data["compound"].str.contains("hex")
data["is_small"] = np.where(table, "Yes", "No")
median = data["flashpoint"].median()
flashpoint = np.where(data["flashpoint"] > median, "Yes","No")
data["high_flashpoint"] = flashpoint 
table = (data["smiles"].str.contains('+', regex = False)) | (data["smiles"].str.contains('-', regex = False))
data["is_ionic"] = np.where(table, "Yes", "No")
data["is_alkene"] = np.where(data["smiles"].str.contains("C=C"), "Yes", "No")
data = data.replace(yesno) #TODO mention in data CLEAN
#sun_df
#["is_silicon","is_metallic","is_tin","is_acid","is_saturated","is_chloro","is_amino","is_OH","is_small","high_flashpoint","is_alkene"]
sun_df = data.copy()
sun_df["is_silicon"] = np.where(sun_df["is_silicon"] == "Yes", "Has Silicon","No Silicon")
sun_df["is_metallic"] = np.where(sun_df["is_metallic"] == "Yes", "Is Metallic","Not Metallic")
sun_df["is_tin"] = np.where(sun_df["is_tin"] == "Yes", "Has Tin","No Tin")
sun_df["is_acid"] = np.where(sun_df["is_acid"] == "Yes", "Acidic","Not Acidic")
sun_df["is_saturated"] = np.where(sun_df["is_saturated"] == "Yes", "Is Saturated","Not Saturated")
sun_df["is_chloro"] = np.where(sun_df["is_chloro"] == "Yes", "Has Chlorine","No Chlorine")
sun_df["is_amino"] = np.where(sun_df["is_amino"] == "Yes", "Has Amino Group","No Amino Group")
sun_df["is_OH"] = np.where(sun_df["is_OH"] == "Yes", "Is Alkaline","Is Alkaline")
sun_df["is_small"] = np.where(sun_df["is_small"] == "Yes", "Small","Large")
sun_df["high_flashpoint"] = np.where(sun_df["high_flashpoint"] == "Yes", "High Flashpoint","Low Flashpoint")
sun_df["is_alkene"] = np.where(sun_df["is_alkene"] == "Yes", "Has C=C (Alkene)","No C=C")
names = {
    "Flashpoint": "flashpoint",
    "Has Silicon": "is_silicon",
    "Is Metallic": "is_metallic",
    "Has Tin": "is_tin",
    "Is Acidic": "is_acid",
    "Is Saturated": "is_saturated",
    "Has Chlorine": "is_chloro",
    "Has Amino-group": "is_amino",
    "Has Hydroxyl-group": "is_OH",
    "Is Small": "is_small",
    "High Flashpoint": "high_flashpoint",
    "Is Alkene": "is_alkene"
        }
reverse = {v:k for k,v in names.items()}
true_names = ["flashpoint","is_silicon","is_metallic","is_tin","is_saturated","is_chloro","is_amino","is_OH","is_small","high_flashpoint","is_alkene"]
choices_names_flashpoint = ["Flashpoint", "Has Silicon", "Is Metallic", "Has Tin", "Is Acidic", "Is Saturated", "Has Chlorine", "Has Amino-group", "Has Hydroxyl-group", "Is Small", "High Flashpoint", "Is Alkene"]
choices_names_no_flashpoint = ["Has Silicon", "Is Metallic", "Has Tin", "Is Acidic", "Is Saturated", "Has Chlorine", "Has Amino-group", "Has Hydroxyl-group", "Is Small", "High Flashpoint", "Is Alkene"]


def fit_distribution(data, target):
    numvars = [value for value in data.columns if value == "flashpoint"]
    catvars = [value for value in data.columns if value not in numvars + [target]]
    is_prior = data[target].value_counts(normalize = True)[1]
    not_prior = data[target].value_counts(normalize = True)[0]
    is_data = data[data[target] == "Yes"].copy()
    not_data = data[data[target] == "No"].copy()
    
    is_results = {}
    not_results = {} 
    for val in numvars:
        mu = np.mean(is_data[val])
        sigma = np.std(is_data[val])
        dist_is = stats.norm(mu, sigma)
        is_results[val] = dist_is.pdf(data[val])

        mu = np.mean(not_data[val])
        sigma = np.std(not_data[val])
        dist_not = stats.norm(mu, sigma)
        not_results[val] = dist_not.pdf(data[val])
        
    for val in catvars:
        dict_is = dict(is_data[val].value_counts(normalize = True))
        dict_not = dict(not_data[val].value_counts(normalize = True))
        is_results[val] = data[val].map(dict_is)
        not_results[val] = data[val].map(dict_not)
    
    is_frame = pd.DataFrame.from_dict(is_results)
    not_frame = pd.DataFrame.from_dict(not_results)
    
    prob_is = is_prior * is_frame.prod(axis=1)
    prob_not = not_prior * not_frame.prod(axis=1)

    df = pd.DataFrame()
    df["predicted"] = (prob_is > prob_not) * 1
    df["actual"] = np.where(data[target] == "Yes",1,0)
    
    accuracy = sum(df["predicted"] == df["actual"])/len(df["actual"])
    return df, accuracy


with st.sidebar: 
	selected = option_menu(
		menu_title = 'Navigation Panel',
		options = ['Introduction', 'Data Cleaning','Data Exploration','Naive Bayes','Data Analysis', 'Conclusion', 'Bibliography'],
		menu_icon = 'box-fill',
		icons = ['book', 'person-rolodex','search','boxes','bar-chart', 
		'check2-circle','card-text'],
		default_index = 0,
		)
    
if selected == "Introduction":
    st.markdown("# Introduction")
    st.write("The goal for this casestudy is to make a naive bayes prediction model that can predict wether or not a molecule is acidic or not based upon its name and a smattering of other factors (e.g size, fuiunctional groups) to a high degree of accuracy (90%+). Hopefully, this would aid the scientific world to advance by decreaseing the time needed for acid-base testing.")
    st.write("Why is this important? Well because in pharmaceutical research, the acidity of a molecule can provide insights into its behavior in biological systems. Acidic molecules may exhibit specific interactions with target receptors, enzymes, or other biological molecules. Predicting acidity helps in identifying potentially active compounds and designing drug candidates with desired properties<sup>1</sup>.",unsafe_allow_html=True)
    
if selected == "Data Cleaning":
    st.markdown("# Data Cleaning")
    st.write("#### For this dataset, although there are already 4 columns of categorical values, I will need to create more from other text-based columns to ensure best prediction results.")
    st.table(data.head(5))
    st.write("https://github.com/kjappelbaum/awesome-chemistry-datasets")
    st.markdown("## Pre-existing Variables")
    st.write("compound: The technical name of the chemical/molecule in question")
    st.write("flashpoint: The point (in celcius) lowest temperature at which a liquid emits enough vapor to ignite when exposed to air measured in degrees kelvin")
    st.write("pure substance: wether or not the molecule is pure... redundant as it is all true")
    st.write("smiles: The functional groups (notable parts that gives the molecule its properties) listed in standard notation")
    st.write("source: Which website the molecules' data are scraped from. E.g pubchem")
    st.write("is_silicone: Wether or not the molecule contains silicon atoms/ions")
    st.write("is_metallic: Wether or not the molecule is metallic or not")
    st.write("is_tin: Wether or not the molecule contains tin. Only 50 out of 10k of them are though")
    st.write("is_acid: Wether or not the molecule dissolve in water to form solutions with a pH lower than 7.")
    st.write("datatype: How the information on the molecule is soruced")
    st.markdown("## New Categorical Variables")
    st.write("is_saturated: Is the molecule saturated or not? The names of all saturated organic compounds end with -ane/-anol. ")
    st.code('data["is_saturated"] = np.where(data["compound"].str.contains("an"), "Yes", "No")', language='python')
    st.write("is_chloro: Does the molecule have chlorine/chloride in it? If there is a chlorine/chloride, the molecule name will contain 'chloro'. ")
    st.code('data["is_chloro"] = np.where(data["compound"].str.contains("chloro"), "Yes", "No")', language='python')
    st.write('is_amino: Does the molecule contain the functional group of NH2? If there is NH2, then the molecule will have the word amino/amine in it')
    st.code('data["is_amino"] = np.where(data["compound"].str.contains("amin"), "Yes", "No")', language='python')
    st.write('is_OH: Does the molecule contain the functional group of OH? (hydroxyl group)? If there is OH, then the molecule will have the word hydro in it')
    st.code('data["is_OH"] = np.where(data["compound"].str.contains("hydro") | data["compound"].str.contains("Hydro"), "Yes", "No")', language='python')
    st.write('is_small: If the molecule is small or large? If the name has the prefix for 1-6 numbers of carbons (e.g eth for 2 carbon atoms)')
    st.code("""table = data["compound"].str.contains("meth") | data["compound"].str.contains("eth") | data["compound"].str.contains("but")  | data["compound"].str.contains("pro") | data["compound"].str.contains("pen") | data["compound"].str.contains("hex")
\n data["is_small"] = np.where(table, "Yes", "No")""", language='python')
    st.write("is_alkene: If the molecule is an alkene or not? If there is a c=c smile in the molecule, then it is an alkene!")
    st.code('data["is_alkene"] = np.where(data["smiles"].str.contains("C=C"), "Yes", "No")', language='python')
    st.write('is_ionic: If the molecule is ionic or not? If there is a + or a -, the molecule is charged and therefore is ionic')
    st.code("""table = (data["smiles"].str.contains('+', regex = False)) | (data["smiles"].str.contains('-', regex = False))
\n data["is_ionic"] = np.where(table, "Yes", "No")""")

if selected == "Data Exploration":
    st.markdown("# Data Exoploration")
    st.markdown("### Some interactive test graphs!")
    st.markdown("### Histogram.exe!")
    col,col1=st.columns([3,5])      
    col.header("Histogram for Categorical Values")
    
    with st.form("Submit"):
        option1 = col.selectbox("Which value do you want to see the distribution for?", choices_names_flashpoint)
        agree = col.checkbox('show data as noramlized percentage %')
        submitted=st.form_submit_button("Submit to generate your histogram")
        number = 20
        if option1 == "Flashpoint":
            check1=col.checkbox("Control bin size",key=4)
            if check1:
                number=col.number_input('Insert a number',min_value=10,max_value=30,step=5)
        if submitted:
            if agree:
                fig2 = px.histogram(data,x=names[option1],
                   barmode="group",
                   histnorm="percent", color_discrete_sequence=px.colors.qualitative.Alphabet,nbins = number,labels = reverse,
                   title="Distribution of the variable: '" + option1 + "'", height = 600)
                col1.plotly_chart(fig2)
                col1.write("See the graph in is_saturated, we can see that the Yes and No are quite evenly balanced relative to all the other variables!")
            else:
                fig2 = px.histogram(data,x=names[option1],
                   barmode="group", color_discrete_sequence=px.colors.qualitative.Alphabet,nbins = number,labels = reverse,
                   title="Distribution of the variable: '" + option1 + "'", height = 600)
                col1.plotly_chart(fig2)
                col1.write("See the graph in is_saturated, we can see that the Yes and No are quite evenly balanced relative to all the other variables!")
           
    col2,col3=st.columns([3,5])      
    col2.header("Histogram for Categorical Values with is_acid")
    with st.form("Submit1"):
        option2 = col2.selectbox("Which value do you want to be the categorical value?", choices_names_no_flashpoint)
        agree1 = col2.checkbox('show data as noramlized percentage %',key=1233)
        bar_norm = None
        hist_norm = None
        if agree1:
            option9 = col2.selectbox("Which mode of normalization?", ["bar","histogram"])
            if option9 == "bar":
                bar_norm = "percent"
            else:
                hist_norm = "percent"
        option3 = col2.selectbox("Which mode?", ["group","overlay","relative"])
        
        submitted=st.form_submit_button("Submit to generate your histogram")
        if submitted:
            if agree1:
                fig3 = px.histogram(data,x=names[option2],color = 'is_acid',barmode=option3, barnorm=bar_norm,histnorm=hist_norm, color_discrete_sequence=px.colors.qualitative.Alphabet,labels = reverse,
                                    title="Distribution of the variable: '" + option2 + "' where color = 'is acidic'", height = 600)
                col3.plotly_chart(fig3)
            else:
                fig3 = px.histogram(data,x=names[option2],color = 'is_acid',barmode=option3, color_discrete_sequence=px.colors.qualitative.Alphabet,labels = reverse,
                                    title="Distribution of the variable: '" + option2 + "' where color = 'is acidic'", height = 600)
                col3.plotly_chart(fig3)
    
    col4,col5=st.columns([3,5])      
    col4.header("Histogram for Categorical Values with is_acid against flashpoint")
    with st.form("Submit2"):
        option4 = col4.selectbox("Which value do you want to be the categorical value against flashpoint?", np.setdiff1d(choices_names_no_flashpoint, ["Is Acidic"]))
        agree2 = col4.checkbox('show data as noramlized percentage %', key="23492834asd")
        option5 = col4.selectbox("Which mode?", ["group","overlay","relative"],key="238yu9hfg9sdhf")
        option7 = col4.selectbox("Which function?", ["avg","min","max"])
        submitted=st.form_submit_button("Submit to generate your histogram")
        if submitted:
            if agree2:
                fig4 = px.histogram(data,x=names[option4],y='flashpoint',color = 'is_acid',barmode=option5, histnorm="percent", histfunc=option7, color_discrete_sequence=px.colors.qualitative.Alphabet,labels = reverse,
                                    title="Plotting the values of '" + option4 + "' against flashpoint", height = 600)
                col5.plotly_chart(fig4)
            else:
                fig4 = px.histogram(data,x=names[option4],y='flashpoint',color = 'is_acid',barmode=option5, histfunc=option7, color_discrete_sequence=px.colors.qualitative.Alphabet,labels = reverse,
                                    title="Plotting the values of '" + option4 + "' against flashpoint", height = 600)
                col5.plotly_chart(fig4)
        
        
    col6,col7=st.columns([2,5])      
    col6.header("Sunburst graph with Is Acid as base")   
    with st.form("Submit3"):
        option6 = col6.multiselect("Which values do you want to see on the sunburst graph?", choices_names_no_flashpoint, max_selections = 2)
        option6_list = []
        option6_name = []
        for value in option6:
            option6_list.append(names[value])
            option6_name.append(reverse[names[value]])
        option8 = col6.selectbox("What mode do you want text to be displayed as? ", ["percent root", "percent entry", "percent parent"])
        submitted = st.form_submit_button("Submit to generate your surburst graph")
        if submitted:
            fig3 = px.sunburst(sun_df, path=["is_acid"] + option6_list, color='is_acid',labels={"is_acid":"ACIDIC"},height=800,width=800
                               ,color_discrete_sequence=px.colors.qualitative.Alphabet)
            if len(option6_list) == 0:
                fig3.update_layout(title="Sunburst graph of Is Acidic")
            elif len(option6_list) == 1:
                fig3.update_layout(title="Sunburst graph of Is Acidic followed by " + option6_name[0])
            else:
                fig3.update_layout(title="Sunburst graph of Is Acidic followed by " + option6_name[0] + " followed by " + option6_name[1])
            fig3.update_layout(showlegend = True)
            #['percent root', 'percent entry', 'percent parent]
            fig3.update_traces(textinfo="label+"+option8)
            col7.plotly_chart(fig3)
    
    
    st.markdown("### Chi-Squared Test!")
    option2 = 'is_metallic'
    option2 = st.selectbox("Select a value to see the chi-squared and probability compared to the 'is_acid' variable", ["is_silicon","is_metallic","is_tin","is_acid","is_saturated","is_chloro","is_amino","is_OH","is_small","high_flashpoint","is_alkene"])
    chisquare = stats.chi2_contingency(pd.crosstab(data['is_acid'], data[option2]))[:2] 
    st.write("Chisquare:", str(chisquare[0]))
    st.write("Probability:", str(chisquare[1]))
    if chisquare[1] < 0.05:
        st.write("Assuming a p-value of 0.05, this data is significant!!!")
    else:
        st.write("Assuming a p-value of 0.05, this data is insignificant!!!")


if selected == 'Naive Bayes':
    st.markdown("# Naive Bayes")
    st.markdown("### What is Naive Bayes?")
    st.write("Naïve Bayes is one of the fast and easy machine learning algorithms to predict values in datascience. It can and is often used for Binary Classifications (only two possible values) and prerforms well as compared to the other Algorithms. It first assumes the fact that all the variables are independent to each other, then by using probability theorems it works out the probabilities for the two outcomes, selecting the one with the highest probability.")
    
    with st.form("Submit4"):
        options = st.multiselect('What columns do you want to be put into the prediction?', choices_names_flashpoint)
        submitted = st.form_submit_button("Submit to generate predictions")
        if submitted:
            st.write("You selected:", options)
            options.append('Is Acidic')
            names_options = [names[val] for val in options]
            selected_df = data[names_options]
            returned = fit_distribution(selected_df, 'is_acid')
            st.write(returned[0])
            st.write(returned[1])
    
    

    with st.expander("Click to see code"):
        st.code("""def fit_distribution(data, target):
            numvars = [value for value in data.columns if value == "flashpoint"]
            catvars = [value for value in data.columns if value not in numvars + [target]]
            is_prior = data[target].value_counts(normalize = True)[1]
            not_prior = data[target].value_counts(normalize = True)[0]
            is_data = data[data[target] == "Yes"].copy()
            not_data = data[data[target] == "No"].copy()
            
            is_results = {}
            not_results = {} 
            for val in numvars:
                mu = np.mean(is_data[val])
                sigma = np.std(is_data[val])
                dist_is = stats.norm(mu, sigma)
                is_results[val] = dist_is.pdf(data[val])

                mu = np.mean(not_data[val])
                sigma = np.std(not_data[val])
                dist_not = stats.norm(mu, sigma)
                not_results[val] = dist_not.pdf(data[val])
                
            for val in catvars:
                dict_is = dict(is_data[val].value_counts(normalize = True))
                dict_not = dict(not_data[val].value_counts(normalize = True))
                is_results[val] = data[val].map(dict_is)
                not_results[val] = data[val].map(dict_not)
            
            is_frame = pd.DataFrame.from_dict(is_results)
            not_frame = pd.DataFrame.from_dict(not_results)
            
            prob_is = is_prior * is_frame.prod(axis=1)
            prob_not = not_prior * not_frame.prod(axis=1)

            df = pd.DataFrame()
            df["predicted"] = (prob_is > prob_not) * 1
            df["actual"] = np.where(data[target] == "Yes",1,0)
            
            accuracy = sum(df["predicted"] == df["actual"])/len(df["actual"])
            return df, accuracy""")
if selected == 'Data Analysis':
    st.markdown("# Data Analysis")
    st.write("Add the box plot with flash")
    agree3 = st.checkbox("Do you want the log the y axis?")
    #TODO bosh the box plot with flashpoint
    chisquares = [round(stats.chi2_contingency(pd.crosstab(data['is_acid'], data[val]))[0],3) for val in np.setdiff1d(true_names, ["flashpoint","is_acid"])]
    pvals = [stats.chi2_contingency(pd.crosstab(data['is_acid'], data[val]))[1] for val in np.setdiff1d(true_names, ["flashpoint","is_acid"])]
    pvals = [f'{p:.3g}' for p in pvals]
    fig5 = px.bar(x =pd.Series(np.setdiff1d(true_names, ["flashpoint","is_acid"])).replace(reverse),y=chisquares,log_y=agree3, text=chisquares, hover_name = pvals)
    fig5.update_xaxes(categoryorder="total descending")
    st.plotly_chart(fig5)
    
    st.markdown("### Box plot for flashpoint and is_acid")
    fig6 = px.box(data,x="flashpoint",y="is_acid", labels = reverse)
    st.plotly_chart(fig6)
if selected == "Conclusion":
    st.markdown("# Conclusion")
    st.write("To summarize, there are 5 factors that work the best predicting the the acidity. Those 5 include Flashpoint, Is Metallic, Has Silicon, Is Saturated and Has Amino Group. These five both have a p-value of 0 (ronded to 5 devcimal places) and the flashpoint having a chisquare value of 503. ")
    st.write("Through my systematic experimentation... My baseline model has an accuracy 91%")
    st.write("However, it turns out that due to the heavy imbalance of my dataset (91%:9%) the prior values used in the naive bayes equation heavily skews the results causing not acid to be selected every single time. To counteract this with unconventional ways, I replaced the heavy imbalance with a closer value: with the priors being 0.4 and 0.6. This yielded better results.")
    
if selected == "Bibliography":
    st.markdown("# Bibliography")
    st.write("[1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3641858/")
