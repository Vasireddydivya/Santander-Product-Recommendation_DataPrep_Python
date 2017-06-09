import numpy as np
import pandas as pd
#from tsne import bh_sne
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
#pylab.rcParams['figure.figsize'] = (10, 6)
def convert_int(dataframe,colname):
    dataframe[colname]=pd.to_numeric(dataframe[colname],errors='coerce')


limit_rows=1000000
df=pd.read_csv("C:/Users/vasir/Desktop/ADM/train_ver2.CSV",nrows=limit_rows)
df.head()
unique_ids   = pd.Series(df["ncodpers"].unique())
limit_people = 5.2e2
unique_id    = unique_ids.sample(n=limit_people)
df           = df[df.ncodpers.isin(unique_id)]
print df.describe()

#print df.dtypes
print df['fecha_dato'].dtype

#converting the fecha_dato and fecha_alta to DateTime
df["fecha_dato"]=pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df["fecha_alta"]=pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
print df['fecha_dato'].unique()
#
##age is string value in data so converting the age to numeric value
convert_int(df,'age')
print df.dtypes
df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month
#
##checking for null values
print df.isnull().any()

#Data Cleaning
with plt.rc_context(dict(sns.axes_style("whitegrid"),
                         **sns.plotting_context("notebook",font_scale=1.5))):
    sns.distplot(df['age'].dropna(),bins=100,kde=False)
    sns.plt.title("Age Distribution")
    plt.ylabel("Count")
#I found few outliers below 20 and above 100, so we need to uniformly distribute the graph and replace the NA's with mean or median
#trying with median
df.loc[df.age<18,"age"]    = df.loc[(df.age>=18) & (df.age<=30),"age"].median(skipna=True)
df.loc[df.age>100,"age"]   = df.loc[(df.age>=30) & (df.age<=100),"age"].median(skipna=True)
df["age"].fillna(df["age"].median(),inplace=True)
df["age"] = df["age"].astype(int)

df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
df["age"].fillna(df["age"].mean(),inplace=True)
df["age"] = df["age"].astype(int)

with plt.rc_context(dict(sns.axes_style("whitegrid"),
                         **sns.plotting_context("notebook",font_scale=1.5))):
    sns.distplot(df['age'].dropna(),bins=80,kde=False)
    sns.plt.title("Age Distribution")
    plt.ylabel("Count")
#Both median and mean plots are almost similar i will use median value because we will not have bias issues.

print df.isnull().any()

#customer seniority and and new customer are correlated because customer seriority is in months and new customer will be '1' if he registered in past 6 months
#check the NA count for both columns
df['ind_nuevo'].isnull().sum()
#convert the customer seniority to int

convert_int(df,'antiguedad')

print df['antiguedad'].dtype

def find_nullVal_Count(dataframe,colname):
    return dataframe[colname].isnull().sum()
    
val=find_nullVal_Count(df,'ind_nuevo')
print val

df['ind_nuevo'].isnull().sum()

months_active=df.loc[df["ind_nuevo"].isnull(),:].groupby("ncodpers",sort=False).size()
months_active.max()

df.loc[df['ind_nuevo'].isnull(),"ind_nuevo"]=1

print df.isnull().any()      

df['antiguedad'].isnull().sum()
print df['antiguedad'].dtype

print df.loc[df['antiguedad'].isnull(),"ind_nuevo"].describe()

df.loc[df['antiguedad'].isnull(),"antiguedad"]=df.antiguedad.min()
df.loc[df['antiguedad']<0,"antiguedad"]=0

print df.isnull().any() 
#indrel
df['indrel'].isnull().sum()
pd.Series([i for i in df.indrel]).value_counts()

df.loc[df['indrel'].isnull(),"indrel"]=1
print df.isnull().any()       

#tipodom,customer's province
df.drop(['tipodom','cod_prov'],axis=1,inplace=True)
print df.isnull().any() 

#fecha_alta
dates=df.loc[:,'fecha_alta'].sort_values().reset_index()
date_value=int(np.median(dates.index.values))
print date_value

df.loc[df['fecha_alta'].isnull(),"fecha_alta"]=dates.loc[date_value,"fecha_alta"]
df["fecha_alta"].describe()

print df.isnull().any()  

df['nomprov'].unique()
df.loc[df['nomprov']=='CORU\xc3\x91A, A',"nomprov"]="CORUNA, A"
df.loc[df['nomprov'].isnull(),"nomprov"]="UNKNOWN"

df['nomprov'].unique()
print df.isnull().any() 

df['ind_nom_pens_ult1'].isnull().sum()
df.loc[df['ind_nom_pens_ult1'].isnull(),"ind_nom_pens_ult1"]=0
df.loc[df['ind_nomina_ult1'].isnull(),"ind_nomina_ult1"]=0

print df.isnull().any()
pd.Series([i for i in df.indfall]).value_counts()

df.loc[df['indfall'].isnull(),"indfall"]='N'
pd.Series([i for i in df.tiprel_1mes]).value_counts()
df.loc[df['tiprel_1mes'].isnull(),"tiprel_1mes"]='A'
df.tiprel_1mes = df.tiprel_1mes.astype("category")
map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "P",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}
df.indrel_1mes.fillna("P",inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
df.indrel_1mes = df.indrel_1mes.astype("category")

print df.isnull().any()
pd.Series([i for i in df.canal_entrada]).value_counts()
missing_col=['ind_empleado','pais_residencia','sexo','canal_entrada','conyuemp']
for col in missing_col:
    df.loc[df[col].isnull(),col]='UNKNOWN'
print df.isnull().any()

pd.Series([i for i in df.indext]).value_counts()
df.loc[df['indext'].isnull(),"indext"]='N'

print df.isnull().any()
pd.Series([i for i in df.indresi]).value_counts()
df.loc[df['indresi'].isnull(),"indresi"]='S'

print df.isnull().any()
pd.Series([i for i in df.ult_fec_cli_1t]).value_counts()

pd.Series([i for i in df.ind_actividad_cliente]).value_counts()
print df['ult_fec_cli_1t'].isnull().sum()

df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = df["ind_actividad_cliente"].median()

print df['ult_fec_cli_1t'].dtype


#Data Visualizations

import numpy as np
import pandas as pd
#from tsne import bh_sne
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.charts import Histogram,Bar
from bokeh.io import gridplot, output_file, show
from bokeh.plotting import figure
from bokeh.layouts import row

#%matplotlib inline
#pylab.rcParams['figure.figsize'] = (10, 6)
def convert_int(dataframe,colname):
    dataframe[colname]=pd.to_numeric(dataframe[colname],errors='coerce')


limit_rows=1000000
df=pd.read_csv("C:/Users/vasir/Desktop/ADM/train_ver2.CSV",nrows=limit_rows)


convert_int(df,'age')
print df['age'].dtype
df['age']=df['age'].fillna(-1);
cols=['age']
df[cols]=df[cols].applymap(np.int64)
df_frac=df.sample(frac=0.01)
p_age=Histogram(df_frac,values='age',title="Age Distribution")
#show(p_age)


dffrac1=df_frac.dropna(subset=['sexo'],how='any')
dffrac1.head()
#dffrac1['sexo']=dffrac1['sexo'].astype('category')
p=Bar(dffrac1,'sexo',title="Sex")
#show(p)


dffrac2=df_frac.dropna(subset=['renta'],how='any')
bar_renta=Bar(dffrac2,values='renta',label='nomprov',agg='mean',title="City Vs Renta",legend=False,  plot_width=800)
#show(bar_renta)

features_columns=df.filter(regex='ind_+.*ult.*');
features=features_columns.columns.values;
#print features;

df1=df[features]
feature=features.tolist();
print feature;
df_na=df_frac.dropna(subset=['ind_nom_pens_ult1','ind_nomina_ult1'],how='any')

df_bar=Bar(df_na,label='ind_nom_pens_ult1',values='age',group='sexo',title="Sex vs ind_nom_pens_ult1",  plot_width=200)
df_bar1=Bar(df_na,label='ind_nomina_ult1',values='age',group='sexo',title="Sex vs ind_nomina_ult1",legend=False,  plot_width=200)
df_bar2=Bar(df_na,label='ind_ahor_fin_ult1',values='age',group='sexo',title="Sex vs ind_ahor_fin_ult1",legend=False,  plot_width=200)
df_bar3=Bar(df_na,label='ind_aval_fin_ult1',values='age',group='sexo',title="Sex vs ind_aval_fin_ult1",legend=False,  plot_width=200)
df_bar4=Bar(df_na,label='ind_cco_fin_ult1',values='age',group='sexo',title="Sex vs ind_cco_fin_ult1",legend=False,  plot_width=200)
df_bar5=Bar(df_na,label='ind_cder_fin_ult1',values='age',group='sexo',title="Sex vs ind_cder_fin_ult1",legend=False,  plot_width=200)
df_bar6=Bar(df_na,label='ind_cno_fin_ult1',values='age',group='sexo',title="Sex vs ind_cno_fin_ult1",legend=False,  plot_width=200)
df_bar7=Bar(df_na,label='ind_ctju_fin_ult1',values='age',group='sexo',title="Sex vs ind_ctju_fin_ult1",legend=False,  plot_width=200)
df_bar8=Bar(df_na,label='ind_ctma_fin_ult1',values='age',group='sexo',title="Sex vs ind_ctma_fin_ult1",legend=False,  plot_width=200)
df_bar9=Bar(df_na,label='ind_ctop_fin_ult1',values='age',group='sexo',title="Sex vs ind_ctop_fin_ult1",legend=False,  plot_width=200)
df_bar10=Bar(df_na,label='ind_ctpp_fin_ult1',values='age',group='sexo',title="Sex vs ind_ctpp_fin_ult1",legend=False,  plot_width=200)
df_bar11=Bar(df_na,label='ind_deco_fin_ult1',values='age',group='sexo',title="Sex vs ind_deco_fin_ult1",legend=False,  plot_width=200)
df_bar12=Bar(df_na,label='ind_deme_fin_ult1',values='age',group='sexo',title="Sex vs ind_deme_fin_ult1",legend=False,  plot_width=200)
df_bar13=Bar(df_na,label='ind_dela_fin_ult1',values='age',group='sexo',title="Sex vs ind_dela_fin_ult1",legend=False,  plot_width=200)
df_bar14=Bar(df_na,label='ind_ecue_fin_ult1',values='age',group='sexo',title="Sex vs ind_ecue_fin_ult1",legend=False,  plot_width=200)

df_bar15=Bar(df_na,label='ind_fond_fin_ult1',values='age',group='sexo',title="Sex vs ind_fond_fin_ult1",legend=False,  plot_width=200)
df_bar16=Bar(df_na,label='ind_hip_fin_ult1',values='age',group='sexo',title="Sex vs ind_hip_fin_ult1",legend=False,  plot_width=200)
df_bar17=Bar(df_na,label='ind_plan_fin_ult1',values='age',group='sexo',title="Sex vs ind_plan_fin_ult1",legend=False,  plot_width=200)
df_bar18=Bar(df_na,label='ind_pres_fin_ult1',values='age',group='sexo',title="Sex vs ind_pres_fin_ult1",legend=False,  plot_width=200)
df_bar19=Bar(df_na,label='ind_reca_fin_ult1',values='age',group='sexo',title="Sex vs ind_reca_fin_ult1",legend=False,  plot_width=200)
df_bar20=Bar(df_na,label='ind_tjcr_fin_ult1',values='age',group='sexo',title="Sex vs ind_tjcr_fin_ult1",legend=False,  plot_width=200)

df_bar21=Bar(df_na,label='ind_valo_fin_ult1',values='age',group='sexo',title="Sex vs ind_valo_fin_ult1",legend=False,  plot_width=200)
df_bar22=Bar(df_na,label='ind_viv_fin_ult1',values='age',group='sexo',title="Sex vs ind_viv_fin_ult1",legend=False,  plot_width=200)
df_bar23=Bar(df_na,label='ind_recibo_ult1',values='age',group='sexo',title="Sex vs ind_recibo_ult1",legend=False,  plot_width=200)

show(df_bar)

#fig=figure();

#fig.add_glyphs(df_bar.get_glyphs())
#fig.add_glyphs(bar_renta.get_glyphs())
#fig.add_glyphs(p_age.get_glyphs())

output_file('visulize.html')
p=gridplot([df_bar,df_bar1,p_age],[df_bar2,df_bar3,df_bar4],[df_bar5,df_bar6,df_bar7])
show(p)

df['ind_ahor_fin_ult1'].isnull().sum()
df['ind_nom_pens_ult1'].isnull().sum()
df['ind_nomina_ult1'].isnull().sum()

df['ind_recibo_ult1'].isnull().sum()
