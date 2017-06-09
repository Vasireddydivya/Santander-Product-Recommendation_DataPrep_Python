library(dplyr)
library(tidyr)
library(data.table)
library(ggplot2)
library(lubridate)
library(fasttime)

#ggplot theme
my_theme<-theme_bw()+theme(axis.title = element_text(size=24),plot.title = element_text(size=36),axis.text = element_text(size=16))
my_theme_dark<-theme_dark()+theme(axis.title = element_text(size=24),plot.title = element_text(size = 36),axis.text = element_text(size=16))

#loading the data
set.seed(1)
setwd("C:/Users/vasir/Desktop/ADM")
df<-fread("train_ver2.csv",nrows = 5000000)
test<-fread("test_ver2.csv")

#features into vector
features<-names(df)[grepl("ind_+.*ult.*",names(df))]

#Adding label for each product and month that indicates whether a customer added, dropped or maintained that service.
df<-df %>% arrange(fecha_dato) %>% as.data.table()
df$month.id<-as.numeric(factor(df$fecha_dato))
df$previous.month.id<-df$month.id-1

test$month.id<-max(df$month.id)+1
test$previous.month.id<-max(df$month.id)

#merging the train and test data.
test<-merge(test,df[,names(df) %in% c(features,'ncodpers','month.id'),with=FALSE],by.x = c("ncodpers","previous.month.id"),by.y = c("ncodpers","month.id"),all.x = TRUE)

#converting the fecha_dato and fecha_alta to date format
df[,fecha_dato:=fastPOSIXct(fecha_dato)]
df[,fecha_alta:=fastPOSIXct(fecha_alta)]
unique(df$fecha_dato)

#Customers buy products at certain months like cristimas bonues
df$month<-month(df$fecha_dato)


#checking for missing values
dfNA<-data.frame(sapply(df,function(x) any(is.na(x))))

#Data Cleaning
ggplot(df,aes(x=age))+geom_bar(alpha=0.75,fill='tomato',color='black')+ggtitle("Age Distribution")+my_theme

#There are customers who with age less than 15 and some customers with age more than 100. The distribution of plot is bimodal.
#One peak is at university aged students and another peak is around middle age.

#age
df$age[(df$age<18)]<-median(df$age[(df$age>=18) & (df$age<=30)],na.rm = TRUE)
df$age[(df$age>100)]<-median(df$age[(df$age>=30) & (df$age<=100)],na.rm = TRUE)
df$age[is.na(df$age)]<-median(df$age,na.rm = TRUE)

#ind_neuvo
sum(is.na(df$ind_nuevo))
#This parameter indicates whether the customer is new or not. So, we will check how many months the customer is active.
months.active<-df[is.na(df$ind_nuevo),] %>% group_by(ncodpers) %>% summarise(months.active=n()) %>% select(months.active)
max(months.active)
df$ind_nuevo[is.na(df$ind_nuevo)]<-1

#antiguedad
sum(is.na(df$antiguedad)) #same number again
summary(df[is.na(df$antiguedad),]%>% select(ind_nuevo))
#the same customers, so replace the 
new.antiguedad<-df %>% dplyr::select(ncodpers,month.id,antiguedad) %>% dplyr::group_by(ncodpers) %>% dplyr::mutate(antiguedad=min(antiguedad,na.rm=T),month.id-6) %>% ungroup() %>% dplyr::arrange(ncodpers) %>% dplyr::select(antiguedad)
df<-df %>% arrange(ncodpers)
df$antiguedad<-new.antiguedad$antiguedad
df$antiguedad[df$antiguedad<0]<- -1

elapsed_months<-function(end_date,start_date){
  12*(year(end_date)-year(start_date))+(month(end_date)-month(start_date))
}

recalculated.antiguedad<-elapsed_months(df$fecha_dato,df$fecha_alta)
df$antiguedad[!is.na(df$fecha_alta)]<-recalculated.antiguedad[!is.na(df$fecha_alta)]
df$ind_nuevo<-ifelse(df$antiguedad<=6,1,0)

#fecha_alta
df$fecha_alta[is.na(df$fecha_alta)]<-median(df$fecha_alta,na.rm=TRUE)

#indrel
table(df$indrel)
df$indrel[is.na(df$indrel)]<-0

#ind_actividad_cliente
df$ind_actividad_cliente[is.na(df$ind_actividad_cliente)]<-median(df$ind_actividad_cliente,na.rm = TRUE)

#Deleteing the two predictors 'cod_prov' and 'tipodom' which are needed because the nomprov predictor will cover the city.
df<-df %>% select(-cod_prov,-tipodom)

#renta
sum(is.na(df$renta))
#There are so many NA's for this parameter. So, checking the median income with respect to city and replacing the NA values with median income w.r.t particular city
df %>% filter(!is.na(renta)) %>% group_by(nomprov) %>% summarise(med.income=median(renta)) %>% arrange(med.income) %>% mutate(city=factor(nomprov,levels=nomprov)) %>% ggplot(aes(x=city,y=med.income))+geom_point(color='#b60c1e')+xlab("City")+ylab("Median income")+my_theme+geom_text(aes(x=city,y=med.income,label=city),angle=90,hjust=-.25)+ggtitle("Income Distribution by City")

new.incomes<-df %>% select(nomprov) %>% merge(df %>% group_by(nomprov) %>% summarise(med.income=median(renta,na.rm=TRUE)),by='nomprov') %>% select(nomprov,med.income) %>% arrange(nomprov)
df<-df %>% arrange(nomprov)
df$renta[is.na(df$renta)]<-new.incomes$med.income[is.na(df$renta)]

df$renta[is.na(df$renta)]<-median(df$renta,na.rm = TRUE)

#ind_nomina_ult1,ind_nom_pens_ult1
df$ind_nom_pens_ult1[is.na(df$ind_nom_pens_ult1)]<-0
df$ind_nomina_ult1[is.na(df$ind_nomina_ult1)]<-0

#creating lag feature
create.lag.feature<-function(df, 
                             feature.name,
                             months.to.lag=1,
                             by=c("ncodpers","month.id"),
                             na.fill=NA){
  df.sub<-df[,mget(c(by,feature.name))]
  names(df.sub)[names(df.sub)==feature.name]<-"original.feature"
  original.month.id<-df.sub$month.id
  added.names<-c()
  for(month.ago in months.to.lag){
    print(paste("Collecting information on",feature.name,month.ago,"month(s) ago"))
    colname<-paste("lagged.",feature.name,".",month.ago,"months.ago",sep = "")
    added.names=c(colname,added.names)
    df.sub<-merge(df.sub,df.sub[,.(ncodpers,
                                   month.id=month.ago+original.month.id,
                                   lagged.feature=original.feature)],
                  by=by,
                  all.x=TRUE,
                  sort=FALSE)
    names(df.sub)[names(df.sub)=="lagged.feature"]<-colname
  }
  dt<-merge(df,df.sub[,c(by,added.names),with=FALSE],by=by,all.x = TRUE,sort = FALSE)
  dt[is.na(dt)]<-na.fill
  return(dt)
}

df<-as.data.table(df)
df<-create.lag.feature(df,'ind_actividad_cliente',1:8,na.fill=0)

df[,last.age:=lag(age),by="ncodpers"]
df$turned.adult <- ifelse(df$age==20 & df$last.age==19,1,0)

df<-as.data.frame(df)

#Dividing the datasets into test and train
test<-df %>% filter(month.id==max(month.id))
df<-df %>% filter(month.id<max(month.id))

write.csv(df,"cleaned_train.csv",row.names=FALSE)
write.csv(test,"cleaned_test.csv",row.names=FALSE)

#Data Visualizations
features<-names(df)[grepl("ind_+.*ult.*",names(df))]
df[,features]<-lapply(df[,features],function(x)as.integer(round(x)))
df$totalservices<-rowSums(df[,features],na.rm = TRUE)
df<-df %>% arrange(fecha_dato)
df$month.id<-as.numeric(factor(df$fecha_dato))
df$next.month.id<-df$month.id+1

#Adding the meaningful labels for differences in the months
status.change<-function(x){
  if(length(x)==1){
    label=ifelse(x==1,"Added","Maintained")
  }else{
    diffs<-diff(x)
    diffs<-c(0,diffs)
    label<-rep("Maintained",length(x))
    label<-ifelse(diffs==1,'Added',ifelse(diffs==-1,"Dropped","Maintained"))
  }
  label
}

#Applying this function on all the features
df[,features]<-lapply(df[,features],function(x)return(ave(x,df$ncodpers,FUN = status.change)))

interesting<-rowSums(df[,features]!='Maintained')
df<-df[interesting>0,]
df<-df %>% gather(key=feature,value=status,ind_ahor_fin_ult1:ind_recibo_ult1)
df<-filter(df,status!='Maintained')
head(df)

totals.by.feature<-df %>% group_by(month,feature) %>% summarise(counts=n())
df %>% 
  group_by(month,feature,status) %>% summarise(counts=n()) %>% ungroup() %>% 
  inner_join(totals.by.feature,by=c('month','feature')) %>% 
  mutate(counts=counts.x/counts.y) %>% 
  ggplot(aes(y=counts,x=factor(month.abb[month],levels=month.abb[seq(12,1,-1)])))+
  geom_bar(aes(fill=status),stat = 'identity')+
  facet_wrap(facets = ~feature,ncol=6)+
  coord_flip()+
  my_theme_dark+ylab("count")+xlab("")+ggtitle("Relative Service By Month")+
  theme(axis.text = element_text(size=10),legend.text = element_text(size=14),strip.text = element_text(face='bold'),legend.title = element_blank())+
  scale_fill_manual(values = c('cyan','magenta'))

#product changes over year

#Feature Engineering
df<-fread("cleaned_train.csv")
labels<-names(df)[grepl("ind_+.*_+ult",names(df))]
cols<-c("ncodpers","month.id","previous.month.id",labels)
df<-df[,names(df) %in% cols, with=FALSE]
df<-merge(df,df,by.x = c("ncodpers","previous.month.id"),by.y = c("ncodpers","month.id"),all.x = TRUE)
df[is.na(df)]<-0

products<-rep("",nrow(df))
for(lable in labels){
  colx<-paste0(label,".x")
  coly<-paste0(label,".y")
  diffs<-df[,.(get(colx)-get(coly))]
  products[diffs>0]<-paste0(products[diff>0],label,sep=' ')
}
df<-df[,.(ncodpers,month.id,products)]
write.csv(df,"purchased_products.csv",row.names = FALSE)

#feature-purchase-frequency
df<-fread("cleaned_train.csv")
labels<-names(df)[grepl("ind_+.*_+ult",names(df))]
cols<-c("ncodpers","month.id","previous.month.id",labels)
df<-df[,names(df) %in% cols, with=FALSE]
df<-merge(df,df,by.x = c("ncodpers","previous.month.id"),by.y = c("ncodpers","month.id"),all.x = TRUE)
df[is.na(df)]<-0

products<-rep("",nrow(df))
num.trasactions<-rep(0,nrow(df))
purchase.frequencies<-data.frame(ncodpers=df$ncodpers,month.id=(df$previous.month.id+2))
for(label in labels){
  colx<-paste0(label,".x")
  coly<-paste0(label,".y")
  diffs<-df[,.(ncodpers,month.id,change=get(colx)-get(coly))]
  num.trasactions<-num.trasactions+as.integer(diffs$change!=0)
  diffs[diffs<0]<-0
  setkey(diffs,ncodpers)
  d<-diffs[,.(frequency=cumsum(change)),by=ncodpers]
  purchase.frequencies[[paste(label,"_purchase.count",sep="")]]<-d$frequency
}
purchase.frequencies$num.transactions<-num.trasactions
purchase.frequencies<-purchase.frequencies %>% dplyr::group_by(ncodpers) %>% dplyr::mutate(num.transactions=cumsum(num.transactions))

write.csv(purchase.frequencies,"purchase.frequencies.csv",row.names = FALSE)

#Number of months since products was last owned
months.since.owned<-function(dt,products,months.to.search,default.value=999){
  for(product in products){
    print(paste("Finding months since owning",product))
    colname<-paste(product,".last.owned",sep="")
    for(month.ago in seq(months.to.search,1,-1)){
      cur.colname<-paste(product,"_",month.ago,"month_ago",sep="")
      dt[[colname]][dt[[cur.colname]]==1]<-month.ago
    }
  }
  return(dt)
}


#