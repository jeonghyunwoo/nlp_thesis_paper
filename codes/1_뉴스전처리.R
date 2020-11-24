# commodity, economy news전처리해서 저장한다 ####
library(tidyverse)
library(lubridate)
library(textfeatures)
library(tictoc)
# 추출기준일: 2020.8.23
lct <- Sys.getlocale("LC_TIME") # Korean_Korea.949
Sys.setlocale("LC_TIME","C") 
fl=list.files('data','comm_nws.*csv',full.names=T)
df1 = map_dfr(fl,read_csv) %>% 
  mutate(press = str_remove(press,'By ')) %>% 
  mutate(date = ifelse(str_detect(date,'hour|ago'),'Aug 23, 2020',date)) %>% 
  mutate(dat = as.Date(date,'%b %d, %Y')) %>% 
  mutate(weekday = weekdays(dat,T)) 
Sys.setlocale("LC_TIME",lct)
saveRDS(df1,'data/commodity_news_raw.rds')
# df2 : df1중복제거 ####
df1 = read_rds('data/commodity_news_raw.rds')
df2 = distinct(df1,press,text,.keep_all = T) %>% 
  mutate(textlen = nchar(text))
df2 = arrange(df2,dat) %>% 
  mutate(id = row_number()) %>% 
  relocate(id,.before=1)
saveRDS(df2,'data/commodity_news_전처리.rds')
dim(df2) # 29169, 다른 언론사에 중복기사 있음 

# economy news ####
lct <- Sys.getlocale("LC_TIME") # Korean_Korea.949
Sys.setlocale("LC_TIME","C") 
fl=list.files('data','econ_nws.*csv',full.names=T)
df1 = map_dfr(fl,read_csv) %>% 
  mutate(press = str_remove(press,'By ')) %>% 
  mutate(date = ifelse(str_detect(date,'hour|ago'),'Aug 29, 2020',date)) %>% 
  mutate(dat = as.Date(date,'%b %d, %Y')) %>% 
  mutate(weekday = weekdays(dat,T)) 
Sys.setlocale("LC_TIME",lct)
saveRDS(df1,'data/economy_news_raw.rds')
# df2 : df1중복제거 ####
df2 = distinct(df1,href,.keep_all = T) %>% 
  mutate(textlen = nchar(text)) %>% 
  filter(!is.na(text))
df2 = arrange(df2,dat) %>% 
  mutate(id = row_number()) %>% 
  relocate(id,.before=1)
saveRDS(df2,'data/economy_news_전처리.rds')
dim(df2) # 32330 9

# textfeatures ####
tic()
df2feat = textfeatures(df2,normalize = F)
toc()
# 1129.7 sec elapsed
df2feat = bind_cols(
  select(df2,id,dat),
  df2feat
)
saveRDS(df2feat,'data/commodity_news_feature.rds')
# df2feat = read_rds('data/comm_nws_feat.rds')


# rnewsflow ####
library(quanteda)
library(RNewsflow)
corp = corpus(df2[1:1000,] %>% select(-href,-date),
              docid_field = 'id',text_field = 'text')


# spacyr NER ####
# quanteda::corpus --> spacyr::spacy_parse 
# cleanNLP spacy는 안된다 
library(spacyr)
spacy_initialize(model = 'en_core_web_sm')
corp = corpus(slice(df2,1:10) %>% select(-href,-date),docid_field = 'id',text_field = 'text')
ett = spacy_parse(corp,nounphrase = T) %>% as_tibble()
count(ett,ent=str_remove(entity,'_B|_I'),sort=T) %>% 
  na_if("") %>% 
  drop_na()


ents = df2 %>% 
  sample_n(100) %>%
  pull(text) %>% 
  spacy_parse(pos=F,lemma=F) %>% 
  as_tibble() %>% 
  mutate(enty = na_if(entity,"")) %>% 
  drop_na()


# 
library(quanteda)
df = read_rds('data/commodity_news_전처리.rds')
txt = df$text[1]
tokens(txt,what='sentence')
tokens(txt) %>% tokens_compound(pattern = phrase(c('German media','Dominic Cognata')))
corp = corpus(df,docid_field = 'id',text_field = 'text')
docvars(corp,'year') = lubridate::year(df$dat)
corp_2018 = corpus_subset(corp,year==2018)
summary(corp_2018)
dfmat = dfm(corp_2018,remove=stopwords('en'),remove_punct=T,stem = T)
dfmat[,1:5]
topfeatures(dfmat,n=10)
textplot_wordcloud(dfmat,min_count=10)
dfm2011 <- dfm(corpus_subset(corp,year==2011),remove=stopwords('en'),
               remove_punct=T,stem=T)
dfm2019 <- dfm(corpus_subset(corp,year==2019),remove=stopwords('en'),
               remove_punct=T,stem=T)
textstat_simil(dfm2011,dfm2019,margin='documents',method='cosine')
vixf = list.files('data','vix',full.names=T)
vix = read_csv(vixf,skip=1) %>% 
  janitor::clean_names() %>% 
  mutate(date = mdy(date))
windowsFonts(ng = windowsFont('NanumGothic'))
vix %>% 
  ggplot(aes(date,vix_close))+
  geom_line(color='steelblue')+
  theme_grey(10,'ng')
df2feat = read_rds('data/commodity_news_feature.rds')
df2vec = df2feat %>% 
  group_by(dat) %>% 
  summarise(across(starts_with('w'),mean)) %>% 
  ungroup()
glimpse(df2vec)
vec2vix = vix %>% 
  inner_join(df2vec,by=c('date'='dat'))
dim(vec2vix)
head(vec2vix)
# pca 만들어보자 