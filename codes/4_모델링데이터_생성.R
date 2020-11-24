library(tidyverse)
library(lubridate)
library(explore)
library(scales)
last_clip <- function() .Last.value %>% clipr::write_clip()
top <- read_csv('data/topic_prob.csv')
# tr_te: 1(train,~2020.2), 2(test,2020.3~)
# spx_target : from make_targets.R
# target: p7(7일후 spx종가),t7(7일후 spx가격상승여부),r7(7일후가격 기준 수익률)
toptg <- top %>% 
  left_join(select(spx_target,date,p7,t7,r7),by=c('dat'='date')) %>% 
  mutate(t7 = fct_relevel(t7,'down')) %>% 
  rowwise() %>% 
  mutate(maxp = max(c_across(topic1:topic14)),
         topic = which(c_across(topic1:topic14)==maxp)[1])
toptg
group_by(toptg,topic) %>% 
  summarise(upp = mean(t7=='up',na.rm=T),
            n = n())
# 일별 기사갯수 ####
day_nws_count <- count(toptg,date=dat)
describe(day_nws_count,n)
count(day_nws_count,n)
# 일평균 기사갯수 
mean(day_nws_count$n) # 15.5
group_by(day_nws_count,year=year(date)) %>% 
  summarise(avg = mean(n))
last_clip()
# 연도별 기사갯수 
count(toptg,year=year(dat))
last_clip()

# 일별 토픽갯수(토픽다양성) ####
day_uniq_topic <- group_by(toptg,date=dat) %>% 
  summarise(n = n(),
            net_topic = n_distinct(topic))
day_uniq_topic %>% 
  ggplot(aes(date,net_topic))+
  geom_point(color='steelblue')
day_uniq_topic %>% 
  mutate(year = year(date)) %>% 
  ggplot(aes(year,net_topic))+
  stat_summary(fun='mean',color='steelblue',geom='line')+
  stat_summary(fun='mean',color='steelblue',geom='point')+
  scale_x_continuous(breaks=2014:2020)

day_uniq_topic %>% 
  mutate(year = year(date)) %>% 
  ggplot(aes(year,net_topic))+
  stat_summary(fun.data='mean_cl_boot',geom='errorbar',color='firebrick',width=0.1)+
  stat_summary(fun='mean',color='steelblue',geom='line')+
  stat_summary(fun='mean',color='steelblue',geom='point')+
  scale_x_continuous(breaks=2014:2020)
ggsave('plot/uniq_topic_trend.png',width=10,height=6,units='cm')

day_uniq_topic %>% 
  mutate(month = format(date,'%Y%m')) %>% 
  group_by(year = year(date)) %>% 
  summarise(avg = mean(net_topic))
last_clip()
# 
tr <- filter(toptg,tr_te==1) %>% filter(year(dat)>=2016) %>% 
  select(topic1:topic14,target=t7)
te <- filter(toptg,tr_te==2) %>% 
  select(topic1:topic14,target=t7)
# sentiment score ####
# from py/sentiment.ipynb
# vader sentiment score와 textblob의 polarity score 
vsent <- read_csv('data/vsent.csv')
# tosent : topic+sentiment ####
tosent <- left_join(vsent,toptg,by='id') %>% 
  select(id,date=dat,topic,everything())
ggplot(tosent,aes(factor(topic),compound))+
  geom_boxplot()
tosent %>% 
  mutate(topic=factor(topic)) %>% 
  sample_frac(.5) %>% 
  ggplot(aes(compound,polarity,color=topic))+
  geom_point()+
  facet_wrap(~topic)
library(correlationfunnel)
tr <- filter(tosent,tr_te==1) %>% filter(year(date)>=2016) %>% 
  select(topic:topic14,target=t7)
explore(tr)
# 
tosent %>% 
  filter(tr_te==1) %>% 
  ggplot(aes(t7,compound))+
  geom_boxplot()+
  facet_wrap(~topic)
# 
library(broom)
ttsent <- drop_na(tosent,t7) %>% filter(year(date)>=2016)
tt_comp <- map_dfr(1:14,~t.test(compound~t7,filter(ttsent,topic==.x),
                                alternative='two.sided') %>% 
                tidy %>% 
                add_column(topic=.x,.before=1)) %>% 
  select(topic,estimate1,estimate2,t=statistic,p.value,alternative)
tt_comp
last_clip()
# 
tt_pola <- map_dfr(1:14,~t.test(polarity~t7,filter(ttsent,topic==.x),
                                alternative='two.sided') %>% 
                     tidy %>% 
                     add_column(topic=.x,.before=1)) %>% 
  select(topic,estimate1,estimate2,t=statistic,p.value,alternative)
tt_pola
last_clip()
# 
alt = 'less'
tt_subj <- map_dfr(1:14,~t.test(subjectivity~t7,filter(ttsent,topic==.x),
                                alternative=alt) %>% 
                     tidy %>% 
                     add_column(topic=.x,.before=1)) %>% 
  select(topic,estimate1,estimate2,t=statistic,p.value,alternative)
tt_subj
last_clip()
# topic별 비중 ####
count(tosent,topic) %>% mutate(pct=percent(n/sum(n)))
last_clip()
# neg 
map_dfr(1:14,~t.test(neg~t7,filter(ttsent,topic==.x),
                     alternative='two.sided') %>% 
          tidy %>% 
          add_column(topic=.x,.before=1)) %>% 
  select(topic,estimate1,estimate2,t=statistic,p.value,alternative)
last_clip()
# neu
map_dfr(1:14,~t.test(neu~t7,filter(ttsent,topic==.x),
                     alternative='two.sided') %>% 
          tidy %>% 
          add_column(topic=.x,.before=1)) %>% 
  select(topic,estimate1,estimate2,t=statistic,p.value,alternative)
last_clip()
# pos
map_dfr(1:14,~t.test(pos~t7,filter(ttsent,topic==.x),
                     alternative='two.sided') %>% 
          tidy %>% 
          add_column(topic=.x,.before=1)) %>% 
  select(topic,estimate1,estimate2,t=statistic,p.value,alternative)
last_clip()

# tosent + doc2vec ####
lstm_data <- read_csv('data/lstm_data1.csv')
glimpse(lstm_data)
# 유효한 토픽만 남기고, 날짜별로 모든 점수와 벡터를 평균한다  
dfcand2 <- read_rds('data/ecvec_r1.rds') %>% 
  select(id,date,v1:v300) %>% 
  left_join(
    select(tosent,id,topic:subjectivity,p7,t7,r7,tr_te)
  ) %>% 
  select(id,date,topic:tr_te,v1:v300) %>% 
  filter(year(date)>=2016) %>% 
  filter(topic %in% c(3,6,7,11,12,13)) %>% 
  group_by(date) %>% 
  summarise(target = last(t7),
            tr_te = last(tr_te),
            across(neg:subjectivity,list(avg=mean,std=sd)),
            across(v1:v300,mean)) %>% 
  mutate(across(neg_avg:subjectivity_std,~replace_na(.,0)))
dfcand2 %>% write_csv('data/lstm_data2.csv')
# dfcand3: 기사선별작업 없음 
dfcand3 <- read_rds('data/ecvec_r1.rds') %>% 
  select(id,date,v1:v300) %>% 
  left_join(
    select(tosent,id,topic:subjectivity,p7,t7,r7,tr_te)
  ) %>% 
  select(id,date,topic:tr_te,v1:v300) %>% 
  filter(year(date)>=2016) %>% 
  group_by(date) %>% 
  summarise(target = last(t7),
            tr_te = last(tr_te),
            across(neg:subjectivity,list(avg=mean,std=sd)),
            across(v1:v300,mean)) %>% 
  mutate(across(neg_avg:subjectivity_std,~replace_na(.,0)))
dfcand3 %>% write_csv('data/lstm_data3.csv')
# 기사선별 t7,p7,r7 
dfcanda <- read_rds('data/ecvec_r1.rds') %>% 
  select(id,date,v1:v300) %>% 
  left_join(
    select(tosent,id,topic:subjectivity,p7,t7,r7,tr_te)
  ) %>% 
  select(id,date,topic:tr_te,v1:v300) %>% 
  filter(year(date)>=2016) %>% 
  filter(topic %in% c(3,6,7,11,12,13)) %>% 
  group_by(date) %>% 
  summarise(t7 = last(t7),
            p7 = last(p7),
            r7 = last(r7),
            tr_te = last(tr_te),
            across(neg:subjectivity,list(avg=mean,std=sd)),
            across(v1:v300,mean)) %>% 
  mutate(across(neg_avg:subjectivity_std,~replace_na(.,0)))
dfcanda %>% write_csv('data/lstm_data_alltarget.csv')
# 기사미선별 t7,p7,r7
dfcanda1 <- read_rds('data/ecvec_r1.rds') %>% 
  select(id,date,v1:v300) %>% 
  left_join(
    select(tosent,id,topic:subjectivity,p7,t7,r7,tr_te)
  ) %>% 
  select(id,date,topic:tr_te,v1:v300) %>% 
  filter(year(date)>=2016) %>% 
  group_by(date) %>% 
  summarise(t7 = last(t7),
            p7 = last(p7),
            r7 = last(r7),
            tr_te = last(tr_te),
            across(neg:subjectivity,list(avg=mean,std=sd)),
            across(v1:v300,mean)) %>% 
  mutate(across(neg_avg:subjectivity_std,~replace_na(.,0)))
dfcanda1 %>% write_csv('data/lstm_data3_alltarget.csv')
# lstm_data4 (from ld4) #### 
idx = read_rds('data/indices/all_idx.rds')
nerdv = read_csv('data/nerdv.csv')
spvx = filter(idx,gb %in% c('spx','cboe_vix')) %>% 
  select(date,gb,price) %>% 
  pivot_wider(names_from=gb,values_from=price) %>% 
  rename(vix = cboe_vix)
ld4 <- read_rds('data/ecvec_r1.rds') %>% 
  select(id,date,v1:v300) %>% 
  left_join(
    select(tosent,id,topic:subjectivity)
  ) %>% 
  filter(year(date)>=2016) %>% 
  left_join(spvx,by='date') %>% 
  fill(c(vix,spx),.direction='up') %>% 
  drop_na(spx)
select(ld4,date,vix,spx)
# 감성과 가격지수 상관분석 
library(GGally)
ggpairs(select(ld4,compound,polarity,spx,vix))
data(tips, package='reshape')
ggally_cross(tips, aes(smoker,sex))
ggally_table(tips,aes(smoker,sex,color=smoker))
library(corrr)
select(ld4,compound,polarity,spx,vix,v1) %>% 
  correlate()

library(forecast)
comp = ld4$compound
pola = ld4$polarity
spx = ld4$spx
vix = ld4$vix
ccf(comp,pola)
ccf(comp,vix)
ccf(pola,vix)
ccf(comp,spx)
ccf(pola,spx)
# topic별로 평균낸후 감성과 지수 상관분석 
count(ld4,topic)
bytopic = ld4 %>% 
  group_by(date,topic) %>% 
  summarise(across(c(neg,pos,neu,compound,polarity),mean),
            across(c(spx,vix),last)) %>% 
  ungroup()

corr = bytopic %>% 
  group_by(topic) %>% 
  nest() %>% 
  mutate(corr = map2(topic,data,~correlate(select(.y,neg,pos,neu,compound,polarity,spx,vix)) %>% 
                       focus(spx,vix) %>% 
                       add_column(topic = .x,.before=1))) %>% 
  pull(corr) %>% 
  bind_rows() %>% 
  arrange(rowname,topic)
corr
last_clip()
corr %>% 
  filter(rowname=='compound') %>% 
  arrange(desc(abs(vix)))
last_clip()
# 
count(ld4,topic)
last_clip()
# topic_sent : topic별 compound, polarity 평균점수 ####
topic_sent = ld4 %>%
  select(date,topic,compound,polarity) %>% 
  group_by(date,topic) %>% 
  summarise(across(c(compound,polarity),mean)) %>% 
  pivot_wider(names_from=topic,values_from=c(compound,polarity),
              values_fill = 0)
# daily_sent : 일자별 감성점수의 평균,표준편차 
daily_sent = ld4 %>% 
  group_by(date) %>% 
  summarise(across(c(neg,pos,neu,compound,polarity,subjectivity),list(avg=mean,std=sd))) %>% 
  mutate(across(neg_avg:subjectivity_std,~replace_na(.,0))) %>% 
  ungroup()
# daily_d2v : 일자별 평균 doc2vec
daily_d2v = ld4 %>% 
  group_by(date) %>% 
  summarise(across(v1:v300,mean)) %>% 
  ungroup()
# daily_target: 일자별 target ####
range(daily_d2v$date)
daily_target = idx %>% 
  filter(gb %in% c('spx','cboe_vix')) %>% 
  select(gb,date,price) %>% 
  pivot_wider(names_from=gb,values_from=price) %>% 
  rename(vix=cboe_vix)

# lstm_data4 ####
lstm_data4 = topic_sent %>% 
  left_join(daily_sent,by='date') %>% 
  left_join(daily_d2v,by='date') %>% 
  left_join(daily_target,by='date') %>% 
  select(date,spx,vix,everything()) %>% 
  ungroup() %>% 
  fill(c(spx,vix),.direction='up')
write_csv(lstm_data4,'data/lstm_data4.csv')

# daily_ner: 일자별 nerdv 평균 ####
daily_ner = nerdv %>% 
  group_by(date=dat) %>% 
  summarise(across(ndv1:ndv50,mean)) %>% 
  ungroup()
# lstm_data5: lstm_data4 + nerdv ####
lstm_data5 = lstm_data4 %>% 
  left_join(daily_ner,by='date')
dim(lstm_data5)
write_csv(lstm_data5,'data/lstm_data5.csv')


smp = lstm_data4 %>% 
  select(date,compound_4:subjectivity_std,v1:v300,vix) %>% 
  mutate(vix1 = lead(vix)) %>% 
  filter(year(date) %in% c(2016,2017))
library(randomForest)
rf1 = randomForest(vix1~.-date-vix,data=smp,importance=T)
varImpPlot(rf1)
imp = importance(rf1) %>% as.data.frame() %>% rownames_to_column() %>% as_tibble()
imp = arrange(imp,desc(`%IncMSE`))
head(imp$rowname,10) %>% str_c("'",.,"'",collapse=',')
last_clip()
# ccf ####
library(broom)
test = read_csv('data/test_topsent_vix2018.csv')
test = read_csv('data/test_vix2018.csv')
ccf1 = ccf(test$truth,test$pred) %>% tidy
filter(ccf1,acf==max(acf)) # real이 3일 앞선다 
# vix : 시차 2일
# vix+sentiment: 시차 3일 

# 성과분석 비교 ####
library(pacman)
p_load(tidyverse,clipr,lubridate)
pef1 = read_clip_tbl() %>% as_tibble()
pef2 = read_clip_tbl() %>% as_tibble()
pef3 = read_clip_tbl() %>% as_tibble()
pef = bind_rows(pef1,pef2,pef3)
pef %>% 
  filter(lookback==7) %>% 
  ggplot(aes(feat,rmse))+
  geom_boxplot()
pef %>% 
  filter(lookback==7) %>% 
  filter(test==2018)
