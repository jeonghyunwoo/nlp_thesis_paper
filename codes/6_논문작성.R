# 논문 보고서 
path = '논문작성/논문삽입 그림'
ecnws = read_rds('data/economy_news_전처리.rds')
dim(ecnws) # 19230 9
# 1.서 론 
# 2.데이터 및 분석방법 ####
windowsFonts(ng = windowsFont('NanumGothic'))
count(ecnws,year = year(dat)) %>% 
  mutate(year = factor(year)) %>% 
  ggplot(aes(year,n))+
  geom_col(fill='steelblue')+
  geom_text(aes(label=n),family='ng',vjust=-1,size=3)+
  # coord_flip()+
  scale_y_continuous(expand = expansion(mult=c(0,.1)))+
  labs(y='')+
  theme_tufte(10,'ng')
# 연도별 일평균 기사수 
ecnws %>% 
  group_by(dat) %>% 
  summarise(n = n()) %>% 
  group_by(year = factor(year(dat))) %>% 
  summarise(avgn = mean(n)) %>% 
  ggplot(aes(year,avgn))+
  geom_col(fill='grey')+
  geom_text(aes(label=round(avgn,1)),family='ng',vjust=-1,size=3)+
  scale_y_continuous(expand = expansion(mult=c(0,.1)))+
  labs(y='')+
  theme_tufte(10,'ng')
# VIX, SPX 추이 
idx = read_rds('data/indices/all_idx.rds')
idx %>% 
  filter(gb %in% c('spx','cboe_vix')) %>% 
  mutate(gb = ifelse(gb=='cboe_vix','VIX','S&P')) %>% 
  mutate(gb = fct_relevel(gb,'VIX')) %>% 
  ggplot(aes(date,price,color=gb))+
  geom_line()+
  geom_vline(xintercept = ymd(c(20140101,20160101,20180101,20200101)),
             color='grey',linetype=2)+
  facet_wrap(~gb,scales='free')+
  theme_few(10,'ng')+
  theme(legend.position = 'none',
        strip.text = element_text(size=13))
# 950 300

# vixud1,spxud1 성과분석 ####
last_clip = function() .Last.value %>% clipr::write_clip()
ud = list.files('data','*ud1_perfs.csv',full.names=T)
vixud = read_csv(ud[2]) %>% select(-X1)
spxud = read_csv(ud[1]) %>% select(-X1)
vixud = vixud %>% 
  mutate(feature = fct_relevel(feature,'topic_sents','total_sents','d2v','d2vpca','tx1'))
spxud = spxud %>% 
  mutate(feature = fct_relevel(feature,'topic_sents','total_sents','d2v','d2vpca','tx1'))
vixud %>% 
  select(feature,lookback,value=recall) %>% 
  pivot_wider(names_from=feature,values_from=value)
last_clip()
spxud %>% 
  select(feature,lookback,value=recall) %>% 
  pivot_wider(names_from=feature,values_from=value)
last_clip()
# tanh 성과분석 ####
ud2 = list.files('data','*ud1_tanh_perfs.csv',full.names = T)
vixud2 = read_csv(ud2[2]) %>% select(-X1)
spxud2 = read_csv(ud2[1]) %>% select(-X1)
vixud2 = vixud2 %>% 
  mutate(feature = fct_relevel(feature,'topic_sents','total_sents','d2v','d2vpca','tx1'))
spxud2 = spxud2 %>% 
  mutate(feature = fct_relevel(feature,'topic_sents','total_sents','d2v','d2vpca','tx1'))
vixud2 %>% 
  select(feature,lookback,value=recall) %>% 
  pivot_wider(names_from=feature,values_from=value)
last_clip()
spxud2 %>% 
  select(feature,lookback,value=recall) %>% 
  pivot_wider(names_from=feature,values_from=value)
last_clip()
# 주가예측 논문 
library(tidyverse)
paper = list.files('d:/논문자료_황택주','*.pdf',recursive=T)
length(paper)
paper = tibble(title=paper)
paper %>% 
  filter(str_detect(title,'주가|예측|VIX')) %>% 
  filter(str_detect(title,'주가|금융'))
  # mutate(title=str_extract(title,'\\[.*\\].pdf'))
.Last.value %>% clipr::write_clip()
# 
list.files('data','lstm*')
ld5 = read_csv('data/lstm_data5.csv')
names(ld5)
p1=ld5 %>% 
  ggplot(aes(date,compound_avg))+
  geom_line()
p4=ld5 %>% 
  ggplot(aes(date,polarity_avg))+
  geom_line()
ld5 %>% 
  ggplot(aes(date,subjectivity_avg))+
  geom_line()
p2=ld5 %>% 
  ggplot(aes(date,vix))+
  geom_line()
p3=ld5 %>% 
  ggplot(aes(date,compound_std))+
  geom_line()
library(patchwork)
p1/p2/p4
library(corrr)
library(lubridate)
last_clip = function() .Last.value %>% clipr::write_clip()  
# compound 상관관계 ####
ld5 %>% 
  mutate(year=year(date)) %>% 
  select(year,vix:compound_8,compound_avg) %>% 
  group_by(year) %>% 
  nest() %>% 
  mutate(cor = map(data,
                   ~correlate(.x) %>% 
                     focus(vix) %>% 
                     arrange(desc(abs(vix))))) %>% 
  unnest(cor) %>% 
  select(-data) %>% 
  pivot_wider(names_from=year,values_from=vix)
last_clip()
# vix와 모든 감성점수 상관관계 ####
ld5 %>% 
  mutate(year=year(date)) %>% 
  select(year,vix,compound_4:subjectivity_std) %>% 
  group_by(year) %>% 
  nest() %>% 
  mutate(cor = map(data,
                   ~correlate(.x) %>% 
                     focus(vix))) %>% 
  unnest(cor) %>% 
  select(-data) %>% 
  pivot_wider(names_from=year,values_from=vix) %>% 
  arrange(rowname)
    
last_clip()
# spx와 모든 감성점수 상관관계 ####
ld5 %>% 
  mutate(year=year(date)) %>% 
  select(year,spx,compound_4:subjectivity_std) %>% 
  group_by(year) %>% 
  nest() %>% 
  mutate(cor = map(data,
                   ~correlate(.x) %>% 
                     focus(spx))) %>% 
  unnest(cor) %>% 
  select(-data) %>% 
  pivot_wider(names_from=year,values_from=spx) %>% 
  arrange(rowname)
last_clip()
# vix,spx와 모든 감성점수 상관관계 ####
ld5 %>% 
  select(spx:ndv50) %>% 
  correlate() %>% 
  focus(vix,spx) %>% 
  arrange(rowname)
last_clip()
# 표1 vix,spx 기술통계 ####
library(tidyverse)
library(broom)
vsx <- read_rds('data/indices/all_idx.rds') %>% 
  filter(gb %in% c('spx','cboe_vix'))
glimpse(vsx)
summary(vsx)
ecnws = read_rds('data/economy_news_전처리.rds')
library(lubridate)
ecnws = ecnws %>% 
  filter(dat<=ymd(20200831))
summarise(ecnws,across(dat,list(from=min,to=max)))
n_distinct(ecnws$dat)
nrow(ecnws)
df = read_csv('data/lstm_data5.csv')
tail(df)
# 연도별 기사갯수 ####
windowsFonts(ng = windowsFont('NanumGothic'))
library(ggforce)
library(ggthemes)
library(patchwork)
library(lubridate)
library(scales)
library(tidyverse)
p1 = ecnws %>% 
  mutate(year = year(dat) %>% as.character) %>% 
  count(year) %>% 
  ggplot(aes(year,n))+
  geom_col(fill='lightsteelblue')+
  geom_text(aes(label=n),size=3,family='ng',
            vjust=-1)+
  geom_vline(xintercept = 2.5,color='red',linetype=2)+
  labs(x='',y='')+
  ggtitle('연도별 총 기사 개수')+
  scale_y_continuous(expand=expansion(mult=c(0,.2)))+
  theme_tufte(11,'ng')+
  theme(axis.line.x = element_line(color='grey30'),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        plot.title = element_text(face='bold',size=11))
p2 = ecnws %>% 
  count(dat) %>% 
  mutate(year = year(dat) %>% as.character) %>% 
  group_by(year) %>% 
  summarise(avg = mean(n)) %>% 
  ggplot(aes(year,avg))+
  geom_col(fill='lightsteelblue')+
  geom_text(aes(label=comma(avg,accuracy=.1)),size=3,family='ng',
            vjust=-1)+
  geom_vline(xintercept = 2.5,color='red',linetype=2)+
  labs(x='',y='')+
  ggtitle('연도별 일평균 기사 개수')+
  scale_y_continuous(expand=expansion(mult=c(0,.2)))+
  theme_tufte(11,'ng')+
  theme(axis.line.x = element_line(color='grey30'),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        plot.title = element_text(face='bold',size=11))
p1+p2
ggsave('논문작성/논문삽입 그림/기사갯수통계.png',dpi=2000,
       width = 14, height = 6, units='cm')
# 수집 기사 예시 
ecnws %>% filter(title!=text) %>% 
  slice_head(n=3) %>% 
  bind_rows(ecnws %>% 
              filter(title!=text) %>% 
              slice_tail(n=3)) %>% 
  select(date=dat,title,text) %>% 
  write_csv('data/기사샘플.csv')
last_clip()
ecnws %>% 
  sample_n(6) %>% 
  select(date=dat,title,text) %>% 
  arrange(date) %>% 
  write_csv('data/기사샘플.csv')
# 가격지수 트렌드 ####
vsx %>% 
  mutate(gb = ifelse(gb=='cboe_vix','VIX','S&P 500')) %>% 
  mutate(gb = fct_relevel(gb,'VIX')) %>% 
  ggplot(aes(date,price))+
  geom_line(color='steelblue')+
  geom_vline(xintercept = ymd(20200101),color='grey',
             linetype=2)+
  scale_y_continuous(expand = expansion(mult=.1))+
  facet_wrap(~gb,scales='free')+
  labs(x='',y='')+
  theme_tufte(11,'ng')+
  theme(legend.position = 'none',
        strip.text = element_text(face='bold',size=10),
        axis.line = element_line(color='grey'))
ggsave('논문작성/논문삽입 그림/가격지수trend.png',dpi=2000,
       width = 14, height = 6, units='cm')

# 벡터차원
select(df,neg_avg:subjectivity_std) %>% dim

# train,test up/down ####
vud = read_csv('data/cl_vixud_trte.csv') %>% 
  select(gb,vixud1)
sud = read_csv('data/cl_spxud_trte.csv') %>% 
  select(gb,spxud1)
vud %>% 
  count(gb,vixud1) %>% 
  pivot_wider(names_from=gb,values_from=n)
last_clip()
sud %>% 
  count(gb,spxud1) %>% 
  pivot_wider(names_from=gb,values_from=n)
last_clip()
# vix1 실험결과 ####
library(clipr)
vx1rst = read_clip_tbl() %>% as_tibble()
vx1rst = vx1rst %>% 
  pivot_longer(cols=-특징변수.조합,names_to='lookback',values_to='rmse') %>% 
  set_names(c('feat','lookback','rmse')) %>% 
  mutate(lookback=str_remove(lookback,'X')) %>% 
  group_by(feat) %>% 
  mutate(avg = mean(rmse))
vx1rst %>% write_csv('논문작성/vx1rst.csv')
vx1rst = read_csv('논문작성/vx1rst.csv')
vx1rst %>% 
  ggplot(aes(reorder(feat,avg),rmse))+
  geom_boxplot()+
  # geom_rangeframe()+
  stat_summary(fun='mean',geom='point',shape=21,color='red')+
  labs(x='',y='RMSE')+
  scale_x_discrete(guide=guide_axis(n.dodge=2))+
  # annotate('text',x=1,y=21,label='avg.',family='ng',
  #          size=3,color='red')+
  annotate('text',x=1,y=20,label='Best',family='ng',
           fontface='bold',color='firebrick')+
  theme_grey(11,'ng')
  # theme_tufte(11,'ng')
ggsave(file.path(path,'실험결과vix1.png'),dpi=3000,
       width=10,height=6,units='cm')
# spx1 실험결과 ####
sp1rst = read_clip_tbl() %>% as_tibble()
sp1rst = sp1rst %>% 
  pivot_longer(cols=-특징변수.조합,names_to='lookback',values_to='rmse') %>% 
  set_names(c('feat','lookback','rmse')) %>% 
  mutate(lookback=str_remove(lookback,'X')) %>% 
  group_by(feat) %>% 
  mutate(avg = mean(rmse))
sp1rst %>% write_csv('논문작성/sp1rst.csv')
sp1rst %>% 
  ggplot(aes(reorder(feat,avg),rmse))+
  geom_boxplot()+
  stat_summary(fun='mean',geom='point',shape=21,color='red')+
  labs(x='',y='RMSE')+
  scale_x_discrete(guide=guide_axis(n.dodge=2))+
  annotate('text',x=1.7,y=487,label='Best',family='ng',
           fontface='bold',color='firebrick')+
  annotate('point',x=1,y=486.06,shape=16,color='black')+
  theme_grey(11,'ng')
ggsave(file.path(path,'실험결과spx1.png'),dpi=3000,
       width=10,height=6,units='cm')
# vixud1 실험결과 ####
vxud1 = read_clip_tbl() %>% as_tibble()
vxud1 = vxud1 %>% 
  fill(metric,.direction='down') %>% 
  pivot_longer(cols=-c(metric,lookback),
               names_to='feat',
               values_to='val')
vxud1 %>% write_csv('논문작성/vxud1rst.csv')
vxud1 %>% 
  pivot_wider(names_from=metric,values_from=val) %>% 
  select(feat,lookback,f1,acc) %>% 
  pivot_longer(cols=c(f1,acc)) %>% 
  filter(name=='f1') %>% 
  pivot_wider(id_cols=feat,names_from=lookback,values_from=value)
last_clip()
vxud1 %>% 
  pivot_wider(names_from=metric,values_from=val) %>% 
  select(feat,lookback,f1,acc) %>% 
  pivot_longer(cols=c(f1,acc)) %>% 
  filter(name=='acc') %>% 
  pivot_wider(id_cols=feat,names_from=lookback,values_from=value)
last_clip()
vxud1 %>% 
  filter(metric=='f1') %>% 
  mutate(feat=str_replace(feat,'tx','comb.')) %>% 
  group_by(feat) %>% 
  mutate(avg = mean(val)) %>% 
  ggplot(aes(reorder(feat,-avg),val))+
  geom_boxplot()+
  stat_summary(fun='mean',geom='point',shape=21,color='firebrick')+
  labs(x='',y='F1')+
  scale_x_discrete(guide=guide_axis(n.dodge=2))+
  # annotate('text',x=1.7,y=487,label='Best',family='ng',
  #          fontface='bold',color='firebrick')+
  # annotate('point',x=1,y=486.06,shape=16,color='black')
  theme_grey(11,'ng')
ggsave(file.path(path,'실험결과vxud1.png'),dpi=3000,
       width=10,height=6,units='cm')
# spxud1 실험결과 ####
spxud1 = read_clip_tbl() %>% as_tibble()
spxud1 = spxud1 %>% 
  fill(metric,.direction='down') %>% 
  pivot_longer(cols=-c(metric,lookback),
               names_to='feat',
               values_to='val')
spxud1 %>% write_csv('논문작성/spxud1rst.csv')
spxud1 %>% 
  pivot_wider(names_from=metric,values_from=val) %>% 
  select(feat,lookback,f1,acc) %>% 
  pivot_longer(cols=c(f1,acc)) %>% 
  filter(name=='f1') %>% 
  pivot_wider(id_cols=feat,names_from=lookback,values_from=value)
last_clip()
