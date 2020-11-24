library(pacman)
p_load(tidyquant,janitor,lubridate,tidyverse,skimr,naniar,patchwork)
# economy news
ecn = read_rds('data/economy_news_전처리.rds')
range(ecn$dat) # 2014.1.22~2020.9.1
count(ecn,press)

fl = list.files('data/indices','csv',full.names=T)
snp = read_csv(fl[str_detect(fl,'500')]) %>%
  mutate(date = as.Date(날짜,format='%Y년 %m월 %d일')) %>% 
  arrange(date) %>% 
  relocate(date,.before=1) %>% 
  select(date,종가:저가) %>% 
  set_names(c('date','price','open','high','low'))
lct = Sys.getlocale("LC_TIME") # Korean_Korea.949
Sys.setlocale("LC_TIME","C") 
vix = read_csv(fl[str_detect(fl,'Volatility Index')]) %>% 
  clean_names() %>% 
  select(date:low) %>% 
  mutate(date = mdy(date)) %>% 
  arrange(date)
Sys.setlocale("LC_TIME",lct) 
# 
windowsFonts(ng = windowsFont('NanumGothic'))
theme_set(theme_tq(10,'ng'))
p1 = ggplot(vix,aes(date,price))+
  geom_line()+
  ggtitle('vix')
p2 = ggplot(snp,aes(date,price))+
  geom_line()+
  ggtitle('s&p 500')
p1/p2

# 
source('d:/r/get_table.r',encoding='utf-8')
tic()
url = 'https://www.investing.com/indices/smallcap-2000-historical-data'  
# url = "https://www.investing.com/indices/volatility-s-p-500-historical-data"
# url = "https://www.investing.com/indices/us-30-historical-data"
# url = "https://www.investing.com/indices/nasdaq-composite-historical-data"
# url = "https://www.investing.com/indices/us-spx-500-historical-data"
# url = "https://www.investing.com/commodities/crude-oil-historical-data"
# url = "https://www.investing.com/commodities/brent-oil-historical-data"
# url = "https://www.investing.com/commodities/natural-gas-historical-data"
# url = "https://www.investing.com/commodities/gold-historical-data"
# url = "https://www.investing.com/commodities/silver-historical-data"
# url = "https://www.investing.com/commodities/copper-historical-data"
a = get_table(url,sleep=10)
toc()
lct = Sys.getlocale("LC_TIME") # Korean_Korea.949
Sys.setlocale("LC_TIME","C") 
a = a %>% 
  mutate(date = mdy(date)) %>% 
  mutate_at(vars(price:low),~str_remove_all(.,',') %>% as.numeric) %>% 
  arrange(date) %>% 
  as_tibble()
Sys.setlocale("LC_TIME",lct) 
a %>% 
  saveRDS('data/indices/smallcap.rds')

library(fs)
all_idx = map_dfr(dir_ls('data/indices'),
                 function(x){
                   fn = path_file(x) %>% path_ext_remove()
                   a = read_rds(x) %>% 
                     add_column(gb=fn,.before=1)
                   return(a)
                 })
saveRDS(all_idx,'data/indices/all_idx.rds')
all_idx %>% 
  select(date,gb,change_percent) %>% 
  mutate(change_percent=str_remove(change_percent,'%') %>% 
           as.numeric()) %>% 
  pivot_wider(names_from=gb,values_from=change_percent) %>% 
  select(-date) %>% 
  corrr::correlate() %>%
  corrr::shave() %>% 
  mutate_if(is.numeric,~round(.,3))
skimr::skim(all_idx)
all_idx %>% 
  ggplot(aes(date,price,color=gb))+
  geom_line()+
  facet_wrap(~gb,scales='free_y')+
  theme_minimal(10)+
  theme(legend.position = 'none')
# 채권쪽 
urls = c(
  "https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data",
  "https://www.investing.com/rates-bonds/u.s.-30-year-bond-yield-historical-data",
  "https://www.investing.com/rates-bonds/u.s.-5-year-bond-yield-historical-data",
  "https://www.investing.com/rates-bonds/u.s.-3-month-bond-yield-historical-data",
  "https://www.investing.com/rates-bonds/us-10-yr-t-note-historical-data",
  "https://www.investing.com/rates-bonds/euro-bund-historical-data",
  "https://www.investing.com/rates-bonds/10-2-year-treasury-yield-spread-historical-data"
)
tic()
bonds = map_dfr(urls,function(x){
  a = get_table(x,sleep=10) %>% 
    add_column(gb=path_file(x),.before=1) %>% 
    as_tibble()
  print(str_c(path_file(x),' complete'))
  return(a)
})
toc()
lct = Sys.getlocale("LC_TIME") # Korean_Korea.949
Sys.setlocale("LC_TIME","C") 
bond_yld = bonds %>% 
  mutate(date = mdy(date)) %>% 
  arrange(gb,date) %>% 
  mutate(gb = str_remove(gb,'-historical-data|bond-yield'))
Sys.setlocale("LC_TIME",lct) 
bond_yld %>% 
  saveRDS('data/indices/bond_yld.rds')
count(bond_yld,gb)
all_idx_yld = bind_rows(all_idx,byld)
saveRDS(all_idx_yld,'data/indices/all_idx_yld.rds')
