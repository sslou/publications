# To run script in Rstudio console: source('NSQIP_clean.R')

library(dplyr)
library(readr)
library(data.table)

mutate_puf = function(puf) {
  df = puf %>%
    # Assign missing values
    mutate( 
      Age = as.numeric(substr(Age, start=1, stop=2)), # 90+ makes this var a fctr rather than int
      ASA = as.numeric(substr(ASACLAS, start=1, stop=1)), # will assign NAs as well
      ELECTSURG = na_if(ELECTSURG, 'Unknown'),
      DIABETES = na_if(DIABETES, 'NULL'),
      PRSODM = na_if(PRSODM, -99),
      PRCREAT = na_if(PRCREAT, -99),
      PRALBUM = na_if(PRALBUM, -99),
      PRBILI = na_if(PRBILI, -99),
      PRHCT = na_if(PRHCT, -99),
      PRPLATE = na_if(PRPLATE, -99),
      PRPTT = na_if(PRPTT, -99),
      PRINR = na_if(PRINR, -99),
      PRWBC = na_if(PRWBC, -99),
      DOTHBLEED = na_if(DOTHBLEED, -99),
    ) %>%
    # Recode variables
    mutate( 
      SDSA = case_when(
        is.na(HtoODay) ~ NA_real_,
        HtoODay == 0 ~ 1,
        HtoODay > 0 ~ 0 ),
      SEX = recode(as.character(SEX), 'female' = 1, 'male' = 0), # will fill NA if it doesnt take these vals
      INOUT = recode(as.character(INOUT), 'Inpatient' = 1, 'Outpatient' = 0),
      EMERGNCY = recode(as.character(EMERGNCY), 'Yes' = 1, 'No' = 0),
      ELECTSURG = recode(as.character(ELECTSURG), 'Yes' = 1, 'No' = 0),
      DIABETES = recode(as.character(DIABETES), 'NO' = 0, 'NON-INSULIN' = 1, 'INSULIN' = 2),
      SMOKE = recode(as.character(SMOKE), 'Yes' = 1, 'No' = 0),
      HXCOPD = recode(as.character(HXCOPD), 'Yes' = 1, 'No' = 0),
      HXCHF = recode(as.character(HXCHF), 'Yes' = 1, 'No' = 0),
      HYPERMED = recode(as.character(HYPERMED), 'Yes' = 1, 'No' = 0),
      RENAFAIL = recode(as.character(RENAFAIL), 'Yes' = 1, 'No' = 0),
      DIALYSIS = recode(as.character(DIALYSIS), 'Yes' = 1, 'No' = 0),
      STEROID = recode(as.character(STEROID), 'Yes' = 1, 'No' = 0),
      BLEEDDIS = recode(as.character(BLEEDDIS), 'Yes' = 1, 'No' = 0)
    ) %>%
    select(PUFYEAR, CaseID, SEX, PRNCPTX, CPT, INOUT, ASA, SDSA, EMERGNCY, ELECTSURG,
           Age, HEIGHT, WEIGHT,
           DIABETES, SMOKE, HXCOPD, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, STEROID, BLEEDDIS,
           PRSODM, PRCREAT, PRALBUM, PRBILI, PRHCT, PRPLATE, PRPTT, PRINR, PRWBC,
           OTHBLEED, NOTHBLEED, DOTHBLEED)
  
  # Compute % transfused
  df = df %>%
    group_by(CPT) %>%
    mutate(
      count = n(),
      percent_transfused = sum(NOTHBLEED)/n() *100
    )
  return(df)
}

clean_puf16 = function(puf) { # works for PUF16-18
  df = puf %>%
    select(PUFYEAR, CaseID, SEX, PRNCPTX, CPT, INOUT, HtoODay, EMERGNCY, ELECTSURG,
           ASACLAS, Age, HEIGHT, WEIGHT, 
           DIABETES, SMOKE, HXCOPD, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, STEROID, BLEEDDIS, 
           PRSODM, PRCREAT, PRALBUM, PRBILI, PRHCT, PRPLATE, PRPTT, PRINR, PRWBC,
           OTHBLEED, NOTHBLEED, DOTHBLEED) 
  
  return(mutate_puf(df))
}

clean_puf19 = function(puf) { # works for PUF19 and onwards presumably
  df = puf %>%
    select(PUFYEAR, CASEID, SEX, PRNCPTX, CPT, INOUT, HTOODAY, EMERGNCY, ELECTSURG,
           ASACLAS, AGE, HEIGHT, WEIGHT, 
           DIABETES, SMOKE, HXCOPD, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, STEROID, BLEEDDIS, 
           PRSODM, PRCREAT, PRALBUM, PRBILI, PRHCT, PRPLATE, PRPTT, PRINR, PRWBC,
           OTHBLEED, NOTHBLEED, DOTHBLEED) %>%
    mutate( # puf 19 renamed these variables
      Age = AGE,
      HtoODay = HTOODAY,
      CaseID = CASEID
    )
  df = mutate_puf(df)
  df = df %>%
    mutate(NOTHBLEED_d0 = case_when(NOTHBLEED == 1 & DOTHBLEED < 1 ~ 1,
                               NOTHBLEED == 1 & DOTHBLEED >= 1 ~ 0,
                               TRUE ~ 0
                               )
    )
  return(df)
}



# Read raw data and clean
puf = data.table::fread('/storage1/fs1/lu/Active/ehrlog/blood_product/raw_data/acs_nsqip_puf16.txt', header=T, sep='\t')
df = clean_puf16(puf)
data.table::fwrite(df, '/storage1/fs1/lu/Active/ehrlog/blood_product/puf16_lite.csv')

puf = data.table::fread('/storage1/fs1/lu/Active/ehrlog/blood_product/raw_data/acs_nsqip_puf17.txt', header=T, sep='\t')
df = clean_puf16(puf)
data.table::fwrite(df, '/storage1/fs1/lu/Active/ehrlog/blood_product/puf17_lite.csv')


puf = data.table::fread('/storage1/fs1/lu/Active/ehrlog/blood_product/raw_data/acs_nsqip_puf18_v2.txt', header=T, sep='\t')
df = clean_puf16(puf)
data.table::fwrite(df, '/storage1/fs1/lu/Active/ehrlog/blood_product/puf18_lite.csv')


puf = data.table::fread('/storage1/fs1/lu/Active/ehrlog/blood_product/raw_data/acs_nsqip_puf19.txt', header=T, sep='\t')
df = clean_puf19(puf)
data.table::fwrite(df, '/storage1/fs1/lu/Active/ehrlog/blood_product/puf19_lite_v3.csv')


# Read cleaned data and concatenate
df16 = data.table::fread('/storage1/fs1/lu/Active/ehrlog/blood_product/puf16_lite.csv')
df17 = data.table::fread('/storage1/fs1/lu/Active/ehrlog/blood_product/puf17_lite.csv')
df18 = data.table::fread('/storage1/fs1/lu/Active/ehrlog/blood_product/puf18_lite.csv')

df = rbindlist(list(df16, df17, df18), use.names = T)

df = df %>%
  mutate(
    NOTHBLEED_d0 = case_when(
      NOTHBLEED == 1 & DOTHBLEED < 1 ~ 1,
      NOTHBLEED == 1 & DOTHBLEED >= 1 ~ 0,
      TRUE ~ 0
    )
  ) %>%
  group_by(CPT) %>%
  mutate(
    count = n(),
    percent_transfused = sum(NOTHBLEED_d3)/n() *100
  )

data.table::fwrite(df, '/storage1/fs1/lu/Active/ehrlog/blood_product/puf16-18_lite_v4.csv')