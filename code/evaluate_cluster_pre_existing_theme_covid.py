import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer, SentencesDataset, models, losses, InputExample
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import LabelSentenceReader, InputExample
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import json
import pandas as pd
from sentence_transformers import datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from scipy.spatial import distance
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer


# fb = pd.read_csv('data/pre_existing_covid.csv') 

# ## Get covered list
# # fb_thr = fb.loc[(fb.dist_p <= 0.6)] #threshold 0.6
# #fb_thr = fb.loc[(fb.dist_p <= 0.5)] #threshold 0.5
# #fb_thr = fb.loc[(fb.dist_p <= 0.4)] #threshold 0.4
# # fb_thr = fb.loc[(fb.dist_p <= 0.3)] #threshold 0.3

# # #get uncover ads
# # fb_thr = fb.loc[(fb.dist_p > 0.6)] #threshold 0.6
# fb_thr = fb.loc[(fb.dist_p > 0.5)] #threshold 0.5
# #fb_thr = fb.loc[(fb.dist_p > 0.4)] #threshold 0.4
# #fb_thr = fb.loc[(fb.dist_p > 0.3)] #threshold 0.3

# fb_thr = fb_thr.reset_index(drop=True)
# print(fb_thr.shape)
# print("number of unique theme using threshold :: ", fb_thr['pred_theme'].nunique()) 
# print("number of unique phrase using threshold  :: ", fb_thr['pred_phrase'].nunique()) 

# # # # ##calculate accuracy & f1 score with threshold
# # from sklearn.metrics import f1_score
# # acc = f1_score(fb_thr['gt_theme'].values, fb_thr['pred_theme'].values, average='micro')
# # f1 = f1_score(fb_thr['gt_theme'].values, fb_thr['pred_theme'].values, average='macro')
# # print("accuracy, macro avg f1 score :: ", acc , f1)


# # # ##calculate accuracy & f1 score all
# # from sklearn.metrics import f1_score
# # acc = f1_score(fb['gt_theme'].values, fb['pred_theme'].values, average='micro')
# # f1 = f1_score(fb['gt_theme'].values, fb['pred_theme'].values, average='macro')
# # print("accuracy, macro avg f1 score all :: ", acc , f1)

# ### save covered ads into new file
# #fb_thr.to_csv('data/assignment_0.6.csv', index = False) 
# #fb_thr.to_csv('data/assignment_0.5.csv', index = False) #good for Prof
# #fb_thr.to_csv('data/assignment_0.4.csv', index = False)
# #fb_thr.to_csv('data/assignment_0.3.csv', index = False)


# # ## save uncoverd ads into a new file
# #fb_thr.to_csv('data/uncover_0.6.csv', index = False) 
# fb_thr.to_csv('data/uncover_0.5_covid.csv', index = False) 
# #fb_thr.to_csv('data/uncover_0.4.csv', index = False) 
# #fb_thr.to_csv('data/uncover_0.3.csv', index = False) 

# #fb_thr.to_csv('data/all_ads_uncover_0.6.csv', index = False) 
# #fb_thr.to_csv('data/gt_ads_uncover_0.5.csv', index = False) 
# # fb_thr.to_csv('data/gt_ads_uncover_0.4.csv', index = False) 

# #print(fb_thr['pred_phrase'].unique())
# cover_list = fb_thr['pred_phrase'].unique()
# print(len(cover_list)) #105

# #What is the average & median distance
# avg_dis = fb_thr['dist_p'].mean(axis = 0)
# med_dis = fb_thr['dist_p'].median(axis = 0)
# print("avg distance using threshold ", avg_dis)
# print("median distance using threshold ", med_dis)

#we have 15 themes for covid
themes = {"GovDistrust": [
    "lack of trust in the government",
    "Fuck the government",
    "The government is a total failure",
    "Never trust the government",
    "Biden is a failure",
    "Biden lied people die",
    "The government and Fauci have been dishonest",
    "The government always lies",
    "The government has a strong record of screwing things up",
    "The government is good at screwing things up",
    "The government is screwing things up",
    "The government is lying",
    "The government only cares about money",
    "The government doesn't work logically",
    "Do not trust the government",
    "The government doesn't care about people’s health",
    "The government won't tell you the truth about the vaccine",
    "Biden will not hold China accountable"
  ],
   
   "GovTrust": [
    "We trust the government",
    "Biden is tackling covid",
    "The government cares for people",
    "We are thankful to the government for the vaccine availability",
    "Hats off to the government for tackling the pandemic",
    "It is a good thing to be skeptical of the government, but they are right about the covid vaccine",
    "It is a good thing to be skeptical of the government, but they haven’t lied about the covid vaccine",
    "The government can be corrupt, but they are telling the truth about the covid vaccine",
    "The government can be corrupt, but they are not lying about the covid vaccine",
    "Biden is helping to end the pandemic",
    "Trump initiated covid vaccine",
    "Biden will hold China accountable"
  ],

    "VaccineRollout": [
    "Vaccine appointment is available",
    "Schedule your vaccine appointment",
    "No appointment needed",
    "Walk-in vaccine clinic is available",
    "Drive through vaccine site is available",
    "Mobile clinic is available here",
    "Vaccine clinic has been set up",
    "New vaccine center has been opened",
    "CDC recommends vaccine for kids",
    "FDA authorized covid vaccine for children ages 5 to 11 years old"
  ],
           
  "VaccineSymptom": [
    "I got fever after taking the vaccine",
    "Know the vaccine symptom",
    "Covid vaccines can cause blood clots",
    "The vaccine is dangerous for people with medical conditions",
    "I won't take the vaccine due to medical reasons",
    "The vaccine has side effects"
  ],
           
 "VaccineEquity": [
    "Vaccine should be available for everyone",
    "Vaccine should be free of cost",
    "Everyone has right to get free covid vaccine",
    "We don't have equal access to vaccine",
    "We should ensure vaccine access to vulnerable communities "
  ],
          
  "VaccineStatus": [
    "Half of the population are fully vaccinated",
    "Here is the vaccine statistics",
    "Update of vaccine status",
    "Covid-19 update",
    "Sign up fod vaccine update",
    "We need volunteers for vaccine site",
    "Vaccination rate is slow",
    "Vaccination rate is high",
    "Covid vaccine information is here",
    "Infection rate is lower",
    "Covid death is real",
    "The pandemic is not a lie, hospitalizations are on the rise"
  ],

  "EncourageVaccination": [
    "Get your vaccine",
    "Get the shot",
    "Get jabbed",
    "Protect our community by getting vaccinated",
    "I encourage everyone to get their shot when they can",
    "We can all do our part",
    "Getting vaccinated is the best way to protect yourself and everyone around you",
    "Your shot matters",
    "Take your best shot",
    "Be the part of the solution",
    "Win lottery by getting vaccinated",
    "Sleeve up",
    "Collect your vaccine incentives",
    "Take the vaccine for your family, for your friends, for your country",
    "Get vaccinated",
    "Get free vaccine and free lunch",
    "Call us and we will answer your questions regarding covid vaccine",
    "Get boosted",
    "Vax up",
    "Vaccines save lives",
    "Do your part to stop pandemic",
    "This is our best shot",
    "Join a discussion about vaccine"
  ],
     
           
  "VaccineMandate": [
    "Forcing people to take experimental vaccines is oppression",
    "The vaccine has nothing to do with Covid-19, it's about the vaccine passport and tyranny",
    "The vaccine mandate is unconstitutional",
    "I'm not against the vaccine but I am against the mandate",
    "I have freedom to choose not to take the vaccine",
    "I choose not to take the vaccine",
    "I am free to refuse the vaccine",
    "It is not about covid, it is about control",
    "My body my choice",
    "Medical segregation based on vaccine mandates is discrimination",
    "The vaccine mandate violates my rights",
    "Falsely labeling the injection as a vaccine is illegal",
    "Firing over vaccine mandates is oppression",
    "Vaccine passports are medical tyranny",
    "I won't let the government tell me what I should do with my body",
    "I won't have the government tell me what to do",
    "The vaccine mandate is not oppression because vaccines lower hospitalizations and death rates",
    "The vaccine mandate is not oppression because it will help to end this pandemic",
    "The vaccine mandate will help us end the pandemic",
    "We need a vaccine mandate to end this pandemic",
    "I support vaccine mandate",
    "If you don't get the vaccine based on your freedom of choice, don’t come crawling to the emergency room when you get COVID",
    "If you refuse a free FDA-approved vaccine for non-medical reasons, then the government shouldn't continue to give you free COVID tests",
    "You are free not to take the vaccine, businesses are also free to deny you entry",
    "You are free not to take the vaccine, businesses are free to protect their customers and employees",
    "If you choose not to take the vaccine, you have to deal with the consequences",
    "If it is your body your choice, then insurance companies should stop paying for your hospitalization costs for COVID", 
    "Check vaccine card for events",
    "Airlines require vaccine passport",
    "Proof of vaccine is required",
    "Vaccine passport is useful for reopen",
    "Support vaccine passport",
    "We don't support vaccine passport",
    "Workers have declined COVID-19 Vaccine"
  ],
  
  "VaccineReligion": [
    "The vaccine is against religion",
    "The vaccines are the mark of the beast",
    "The vaccine is a tool of Satan",
    "The vaccine is haram",
    "The vaccine is not halal",
    "I will protect my body from a man made vaccine",
    "I put it all in God's hands",
    "God will decide our fate",
    "Allah will protect us",
    "The vaccine contains bovine, which conflicts with my religion",
    "The vaccine contains aborted fetal tissue which is against my religion",
    "The vaccine contains pork, muslims can't take the vaccine",
    "Jesus will protect me",
    "The vaccine doesn't protect you from getting or spreading Covid, God does",
    "The covid vaccine is another religion",
    "The vaccine is not against religion, get the vaccine",
    "No religion ask members to refuse the vaccine",
    "Religious exemptions are bogus",
    "When turning in your religious exemption forms for the vaccine, remember ignorance is not a religion",
    "Disregard for others' lives isn't part of your religion",
    "Jesus is trying to protect us from covid by divinely inspiring scientists to create vaccines"
  ],
  
  "VaccineEfficacy": [
    "The vaccine works",
    "The vaccine is safe",
    "Vaccines do work, ask a doctor or consult with an expert",
    "The covid vaccine helps to stop the spread",
    "Unvaccinated people are dying at a rapid rate from COVID-19",
    "There is a lot of research supporting that vaccines work",
    "The research on the covid vaccine has been going on for a long time",
    "Millions have been vaccinated with only mild side effects",
    "Millions have been safely vaccinated against covid",
    "The benefits of the vaccine outweigh its risks",
    "Vaccine is safe for pregnant woman",
    "The vaccine has benefits",
    "The vaccine is safe for women and kids",
    "The vaccine won't make you sick",
    "The vaccine isn't dangerous",
    "The vaccine won't kill you",
    "The covid vaccine isn't a death jab",
    "The covid vaccine doesn't harm women and kids",
    "COVID-19 vaccine ingredients are safe"
  ],
           
  
  "VaccineDevelopment": [
    "Covid vaccine research has been going on for a while",
    "Plenty of research has been done on the covid vaccine",
    "The technologies used to develop the COVID-19 vaccines have been in development for years to prepare for outbreaks of infectious viruses",
    "The testing processes for the vaccines were thorough didn't skip any steps",
    "The vaccine received FDA approval",
    "Vaccine uses mRNA technology",
    "the vaccine is not properly tested, it has been developed too quickly",
    "Covid-19 vaccines have not been through the same rigorous testing as other vaccines",
    "The covid vaccine is experimental",
    "The covid vaccine was rushed through trials",
    "The approval of the experimental vaccine was rushed"
  ],

  "CovidPlan": [
   "Expand vaccine distribution",
   "We are working on vaccine distribution",
   "Covid rescue plan",
   "Cash relief",
   "More covid testing center",
   "Seting up vaccine clinic",
   "Support small business",
   "Reopen country",
   "Rebuilding economy",
   "Covid stimulus check",
   "Reopen school",
   "Unemployment benefit",
   "Expand mask and PPE supply" 
  ],

  "VaccineMisinformation": [
    "Animal shelters are empty because Dr Fauci allowed experimenting of various Covid vaccines/drugs on dogs and other domestic pets",
    "Fauci tortures dogs and puppies",
    "The covid vaccine is a ploy to microchip people",
    "Bill Gates wants to use vaccines to implant microchips in people",
    "Globalists support a covert mass chip implantation through the covid vaccine",
    "There is aborted fetal tissue in the Covid Vaccines",
    "Covid vaccines contain aborted fetal cells",
    "The covid vaccine will make you sterile",
    "Covid vaccine will affect your fertility",
    "The vaccine will not make you sterile",
    "The covid vaccine will not affect your fertility",
    "No difference if fertility rate has been found between vaccinated and unvaccinated people",
    "Vaccines were tested on fetal tissues, but do not contain fetal cells",
    "Vaccines do not contain aborted fetal cells",
    "Vaccine misinformation is floating around",
    "Don't believe in vaccine misinformation",
    "Don't trust vaccine conspiracy"
    ],

    "NaturalImmunity": [
    "Natural methods of protection against the disease are better than vaccines",
    "Herd immunity is broad, protective, and durable",
    "Natural immunity has higher level of protection than the vaccine",
    "Embrace population immunity",
    "I trust my immune system",
    "I have antibodies I do not need the vaccine",
    "Natural immunity is effective",
    "Natural immunity would require a lot of people getting sick",
    "Experts recommend the vaccine over natural immunity",
    "The vaccine has better long term protection than to natural immunity",
    "Natural immunity is not effective",
    "Experts aren’t sure how long hybrid immunity lasts",
    "Natural immunity is highly variable"
  
    ],

    "Vote": [
    "Please vote",
    "Your vote matters",
    "Go vote",
    "Vote today"
    
  ]

}

# ### find unmatched talking points 

# food_list=list(themes.values())
# flat_list = [item for sublist in food_list for item in sublist]
# print("total number of talking points : ", len(flat_list)) ##107

# #print("No match elements: ", set(flat_list)-set(cover_list), len(set(flat_list)-set(cover_list)))

# # ints_list = list(set(flat_list).intersection(cover_list))
# # print(len(ints_list))
# sys.exit()


#Paraphrase Model
df = pd.read_csv('data/unq_all_covid_fb_ad.csv')

#print(df)  ## ad_creative_body, paraphrase_gpt3
#sys.exit()

#model = SentenceTransformer('./sbert_climate_gt_epoch1')

model = SentenceTransformer('all-mpnet-base-v2') ##original sbert model

#model = SentenceTransformer('./sbert_climate_unlabel') #fine-tuned sbert model

def compute_theme_embeddings(model, themes):
    theme_embeddings = {}
    for theme in themes:
        embeddings = []
        for phrase in themes[theme]:
            embeddings.append(model.encode([phrase.lower()])[0])
        theme_embeddings[theme] = embeddings
    return theme_embeddings

theme_embeddings = compute_theme_embeddings(model, themes)

def get_ad_theme(model, ad, themes, theme_embeddings):
    ad_embed = model.encode(ad)
    min_dis = []; curr_themes = []; curr_phrases = []
    phrase_dis = []
    max_dis = []
    #max_prob_per_thm =[]; prob_max_index_per_thm = []; phrase_max_prob = []
    prob_index = []
    for theme, val in themes.items(): #each theme
        #print(themes[theme][0])
        #sys.exit()
        # phrases = []
        dis = []

        for i in range (0, len(val)): #phrases in each theme
            cosine_dis = distance.cdist(ad_embed.reshape(1 , 768), theme_embeddings[theme][i].reshape(1 , 768), 'cosine')
            #print("cosine_dis", cosine_dis[0][0])
            dis.append(cosine_dis)
            phrase_dis.append(cosine_dis)
            curr_phrases.append(val[i])


        #print(len(dis), dis)
        # print(len(curr_phrases), curr_phrases)
        #soft = torch.softmax(torch.FloatTensor(dis), dim=0) #convert list to tensor to pass it trough softmax
        #print("soft",soft.squeeze(), soft.squeeze().size())
        #prob = soft.squeeze().tolist() #use squeeze to remove extra 1 dimension then convert tensor to list  
        #print(prob) #list
        
        #max_prob_per_thm.append(np.array(prob).max(axis = 0))
        #prob_max_index_per_thm.append(np.array(prob).argmax(axis=0).item())
        #phrase_max_prob.append(themes[theme][np.array(prob).argmax(axis=0).item()])
        #print("prob_max_index_per_thm", prob_max_index_per_thm)
        # print(max_prob)
        #sys.exit()
        #print("dis", dis, len(dis))
        min_dis.append(np.array(dis).min(axis=0).item())
        #for negative example
        max_dis.append(np.array(dis).max(axis=0).item())

        curr_themes.append(theme)
        #print("min", np.array(dis).argmin(axis=0).item())
        #prob_index.append(prob[np.array(dis).argmin(axis=0).item()])

    #print("phrase_dis", phrase_dis, len(phrase_dis))
    #print(min_dis, len(min_dis))  #13 ta value      
    min_index = np.array(min_dis).argmin(axis=0).item()
    #for negative example
    max_index = np.array(max_dis).argmax(axis=0).item()
    #print("min_index", min_index) #index 7
    min_dist = min_dis[min_index]
    #print("min_dist", min_dist) # 0.6902957994808554
    phrs_min_index = np.array(phrase_dis).argmin(axis=0).item()
    #for negative example
    phrs_max_index = np.array(phrase_dis).argmax(axis=0).item() 
    #print("phrs_min_index", phrs_min_index)
    #print("prob_index", prob_index, len(prob_index))
    #prob_score = prob_index[min_index]
    #print("prob_score", prob_score)

    # print(max_prob_per_thm,len(max_prob_per_thm)) #15
    # print("prob_max_index_per_thm", prob_max_index_per_thm,len(prob_max_index_per_thm))
    # print("phrase_max_prob", phrase_max_prob, len(phrase_max_prob))
    # max_prob_total=np.array(max_prob_per_thm).max(axis = 0)
    # max_prob_index_total=np.array(max_prob_per_thm).argmax(axis = 0)
    #print("max_prob_total, max_prob_index_total", max_prob_total, max_prob_index_total)
    
    theme = curr_themes[min_index]
    phrase = curr_phrases[phrs_min_index]
    #for negative example
    neg_theme = curr_themes[max_index]
    neg_phrase = curr_phrases[phrs_max_index]
    # theme_prob = curr_themes[max_prob_index_total]
    #sys.exit()
    return theme, min_dist, phrase, neg_theme, neg_phrase

thm = {}
phrs = {}
score = {}
neg_thm = {}
neg_phrs = {}
for i in range(0,df.shape[0]):
    theme, dist, phrase, neg_theme, neg_phrase = get_ad_theme(model, df['ad_creative_body'][i].lower(), themes, theme_embeddings)
    thm[df['id'][i]] = theme
    phrs[df['id'][i]] = phrase
    score[df['id'][i]] = dist
    neg_thm[df['id'][i]] = neg_theme
    neg_phrs[df['id'][i]] = neg_phrase


     
ads = df
#print(ads)
ad_id = []
text = []
fe = []
page = []
time = []
imp = []
spend = []
region = []
demo = []
theme_p = []
neg_theme_p = []
stance = []
theme_gt = []
phrase_p = []
neg_phrase_p = []
dist_p = []
label = []
count1 = 0
count0 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
prob_score = []
avg_imp =[]
avg_spend = []
des = []
for i in range(0, ads.shape[0]):
    #print("ads.id[i]", ads.id[i], type(ads.id[i]))
    #id = str(ads.id[i]).split('.')[0]
    id = ads.id[i]
    if id in thm:
        ad_id.append(id)
        #print('id', id)
        theme_p.append(thm[id]) 
        phrase_p.append(phrs[id]) 
        dist_p.append(score[id]) 
        neg_theme_p.append(neg_thm[id]) 
        neg_phrase_p.append(neg_phrs[id])
        if (score[id] > 0.5): ##provide threshold 
            label.append(0.0)
            count0 = count0 + 1
        elif(score[id] <= 0.5):
            label.append(1.0)
            count1 = count1 + 1

        if (score[id] > 0.4): ##provide threshold 
            count2 = count2 + 1
        elif(score[id] <= 0.4):
            count3 = count3 + 1

        if (score[id] > 0.3): ##provide threshold 
            count4 = count4 + 1
        elif(score[id] <= 0.3):
            count5 = count5 + 1

        if (score[id] > 0.6): ##provide threshold 
            count6 = count6 + 1
        elif(score[id] <= 0.6):
            count7 = count7 + 1
        # prob_score.append(prob_them[id])
        # if (score[id] > 0.5): ##provide threshold 
        #     label.append(0.0)
        #     count0 = count0 + 1
        # else:
        #     label.append(1.0)
        #     count1 = count1 + 1
        #stance.append(ads['stance'][i])
        text.append(ads['ad_creative_body'][i])
        fe.append(ads['funding_entity'][i])
        # ##For gt
        # stance.append(ads['stance'][i])
        #theme_gt.append(ads['pred_theme'][i])
        

        #time.append(ads['date'][i])
        #imp.append(ads['impressions'][i])
        #spend.append(ads['spend'][i])
        #region.append(ads['region_distribution'][i])
        #demo.append(ads['demographic_distribution'][i])
        #avg_imp.append(ads['avg_imp'][i])
        #avg_spend.append(ads['avg_spend'][i])
        #des.append(ads['ad_creative_link_description'][i])
print(len(theme_p), len(ad_id),  len(fe))     #29862 29862 29862 29862 29862 29862 29862 29862 29862  
        
# print("number of assignments <= 0.5 (closer): ", count1)
# print("number of assignments > 0.5 (farther) : ", count0)

print("number of assignments <= 0.6 (closer): ", count7)
print("number of assignments > 0.6 (farther) : ", count6)
print("number of assignments <= 0.5 (closer): ", count1)
print("number of assignments > 0.5 (farther) : ", count0)
print("number of assignments <= 0.4 (closer): ", count3)
print("number of assignments > 0.4 (farther) : ", count2)
print("number of assignments <= 0.3 (closer): ", count5)
print("number of assignments > 0.3 (farther) : ", count4)



predicted_fb = pd.DataFrame({'id': ad_id, 'ad_creative_body': text,  'pred_theme' : theme_p, 'pred_phrase' : phrase_p, 'dist_p' : dist_p, 
                             
                            'funding_entity':fe, 'neg_theme' : neg_theme_p, 'neg_phrase' : neg_phrase_p})


# # ##For gt sanity check
# predicted_fb = pd.DataFrame({'id': ad_id, 'ad_creative_body': text, 'gt_theme' : theme_gt, 'pred_theme' : theme_p, 'pred_phrase' : phrase_p, 'dist_p' : dist_p, 
                             
#                             'funding_entity':fe})


print("number of unique theme w/o threshold :: ", predicted_fb['pred_theme'].nunique()) 
print("number of unique phrase w/o threshold ::", predicted_fb['pred_phrase'].nunique()) 

predicted_fb.to_csv('data/pre_existing_covid.csv', index = False) 

