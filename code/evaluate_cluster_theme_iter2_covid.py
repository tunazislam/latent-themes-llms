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


# fb = pd.read_csv('data/covid_thm_iter1.csv') 

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
# fb_thr.to_csv('data/uncover_0.5_covid_iter1.csv', index = False) 
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

#we have 23 themes for covid
themes = {
    "GovDistrust": [
    "Joe Biden is destroying America. He has opened our borders during a pandemic. He's divided this country racially and has created a war between the vaccinated and unvaccinated. He has embarrassed our military and turned our allies against us.",
    "Joe Biden has failed the American people and now he is trying to cover it up. Listen to what I had to say today on Fox Business.",
    "President Biden is at record lows in the polls thanks to his border debacle, Covid disaster, and uncanny ability to divide America. He is, without question, the worst president in U.S. history.",
    "Fauci has lied to Congress and the American Public for over 19-months. It's time to expose his massive cover-up and put him in jail.",
    "Dr. Fauci, the CDC, and Joe Biden have yet to explain why vaccinated people need protection from the unvaccinated."
  ],
   
   "GovTrust": [
    "We thank the Administration & Congress for direct COVID-19 vaccine access for our patients.",
    "Donald Trump is urging all Americans to receive the Covid-19 vaccine.",
    "President Biden continues to lead our vaccination effort as we look to put COVID-19 behind us and build back better.",
    "Thank you to local, state and federal officials for prioritizing our residents and staff members to receive the COVID-19 vaccine. We are vaccinated! #GetVaccinatedLA",
    "SOME OF PRESIDENT BIDEN'S ACCOMPLISHMENTS SO FAR (UPDATED) \n Spread the word. \n 1. Rejoined the Paris Agreement, an agreement within the United Nations Framework Convention on Climate Change (UNFCCC), on climate change mitigation, adaptation, and finance, signed in 2016. (Trump withdrew the U.S. from this agreement.) \n 2. In a letter to UN Secretary-General António Guterres, Biden rescinded Donald Trump's July 2020 intended U.S. withdrawal from the World Health Organization. \n 3. In January, signed ten executive orders related to the country's COVID-19 response and vaccination efforts. These included Ensuring a Data-Driven Response to COVID-19 and Future High-Consequence Public Health Threats, Establishing the COVID-19 Pandemic Testing Board and Ensuring a Sustainable Public Health Workforce for COVID-19 and Other Biological Threats, Supporting the Reopening and Continuing Operation of Schools and Early Childhood Education Providers, and activating the Defense Production Act in order to speed up vaccine distribution. \n 4. Signed an executive order to increase the minimum wage for federal employees to $15 per hour.\n 5. Withdrew Trump's travel ban from Muslim-majority countries.\n 6. Signed the American Rescue Plan into law. The American Rescue Plan Act of 2021, also called the COVID-19 Stimulus Package or American Rescue Plan, is a $1.9 trillion economic stimulus bill passed by the 117th United States Congress and signed into law by President Biden on March 11, 2021, to speed up the United States' recovery from the economic and health effects of the COVID-19 pandemic and the ongoing recession. First proposed on January 14, 2021, the package builds upon many of the measures in the CARES Act from March 2020 and in the Consolidated Appropriations Act, 2021, from December. \n 7. Created a task force to reunite children separated from their families as a result of the Trump administration's policy of taking children and even babies away from their parents to discourage refugees from coming to the southern U.S. border. The Biden Administration has reunited over 50 families and has recently launched a web portal, together.gov, that will allow parents to contact the U.S. government to expedite the process of reunification. \n 8. Announced sanctions on Russia over the worst-ever hack of U.S. government agencies. (Stands in contrast to Trump's extreme reluctance to do the same.) \n 9. Biden followed through on his promise to end the war in Afghanistan, America's longest war. Biden and the U.S. military did an extraordinary job of evacuating more than 120,000 Americans and Afghan allies in 2 weeks. \n 10. Made all adults eligible for COVID-19 vaccination by April 19th, 2 weeks ahead of schedule. As of now 70% of adults in the U.S. have had at least one COVID-19 shot. Although there's been a recent increase due to people who still haven't been vaccinated contracting the Delta variant, overall the number of daily new cases of COVID-19 in the U.S. has fallen substantially since Biden became president. \n 11. Added over 3 million jobs in the first 6 months of his presidency. \n 12. Signed the COVID-19 Hate Crimes Act. Prompted by a nearly 150% increase in hate crimes against Asian Americans after the breakout of the COVID-19 pandemic, the legislation, introduced by Rep. Grace Meng, D-N.Y., and Sen. Mazie Hirono, D-Hawaii, aims to make the reporting of hate crimes more accessible at the local and state levels by boosting public outreach and ensuring reporting resources are available online in multiple languages. It also directs the Department of Justice to designate a point person to expedite the review of hate crimes related to COVID-19 and authorizes grants to state and local governments to conduct crime-reduction programs to prevent and respond to hate crimes. A big change from Trump calling COVID-19 the 'China virus' and the 'Kung Flu.' \n 13. Suspended oil drilling leases in the Arctic National Wildlife Refuge. \n 14. Signed an executive order expanding voting access that directs the heads of all federal agencies to submit proposals for their respective agencies to promote voter registration and participation within 200 days, while assisting states in voter registration under the National Voter Registration Act. In addition, the order instructs the General Services Administration to modernize the federal government's Vote.gov portal. \n 15. Signed an executive order to restore collective bargaining power for federal workers. \n 16. The American Rescue Plan fulfills Biden's promise to expand the Affordable Care Act, for two years. With its expanded subsidies for health plans under the Affordable Care Act, the coronavirus relief bill makes insurance more affordable, and puts health care on the ballot in 2022. \n 17. Issued an executive order that promotes more competition in the economy. It urges agencies to crack down on anti-competitive practices in sectors from agriculture to drugs and labor. Fully implemented, the effort will help lower Americans' internet costs and lower drug prices, among other benefits. The order instructs antitrust agencies to focus on labor, healthcare, technology and agriculture as they address a laundry list of issues that have irritated consumers, and in the case of drug prices, has bankrupted some. \n 18. Signed the Crime Victims Fund Act, a bill to bolster a fund used to support victims of domestic violence, sexual assault and other crimes. \n 19. Signed a bill making Juneteenth the eleventh federal holiday. Juneteenth, now officially Juneteenth National Independence Day, commemorats the emancipation of African-American slaves. It is also often observed for celebrating African-American culture. \n 20. Signed an executive order to increase refugee admissions and allow a safe haven for about 125,000 refugees. \n 21. Cancelled student loan debt for more than 300,000 Americans with severe disabilities that leave them unable to earn significant incomes. \n 22. Announced 'Path out of the Pandemic,' a plan which includes new wide-ranging requirements for federal employees and employees of companies with more than 100 workers to receive the COVID-19 vaccine or undergo regular testing at least once a week, with no option of testing for unvaccinated federal employees. \n 23. Biden is currently working with Congress to pass 2 major bills that will invest in infrastructure as well as other priorities such as providing 2 free years of community college, child care and universal pre-k, Medicare expansion, an extended child tax credit, cutting prescription drug prices, paid family and medical leave and combatting climate change. Liberal and moderate Democrats, with help from Biden, have been making steady progress in working out their differences to pass these bills."
  ],

    "VaccineRollout": [
    "The COVID-19 vaccines are FDA approved for children ages 5-11 years old. Pediatrician Dr. Raman trusts the COVID-19 vaccine.",
    "Reduce your child's risk of severe illness from COVID. Everyone ages 5+ is vaccine eligible—get them the protection they need.",
    "Last week, COVID vaccines were made available for kids ages 5 to 11 and will be available at this month's drive-in events.",
    "The COVID-19 vaccine is now available for kids ages 5 -11! All children deserve protection against COVID-19. #ThisIsOurShot to get kids safely back to school, sports, and spending time with the people they love. #CallYourPediatrician today!",
    "The COVID-19 vaccine is available for kids ages 5 -17! All children deserve protection against COVID-19. #ThisIsOurShot to help kids get safely back to activities and to being with the people they love.",
    ##new from cluster iter1
    "News for Missouri - Where some metro counties stand on COVID-19 vaccine rollout. Install SmartNews for free to read more.",
    "Bronx Residents Line Up to be the First to Receive Johnson & Johnson COVID-19 Vaccine. Install SmartNews for free to read more.",
    "News for West Virginia - State health officials urge West Virginians to get a COVID-19 vaccine. Install SmartNews for free to read more.",
    "Residents, staff at Kennebunk senior living facility receive COVID-19 vaccine. Install SmartNews for free to read more.",
    "East Baton Rouge Parish School System employees next in line for COVID-19 vaccine. Install SmartNews for free to read more."
  ],
           
  "VaccineSymptom": [
    "First Reported Cases of Blood Clots Causing Stroke in Young Adults Following COVID-19 Vaccination.",
    "The EMA found a possible link between the Johnson & Johnson COVID-19 vaccine and 'unusual blood clots' and wants a warning attached.",
    "The U.S. recommended a “pause” in using the single-dose Johnson & Johnson COVID-19 vaccine to investigate reports of rare but potentially dangerous blood clots.",
    "Public health officials lift pause on Johnson & Johnson COVID vaccine. Learn more about symptoms of potential rare blood clots.",
    "On April 13, the CDC and U.S. Food and Drug Administration, who oversee vaccines, recommended a temporary pause in use of the Johnson & Johnson vaccine. The reason was reported cases of abnormal blood clotting with low platelet counts (thrombosis) with thrombocytopenia syndrome, or TTS, in people who received this vaccine. The Johnson & Johnson vaccine 💉 was determined to be safe after an in-depth review. However, it may be more dangerous for some people. Women ages 18-49, who are most at risk for TTS, may choose to get a different COVID-19 vaccine.  Right now, it looks like these specific types of blood clots are an extremely rare reaction. Over 8 million people have gotten the vaccine, and only 17 adverse events of this type have been reported. BMC continues to strongly encourage all patients and community members to get vaccinated with whichever vaccine is available to them. We understand that you may have questions and concerns. Please view our stories today, or visit https://bit.ly/2Th6Gza for more information."
  ],
           
 "VaccineEquity": [
    "We’ve got to ensure vaccines go to those who need them the most.",
    "Making the vaccine more equitable among emerging disparities.",
    "As the pandemic continues to spread, the United States needs to take immediate action to protect vulnerable people around the world. The United States must work to make worldwide equitable vaccine access a reality. Make your voice heard—call on the United States to make COVID-19 vaccines accessible to all.",
    "'No one is safe until everyone is safe, and we will only bring the pandemic under control when vaccines are available equally to all people.'Anne-Marie Grey, CEO of USA for UNHCR, and Mary Maker explain the importance of vaccine equity and the steps we must take to overcome this pandemic. #WorldHealthDay",
    "Help us advocate for equitable Covid-19 vaccine access! 💉 Health care is a human right — no one should be left behind. Your gift today will power Human Rights Watch’s best work to hold governments and pharmaceutical companies accountable for ensuring vaccine distribution respects human rights.",
    ####new from cluster iter2
    "In 2021, we invested in programs and hosted conversations aimed at improving health equity. This year, our work continues.",
    "KIDS LIVES MATTER INTL. & POWER4LIFE IS LAUNCHING SOMETHING NEW... ""PRAISE IN THE PARK"" FOR THE ENTIRE FAMILY SUNDAY, SEPTEMBER 19TH @2PM @CUNEY HOMES (3260 TRUXILLO ST.)""!! We have a great opportunity to gather the communities we serve, and their families in need that we serve to lift up a great CELEBRATION ON THE COURT!!! \n ____We have partnered with Bread of Life Inc., Amazing 102.5FM, Baylor College of Medicine, UofH Community Health Workers Initiative & much more. Attendees can even sign your student(s) up to participate in free special STEM camps sponsored by UH and Kids' Lives Matter International. YOU MUST ARRIVE EARLY -First Come, First Serve- WE WILL START ON TIME!! Sign your kids up also to receive gifts for the upcoming holidays. VIEW THE FLYER FOR MORE DETAILS… ____ Van transportation at the following UPDATED times from select Locations…~FREE TRANSPORTATION PICK-UP TIMES from LISTED Locations' Comm. Bldg.: •Clayton Homes at 12PM, •Kennedy Place at 12:20 PM, & •Kelly Village at 12:40 PM~ _____VOLUNTEERS: We need your help to serve and bless our community residents, so that they will have the opportunity to receive plenty of personal care items, hot meals, PLUS they can qualify for new beds and laptops for students, and Gift cards for groceries or anything else they need. PLEASE BRING OTHERS TO VOLUNTEER AND BRING ANY NEW OR GENTLY WORN & CLEAN CLOTHES and SHOES. **ARRIVE AT 12:30PM PROMPT TO VOLUNTEER. For inquiries, please call or text Min Cher-Von 832.229.8242 or Pastor Traylor 832.722.4538._____ \n WANT TO GIVE TO BLESS THOSE WE SERVE, DO SO AT: \n ✅ https://cash.me/$KidsLivesMatterIntl  https://cash.me/$P4LOnline \n ✅ KLM Zelle Account: KidsLivesMatterintl@gmail.com Visit our websites: www.power4lifeministries.org OR www.kidslivesmatterintl.org \n ✅ Text your donation amount to 832-981-3301 \n ✅ PayPal: http://paypal.me/P4LMinistries",
    "Lorena Gonzalez Recognizes ERC as Non-Profit Of The Year For Assembly District 80! We're proud to recognize Employee Rights Center as the 80th Assembly District’s Nonprofit of the Year! ERC has done so much over the last 20 years to advocate and provide representation for some of the most vulnerable and marginalized workers in our district. When the pandemic hit, we saw so many left unemployed and confused about how to access support services. ERC was one of the first organizations that worked to transform its normal operation into a resource hub to meet the immediate needs of the community. Just in the last year, ERC has helped organize Census outreach events, COVID vaccination sites, assisted with filling out and submitting EDD claims, partnered with other organizations to provide rental and utility assistance, and helped establish a pantry to distribute food to those in need. As part of its broader mission, the nonprofit also provided resources for disadvantaged workers to stand up against wage theft, health and safety issues in the workplace, and discrimination, regardless of their immigration status. That’s just a few examples of the countless ways ERC has supported and helped provide relief to working families in our community during such a difficult year. Please join us in thanking ERC for their service, dedication, and resilience!"
  ],
          
  "VaccineStatus": [
    "Jefferson County News: Local doctor has message to public after getting COVID vaccine. Install news app trusted by millions to stay informed of latest Jefferson County local news!",
    "Chemung County COVID-19 vaccine tracker: 30% fully vaccinated. Install SmartNews for free to read more.",
    "Wyandotte County News: Wyandotte County opens another COVID-19 vaccine site. Install news app trusted by millions to stay informed of latest Wyandotte County local news!",
    "Alamance County News: Alamance County Health Department to begin taking appointments for COVID-19 vaccine; opening hotline.",
    "Mecosta County News: Health department breaks down COVID-19 vaccine scheduling. Install news app trusted by millions to stay informed of latest Mecosta County local news!"
  ],

  "EncourageVaccination": [
    "Get your vaccine today!",
    "Get vaccinated, it’s our best shot for a strong healthy community.",
    "Find out where to get a vaccine near you.",
    "Help Build a Healthier Community and Register for a Vaccine Today! Click Here to Learn More.",
    "The best way to protect yourself and those around you is by getting vaccinated! AltaMed Health Services",
     ##new from cluster iter2
    "Get your COVID-19 vaccine on the Las Vegas Strip! We’ll have your choice of COVID-19 vaccines (one-shot Janssen for 18+, 2-dose Pfizer for 12+ or 2-dose Moderna for 18+) this Saturday, from 10 a.m. to 6 p.m., at Double Barrel at Park MGM. Nevadans and those visiting Las Vegas are welcome, and COVID-19 testing will also be available. Get protected from COVID-19, and enjoy a free beverage of choice (alcoholic or non, age restrictions apply) while you wait your 15-minute observation time. If you're already vaccinated and you bring a friend to get vaccinated, you're both entered to win raffle prizes. Plus don’t forget: All COVID-19-vaccinated Nevadans are entered to win millions in cash/prizes during Vax Nevada Days!",
    "Immunize Nevada, in partnership with the city of Fernley, is hosting a COVID-19 vaccine clinic for those 12+ with the Janssen, Moderna and Pfizer COVID-19 vaccines. Those individuals that receive their vaccines at this event will be able to swim for free that day at the Fernley Swimming Pool. Fernley Swimming Pool | 12+ COVID-19 Vaccine Clinic | No appointment necessary",
    "Appointments are no longer required at the Miami-Dade County vaccination sites. The Aventura Mall Drive Thru site is open 7 days a week from 10 a.m. – 7 p.m. 19501 Biscayne Blvd. Aventura, FL 33180 Enter on NE 29 CT near Citibank For a faster registration process on-site, residents are encouraged to sign up online for appointments available both same-day and in coming days by visiting miamidade.gov/vaccine or calling 305-614-2014. #VaxUpMiami",
    "Rock Park | 12+ COVID-19 Vaccine Clinic | No appointment necessary \n Immunize Nevada is hosting a COVID-19 vaccine clinic for those 12+ with the Janssen, Moderna and Pfizer COVID-19 vaccines.",
    "Appointments are no longer required at the Miami-Dade County vaccination sites. \n The Zoo Miami Drive Thru site is open 7 days a week from 8 a.m. – 8 p.m. 12400 SW 152nd Street, Miami, FL 33177 \n For a faster registration process on-site, residents are encouraged to sign up online for appointments available both same-day and in coming days by visiting miamidade.gov/vaccine or calling 305-614-2014. #VaxUpMiami"
  ],
     
           
  "VaccineMandate": [
    "Vaccine Passports... a not so dystopian future.",
    "'Vaccine passports restrict the free flow of commerce during a time when life and the economy are returning to normal,' Little said. 'Vaccine passports threaten individual freedom and patient privacy.'",
    "I’m opposed to vaccine mandates, what do you think?",
    "I'm not against the vaccine but I am against the mandate",
    "COVID Vaccine Passports: Unconstitutional, Un-American & Unacceptable",
    ##new from cluster iter1
    "On Tuesday, I heard about Fresa’s illegally (see SB 968 from regular session) requiring vaccine passports to enter. I called to verify. After they confirmed, I told them it was against the law. They did not agree. Complaints were submitted to TABC on them and another restaurant. When their liquor licenses are threatened, they comply! The new law in action, it worked!",
    "News for Maryland - MCPS employee files lawsuit over COVID-19 vaccine mandate. Install SmartNews for free to read more.",
    "Had the opportunity to speak yesterday, broadcast on Channel 3000 news in Madison, about potential litigation against Wisconsin health care providers based on their mandatory COVID-19 vaccination policies for employees. Give it a watch if you're so inclined and let me know your thoughts as well. Thanks much. #healthcare #litigation #covid",
    "More than 100 new #laws that passed during the 2021 #Florida #legislative session took effect this month. Here are five—from Covid vaccine “passports” to assessing the “intellectual freedom and viewpoint diversity” at state colleges and universities—that will have a direct impact in your life. \n 1. Outraged by an unfounded perception that social media companies are biased against conservatives, Republican lawmakers passed a measure (SB 7072) that seeks to prohibit social media companies from banning political candidates that violate their terms of agreement. Companies could face fines of $250,000 a day for statewide candidates. \n 2. New bill (HB 529) requires all K-12 public schools to hold moments of silence each day. Gov. Ron DeSantis stated that “our founding fathers” did not believe in “the idea you can just push God out of every institution.” Although teachers cannot recommend what students do with that time, opponents believe the measure promotes prayer in schools. \n 3. The GOP-controlled Legislature passed HB 233, a measure that requires conducting surveys to assess the “intellectual freedom and viewpoint diversity” at state colleges and universities. Those who oppose it fear that Holocaust deniers, alt-right racists and QAnon adherents will seek to peddle false theories in classrooms. \n 4. Following DeSantis’ lead, lawmakers approved SB 2006, a measure that prevents businesses, schools, and government agencies from requiring people to show documentation—so-called “vaccine passports”—certifying COVID-19 vaccinations, before entering the premises. The law also gives the governor power to override local orders during health crises, despite Florida still being among the states with the highest number of infections. \n 5. The Legislature approved SB 1884, a measure that broadens a 2011 law threatening tough penalties against local governments that impose gun regulations. #florida #floridalaw #floridabill",
    ##new from cluster iter2
    "Why is the City of Phoenix so comfortable with such a blatant abuse of power - especially one that threatens our first responders during a time when violent crime is surging? No free people should ever be compelled by force of government into certain medical decisions they disagree with!",
    "Can you believe it? Democrat elected officials are forcing our police officers out of the job over the Covid-19 Vaccine.",
    "This is outrageous and is happening on Greg Abbott’s watch. He knows his executive order doesn’t work. So do the members of the Texas House and Senate who have joined the Texas GOP in calling for a special session to outlaw vaccine mandates. Why won’t Abbott call a special session? https://texasscorecard.com/state/rice-university-mandates-employee-covid-19-vaccinations/",
    "At least one Recall Gavin 2020 gubernatorial candidate believes the governor has overstepped his authority in requiring all state workers and healthcare workers to show proof of full vaccination or test negative weekly for COVID-19.",
    "OGLES DEFIES BIDEN ADMINISTRATION! ... Stating 'your healthcare is your business, and I will willfully go to jail before forcing any of you to take the vaccine.'... This is not the first time Mayor Andy Ogles has defied the Biden administration on COVID-19 mandates... https://tennesseestar.com/2022/01/11/maury-county-mayor-andy-ogles-says-hell-go-to-jail-before-he-forces-county-employees-to-take-covid-19-vaccine/"
  ],
  
  "VaccineReligion": [
    "Does your religion require vaccination? Check out this encouraging word from Pastor Bunjee Garrett. www.csa.church",
    "No major religious denomination opposes vaccination , but religious exemptions may still complicate mandates",
    "Is your school or employer forcing you to get a vaccine? Does receiving the vaccine violate your religious rights? PJI is here to help! Click the link below to receive your free copy of PJI’s vaccine resources!",
    "The leaders of the South Dakota Catholic Conference stated this week that they oppose vaccine mandates, and that Catholics can object to a Covid-19 shot on religious principles. The bishops of South Dakota stated: “One may accept Covid-19 vaccines in good conscience if certain conditions are met, but doing so is not a universal moral duty.”⠀Click the link in our bio for the full article.",
    "When Conway Regional Health System noted an unusual uptick in vaccine exemption requests that cited the use of fetal cell lines in the development and testing of the vaccines, the administration gave a religious attestation form to those requesting the exemption. However, the form also included 30 commonly used medicines that fall into the same category as the COVID-19 vaccine in their use of fetal cell lines."
  ],
  
  "VaccineEfficacy": [
    "The vaccine is effective.",
    "Are the COVID-19 vaccine safe and effective for pregnant women? Hear from pregnant mom Lindsay on what her doctor told her.",
    "The vaccine is an effective and powerful tool that is widely available. It dramatically reduces your chances of getting COVID.",
    "The vaccine is essential.",
    "New data from AP shows that more than 99% of COVID-related deaths in the U.S. are among unvaccinated people. With this data, BMC's infectious diseases physician, Dr. Cassandra Pierre urges other healthcare workers to talk to their unvaccinated and undecided patients to answer their questions about the vaccine. 'The race is not over for our patients.' Read more on HealthCity:"
  ],
           
  
  "VaccineDevelopment": [
    "The development of the COVID-19 vaccines did not cut corners on testing for safety and efficacy. The vaccines were made using processes that have been developed and tested over many years, and which are designed to make — and thoroughly test — vaccines quickly in case of an infectious disease pandemic like we are seeing with COVID-19.",
    "While the COVID-19 vaccines are relatively new - the technology and science behind the vaccines have been in development for decades. In the video below, we demonstrate how years of vaccine research and advanced technology allowed researchers and scientists worldwide to be prepared to develop an mRNA vaccine that could help fight the spread of a global infectious disease.",
    "'The speed in development resulted because of its world-wide impact — this country was paralyzed, we mobilized everything to make this work. COVID was everywhere.' BMC's infectious diseases expert, Dr. Sabrina Assoumou shares further insight into the development of the COVID-19 vaccines. Read more with Bay State Banner:",
    "“The FDA’s approval of this vaccine is a milestone as we continue to battle the COVID-19 pandemic. While this and other vaccines have met the FDA’s rigorous, scientific standards for emergency use authorization, as the first FDA-approved COVID-19 vaccine, the public can be very confident that this vaccine meets the high standards for safety, effectiveness, and manufacturing quality the FDA requires of an approved product,” said Acting FDA Commissioner Janet Woodcock, M.D.",
    "The world has responded to the worst pandemic in a century by revolutionizing vaccine development. But it didn’t happen overnight. Swipe to learn why Covid-19 vaccines were decades in the making."
  ],

  "CovidPlan": [
   "COVID isn’t over. Let’s plan how to get back to normal, safely and quickly.",
   "The pandemic has been a spotlight on all the cracks in our society and The America Rescue Plan is here to help.",
   "Congress passed the American Rescue Plan—a bill that will help fight the effects of the COVID-19 pandemic. From funding vaccine production to get more shots in arms, to providing more money for individuals and families, it is a critical first step to getting back to normal. Just ask Miren, a child care provider who can now rest easy knowing that she and the children she cares for will have access to the resources they need to stay safe. Learn more about how the American Rescue Plan is supporting working people by clicking below. 👇👇👇",
   "Is your community interested in becoming a Point of Dispensing (POD) the Covid 19 vaccine?  Visit our website for further information: Westbocacc.com",
   "After lots of pushing from all of the Hudson County elected representatives we are finally getting a big allocation of vaccine doses for our city/county. There will be more information on how to sign up but I wanted to make sure you are aware that next week presents a big opportunity for us to move forward with a large number of vaccinations. https://hudsoncountyview.com/hudson-county-to-triple-prior-covid-19-vaccination-allocations-next-week-with-nearly-21k-doses/"
  ],

  "VaccineMisinformation": [
    "There is no evidence that the COVID-19 vaccine can cause problems with fertility or pregnancy.",
    "COVID-19 vaccines don't hurt fertility. A lot of people have concerns about vaccines. You deserve reliable info. If you're worried, talk to your doctor.",
    "Stand with us in the critical fight to combat COVID-19 vaccine misinformation.",
    "There is no evidence to suggest that the COVID-19 vaccine causes fertility problems in males or females. But we do know that 99% of people who die from COVID-19 are unvaccinated. Determine which COVID-19 vaccine is suitable for you by visiting vipservices.org/covid19vaccine/.",
    "Despite what a popular myth may say, current evidence shows that COVID-19 vaccines do NOT harm male or female fertility! People can safely get pregnant after getting a COVID-19 vaccine."
    ],

    "NaturalImmunity": [
    "Is natural immunity from disease better than immunity from vaccine? Secretary of Health Kim Malsam-Rysdon says, “Both the COVID-19 virus and vaccine are new, but immunity appears to be greater from vaccine.” Choosing vaccination protects everyone. Learn more at doh.sd.gov/COVID",
    "We've heard a lot about herd immunity lately (sorry couldn't help it)...what does it all mean? Follow along via the latest from our Inoculation Nation feature: https://apmresearchlab.org/covid/vaccine-progress",
    "A message from Dr. Bernard H. Eichold II, Health Officer of Mobile County on herd immunity.",
    "Multiple Pfizer Employees' Admit Natural Immunity Against COVID-19 is Better Than Vaccine-induced and Why.",
    "Augusta News: Head of Maine CDC says Delta variant changes herd immunity goals. Install news app trusted by millions to stay informed of latest Augusta local news!"
    ],

    "Vote": [
    "Vote today.",
    "Early voting starts today. April 19th. I would would appreciate your vote and support #roberthfor6 #district6 #sad6 #sanantonio #citycouncil #APRIL19TH #earlyvoting #vote #voteforRobH",
    "Please remember to vote this TUESDAY NOV. 2nd for Doug Lagrange (Supervisor), Dan Leinung and Adam Greenberg (Town Council) and Lisa Williams (Town Clerk). Polls are open from 6 am to 9 pm. Thank you for your support.",
    "Vote April 6th, 2021.",
    "Let your voice be heard."
    
  ],
  ######### 5 New themes after iter 1 #######
   "VaccineRefusalNews": [
    "Lincoln County News: This is How Many People Are Refusing the COVID-19 Vaccine in Missouri. Install news app trusted by millions to stay informed of latest Lincoln County local news!",
    "Randolph County News: This is How Many People Are Refusing the COVID-19 Vaccine in Arkansas. Install news app trusted by millions to stay informed of latest Randolph County local news!",
    "Logan County News: This is How Many People Are Refusing the COVID-19 Vaccine in Arkansas. Install news app trusted by millions to stay informed of latest Logan County local news!",
    "Chicot County News: This is How Many People Are Refusing the COVID-19 Vaccine in Arkansas. Install news app trusted by millions to stay informed of latest Chicot County local news!",
    "Cedar County News: This is How Many People Are Refusing the COVID-19 Vaccine in Missouri. Install news app trusted by millions to stay informed of latest Cedar County local news!"
    ],

   "MaskMandate": [
    "My position on this is clear and remains the same: there will be no mask mandates or shutdowns in Allegany County. As a county, we will continue to strongly support vaccination efforts, but we will not place additional restrictions and burdens on citizens and businesses. https://www.wcbcradio.com/?news=mask-debate-featured-at-commissioners-meeting",
    "FYI. The state superintendent sent this today to each district superintendent today. Attacking your local school board for following Governor Pritzker’s Executive Order is unproductive. It is Governor Pritzker who is responsible for the current policy. Illinois State Board of Education: Office of the Superintendent \n Dear Superintendents:  \n  I know many of you are in a difficult position. The pandemic has required us to navigate changing circumstances and guidance. I deeply appreciate your leadership and the courage and integrity you have shown in tremendously challenging times. Many of you have requested clarity on the action the Illinois State Board of Education will take to enforce the universal indoor masking requirement, and this communication seeks to provide that clarity.  \n  As you know, Executive Order 2021-18, which went into effect last Wednesday, requires that all students, staff, and visitors wear masks indoors in all P-12 schools in Illinois. Governor Pritzker took this action after the Centers for Disease Control and Prevention and the American Academy of Pediatrics recommended it.  \n The purpose of the universal indoor masking requirement is to ensure all students can safely attend school in-person this fall. We know that consistent and correct mask use is the simplest, most effective way to keep students safely in school, where they can learn and grow to their fullest potential. Masks work best when everyone wears one. Research conducted by the CDC found that schools are safe when they have prevention strategies in place, as documented in the updated guidance from ISBE and the Illinois Department of Public Health. \n The Delta variant is causing a surge in cases and hospitalizations, and masking is a critical strategy to protect students’ access to in-person learning and to keep students, staff, and the community around them safe. We don’t throw our umbrella away in a rainstorm because we’re not getting wet. We have to keep our umbrella up until the storm passes.  \n The executive order has the force of law. I understand the pressure some school and district leaders may be facing from community members, and I will provide you with every support to understand, communicate, and comply with the order. \n However, noncompliance is not an option. I will not compromise the health and safety of students or staff, nor will I risk even one child’s life.  \n Local boards of education, schools, and school districts do not have the authority to deny the Governor’s Executive Order requiring universal indoor masking in schools. Doing so not only puts students’ health and safety at risk but also opens the district to extraordinary legal liability – potentially without any insurance to cover damages. I strongly recommend that each district consult with its legal counsel and insurer to fully understand the repercussions. \n Further, ISBE has and will use its regulatory authority, pursuant to 23 Ill Admin. Code 1.20, to ensure school districts protect students and staff; if school districts fail to do so, this risks State recognition. \n A district would have multiple opportunities to remedy the deficiencies that present a health hazard or a danger to students or staff before becoming unrecognized. A district would first have its recognition status changed to “On Probation” and would be asked to submit a corrective action plan. Failure to address the deficiencies would lead to nonrecognition, meaning total loss of access to state funding and loss of the school's ability to engage in any Illinois High School Association and Illinois Elementary School Association athletic competitions. \n These are not steps anyone at ISBE wishes to take nor should these steps be necessary. School districts have the moral and legal obligation to follow public health requirements and guidance to keep their students and staff safe.  \n Wearing a mask is simple, safe, and easy. I know it can be uncomfortable sometimes, but so are football helmets and seatbelts. Sometimes we have to bear a little discomfort for the sake of safety and because it’s the law.  \n We have so many important issues to face as we start the school year, and we want to start the year off with positivity for all our students. I ask that you respect that there is a mask requirement, communicate this to your school communities, and celebrate the return to in-person learning.  \n Thank you for your partnership and support. \n Sincerely,   \n Dr. Carmen I. Ayala  \n State Superintendent of Education  \n Illinois State Board of Education  \n Visit https://www.isbe.net/",
    "New York school mask mandates struck down, it was a great legal opinion, actually one of the best COVID legal opinions I have seen. Let’s talk about what happens next.",
    "This recommendation from the Superintendent was passed by the board at tonight's meeting. The first option was the plan that was approved. I still think masks should be optional now and up to the parent/child or employee. They again referred to the unclear county 'quarantine' These healthcare decisions should always be made by the parent/student and employees. I do not agree with any mandates including masks."
    ],

    "VaccineBrewIncentives": [
    "Still need to get your COVID-19 vaccine? Visit the Brewlab Charleston at 2200 Heriot Street, Charleston, SC for your shot and chase it with a free beverage.",
    "Still need to get your COVID-19 vaccine? Visit Cooper River Brewing at 2201 B Mechanic Street, Charleston, SC for your shot and chase it with a free beverage.",
    "Still need to get your COVID-19 vaccine? Visit Tideland Brewing at 4155 Dorchester Rd Suite C, North Charleston, SC for your shot and chase it with a free beer.",
    "Still need to get your COVID-19 vaccine? Visit Tideland Brewing at 4155 Dorchester Rd Suite C, North Charleston, SC for your shot and chase it with a free beer.",
    "Still need to get your COVID-19 vaccine? Visit Palmetto Brewing Company at 289 Huger St, Charleston, SC for your shot and chase it with a free beverage."
    ],

    "CommunityServiceByCandidate": [
    "👦🏻 Hi everyone, this is Victor Ramirez, candidate for District 2 County Council. 🗓 My team and I have been excited to meet many of you at your home and in your neighborhood and we have had great conversations about issues and concerns that you care about it. I’m glad that we have been able to help residents of District 2. ✅ Such as helping to put speed humps, streetlight, neighborhood clean ups, food banks and hosting covid 19 vaccination clinics to name a few things that we were able to accomplish in 2021. 🤝 Imagine what we can accomplish together when I become the next county council member for District 2. ⚽️ As I often reminded my northwestern High School Soccer Team on their journey to winning a state championship, there is no I in team and that working together you can accomplish great things. 🙏🏻 Together we will accomplish so much more and we will move District 2 forward. 🔊 As always, My promise is to listen to the residents and their needs. If you want to join our team, please visit us at victorramirez.com and remember.😼 “Once a wildcat, Always a wildcat!”",
    "Just wanted to say hello and introduce myself! I'm running for Board of Education later this year. Have been a life long BH resident who you may know as someone who helps clear out your driveway during a storm, details your neighbors car, or just as 'that computer guy,' understandably! Earlier this year while I was booking vaccine appointments the WSJ came and did a quick segment on me that will give you some context. This was when I had just started the BH Senior Vaccination Program. By the end, I'd booked 580+ senior vaccine appointments on behalf of the town when Union County had no appointments. Due to some amazing BH friends, I had help contacting nurses and teachers who needed appointments for their parents. Over 1,500 appointments booked total for residents of BH. Working directly with the Mayor and her brilliant administrator Liza, it was one of the most fulfilling things I've done! Thank you Joanna from the WSJ for this piece.",
    "Just wanted to say hello and introduce myself! I'm running for Board of Education later this year. Have been a life long BH resident who you may know as: someone who helps clear out your driveway during a storm, details your neighbor's car, or just as 'that computer guy,' understandably! Earlier this year while I was booking vaccine appointments the WSJ came and did a quick segment on me that will give you some background. This was when I had just started the BH Senior Vaccination Program. By the end, I'd booked 580+ senior vaccine appointments on behalf of the town, when Union County had no appointments available. Due to some amazing BH friends, I had help contacting nurses and teachers who needed appointments for their parents. Over 1,500 appointments booked total for residents of BH. Volunteered directly for the Mayor and her brilliant administrator Liza on the senior vaccination program, it was one of the most fulfilling things I've done.",
    "Dear Fellow Arlington Residents, I am pleased to announce that I am running for re-election to the Arlington Select Board. The past three years have been an incredible experience. I continue to learn from my colleagues and town residents and grow as a Select Board Member each day. I am proud to represent a diverse, vibrant community, whose residents actively participate in the political process and hold their elected officials to a very high standard. The last term has not gone by without challenges. As a community, we had to solve the problem of a crumbling high school, while maintaining fiscal stability and continuing to provide top notch town services to residents. I am grateful to our voters for overwhelmingly choosing to invest in Arlington’s school children. In light of a national reckoning on racial injustice, we have had many discussions to identify racism that still exists both nationally and here in town. Town officials and staff have collaborated with experts and our residents in efforts to end racial discrimination in Arlington and ensure that we are truly a municipality where all are welcome. We have made significant progress, but there much more work to do. Soaring property values in Arlington have led to a housing affordability crisis in town. Almost universally, officials and residents have expressed a commitment to take action to increase affordable housing. We have engaged in an ongoing effort to review zoning bylaws to identify common sense changes that will help to realize this important goal. In the next year, we will take action to break down barriers to affordable housing and support diversity in Arlington. In the midst of a global pandemic that has shocked our lives, strained families and businesses, and put our population at serious risk, town officials have partnered with residents to take necessary precautions to keep our town safe. I am truly proud of the way Arlington coalesced in response to COVID-19 to ensure that the town remains a safe place to live. I want to thank our town staff, our first responders, and all our essential workers for their incredible efforts during this difficult time. As the vaccine rolls out and we see light at the end of the tunnel, it is imperative for town government and residents to support our local businesses to ensure that they can remain open or re-open when safe to do so. While we have achieved much in last three years, there is significant work ahead of us. With uncertainty in future revenues, both locally and at the state level, we need to find creative ways to maintain fiscal stability and honor the commitments we made to our school aged children and seniors. We need to continue to engage in community conversations about race, learn from our differences, and work toward true racial equity. And, we need to work collaboratively with town officials, residents, and interested parties to alleviate our housing affordability crisis. I look forward to addressing these issues in addition to our ongoing efforts to modernize our transportation infrastructure and protect our open spaces. There is far more that unites our residents than divides us, and through collaboration and mutual respect, I am confident that we can build on our progress and achieve these important goals. I respectfully ask for one of your two votes on April 10th. Thank You! John V. Hurd To read more, volunteer, or donate, please visit: www.re-electjohnhurdselectboard.com or e-mail: reelectjohnhurdselectboard@gmail.com",
    "Joseph Geistman, Head of the Victoria Young Republicans joins us in studio to discuss his run at City Council District 3 seat. Mr. Geistman responds to our questions and gives his thoughts on running for City Council District 3. We had a great time and thank Mr. Geistman for his time and service to our community. You can see the Mayoral and City Council Debates for the City of Victoria at the following links: Youtube: https://www.youtube.com/watch?v=LPGzTW3jgEQ \n Facebook: https://fb.watch/5-yvhUjNj_/ \n These special Election Editions of Meet Victoria with Caleb Shaw are sponsored by Lone Star News Network. Instagram: @shawrealty @calebishaw @meet.victoria @iamthesteveo @lonestarnewsnetwork \n #TeamShaw #Victoriatx #MeetVictoria #MeetVictoriawithCalebShaw #VictoriaEvents #Victoria #victoriatexas #local #localbusiness #calebshaw #shawrealty #txrealestatebroker #elections #citycouncil #localelections #LoneStarNewsNetwork #LSNN."
    ],

    "AdvocatingUnifiedLiberties": [
    "Enjoyed joining Eric Bolling on Newsmax today, talking about how it is critical to resist totalitarian mandates thrown down by petty, tyrannical local elected officials who want to force kids to wear masks and people to be vaccinated. When freedom is lost, it's nearly impossible to get back. Let me know your feedback on this interview and please join our team to ensure we always push back Big Government control.",
    "When times are tough, our freedoms are more vital than ever. This session, we took big steps to ensure your individual liberties are not trampled on! We prohibited the government from shutting down churches and restricting your right to worship in-person, ensured your local elected officials have the final say on health orders and mandates, and guaranteed your local state legislators have a voice in the decisions that affect all of us during statewide emergencies. We also prohibited the government from issuing or requiring COVID-19 vaccine passports. It’s time to stand up for our freedoms!",
    "'Those who will not be governed by God will be ruled by tyrants.'' –William Penn \n The video below is of a Fort Wayne, Indiana, Nurse Practitioner who refused to treat a minor patient who had been diagnosed with sinusitis, ear infection & bronchitis because he wasn't COVID 19 vaccinated. This video is an evil and a disgusting reminder of how depraved our society has become. This type of depravity, which is cheered on by politicians on the left and largely ignored by politicians on the right, is a sad example of how divided we have allowed ourselves to become. \n God commands us to love, honor, and help each other and our Constitution's main charge of the government is the duty to protect and safeguard the rights of its citizens. But as John Adams once famously wrote, 'Our Constitution was made only for a moral and religious people. It is wholly inadequate to the government of any other,' because the U.S. Constitution's design for maximum human liberty works only when the people have a moral compass beyond their own appetites. \n My fellow Texans, we are in the midst of a cultural and spiritual war for the heart, mind, and soul of not just Texas but America. Please join my campaign for Texas House District 57 as I fight to bring a strong, principled and conservative voice to Austin. Visit my website www.poolefortexas.com for more information about my campaign, how to contact me, and how you can support my campaign.",
    "When you wish death upon someone in the name of saving life. When you beat someone bloody in the name of their health.When you mandate that a child risks their future to make you feel safer. When medical professionals are silenced in the name of ‘science’ - the process of testing and challenging hypotheses.Something else is at work. A darker thing, lying in the shadow of your psyche. Your fear turned to rage and projected externally. Subconsciously believing that the solution to what you feel inside, lies in the destruction of the Other. This has been the sentiment at the root of countless atrocities. With pitchforks, fire, and now with pixels. I pray that we do not repeat our sordid history. I pray that we stop dehumanizing ALL people, and instead choose reverence. There are no ‘sheep’. There are no ‘unclean’ domestic terrorists. There are humans making the best choices they can with the information they have been exposed to and their intuition. \n Left, right  \n Red, blue,  \n Vaxxed, unvaxxed. \n We’re all on the same team. A human team, part of the Earth team, part of the divine team, and our team needs us. Shall we fight amongst ourselves until our planet burns? Will righteousness be a comforting blanket when the oceans die and the crops fail? If humans are the problem, humans are also the solution. We need each other. Please, please, I pray we wake up before it is too late. I want my children and their children to dive off waterfalls, go to the Louvre, dive in the barrier reef. We’ll never get there unless we come together. It’s okay to fight for what you believe in, we need to stand for our truth, but may we never forget that behind the beliefs of another is a soul just like ours. It is time to end the violent desire for homogeneity and unite the polarity by remembering our humanity. This is what I stand for. If you feel this too, use the hashtag #unitedpolarity and let’s make it a movement 💚🤞"
    ],

    ######### 4 New themes after iter 2 #######

    "CovidEconomy": [
    "Did you miss our Global Forum on Economic Recovery? If so, watch our CEO, Suzanne Clark, and Bill Gates discuss how the pandemic, the growing importance of digital solutions, and climate change impact our economic future. Watch the full convo now!",
    "If 2020 was a dry run for our next big economic disaster, how’d we do? Ian Bremmer spoke to economic historian Adam Tooze about what lessons can be learned from this dark time, as well as his new book Shutdown: How COVID Shook the World Economy.",
    "Stocks in the Future invites you to attend a free virtual event coming up on Thursday April 29th: Forging Ahead in a Post COVID Economy \n In March 2020, we were all thrust into a new way of doing things. Communities and businesses in Baltimore and beyond have felt the impact of the pandemic in many ways both big and small. Now the COVID vaccine is rolling out, and our country is being led by a new administration. What will our economy look like in the next month? Six months? Year? What should our focus be to make improvements? What are the good things we can look forward to, and what challenges lay ahead? What are we doing at the city, state and national level to improve things? What can we all do as individuals to move things ahead? Our panel discussion will take a look at these questions and share their insights. \n Our Esteemed Panelists* Peter Franchot, Comptroller, State of Maryland, Bill Henry, Comptroller, City of Baltimore Moderator: Jayne Miller, WBAL-TV \n In addition to an informative discussion, we look forward to sharing presentations from SIF alumna, current SIF students, our Executive Director, our Board President and the drawing for a 50/50 raffle! \n Register today to attend Forging Ahead in a Post COVID Economy at 7:00 p.m. on Thursday, April 29, 2021. Join us as our local and state financial leaders share their perspective on the outlook of our economy. We look forward to an educational and insightful discussion. SIF is not immune to the effects of COVID-19. While the event is free to attend, we hope you will consider making a donation of $25 or more. Your gift will support SIF as we strive to continue to provide financial literacy education for middle school students; including more than 1,000 students at 20 schools in Baltimore and the surrounding areas. Registrants will receive a Zoom link and password during the week of the event. Register by emailing us at stocksinthefuture1@gmail.com We will respond on April22nd (and thereafter) with a confirmation of your registration, a ZOOM link, a link to donate to SIF and details of the event. Thank you for your support!",
    "Market Highlights - April 26th, 2021 | Macroeconomic data continue to show a post-winter storm rebound. While Covid increasingly appears to be under control here in the U.S., India has driven global cases to new highs. The combination of pent-up demand, stimulus checks, and increasing vaccinations is contributing to economic boom-like conditions. We continue to monitor mounting inflationary pressures and will be focusing on those companies that have pricing power to protect or sustain profit margins.  The Biden administration is 'going BIG' with deficit spending, seemingly attempting to accomplish all of its policy objectives at once. Meanwhile, we expect the Federal Reserve to continue to maintain its highly accommodative monetary policy for the time being. We are testing limits of how far we can go with massive deficits, debt and central bank balance sheet expansion with uncertain longer-term consequences. We are now roughly one third of the way through earnings season and so far both the rate and magnitude of earnings beats have surprised to the upside, driving further upward revisions to S&P 500 earnings estimates.  Bond yields, meanwhile, have taken a breather, providing highly-valued growth stocks (and home buyers) a reprieve for now. We see stock valuations as full and investor complacency as elevated – preconditions for increased market volatility in the weeks and months ahead. Risks continue to be an unexpected vaccination snafu or a troublesome new Covid variant, an inflation scare that causes a “disorderly” rise in bond yields, changes in Federal Reserve policy stance, and rising taxes. However, we remain broadly constructive on the market on the basis of highly accommodative monetary policy, massive fiscal stimulus, increasing vaccinations and rising S&P profit estimates. READ MORE: https://bit.ly/3nn8ul2 Disclosure: bit.ly/3kwUQuk #InvestmentStrategies #AtlantaGA #Fiduciary #RIA #InvestmentManagement #PortfolioManagement #RiskManagement #Investing #InvestmentAdviser #AssetAllocation #WealthManagement",
    "Wealth Chatter Blog: Late Summer Musings This has been the hottest, most humid and wettest August I can remember in the Susquehanna Valley of Pennsylvania. It’s feels like a Florida summer absent the huge bugs and bad drivers. I’m so ready for college football and cooler weather as the calendar turns next week. Here’s what’s on my mind as Summer 2021 wanes. • For the first time in years, I intentionally skipped writing an investment outlook this summer, was hoping no one would notice. We’re still doing copious research and reading too many investment commentaries, but these are historically peculiar times with no historical context. • We see status quo investment conditions for the balance of 2021 and have our client portfolios in a fully invested position avoiding fixed interest rate credit - investment grade and high yield bonds - as much as possible.• The robust US economic reopening in 2021 is still ascendant. Nearly every business we know is booming; many, however, are still held back by a severe labor pool shortage and supply chain bottlenecks. • More good news arrives next week - federal unemployment benefits FINALLY end on September 4th. It was silly fiscal policy to pay workers more money to not work for so long; the cessation should really help solve the labor crisis.• We’re not overly concerned about the Delta COVID variant derailing economic momentum. Only the unvaccinated are getting really sick and there’s scant interest in a return to mass shutdowns. Moreover, there’s evidence that Delta cases are starting to peak despite the uneasy hospitalization headlines. Get vaccinated for your loved ones.• The FDA approval of the Pfizer vaccine last week was big news. The FDA did not succumb to political pressure to shortcut their normal new drug review process, which lends even more credence the Pfizer vaccine is safe. Immediately after Pfizer received FDA approval, many employers nationwide announced vaccination requirements to come to work. Good.• Mandatory vaccinations coupled with no more gratuitous unemployment benefits should fix the US labor shortage and sustain the economic momentum. (For more go to https://www.harvrock.com/blog.htm?ID=9071)"
    ],

    # "PromoteCommunityVaccination": [
    # "Get your COVID-19 vaccine on the Las Vegas Strip! We’ll have your choice of COVID-19 vaccines (one-shot Janssen for 18+, 2-dose Pfizer for 12+ or 2-dose Moderna for 18+) this Saturday, from 10 a.m. to 6 p.m., at Double Barrel at Park MGM. Nevadans and those visiting Las Vegas are welcome, and COVID-19 testing will also be available. Get protected from COVID-19, and enjoy a free beverage of choice (alcoholic or non, age restrictions apply) while you wait your 15-minute observation time. If you're already vaccinated and you bring a friend to get vaccinated, you're both entered to win raffle prizes. Plus don’t forget: All COVID-19-vaccinated Nevadans are entered to win millions in cash/prizes during Vax Nevada Days!",
    # "Immunize Nevada, in partnership with the city of Fernley, is hosting a COVID-19 vaccine clinic for those 12+ with the Janssen, Moderna and Pfizer COVID-19 vaccines. Those individuals that receive their vaccines at this event will be able to swim for free that day at the Fernley Swimming Pool. Fernley Swimming Pool | 12+ COVID-19 Vaccine Clinic | No appointment necessary",
    # "Appointments are no longer required at the Miami-Dade County vaccination sites. The Aventura Mall Drive Thru site is open 7 days a week from 10 a.m. – 7 p.m. 19501 Biscayne Blvd. Aventura, FL 33180 Enter on NE 29 CT near Citibank For a faster registration process on-site, residents are encouraged to sign up online for appointments available both same-day and in coming days by visiting miamidade.gov/vaccine or calling 305-614-2014. #VaxUpMiami",
    # "Rock Park | 12+ COVID-19 Vaccine Clinic | No appointment necessary \n Immunize Nevada is hosting a COVID-19 vaccine clinic for those 12+ with the Janssen, Moderna and Pfizer COVID-19 vaccines.",
    # "Appointments are no longer required at the Miami-Dade County vaccination sites. \n The Zoo Miami Drive Thru site is open 7 days a week from 8 a.m. – 8 p.m. 12400 SW 152nd Street, Miami, FL 33177 \n For a faster registration process on-site, residents are encouraged to sign up online for appointments available both same-day and in coming days by visiting miamidade.gov/vaccine or calling 305-614-2014. #VaxUpMiami"
    # ],

    "UrgentPoliticalAdvocacy": [
    "WOW: A fed up Republican Senator finally ripped REPUBLICAN governors on air. \n To demand Fox is held liable for vaccine disinformation that’s getting people killed, sign here 👉 https://odaction.com/fox-vaccine \n Follow Brian Tyler Cohen & Occupy Democrats.",
    "Breaking Down the News: Biden’s Broken Border, Woke Capitalism, & States Strike Back What can we believe in the news today? How can we evaluate what we’re seeing and hearing in the news? Today on the show, Dustin & Sean break down the news and separate what’s real and what’s not from the Border, to Woke Corporations, and where States and individuals are turning the tables on an overreaching government. Sponsored by www.OlsonStrategies.com, www.Grassfire.us, & www.10CentTexting.com \n Sign up to get exclusive Political Trade Secrets Insider stuff at www.PoliticalTradeSecrets.com. Join our community by clicking my.community.com/dustinolson OR texting us directly at (202) 918-3345.",
    "🚨🚨🚨BREAKING: The man suspected of being ‘Q’ in QAnon just announced he’s running for our seat 🚨🚨🚨 We knew that our opponents would be extreme, but now that one of the leaders of QAnon is in the race, we need to pull out all of the stops. This is the same group of far-right, white supremacists that propagated dangerous conspiracy theories about the legitimacy of the 2020 Presidential election and the presence of microchips in the COVID-19 vaccine. We’re launching a rapid response fund to make sure our campaign has the resources to fight back and help Democrats keep the House majority. Will you rush a donation to fight back against QAnon?",
    "Hey, did you see this? \n 🔴 LIVE POLL ALERT: We’re missing your response… ⬇️⬇️⬇️ \n 𝟏𝟏:𝟓𝟗 𝐏.𝐌. 𝐃𝐄𝐀𝐃𝐋𝐈𝐍𝐄 | 𝐘𝐎𝐔’𝐕𝐄 𝐁𝐄𝐄𝐍 𝐒𝐄𝐋𝐄𝐂𝐓𝐄𝐃: 𝐃𝐎 𝐘𝐎𝐔 𝐀𝐏𝐏𝐑𝐎𝐕𝐄 𝐎𝐅 𝐏𝐑𝐄𝐒𝐈𝐃𝐄𝐍𝐓 𝐉𝐎𝐄 𝐁𝐈𝐃𝐄𝐍 𝐀𝐍𝐃 𝐕𝐈𝐂𝐄 𝐏𝐑𝐄𝐒𝐈𝐃𝐄𝐍𝐓 𝐊𝐀𝐌𝐀𝐋𝐀 𝐇𝐀𝐑𝐑𝐈𝐒? The Biden-Harris administration has made incredible progress in putting our country back on track after Donald Trump left it in SHAMBLES -- from passing the American Rescue Plan to taking action to help end gun violence to leading the quickest vaccination rollout the world has ever seen for a country our size. \n 𝐌𝐎𝐑𝐄 𝐓𝐇𝐀𝐍 𝟔𝟎% 𝐎𝐅 𝐀𝐌𝐄𝐑𝐈𝐂𝐀𝐍𝐒 𝐀𝐏𝐏𝐑𝐎𝐕𝐄 -- 𝐃𝐎 𝐘𝐎𝐔? A MAJORITY of Americans approve of the bold progress Democrats are making, but Republicans are pulling out all the stops to OBSTRUCT us -- even going so far as to attack our voting rights! The GOP will do whatever it takes to cling to power, even if that means undermining our democracy. \n 𝗧𝗵𝗮𝘁’𝘀 𝘄𝗵𝘆 𝘄𝗲 𝗻𝗲𝗲𝗱 𝘁𝗼 𝗰𝗼𝗻𝗳𝗶𝗿𝗺 𝘁𝗵𝗮𝘁 𝗝𝗼𝗲 𝗮𝗻𝗱 𝗞𝗮𝗺𝗮𝗹𝗮 𝗵𝗮𝘃𝗲 𝘁𝗵𝗲 𝗴𝗿𝗮𝘀𝘀𝗿𝗼𝗼𝘁𝘀 𝗽𝗼𝘄𝗲𝗿 𝘁𝗵𝗲𝘆 𝗻𝗲𝗲𝗱 𝘁𝗼 𝗳𝗶𝗴𝗵𝘁 𝗯𝗮𝗰𝗸. 𝗪𝗲 𝗻𝗲𝗲𝗱 𝘁𝗼 𝗵𝗲𝗮𝗿 𝗳𝗿𝗼𝗺 𝟵𝟴 𝗺𝗼𝗿𝗲 𝘁𝗼𝗽 𝗗𝗲𝗺𝗼𝗰𝗿𝗮𝘁𝘀 𝗹𝗶𝗸𝗲 𝘆𝗼𝘂 -- 𝗯𝘂𝘁 𝗼𝘂𝗿 𝗲𝘅𝗰𝗹𝘂𝘀𝗶𝘃𝗲 𝗹𝗶𝘃𝗲 𝗽𝗼𝗹𝗹 𝗰𝗹𝗼𝘀𝗲𝘀 𝗮𝘁 𝟭𝟭:𝟱𝟵 𝗣.𝗠. 𝘁𝗼𝗻𝗶𝗴𝗵𝘁, 𝗮𝗻𝗱 𝘄𝗲’𝗿𝗲 𝘀𝘁𝗶𝗹𝗹 𝗺𝗶𝘀𝘀𝗶𝗻𝗴 𝘆𝗼𝘂𝗿 𝗿𝗲𝘀𝗽𝗼𝗻𝘀𝗲. 𝗧𝗲𝗹𝗹 𝘂𝘀 𝗔𝗦𝗔𝗣: 𝗗𝗼 𝘆𝗼𝘂 𝗮𝗽𝗽𝗿𝗼𝘃𝗲 𝗼𝗳 𝗣𝗿𝗲𝘀𝗶𝗱𝗲𝗻𝘁 𝗝𝗼𝗲 𝗕𝗶𝗱𝗲𝗻 𝗮𝗻𝗱 𝗩𝗶𝗰𝗲 𝗣𝗿𝗲𝘀𝗶𝗱𝗲𝗻𝘁 𝗞𝗮𝗺𝗮𝗹𝗮 𝗛𝗮𝗿𝗿𝗶𝘀?",
    "Lawyer-Murphy’s FactWars.com projects re-election of incumbent New Jersey Governor Phil Murphy! – No-Labels Moderate Murphy, who unsuccessfully-ran (3,701-Votes: 0.9%) for Congress (CO-5) last-Year against Doug “MIA” Lamborn, as the Moderate No-Labels Party-Nominee On-Ballot Congressional-Candidate, in the hot (104^F) streets of the Springs, today publicly-projected the re-election of incumbent New Jersey Governor Phil Murphy! Murphy, a lifetime-member of the Teamsters, the UFCW, & the AFSCME, as well as a Pro-Bono Civil-Rights Shyster & the admitted-Leader of Antifa (the Anti-Fascist League founded by Future-President Gen. Dwight D. Eisenhower in 1941) who has more Followers than the entire-population of Booger Town (TX), was quoted as saying, “With 84% of precincts reporting, the two candidates are tied in a dead-heat, but most of the outstanding-Votes are in Democrat-counties. My cousin, Phil, won the rare re-election because he supports paid family-leave & free-tuition community-college! Also, he saved thousands of lives by implementing a rigorous vaccination-program! But, what happened to Police-Reform & Voting-Rights? Nothing has changed, except Management! Meet the new Boss, same as the old Boss! We are the No-Labels Party on the Bottom of the Ballot, and We demand to be heard! We are the Party of Peace, Progress, & Prosperity! We are Pro-Medicaid, Pro-Medicare, Pro-Social Security, Pro-Catholic, Pro-Life, Pro-Adoption, Pro-Fetal Viability, Pro-Hispanic, Pro-Dreamers, Pro-Labor, Pro-Union, Pro-Jobs, Pro-Free Speech, Pro-Free Press, Pro-Gun, Pro-Defense, Pro-Cop, Pro-Law & Order, Pro-FBI, Pro-Strong Borders, Pro-Environment, Pro-Net Neutrality, Pro-Balanced Budget, Pro-Business, & most importantly, Pro-Civil Rights! Here at the No-Labels-Party, We believe that Love is Love, No Human is Illegal, Black-Lives Matter, Science is Real, Women’s Rights are Human-Rights, Water is Life, and Kindness is Everything!” If you can hear this, then you are the Resistance & Rebel-Scum! Healthcare for All! No one gets left behind! God Bless America! Murphy for Congress/Speaker (CO-5) & Murphy/Warren 2024! Updates on the War on false-Charges, frame-Jobs, crooked-Cops, corrupt-Prosecutors, & non-lawyer Judges to follow … This Press-Release is a Political-Ad paid for by Dr. Marcus A. Murphy, MBA/JD: On Tuesday, November 3, 2020, We the People made a decision!"
    ],

    "AgainstSocialistPolicy": [
    "What’s A Life Worth? Who Gets To Decide? Nancy Pelosi thinks the GOVERNMENT does! Under Pelosi’s Socialist Drug Takeover Plan, Washington controls the prescription drug market, which means: \n ➡️ FEWER Breakthrough Cures \n ➡️ LESS Innovation \n ☎️ Call Chris Pappas NOW & tell him to stop Pelosi’s Socialist Drug Takeover Plan. Oppose H.R. 3! Click here to learn more >>",
    "What’s A Life Worth? Who Gets To Decide? Nancy Pelosi thinks the GOVERNMENT does! Under Pelosi’s Socialist Drug Takeover Plan, Washington controls the prescription drug market, which means: \n ➡️ FEWER Breakthrough Cures \n ➡️ LESS Innovation \n ☎️ Call Josh Gottheimer NOW & tell him to stop Pelosi’s Socialist Drug Takeover Plan. Oppose H.R. 3! Click here to learn more >>",
    "What’s A Life Worth? Who Gets To Decide? Nancy Pelosi thinks the GOVERNMENT does! Under Pelosi’s Socialist Drug Takeover Plan, Washington controls the prescription drug market, which means: \n ➡️ FEWER Breakthrough Cures \n ➡️ LESS Innovation \n ☎️ Call Elissa Slotkin NOW & tell her to stop Pelosi’s Socialist Drug Takeover Plan. Oppose H.R. 3! Click here to learn more >>",
    "What’s A Life Worth? Who Gets To Decide? Nancy Pelosi thinks the GOVERNMENT does! Under Pelosi’s Socialist Drug Takeover Plan, Washington controls the prescription drug market, which means: \n ➡️ FEWER Breakthrough Cures \n ➡️ LESS Innovation \n ☎️ Call Susan Wild NOW & tell her to stop Pelosi’s Socialist Drug Takeover Plan. Oppose H.R. 3! Click here to learn more >>",
    "What’s A Life Worth? Who Gets To Decide? Nancy Pelosi thinks the GOVERNMENT does! Under Pelosi’s Socialist Drug Takeover Plan, Washington controls the prescription drug market, which means: \n ➡️ FEWER Breakthrough Cures \n ➡️ LESS Innovation \n ☎️ Call Kim Schrier NOW & tell her to stop Pelosi’s Socialist Drug Takeover Plan. Oppose H.R. 3! Click here to learn more >>"
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


#all unq ads
df = pd.read_csv('data/unq_all_covid_fb_ad.csv') #9k ads

#for gt
# df = pd.read_csv('data/covid_vaccine_560_gt_mf_thm.csv') ##550 ads

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
mf = []
gt_thm = []
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
        # mf.append(ads['moral_foundation'][i]) #gt
        # gt_thm.append(ads['ad_theme'][i]) #gt
        
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
# predicted_fb = pd.DataFrame({'id': ad_id, 'ad_creative_body': text,  'pred_theme' : theme_p, 'gt_theme':gt_thm, 'moral_foundation':mf,'pred_phrase' : phrase_p, 'dist_p' : dist_p, 
                             
#                             'funding_entity':fe, 'neg_theme' : neg_theme_p, 'neg_phrase' : neg_phrase_p})


print("number of unique theme w/o threshold :: ", predicted_fb['pred_theme'].nunique()) 
print("number of unique phrase w/o threshold ::", predicted_fb['pred_phrase'].nunique()) 

predicted_fb.to_csv('data/covid_thm_iter2.csv', index = False) 

# predicted_fb.to_csv('data/covid_thm_mf_gt_pred_SBERT.csv', index = False) 

