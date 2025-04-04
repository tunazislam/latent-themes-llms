import os
import openai
import pandas as pd
import sys
import time  # Import the 'time' module to handle rate limit resets
import re
from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Set your OpenAI API key
openai.api_key = ' '  # Replace with your API key

df = pd.read_csv('data/covid_thm_mf_gt_pred_SBERT.csv') 

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.reset_index(drop=True)
print(df)
# sys.exit()
# Initialize an empty list to store the assistant's replies
assistant_replies = []

# Function to make API requests and handle rate limits
def make_api_request(prompt):
    try:
        response = openai.ChatCompletion.create(
          
            model="gpt-4",
            messages=[
	            		{
					      "role": "system",
					      "content": "Don't generate new summary. Choose the most matched summary from the given 23 summaries."
					    },
                     	{
                      	"role": "user", "content": prompt
                      	}
                      
                      ],
            # temperature=0.7,
            temperature=1,
            max_tokens=40,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message["content"]
    except openai.error.RateLimitError as e:
        # Handle rate limit exceeded error by waiting for a fixed time (e.g., 60 seconds)
        print("Rate limit exceeded. Waiting for 60 seconds...")
        time.sleep(60)
        return make_api_request(prompt)  # Retry the request after waiting


def remove_links(text):
    # Define a regular expression pattern to match URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Remove URLs from the text
    text_without_links = re.sub(url_pattern, '', text)

    return text_without_links

def remove_line_breaks_and_links(text):
    # Remove line breaks
    text_without_line_breaks = re.sub(r'\n{2,}', '\n', text)
    removew_leading_indent = re.sub(r'\n\s+', '\n', text_without_line_breaks)  # Remove leading indentation
    
    # Remove links
    text_without_links = remove_links(removew_leading_indent)

    return text_without_links


# Iterate through the DataFrame and send each text as a user message
for index, row in df.iterrows():
    text = row['ad_creative_body']
    cleaned_text = remove_line_breaks_and_links(text).strip()
    # shortened_text = cleaned_text[:300]  # Example of shortening the prompt text
    shortened_text = cleaned_text  # i don't want to short
    user_message = "We have the following 23 summaries:\n\n1. Summary_GovDistrust: \nThe texts express strong criticism towards President Joe Biden and Dr. Anthony Fauci. They accuse President Biden of failing in his duties by mishandling the border and COVID-19 response, causing division within the country, and weakening international alliances. Additionally, Biden's approval ratings are described as suffering due to these issues. Dr. Fauci is accused of lying to Congress and the public, with calls for his imprisonment and accountability for alleged deceptions. The critique extends to the CDC and the Biden administration's policies on vaccinated individuals needing protection from the unvaccinated, questioning the rationale behind these measures.\n\n2. Summary_GovTrust: \nBoth Donald Trump and President Biden have advocated for COVID-19 vaccination efforts in the U.S., with Trump urging Americans to get vaccinated and Biden actively leading the national vaccination campaign, which includes making all adults vaccine-eligible ahead of schedule. Efforts are also supported by various government levels, ensuring broad access to vaccines. Additionally, Biden's broader administration efforts include rejoining the Paris Agreement, rescinding intended withdrawal from the World Health Organization, and enacting numerous executive orders to tackle the pandemic effectively and enhance public health infrastructure.\n\n3. Summary_VaccineRollout: \nCOVID-19 vaccines are now FDA approved and available for children aged 5-17, emphasizing protection and safe participation in school and social activities. Health experts, including pediatricians, advocate for vaccination to reduce the risk of severe illness in children. Community outreach includes drive-in events and school system initiatives to facilitate vaccine access. Meanwhile, state and local news platforms promote updates on vaccine rollout, encouraging residents of all ages to get vaccinated, as seen in various U.S. locations including Missouri, West Virginia, and the Bronx.\n\n4. Summary_VaccineSymptom:\nFollowing reports of rare blood clots (thrombosis with thrombocytopenia syndrome, TTS) in young adults after receiving the Johnson & Johnson COVID-19 vaccine, U.S. health agencies recommended a temporary pause to investigate. The European Medicines Agency (EMA) also acknowledged a possible link between the vaccine and unusual blood clots, recommending a warning label. After review, the vaccine was deemed safe, but with caution advised for women aged 18-49 due to a higher risk of TTS. Despite the pause, public health officials have resumed its use, stressing that such adverse events are extremely rare, with only 17 cases reported out of over 8 million doses.\n\n5. Summary_VaccineEquity:\nAdvocacy for equitable COVID-19 vaccine distribution emphasizes the need to prioritize vulnerable and marginalized populations globally, stressing that health care is a human right. Calls for action urge the U.S. and other entities to ensure vaccines are accessible to all, reflecting a widespread belief that no one is safe until everyone is safe. Organizations like Human Rights Watch are holding governments and pharmaceutical companies accountable, while community initiatives continue to support health equity through various programs and events. Overall, the message is clear: overcoming the pandemic requires making vaccines available equally to all individuals worldwide.\n\n6. Summary_VaccineStatus:\nLocal news updates across various counties in the U.S. emphasize the ongoing COVID-19 vaccination efforts. Jefferson County highlights a local doctor promoting vaccination, while Chemung County reports that 30% of its population is fully vaccinated. Wyandotte County has opened an additional vaccine site to increase accessibility. Alamance County's health department is now taking vaccine appointments via a new hotline, and Mecosta County is providing detailed information on vaccine scheduling. These updates reflect concerted efforts to encourage and facilitate vaccination among the community members.\n\n7. Summary_EncourageVaccination:\nAcross various locations, efforts are being made to facilitate and encourage COVID-19 vaccination. Promotions include no-appointment-needed vaccine clinics, incentives such as free beverages, swimming, and raffle prizes, and convenient settings like the Las Vegas Strip and local community events. Available vaccines include the one-shot Janssen and two-dose options from Pfizer and Moderna, suitable for different age groups. Locations like Miami-Dade County offer easy access at several sites with extended hours and the option to register online or by phone to expedite the process. These initiatives aim to build a healthier community by increasing vaccination rates.\n\n8. Summary_VaccineMandate:\nThere is significant opposition to COVID-19 vaccine mandates and the concept of vaccine passports across various states in the U.S. Critics argue that such requirements infringe on individual freedoms and privacy, posing constitutional concerns. Legal actions and political responses include lawsuits against mandates, state legislation banning vaccine passports, and vocal criticism from public figures who view these measures as an abuse of power. For example, Texas and Florida have passed laws preventing the enforcement of vaccine passports, reflecting a broader national debate on balancing public health needs with personal liberties.\n\n9. Summary_VaccineReligion:\nReligious perspectives on COVID-19 vaccinations vary, with most major denominations not opposing vaccination, yet some individuals seek religious exemptions. For example, leaders from the South Dakota Catholic Conference acknowledge that Catholics can conscientiously object to vaccines based on religious principles, although they also state that vaccination is not a universal moral obligation. Additionally, Conway Regional Health System addressed an increase in vaccine exemption requests due to concerns about the use of fetal cell lines by providing information on other common medicines developed similarly. Overall, the dialogue reflects a complex interaction between religious beliefs and public health policies.\n\n10. Summary_VaccineEfficacy:\nThe COVID-19 vaccine is broadly acknowledged as effective and essential, particularly highlighted by data showing over 99% of COVID-related deaths occur among unvaccinated individuals. Healthcare professionals emphasize the vaccine's critical role in reducing infection risks and controlling the pandemic. Pregnant women, advised by their doctors, are also considered safe recipients of the vaccine. This collective understanding among health experts aims to combat hesitancy and encourage widespread vaccination to protect public health.\n\n11. Summary_VaccineDevelopment:\nThe rapid development of COVID-19 vaccines did not compromise safety or efficacy, as they were built on decades of scientific research and established vaccine technology. Innovations like mRNA vaccines were prepped by years of prior study, enabling a swift response to the global pandemic. The FDA's approval underscores that these vaccines meet rigorous standards for safety, effectiveness, and manufacturing quality. Experts like Dr. Sabrina Assoumou note that the urgent global need for a solution to the pandemic catalyzed the mobilization of scientific resources, resulting in these breakthrough vaccines.\n\n12. Summary_CovidPlan:\nThe ongoing COVID-19 pandemic has highlighted societal vulnerabilities, prompting significant legislative and community responses. The American Rescue Plan, passed by Congress, aims to mitigate pandemic effects through increased funding for vaccine distribution and financial support for individuals and families, ensuring resources like childcare providers remain safe and operational. Efforts to expand vaccine availability are evident, with communities like Hudson County receiving significant increases in vaccine allocations. Additionally, initiatives to establish Points of Dispensing (PODs) for vaccines are underway, emphasizing a coordinated approach to safely and efficiently return to normalcy.\n\n13. Summary_VaccineMisinformation:\nThere is no scientific evidence linking COVID-19 vaccines to fertility issues in either men or women. Healthcare professionals and credible sources continue to address and dispel misinformation regarding the impact of vaccination on fertility. The vaccines are deemed safe for those planning pregnancies, and it is highlighted that the vast majority of COVID-19 deaths occur among the unvaccinated. Individuals concerned about vaccine safety are encouraged to consult their doctors and visit reliable health websites for accurate information.\n\n14. Summary_NaturalImmunity:\nHealth experts, including state health officials and CDC heads, are discussing the nuances of immunity in the context of COVID-19, particularly with the emergence of variants like Delta, which may shift herd immunity thresholds. Secretary of Health Kim Malsam-Rysdon suggests that vaccine-induced immunity appears to be more robust than natural immunity. Meanwhile, debates continue regarding the effectiveness of natural immunity versus vaccine-induced immunity, with some Pfizer employees reportedly favoring natural immunity. Public health messaging consistently emphasizes the benefits of vaccination for broader community protection and achieving herd immunity.\n\n15. Summary_Vote:\nVarious messages encourage voting on specific dates, emphasizing the importance of civic participation. Early voting for Robert H in District 6 starts on April 19th, and another vote is scheduled for April 6th, 2021. Additionally, there is a call to vote on November 2nd for several local officials, including a supervisor and town council members. These reminders highlight the opportunity for individuals to influence their community and government by voting and supporting various candidates across different dates and locations.\n\n16. Summary_VaccineRefusalNews:\nLocal news outlets in Lincoln and Cedar Counties in Missouri, as well as Randolph, Logan, and Chicot Counties in Arkansas, are reporting on the number of people who are refusing the COVID-19 vaccine. These reports aim to keep local residents informed about vaccine acceptance rates within their respective communities, highlighting regional responses to the vaccination efforts. The focus on vaccine refusal rates is part of broader coverage to understand and address public health challenges during the pandemic.\n\n17. Summary_AgainstMaskMandate:\nThe texts discuss the polarized responses to mask mandates in the U.S. during the COVID-19 pandemic, highlighting varying regional approaches and legal interpretations. Allegany County explicitly rejects mask mandates, emphasizing voluntary vaccination without further restrictions. In contrast, the Illinois State Board of Education, following an executive order, enforces universal indoor masking in schools, citing health safety and legal obligations. A legal victory in New York struck down mask mandates, praised as a significant judicial opinion. Meanwhile, personal opinions express a preference for optional masking, advocating for parental and individual choice over government mandates. These perspectives illustrate the complex interplay between public health directives and individual freedoms during the pandemic.\n\n18. Summary_VaccineBrewIncentives:\nIn Charleston, SC, several breweries are offering incentives to encourage COVID-19 vaccination. Brewlab Charleston, Cooper River Brewing, Tideland Brewing, and Palmetto Brewing Company are all participating in a promotion where individuals can receive a free beverage after getting their vaccine shot at these locations. This initiative aims to increase vaccination rates by combining public health efforts with local business support.\n\n19. Summary_CommunityServiceByCandidate:\nCandidates across various districts and positions are actively engaging with their communities to address local concerns and improve public services. Victor Ramirez, running for District 2 County Council, highlights his team's achievements like implementing speed humps and hosting vaccination clinics, emphasizing teamwork and community collaboration. Another candidate managed a significant local vaccination effort, booking over 1,500 appointments for residents. In Arlington, an incumbent running for the Select Board discusses challenges and achievements during their term, focusing on education, racial justice, housing affordability, and pandemic response. Each candidate showcases their commitment to addressing specific community needs and improving local governance.\n\n20. Summary_AdvocatingUnifiedLiberties:\nThe texts highlight a strong opposition to government mandates related to COVID-19, including mask-wearing and vaccination requirements. Speakers argue that such mandates infringe on personal freedoms and civil liberties. There is a significant emphasis on the need to protect individual rights and ensure that local and state authorities retain control over public health decisions rather than centralized government. The discourse also touches on moral and spiritual dimensions, suggesting a broader cultural and societal conflict, where respect for diverse viewpoints and human dignity should prevail. Overall, the sentiment is one of resistance to perceived overreach by government authorities and a call to uphold individual liberties.\n\n21. Summary_CovidEconomy:\nRecent discussions and analyses focus on the economic recovery from the COVID-19 pandemic, highlighting several key elements. In virtual events and forums, experts including Suzanne Clark and Bill Gates have debated the roles of digital solutions and climate change in shaping future economies. Financial commentators and economic historians like Adam Tooze reflect on the lessons from the pandemic, suggesting that the quick adaptation to digital platforms and economic policies have been critical. Reports indicate that while the U.S. shows signs of economic rebound, challenges like inflation, labor shortages, and supply chain issues persist. However, increased vaccination rates and fiscal policies, including stimulus measures, are seen as positive steps towards sustaining economic growth. There's a general consensus that the combination of accommodating monetary policies and strategic fiscal interventions will continue to support the market and economic recovery.\n\n22. Summary_UrgentPoliticalAdvocacy:\nA Republican senator criticizes Republican governors for promoting vaccine misinformation. A podcast discusses evaluation of news related to Biden’s border policy and woke capitalism. The alleged leader of QAnon announces his campaign run and opponents set up a fund to counteract him. An ongoing opinion poll concerning President Biden and Vice President Harris's administration's approval rate is shared. Forecasts point towards the re-election of New Jersey Governor Phil Murphy, as highlighted by Lawyer-Murphy’s FactWars.com. The summary is derived from various separate texts.\n\n23. Summary_AgainstSocialistPolicy:\nThe texts express opposition to Nancy Pelosi's 'Socialist Drug Takeover Plan', suggesting it would result in government control over the prescription drug market and subsequently fewer breakthrough cures and less innovation. The reader is urged to contact various politicians - Chris Pappas, Josh Gottheimer, Elissa Slotkin, Susan Wild, and Kim Schrier - to voice opposition to Pelosi's plan and specifically to block H.R. 3.\n\nText: Kentucky Breaking: ‘I have natural immunity’ : Ky. Sen. Rand Paul will not get COVID vaccine. Stay on top of this incident and other local breaking news with News Break.\nQuestion: Which summary does this text match the most?\nAnswer: Summary_NaturalImmunity\n\nText: The vaccines that work best against the COVID Delta variant\nQuestion: Which summary does this text match the most? \nAnswer: Summary_VaccineEfficacy\n\nText: CCHS encourages everyone who lives or works in Contra Costa and is 16 or older to get a COVID-19 vaccine.\nQuestion: Which summary does this text match the most? \nAnswer: Summary_VaccineRollout\n\nText: "+shortened_text+"\nQuestion: Which summary does this text match the most? \nAnswer: ?"
    token_count = len(tokenizer.encode(user_message))
    #print("Number of tokens:", token_count)
    if (token_count+40) < 4000:
        # Call the function to make the API request and handle rate limits
        assistant_reply = make_api_request(user_message)
        # Append the assistant's reply to the list
        assistant_replies.append(assistant_reply)
    elif (token_count+40) >= 4000:
        # Call the function to make the API request and handle rate limits
        assistant_reply = make_api_request(user_message[:3950])
        # Append the assistant's reply to the list
        assistant_replies.append(assistant_reply)

# Add the assistant's replies to the DataFrame
df['assistant_reply'] = assistant_replies

# Print or save the updated DataFrame with assistant's replies
df.to_csv('data/chatgpt_theme_covid.csv')


# # # # # # # ################# Deal with pred_theme label #################

# import os
# import openai
# import pandas as pd
# import sys
# df = pd.read_csv('data/df_before_event_llms_covid.csv')

# import re


# thm = []
# for i in range(len(df)):
# 	# # Use regular expressions to extract text starting with "Summary_"
# 	# matches = re.findall(r'Summary_\w+', df['assistant_reply'][i])
# 	# if matches:
# 	#     for match in matches:
# 	#         print(match)
# 	#         thm.append(match)
# 	# else:
# 	#     print("No matches found")
# 	#     thm.append('No matches found')

# 	# Use regular expressions to find the first text starting with "Summary_"
#     match = re.search(r'Summary_\w+', df['assistant_reply'][i])
#     if match:
#         thm.append(match.group())  # Append the matched text
#     else:
#         thm.append('No matches found')  # If no match found, append 'No matches found'


	
# df['pred_theme_llms'] = thm
# df.to_csv('data/before_event_llms_thm_covid.csv')






