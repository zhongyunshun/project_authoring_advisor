# prompts used for the question generation task.
project_location = "German Mills" # also "Humber Bay Park East" and "Peacham Cr"
gm_documents_list = ['2015-10-05 Geomorphic Report Final.pdf', '2017 Bercy Wycliffe Workplan.pdf',
                  '2018 Bercy (German Mills Settlers Park) workplan.pdf', '2021 GMSP Sites 2-3 Workplan_v2.pdf',
                  '2021-02-24 Project Brief_90 percent.pdf', 'German Mills - Project File with Appendices-unlocked_Part1.pdf']
question_generation_prompts = f'''
Please review the provided documents in this list {gm_documents_list}, which have been uploaded into the custom GPT environment.
These documents contain information relevant to the Toronto and Region Conservation Authority (TRCA), covering topics such as environmental conservation, project plans, and regulatory guidelines on {project_location}.
Based on the content of these documents, generate 10 insightful questions that:
1. Seek clarification on any complex points or terminologies used within the documents.
2. Inquire about the implications of the TRCA’s projects and initiatives on local wildlife and communities.
3. Ask for details about any strategies or measures outlined in the documents for promoting sustainable development and conservation.
4. Explore the roles and responsibilities of stakeholders involved in the TRCA’s activities.
5. Question any future plans or upcoming projects mentioned in the documents.

Ensure that the questions are diverse, covering various aspects and themes presented across all three documents, to facilitate a comprehensive understanding of the TRCA’s work and objectives.
'''


'''
Generated Questions for German Mills.
'''

# Complex Points and Terminologies Clarification:
gm_question1 = 'Can you explain the methodology used for the "Slope Stability Analysis" detailed in the Peacham Crescent Slope Stabilization Project, and how it impacts project decisions?'

# Local Wildlife and Communities Implications:
gm_question2 = 'What measures are being taken in the Humber Bay Park East Major Maintenance Project to ensure the protection and minimal disruption of local wildlife, particularly aquatic species identified in the baseline inventory?'

# Sustainable Development and Conservation:
gm_question3 = 'How does the Peacham Crescent Slope Stabilization Project incorporate sustainable development practices to ensure long-term environmental conservation?'

# Stakeholders Roles and Responsibilities:
gm_question4 = 'In the German Mills Settlers Park Sanitary Infrastructure Protection Project, what are the specific roles of the various stakeholders, including TRCA, local municipalities, and engineering firms, in ensuring project success?'

# Future Plans and Upcoming Projects:
gm_question5 = 'Are there future phases or expansions planned for the erosion control and slope stabilization efforts discussed in the Humber Bay Park East and Peacham Crescent projects?'

# Environmental and Social Impact:
gm_question6 = 'How does the erosion hazard management in areas like German Mills Settlers Park plan to address potential social impacts, such as access to parklands and community recreational spaces?'

# Technical Approach to Erosion Control:
gm_question7 = 'Could you detail the decision-making process behind choosing specific erosion control structures, such as armourstone headlands and cobble/rubble beaches, in the context of their effectiveness and environmental impact?'

# Monitoring and Maintenance Strategy:
gm_question8 = 'What long-term monitoring and maintenance strategies are outlined for the newly installed erosion control measures in projects like the Humber Bay Park East Major Maintenance Project?'

# Engagement with Indigenous Communities:
gm_question9 = 'In the development and implementation of TRCA projects like those at Peacham Crescent and Humber Bay, how are local Indigenous communities consulted or involved?'

# Impact on Biodiversity:
gm_question10 = 'How do the specified tree removals and restoration plantings in projects like the Peacham Crescent Slope Stabilization Project contribute to or mitigate against biodiversity loss, particularly concerning species at risk such as bats and butternut trees?'

'''
Generated Questions for Humber Bay Park East.
'''
hbpe_documents_list = ['Design Brief - Phase I.pdf', 'Design Brief - Phase II_DRAFT.pdf',
                  'HBPE Project Brief - Phase I.pdf', 'HBPE Project Brief - Phase II.pdf',
                  'Humber Bay Park East Concept Brief.pdf']
# Complex Points and Terminologies Clarification:
hbpe_question1 = 'In the "Design Brief - Phase II" document, what specific engineering challenges and considerations were addressed during the detailed design process for shoreline protection at Humber Bay Park East?'

# Local Wildlife and Communities Implications:
hbpe_question2 = 'How do the shoreline maintenance projects outlined in the "HBPE Project Brief - Phase II" impact the local wildlife, particularly the aquatic species mentioned in the baseline inventory section?'

# Sustainable Development and Conservation:
hbpe_question3 = 'What sustainable development practices are being implemented in the Humber Bay Park East Major Maintenance Project to ensure long-term environmental conservation?'

# Stakeholders Roles and Responsibilities:
hbpe_question4 = 'Can you elaborate on the roles and responsibilities of TRCA and the City of Toronto in the maintenance and administration of Humber Bay Park East as described in the "HBPE Project Brief - Phase I"'

# Future Plans or Upcoming Projects:
hbpe_question5 = 'What are the future phases or expansions planned for Humber Bay Park East, as mentioned in the "HBPE Project Brief - Phase II"'

# Environmental and Social Impact:
hbpe_question6 = 'How does the slope stabilization project on Peacham Crescent plan to address potential social impacts and ensure the safety of the local community'

# Technical Approach to Erosion Control:
hbpe_question7 = 'In the "Humber Bay Park East Concept Brief", how were the different erosion control structures (e.g., armourstone headlands, gravel beaches) chosen based on their effectiveness and environmental impact'

# Monitoring and Maintenance Strategy:
hbpe_question8 = 'What long-term monitoring and maintenance strategies are outlined for the newly installed erosion control measures in projects such as the "HBPE Project Brief - Phase I"'

# Engagement with Indigenous Communities:
hbpe_question9 = 'Are there any specific engagements or consultations with local Indigenous communities detailed in the projects\' planning and implementation phases ?'

# Impact on Biodiversity:
hbpe_question10 = 'How do the restoration plantings described in the "HBPE Project Brief - Phase I" contribute to or mitigate against biodiversity loss, particularly concerning species at risk?'




'''
Generated Questions for Peacham Crescent.
'''

pc_documents_list = ['2015-05-19 Cole AREA E - Peacham Cres.pdf', '2019-04-02 Conceptual Alt. Report Peacham_Final.pdf',
                  '2019-09-26 Peacham Class EA PP.pdf', '2021-09-29 Peacham Cres Project Brief_Final.pdf']

# Clarification on Complex Points or Terminologies:
pc_question1 = 'How does the Slope Stability and Erosion Risk Assessment define "Long-Term Stable Slope Crests" (LTSSC), and what criteria are used to determine these in the context of the Peacham Crescent projects?'

# Implications on Local Wildlife and Communities:
pc_question2 = 'In light of the findings from the Ecology Survey Memorandum for the Peacham Crescent Erosion Control and Slope Stabilization Project, what specific measures are proposed to mitigate impacts on local wildlife, especially concerning species identified in the vicinity?'

# Strategies for Promoting Sustainable Development and Conservation:
pc_question3 = 'Can you detail any strategies outlined in the "2021-09-29 Peacham Cres Project Brief" to promote sustainable development and conservation throughout the Peacham Crescent Slope Stabilization Project?'

# Roles and Responsibilities of Stakeholders:
pc_question4 = 'What are the defined roles and responsibilities of stakeholders involved in the Class Environmental Assessment for the Peacham Crescent Slope Stabilization Project as described in the "2019-09-26 Peacham Class EA PP"?'

# Future Plans or Upcoming Projects:
pc_question5 = 'Are there any future plans or subsequent phases mentioned within the "2021-09-29 Peacham Cres Project Brief" for further slope stabilization efforts beyond the current project scope at Peacham Crescent?'

# Environmental and Social Impact:
pc_question6 = 'What are the anticipated environmental and social impacts of the slope stabilization project, and how does the "2019-09-26 Peacham Class EA PP" propose to address these impacts?'

# Technical Approach to Erosion Control:
pc_question7 = 'Based on the "2019-04-02 Conceptual Alt. Report Peacham_Final", what were the key factors influencing the selection of the preferred conceptual slope stabilization alternatives?'

# Monitoring and Maintenance Strategy:
pc_question8 = 'What specific long-term monitoring and maintenance strategies are recommended in the "2021-09-29 Peacham Cres Project Brief" to ensure the durability and effectiveness of the implemented slope stabilization measures?'

# Engagement with Indigenous Communities:
pc_question9 = 'How were Indigenous communities engaged in the planning and decision-making processes for the Peacham Crescent Slope Stabilization Project, and what outcomes resulted from this engagement?'

# Impact on Biodiversity:
pc_question10 = 'Given the importance of biodiversity conservation, how does the "2021-09-29 Peacham Cres Project Brief" address potential impacts on biodiversity, particularly in relation to bat species and butternut trees identified in the area ?'
