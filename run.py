import os
from cvrec.cv_rec import recommend_job
from cvgenerate.src.utils.resume_generator import generate_resume
import json
import subprocess
from chatbot.QASystem.src.complete_resume_chatbot import chatbot


# start chatbot
chatbot()


# generate job recommendation
print('Generating job recommendation...')
data_list = []
with open('./chatbot/QASystem/log/output/resume.json', 'r', encoding='utf-8') as file:
    user_input = json.load(file)
    for _, value in user_input.items():
        data_list.append(value)
gpt_response = recommend_job(data_list)
print(gpt_response)


# generate resume
print('Generating resume...')
with open('./chatbot/QASystem/log/output/resume.json', 'r', encoding='utf-8') as file:
    user_info = json.load(file)
template_path1 = './cvgenerate/templates/latex_template.tex'
template_path2 = './cvgenerate/templates/latex_template_research.tex'
output_path1 = './output/resume.pdf'
output_path2 = './output/resume_research.pdf'
generate_resume(user_info, template_path1, output_path1)
generate_resume(user_info, template_path2, output_path2)
print('Resume generated successfully!')

# open the generated resume
subprocess.Popen(['open', './output/resume.pdf'])
subprocess.Popen(['open', './output/resume_research.pdf'])