import openai
import requests
import json

"""
根据resume内容生成岗位推荐信息
written by cris
待完善和修改
"""

# 创建OpenAI实例
client = openai.OpenAI(base_url="https://api.sambanova.ai/v1", api_key="39a11005-f501-4f6e-8672-ddda056d453a")
# 设置 Google Custom Search API 密钥和搜索引擎 ID
google_api_key = 'AIzaSyApPWhorx5F0U1imyx6z_uQcgwgITBerLk'
google_cx = '30360e174e4b8461e'


# ========================================LLM对输入的简历信息进行岗位推荐=====================================================

# 准备few-shot样本集
# samples = []
# 打开文件并读取其内容
with open('cvrec/samples.txt', 'r', encoding='utf-8') as file:
    content = file.read()
# 分隔不同的职位条目
entries = content.split('/\n')
# 转化每个条目为字典并加入到列表中
samples = [json.loads(entry) for entry in entries]


# 准备初始对话内容
prompts = [
    {
        "role": "system",
        "content": "你是一个AI简历分析助手，根据候选人的简历内容推荐合适的岗位。"
    },
    {"role": "user", "content": "请根据以下简历内容推荐合适的岗位。"}
]

# 添加few-shot样本到对话内容中
for sample in samples:
    prompts.append({
        "role": "user",
        "content": sample["prompt"]
    })
    prompts.append({
        "role": "assistant",
        "content": sample["completion"]
    })

def llm_rec(resume_str):
    # 添加用户输入到对话内容中
    prompts.append({
        "role": "user",
        "content": resume_str
    })
    
    # 使用LLM模型进行few-shot训练和岗位推荐
    response = client.chat.completions.create(
        model="Meta-Llama-3.1-405B-Instruct",
        messages=prompts,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    
    # 获取初步推荐结果
    initial_recommendation = response.choices[0].message.content 

    return initial_recommendation


# ========================================RAG对简历信息进行推荐=====================================================

# 定义一个函数来通过 Google Custom Search API 获取上下文信息
def retrieve_context(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={google_cx}"
    response = requests.get(search_url)
    results = response.json()
    if 'items' in results:
        snippets = [item['snippet'] for item in results['items']]
        context = " ".join(snippets)
        return context
    else:
        return "No relevant information found."

# 定义一个函数来生成提示（prompt）
def create_prompt(context, question):
    return f"Context: {context}\nQuestion: {question}\nAnswer:"

# 发送请求到LLM模型
def get_answer(prompt):
    response = client.chat.completions.create(
        model="Meta-Llama-3.1-405B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    return response.choices[0].message.content 

# 主逻辑
def get_RAG_ans(query):
    context = retrieve_context(query)
    prompt = create_prompt(context, query)
    answer = get_answer(prompt)
    return answer



# ========================================总结功能=====================================================
# 对LLM推荐结果和RAG推荐结果进行总结然后返回最终的输出

def summarize_recommendations(recommendations):
    prompt = f"以下是推荐的岗位及其原因：\n{recommendations}\n请将这些岗位总结成一个简洁的列表，格式如下：\n\n您适合的岗位为：\n1、岗位名称，原因为：xxx\n2、岗位名称，原因为：xxx\n3、岗位名称，原因为：xxx"
    response = client.chat.completions.create(
        model="Meta-Llama-3.1-405B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    return response.choices[0].message.content 



# ========================================主函数：LLM推荐 + RAG推荐=====================================================

# 定义函数处理输入的简历信息并进行RAG搜索
def recommend_job(resume_info):
    # 将简历信息格式化为字符串
    resume_str = f"基本信息：{resume_info[0]}\n实习经历：{resume_info[1]}\n项目经历：{resume_info[2]}\n相关能力：{resume_info[3]}\n\n推荐岗位："
    
    # 获取LLM推荐信息
    initial_recommendation = llm_rec(resume_str)
    
    # 获取RAG推荐信息
    RAG_query = f"{resume_str} job matching"
    search_response = get_RAG_ans(RAG_query)

    # 综合初步推荐结果和搜索结果
    final_recommendation = f"{initial_recommendation}\n\n根据最新的网页信息，以下是一些相关的岗位机会：\n{search_response}"
    
    # 输出前进行最后的结果总结
    summary = summarize_recommendations(final_recommendation)

    return summary



# ========================================用户输入信息并且返回=====================================================
# 示例简历信息
resume_info = [
    "年龄：25，性别：男，学历：本科，毕业学校：清华大学",
    ["实习经历1：在某互联网公司担任后端开发实习生，负责开发和维护公司内部系统。", "实习经历2：在某金融公司担任数据分析实习生，参与数据清洗和分析工作。"],
    ["项目经历1：开发一个基于Django的电商网站，负责后端开发和数据库设计。", "项目经历2：参与一个基于Flask的API开发项目，负责接口设计和实现。"],
    "相关能力：英语雅思7分，会Python编程、Pytorch等"
]

# 获取岗位推荐
# recommended_job = recommend_job(resume_info)
# print(recommended_job)
