from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import pandas as pd
import re
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

template = """Given this text, decide what is the category the newsgroup post is concerned about. Valid categories are these:
* computer
* recreational
* science
* for-sale ads
* politics
* religion

Text: {post}
Category:
"""
prompt = PromptTemplate(template=template, input_variables=["post"])
llm = Ollama(model="llama2")
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

df = pd.read_csv("df.csv")
df.head()

post = str("Subject: " + df.at[0, "subject"]) + ". \n Message body: " + str(df.at[0, "body"])
pred = llm_chain.invoke(post)['text']
print(pred)

llm_output = pd.read_csv("llm_output.csv", index_col=0)

# Mapping text categories to their respective IDs
text_to_id = {
    "computer": 0,
    "recreational": 1,
    "science": 2,
    "for-sale ads": 3,
    "politics": 4,
    "religion": 5
}

# Function to extract the category from the LLM prediction
def extract_category(prediction):
    match = re.search('the category it belongs to is "(.*)"', prediction)
    if match:
        return text_to_id.get(match.group(1), -1)  # returns -1 if no match is found
    return -1

# Apply the function to convert predictions to category IDs
llm_output['pred_id'] = llm_output['llm_pred'].apply(extract_category)

# Setup K-Fold cross-validation
kfolds = KFold(n_splits=5, shuffle=True, random_state=1234)

# Calculate F1 scores for each fold
llm_scores = []
for _, test_index in kfolds.split(llm_output):
    fold = llm_output.iloc[test_index]
    llm_scores.append(f1_score(fold['label'], fold['pred_id'], average='macro'))

# Output the average F1 Macro Score
print("Average F1 Macro Score for LLM:", np.mean(llm_scores))

import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

mn_nb_scores = [0.71, 0.72, 0.70, 0.69, 0.71] #multinomial niave bayes

t_stat, p_value = ttest_rel(llm_scores, mn_nb_scores)
print("T-statistic:", t_stat)
print("P-value:", p_value)

labels = ['LLM', 'Multinomial NB']
means = [np.mean(llm_scores), np.mean(mn_nb_scores)]

plt.bar(labels, means, yerr=[np.std(llm_scores), np.std(mn_nb_scores)], capsize=5)
plt.ylabel('F1 Macro Score')
plt.title('Comparison of LLM and Multinomial Naive Bayes Performance')
plt.show()
