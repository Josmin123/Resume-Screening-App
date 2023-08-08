import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer(stop_words='english')
nltk.download('punkt')
nltk.download('stopwords')


#loading models

clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText




#web app
def main():
    st.title('Resume Screening App')
    uploaded_file=st.file_uploader('Upload Resume',type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes=uploaded_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #If UTF-8 decoding fails, try decoding with ('latin-1')
            resume_text=resume_bytes.decode('latin-1')

        clean_resume=cleanResume(resume_text)
        clean_resume_tfidf=tfidf.transform([clean_resume])
        prediction_id=clf.predict(clean_resume_tfidf)[0]
        # st.write(prediction_id)

        category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}
        category_name=category_mapping.get(prediction_id,"Unknown")

        st.write("prediction Category:",category_name)
        print(prediction_id)





#python main
if __name__=='__main__':
    main()

    