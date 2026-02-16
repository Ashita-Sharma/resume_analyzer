import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

nlp = spacy.load("en_core_web_md") # basic english model
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_skills(text):

    doc = nlp(text.lower()) #converting all data to lowercase

    #common tech skills keywords
    tech_patterns = r'\b(python|java|javascript|react|node|sql|aws|docker|kubernetes|git|machine learning|data science|api|rest|graphql|typescript|c\+\+|c#|ruby|go|rust|swift|kotlin|flutter|angular|vue|django|flask|spring|tensorflow|pytorch|pandas|numpy|html|css|sass|mongodb|postgresql|mysql|redis|elasticsearch|kafka|jenkins|ci/cd|agile|scrum|jira|linux|bash|powershell|azure|gcp|firebase|heroku|nginx|apache|oauth|jwt|microservices|devops|backend|frontend|fullstack)\b'

    tech_skills = re.findall(tech_patterns, text.lower()) #finding all keywords present in text

    # extracting noun chunks/phrases as potential skills
    noun_skills = [chunk.text for chunk in doc.noun_chunks
                   if 2 <= len(chunk.text.split()) <= 4] #only keep short phrases like "python developer" etc

    return list(set(tech_skills + noun_skills))


def hybrid_match(resume, job_desc):


    # a) semantic similarity using transformers
    #converts words into vectors to find similar words(vectors)
    resume_emb = transformer_model.encode(resume, convert_to_tensor=True)
    job_emb = transformer_model.encode(job_desc, convert_to_tensor=True)
    semantic_score = util.cos_sim(resume_emb, job_emb).item() * 100 #convert to percentage

    # b) keyword-based TF-IDF
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    try: #checks specific reoccurring words
        tfidf_matrix = vectorizer.fit_transform([resume.lower(), job_desc.lower()])
        keyword_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

        feature_names = vectorizer.get_feature_names_out()
        job_tfidf = tfidf_matrix[1].toarray()[0]
        resume_tfidf = tfidf_matrix[0].toarray()[0]

        # get important keywords
        top_indices = job_tfidf.argsort()[-20:][::-1]
        important_keywords = [feature_names[i] for i in top_indices if job_tfidf[i] > 0]

        matched_kw = [kw for kw in important_keywords if resume_tfidf[list(feature_names).index(kw)] > 0]
        missing_kw = [kw for kw in important_keywords if resume_tfidf[list(feature_names).index(kw)] == 0]
    except:
        keyword_score = 0
        matched_kw = []
        missing_kw = []

    # c) skill extraction
    resume_skills = extract_skills(resume)
    job_skills = extract_skills(job_desc)

    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))

    # d) calculate final weighted score
    final_score = round(
        (semantic_score * 0.5) +  # Semantic understanding has 50% weight
        (keyword_score * 0.3) +  # Keyword matching has 30% weight
        ((len(matched_skills) / max(len(job_skills), 1)) * 100 * 0.2),  # Skill matching has remaining 20%
        2
    )

    return { #return final result as a dictionary
        "final_score": final_score,
        "semantic_score": round(semantic_score, 2),
        "keyword_score": round(keyword_score, 2),
        "skill_match_rate": round((len(matched_skills) / max(len(job_skills), 1)) * 100, 2),
        "matched_keywords": matched_kw[:10],
        "missing_keywords": missing_kw[:10],
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "analysis": {
            "total_resume_skills": len(resume_skills),
            "total_job_skills": len(job_skills),
            "overlap": len(matched_skills)
        }
    }