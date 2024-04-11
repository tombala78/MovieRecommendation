#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
import warnings; warnings.simplefilter('ignore')


#getting the movies metadata dataset.
df = pd.read_csv(r"movies_metadata.csv")

print(f"First 50 rows of dataframe:\n{df.head(50)}\n")

# getting the relevant columns of the dataset.
df_t=df[["id","title","original_title","genres","runtime","original_language","overview","tagline","vote_count","vote_average","adult"]]

#Ensuring each cell in "genres" column is converted to an actual list
df_t['genres'] = df_t['genres'].apply(literal_eval)

#Extracting the different ("name" key) genres for each row for the genres column
df_t['genres'] = df_t['genres'].apply(lambda row: [elem['name'] for elem in row] if isinstance(row, list) else [])

print(f"Genres column head:\n{df_t["genres"].head()}\n")

#all the columns and the number of rows that is null in these columns.
df_t.isnull().sum()

#viewing distribution of data
sns.boxplot(x=df['runtime'])
plt.title("Runtime column (not cleaned)")
plt.show()


df_t["runtime"].fillna(np.nan,inplace=True)
q1=df_t["runtime"].quantile(0.25)
q3=df_t["runtime"].quantile(0.75)
iqr=q3-q1
#Defining lower and upper bounds using quantiles to deal with outliers.
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr

#Replacing outliers with np.nan
df_t["runtime"]=df_t["runtime"].apply(lambda x:np.nan if (x>upper_bound or x<lower_bound) else x)

#Replacing np.nan with dataframes median.
df_t["runtime"].fillna(df_t["runtime"].median(),inplace=True)


#Cleaned version of the column "runtime"
sns.boxplot(x=df_t['runtime'])
plt.title("Runtime column (Cleaned)")
plt.show()

#Another plot representation of the "runtime" to viewing the distribution of the data
sns.kdeplot(df_t['runtime'],shade=True)
plt.title("Runtime column (cleaned)")
plt.show()

#As there are no null values, viewing the data using plot to see different answers. If there are no relevant answers replacing them with np.nan
sns.catplot(data=df_t['adult'])
plt.title("Adult column (Not cleaned)")
plt.show()
df_t["adult"]=df_t["adult"].apply(lambda x:x if x=="False" or x=="True" else np.nan)

#Analysing the distribution, "false" is majority.
sns.catplot(data=df_t['adult'])
plt.title("Adult column (Cleaned)")
plt.show()
#As there isn't a lot of irrelevant answers, replacing np.nan values with the majority answer ("False") 
df_t["adult"].fillna("False",inplace=True)

#Dropping the np.nan values.
df_t["title"].dropna(inplace=True)

print(f"df_t first 50 rows:\n{df_t.head(50)}\n")

#Replacing np.nan values with median
df_t["vote_count"].fillna(df_t["vote_count"].median(),inplace=True)
df_t["vote_average"].fillna(df_t["vote_average"].median(),inplace=True)

#As i will use cosine similarity, i replaced all the null values with empty string.
df_t.fillna('')

#If genres are in a list, extracting them to a single string
df_t['genres'] = df_t['genres'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
#converting titles to lowercase.
df_t["title"] = df_t["title"].str.lower()  # Convert titles to lowercase

print(f"Genres columns:\n{df_t["genres"]}\n")

#describe including Object type values
print(f"Description of overview columns:\n{df_t["overview"].describe(include=["O"])}\n")
#replacing top value, No overview found with np.nan and replacing it with empty string
df_t["overview"].replace("No overview found.",np.nan)
df_t.fillna("",inplace=True)

#describe including Object type values
df_t["original_language"].describe(include=["O"])

#getting the description of the columns "tagline"
df_t["tagline"].describe(include=["O"])
df_t["tagline"].fillna("",inplace=True)

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import linear_kernel

#to combine features, making sure that all the datatypes of the columns are string
df_t=df_t.astype(str)
#combining features into a single row (Content-based filtering)
def combine_features(row):
    try:
        # Combine columns into a single string
        combined = f"{row['title']} {row['genres']} {row['runtime']} {row['original_language']} {row['overview']} {row['tagline']} {row['vote_count']} {row['vote_average']} {row['adult']}"
        return combined
    except Exception as e:
        #if any error encountered, returning an empty string
        print(f"Error combining features: {e}")
        return ""
#applying combine_features function to all the columns
df_t['combined_features'] = df_t.apply(combine_features, axis=1)
#making sure there are no null values in dataframe
df_t['combined_features'] = df_t['combined_features'].fillna('')

# Initialize the HashingVectorizer and TfidfTransformer
vectorizer = HashingVectorizer(stop_words='english', n_features=2**18, alternate_sign=False)
tfidf_transformer = TfidfTransformer()

# Combine HashingVectorizer and TfidfTransformer into a pipeline
pipeline = make_pipeline(vectorizer, tfidf_transformer)

# Transform the combined_features to a matrix
tfidf_matrix = pipeline.fit_transform(df_t['combined_features'])

# Function to get recommendations
def get_recommendations(title, num_recommendations=10):
    if title not in df_t['title'].values:
        print(f"Warning: Title '{title}' not found in dataset.\n")
        return []
    
    # Get the index of the movie that matches the title
    idx = df_t.index[df_t['title'] == title].tolist()[0]
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix)
    
    # Get the scores of all movies
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return df_t.iloc[movie_indices][['title', 'overview', 'genres']]

if __name__ == "__main__":
    recommendations = get_recommendations("fight club",10)
    print(recommendations)
    #print(recommendations["title"])
    #print(recommendations["overview"])
    #print(recommendations["genres"])