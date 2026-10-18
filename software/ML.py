# ============================================================
# 1. INSTALL / CHECK REQUIRED LIBRARIES
# ============================================================

# Run this only if the packages are not already installed.
# In Jupyter/Colab, you can use:
# !pip install -U pandas scikit-learn joblib

import pandas as pd
import joblib
import sklearn

print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("qna_data.csv")

# Keep only the first two columns
df = df.iloc[:, :2]

# Give them standard names
df.columns = ["question", "answer"]

print("\n🔍 Preview of dataset:")
print(df.head())


# ============================================================
# 3. CLEAN DATASET
# ============================================================

# Remove empty rows
df.dropna(subset=["question", "answer"], inplace=True)

# Convert to string
df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)

# Remove duplicate Q&A pairs
df.drop_duplicates(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

print("\n✅ Cleaned dataset shape:", df.shape)
print(df.head())


# ============================================================
# 4. SPLIT DATASET
# ============================================================

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

print("\n📊 Dataset Information")
print("----------------------")
print("Total samples :", len(df))
print("Train samples :", len(train_df))
print("Test samples  :", len(test_df))

print("\n📚 Training data preview:")
print(train_df.head())


# ============================================================
# 5. CREATE TF-IDF VECTORIZER
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use training data for the chatbot
qna_df = train_df.copy()

# Reset index
qna_df.reset_index(drop=True, inplace=True)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=1
)

# Convert all questions into TF-IDF vectors
question_vectors = vectorizer.fit_transform(
    qna_df["question"]
)

print("\n✅ TF-IDF vectorizer created successfully.")
print("Number of questions:", len(qna_df))
print("Vector shape:", question_vectors.shape)


# ============================================================
# 6. CHATBOT ANSWER FUNCTION
# ============================================================

def get_most_similar_answer(user_question, threshold=0.20):

    # Convert user question to TF-IDF vector
    user_q_vector = vectorizer.transform([user_question])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        user_q_vector,
        question_vectors
    )

    # Find the most similar question
    max_sim_index = similarities.argmax()

    # Get similarity score
    max_score = similarities[0][max_sim_index]

    # Prevent random answers for unrelated questions
    if max_score < threshold:
        return (
            "Sorry, I could not find a suitable answer "
            "to your question."
        )

    # Return corresponding answer
    return qna_df.iloc[max_sim_index]["answer"]


# ============================================================
# 7. TEST CHATBOT
# ============================================================

print("\n🤖 Chatbot is ready!")
print("Type 'exit' to stop.\n")

while True:

    question = input("❓ Ask a question: ")

    if question.lower().strip() == "exit":
        print("👋 Chatbot stopped.")
        break

    if not question.strip():
        print("⚠️ Please enter a question.")
        continue

    answer = get_most_similar_answer(question)

    print("🤖 Bot:", answer)
    print()


# ============================================================
# 8. SAVE NEW MODEL
# ============================================================

# IMPORTANT:
# This creates a NEW .pkl using your CURRENT
# scikit-learn version.

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

# Save the same Q&A dataset used by the vectorizer
qna_df.to_csv(
    "qna_data.csv",
    index=False
)

print("\n✅ Files saved successfully:")
print("   tfidf_vectorizer.pkl")
print("   qna_data.csv")


# ============================================================
# 9. VERIFY THE SAVED MODEL
# ============================================================

# Load the model again to make sure it works
loaded_vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)

print("\n✅ Saved TF-IDF model successfully reloaded.")

print(
    "Loaded vectorizer vocabulary size:",
    len(loaded_vectorizer.vocabulary_)
)

print("\n🎉 Training and saving completed successfully!")
