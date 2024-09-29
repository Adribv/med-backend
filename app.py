from flask import Flask, request, jsonify
from flask_cors import CORS  
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  


data = {
    'tablet_name': ['Zyrtec', 'Claritin', 'Allegra', 'Benadryl', 'Chlor-Trimeton'],
    'ingredients': [
        'Cetirizine HCl', 
        'Loratadine', 
        'Fexofenadine HCl', 
        'Diphenhydramine HCl', 
        'Chlorpheniramine Maleate'
    ],
    'category': [
        'Antihistamine', 
        'Antihistamine', 
        'Antihistamine', 
        'Sedating Antihistamine', 
        'Sedating Antihistamine'
    ],
    'side_effects': [
        'Drowsiness, dry mouth', 
        'Headache, fatigue', 
        'Nausea, dizziness', 
        'Severe drowsiness, dry mouth', 
        'Dry mouth, blurred vision'
    ]
}


df = pd.DataFrame(data)


df['combined_features'] = df['ingredients'] + ' ' + df['category'] + ' ' + df['side_effects']


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_tablets(tablet_name):
    idx = df[df['tablet_name'] == tablet_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:4]]
    return df['tablet_name'].iloc[sim_indices].tolist()


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    tablet_name = data.get('tablet_name')
    if not tablet_name:
        return jsonify({"error": "No tablet name provided"}), 400
    
    recommendations = recommend_tablets(tablet_name)
    return jsonify({'tablet_name': tablet_name, 'recommendations': recommendations})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Service is live!"}), 200


if __name__ == '__main__':
    app.run(debug=True)
