from flask import Flask, render_template, request
import recommender as rc
import pickle
import pandas as pd

app = Flask(import_name=__name__)

with open('nmf_model.pkl','rb') as file:
        nmf_model = pickle.load(file)

user_item = pd.read_csv('user_item.csv')

#landing page for local host
@app.route('/')
# redirect to landiung page
@app.route('/request')
def query_page():
    return render_template('query.html')

# redirect to landiung page
@app.route('/nmf_results')
def get_results():
    # grab the data from requests
    user_voting = request.args.to_dict()
    user_voting = {key: int(value) for key, value in user_voting.items()}
    df_result = rc.get_recommendation(user_voting, nmf_model, 10, user_item)
    return render_template("result.html", 
        result=df_result['user'].to_dict())

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    print(user_item)
