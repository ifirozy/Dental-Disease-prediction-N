import numpy as np 
from flask import Flask,render_template,request,jsonify
from joblib import load

app = Flask(__name__)
clf = load(filename='../rscv_random_forest.joblib')


##########################################
@app.route('/', methods=['GET', 'POST'])
def index():
    

    int_features = np.array([int(x) for x in request.form.values()])
    if int_features.shape[0] > 1:
        final_features = np.array(int_features).reshape(1,20)
        model_predict = clf.predict(final_features)
        new_predict = model_predict[0]
        if new_predict == 1:

            return render_template('index.html',final_features=new_predict)
        else:
            return render_template('index.html',final_features=new_predict)

    else:
        return render_template('index.html',final_features=2)


@app.route('/form/', methods=['GET', 'POST'])
def form():
    
    int_features = np.array([int(x) for x in request.form.values()])
    if int_features.shape[0] > 1:
        final_features = np.array(int_features).reshape(1,20)
        model_predict = clf.predict(final_features)
        new_predict = model_predict[0]
        if new_predict == 1:

            return render_template('form.html',final_features=new_predict)
        else:
            return render_template('form.html',final_features=new_predict)

    else:
        return render_template('form.html',final_features=2)




@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'),404



###########################################
if __name__ == '__main__':
    app.run(debug=True)
