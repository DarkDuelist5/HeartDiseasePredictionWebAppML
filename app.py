from flask import Flask, render_template, request,  url_for
app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return render_template('test.html')

@app.route('/')
def hello_world2():
    return render_template('heartdisease.html')

@app.route('/predict', methods = ["POST"])
def predict():
    row2 = ['0.0'] * 28
    row = []
    if request.method == 'POST':
        row.append(request.form['age'])
        row.append(request.form['Gender'])
        row.append(request.form['cp'])
        row.append(request.form['trestbps'])
        row.append(request.form['chol'])
        row.append(request.form['fbs'])
        row.append(request.form['restecg'])
        row.append(request.form['thalach'])
        row.append(request.form['exang'])
        row.append(request.form['oldpeak'])
        row.append(request.form['slope'])
        row.append(request.form['ca'])
        row.append(request.form['thal'])
        
        row2[0] = request.form['age']
        row2[1] = request.form['trestbps']
        row2[2] = request.form['chol']
        row2[3] = request.form['thalach']
        row2[4] = request.form['oldpeak']

        if request.form['Gender'] == '0':
            row2[5] = '1'
        elif request.form['Gender'] == '1':
            row2[6] = '1'
        
        if request.form['cp'] == '1':
            row2[7] = '1'
        elif request.form['cp'] == '2':
            row2[8] = '1'
        elif request.form['cp'] == '3':
            row2[9] = '1'
        elif request.form['cp'] == '4':
            row2[10] = '1'

        if request.form['fbs'] == '0':
            row2[11] = '1'
        elif request.form['fbs'] == '1':
            row2[12] = '1'

        if request.form['restecg'] == '0':
            row2[13] = '1'
        elif request.form['restecg'] == '1':
            row2[14] = '1'
        elif request.form['restecg'] == '2':
            row2[15] = '1'

        if request.form['exang'] == '0':
            row2[16] = '1'
        elif request.form['exang'] == '1':
            row2[17] = '1'
        
        if request.form['slope'] == '1':
            row2[18] = '1'
        elif request.form['slope'] == '2':
            row2[19] = '1'
        elif request.form['slope'] == '3':
            row2[20] = '1'
        
        if request.form['ca'] == '0':
            row2[21] = '1'
        elif request.form['ca'] == '1':
            row2[22] = '1'
        elif request.form['ca'] == '2':
            row2[23] = '1'
        elif request.form['ca'] == '3':
            row2[24] = '1'

        if request.form['thal'] == '3':
            row2[25] = '1'
        elif request.form['thal'] == '6':
            row2[26] = '1'
        elif request.form['thal'] == '7':
            row2[27] = '1'


        import numpy as np
        row = np.array(row)
        row = row.astype(float)

        row2 = np.array(row2)
        row2 = row2.astype(float)
        import randomforest2
        a1 = randomforest2.findAns(row)
        
        import knn2
        a2 = knn2.clf.getAns(row2)

        import logistic
        a3 = logistic.lr.findAns3(row2)

        cto=0
        ctz = 0
        if a1==0.0:
            ctz = ctz + 1
        else:
            cto = cto + 1

        if a2==0:
            ctz = ctz + 1
        else:
            cto = cto + 1

        if a3==0:
            ctz = ctz + 1
        else:
            cto = cto + 1
        final = ""
        if(cto>ctz):
            final = 'Presence of heart disease likely'
        else:
            final = 'No heart disease'
        return render_template('result.html', ans1=a1, ans2 = a2, ans3 = a3, ans4 = final)

if __name__ == "__main__":
    app.run(debug=True)


