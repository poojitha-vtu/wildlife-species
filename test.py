from flask import Flask, request, render_template
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

def get_state_data(state):
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host='localhost',
            database='wildlife',
            user='root',
            password=''
        )

        if connection.is_connected():
            # Fetch data based on the selected state
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM WildlifeSanctuaries WHERE stateName = %s", (state,))
            row = cursor.fetchone()

            # Format the data
            if row:
                return row
            else:
                return None

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        # Close database connection
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/')
def index():
    return render_template('one.html')

@app.route('/state_data', methods=['POST'])
def state_data():
    state = request.form['state']
    state_data = get_state_data(state)
    return render_template('state_data.html', state_data=state_data)

if __name__ == '__main__':
    app.run(debug=True,port=8002)


