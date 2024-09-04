import mysql.connector
import os

def get_fitness_goals(user_id):
    connection = mysql.connector.connect(
        host= os.getenv('DB_HOST_IP'),
        user= os.getenv('DB_USER'),
        password= os.getenv('DB_PASSWORD'),
        database= os.getenv('DB_NAME'),
        port= os.getenv('DB_PORT_NUMBER')
    )
    
    cursor = connection.cursor()

    query = "SELECT fitnessGoals FROM users WHERE id = %s"

    cursor.execute(query, (user_id,))

    data = cursor.fetchone() 
    connection.close()

    if data is None:
        raise Exception(f"User with user_id {user_id} not found.")
    
    return data[0]  # fitnessGoals will be in the first position of the tuple
