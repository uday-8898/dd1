from fastapi import HTTPException, Request, Depends
from datetime import datetime, timedelta
from typing import Optional
import mysql.connector
from mysql.connector import Error

# Database configuration
db_config = {
    "host": "mysqlai.mysql.database.azure.com",
    "database": "chatbot",
    "user": "azureadmin",
    "password": "Meridian@123"
}

# Fetch user from the database by user_email
def get_user_from_db(user_email: str) -> Optional[dict]:
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM km_registration WHERE email_id = %s", (user_email,))
            return cursor.fetchone()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Dependency to check if user account is expired
async def expiry_check(request: Request):
    # Extract user email from headers
    user_email = request.headers.get('email')
    if not user_email:
        raise HTTPException(status_code=400, detail="Missing user email in headers.")

    # Fetch user data from database
    user = get_user_from_db(user_email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Check account type and expiry duration
    if not user.get("account_type"):
        doj = user.get("doj")
        
        # Ensure `doj` is a string before parsing, or check if it's already a datetime object
        if isinstance(doj, str):
            doj = datetime.strptime(doj, "%Y-%m-%d %H:%M:%S")
        elif isinstance(doj, datetime):
            pass  # It's already a datetime object, no need to parse
        
        # Calculate expiry duration
        now = datetime.utcnow()
        expiry_duration = now - doj
        if expiry_duration > timedelta(days=7):
            raise HTTPException(status_code=403, detail="User account expired.")

    # Return user data if needed
    return user