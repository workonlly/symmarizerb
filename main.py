from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from postgress import get_user_by_email, insert_user, get_all_users, save_notification_to_db, get_user_notifications
from sqlalchemy.exc import IntegrityError
from datetime import datetime

# Advanced LangChain summarizer
from langchain_summarizer import summarize_notification

app = FastAPI()

# Add CORS middleware for Flutter web support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginRequest(BaseModel):
    email: str
    password: str

class NotificationData(BaseModel):
    package_name: str
    title: str
    content: str
    timestamp: str
    id: str
    raw_text: str
    user_email: str

def summarize_text(package_name: str, title: str, content: str, raw_text: str) -> dict:
    """
    Advanced summarization using external LangChain module
    
    Args:
        package_name: App package name
        title: Notification title  
        content: Notification content
        raw_text: Raw notification text
        
    Returns:
        Dictionary with summary and metadata
    """
    try:
        result = summarize_notification(package_name, title, content, raw_text)
        return result
    except Exception as e:
        # Fallback to simple summary
        simple_summary = content[:100] + "..." if len(content) > 100 else content
        return {
            "summary": simple_summary,
            "strategy": "fallback",
            "confidence": "low",
            "error": str(e)
        }

@app.get("/")
async def root():
    return {"message": "User configuration API is running", "timestamp": datetime.now().isoformat(), "endpoints": ["/users/", "/register/", "/login/", "/notifications/", "/summaries/{email}"]}

@app.get("/users/")
async def fetch_users():
    try:
        users = get_all_users()
        user_list = [{"id": u.id, "email": u.email} for u in users]
        return {"users": user_list, "count": len(user_list), "message": "Users fetched successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/debug/")
async def debug_info():
    """Debug endpoint to check server status"""
    try:
        users = get_all_users()
        return {
            "status": "server_running",
            "users_count": len(users),
            "timestamp": datetime.now().isoformat(),
            "sample_users": [{"id": u.id, "email": u.email} for u in users[:3]]  # Show first 3 users
        }
    except Exception as e:
        return {
            "status": "database_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/register/")
async def register_user(request: LoginRequest):
    try:
        if not request.email or not request.password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        user = get_user_by_email(request.email)
        if user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        new_user = insert_user(request.email, request.password)
        return {"status": "success", "message": f"User registered successfully with ID: {new_user.id}", "user_id": new_user.id}
    except HTTPException:
        raise
    except IntegrityError as e:
        raise HTTPException(status_code=400, detail=f"Database integrity error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/login/")
async def login_user(request: LoginRequest):
    try:
        if not request.email or not request.password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        user = get_user_by_email(request.email)
        
        # If user doesn't exist, create them (register)
        if not user:
            try:
                new_user = insert_user(request.email, request.password)
                return {
                    "status": "success", 
                    "message": "User created and logged in successfully",
                    "user_id": new_user.id,
                    "email": new_user.email
                }
            except IntegrityError as e:
                raise HTTPException(status_code=400, detail=f"Error creating user: {str(e)}")
        
        # If user exists, check password (in production, use hashed passwords)
        if user.password != request.password:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        return {
            "status": "success", 
            "message": "Login successful",
            "user_id": user.id,
            "email": user.email
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/notifications/")
async def receive_notification(notification: NotificationData):
    try:
        print(f"=== Notification Received ===")
        print(f"User: {notification.user_email}")
        print(f"App: {notification.package_name}")
        print(f"Title: {notification.title}")
        print(f"Content: {notification.content}")
        print(f"Time: {notification.timestamp}")
        print(f"Raw Text: {notification.raw_text}")

        # Generate advanced summary using LangChain
        summary_result = summarize_text(
            notification.package_name,
            notification.title, 
            notification.content,
            notification.raw_text
        )
        
        # Extract summary text for database storage
        summary_text = summary_result.get("summary", "Unable to generate summary")
        
        # Save notification to database with summary
        saved_notification = save_notification_to_db(
            notification.user_email,
            notification.package_name,
            notification.title,
            notification.content,
            notification.raw_text,
            notification.id,
            notification.timestamp,
            summary_text
        )

        return {
            "status": "success",
            "message": "Notification received and saved to database",
            "notification_id": saved_notification.id,
            "user_email": notification.user_email,
            "summary": summary_text,
            "summary_metadata": {
                "strategy": summary_result.get("strategy", "unknown"),
                "confidence": summary_result.get("confidence", "medium"),
                "app_type": summary_result.get("app_type", "other"),
                "urgency": summary_result.get("urgency"),
                "sentiment": summary_result.get("sentiment")
            },
            "received_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing notification: {str(e)}")

@app.get("/notifications/{user_email}")
async def get_notifications_for_user(user_email: str, limit: int = 50):
    try:
        notifications = get_user_notifications(user_email, limit)
        notification_list = []

        for notif in notifications:
            notification_list.append({
                "id": notif.id,
                "package_name": notif.package_name,
                "title": notif.title,
                "content": notif.content,
                "summary": notif.summary,
                "timestamp": notif.timestamp.isoformat(),
                "created_at": notif.created_at.isoformat()
            })

        return {
            "user_email": user_email,
            "total_notifications": len(notification_list),
            "notifications": notification_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching notifications: {str(e)}")

@app.get("/summaries/{user_email}")
async def get_summaries_by_date(user_email: str, date: str = None, start_date: str = None, end_date: str = None, limit: int = 50):
    """
    Get user summaries filtered by date(s)
    
    Parameters:
    - user_email: Email of the user
    - date: Specific date (YYYY-MM-DD format) - optional
    - start_date: Start date for range (YYYY-MM-DD format) - optional  
    - end_date: End date for range (YYYY-MM-DD format) - optional
    - limit: Maximum number of results to return
    
    Examples:
    - /summaries/user@example.com?date=2024-11-02 (specific date)
    - /summaries/user@example.com?start_date=2024-11-01&end_date=2024-11-03 (date range)
    - /summaries/user@example.com (all summaries for user)
    """
    try:
        from postgress import get_user_notifications_by_date
        
        if date:
            # Get notifications for specific date
            notifications = get_user_notifications_by_date(user_email, date, None, limit)
        elif start_date and end_date:
            # Get notifications for date range
            notifications = get_user_notifications_by_date(user_email, start_date, end_date, limit)
        elif start_date:
            # Get notifications from start_date onwards
            notifications = get_user_notifications_by_date(user_email, start_date, None, limit)
        else:
            # Get all notifications if no date filter
            notifications = get_user_notifications(user_email, limit)

        summary_list = []
        for notif in notifications:
            summary_list.append({
                "id": notif.id,
                "package_name": notif.package_name,
                "title": notif.title,
                "summary": notif.summary,
                "date": notif.timestamp.date().isoformat(),
                "time": notif.timestamp.time().isoformat(),
                "timestamp": notif.timestamp.isoformat()
            })

        return {
            "user_email": user_email,
            "date_filter": date,
            "date_range": f"{start_date} to {end_date}" if start_date and end_date else None,
            "total_summaries": len(summary_list),
            "summaries": summary_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching summaries: {str(e)}")

@app.put("/database/")
async def update_database():
    date = "2024-01-01"
    message = f"Database updated successfully for date {date}"
    return {"status": "success", "message": message}

@app.put("/aptos/")
async def aptos_functionality():
    return {"status": "success", "message": "Aptos functionality updated"}