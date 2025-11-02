from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime

URL_DATABASE = "postgresql+psycopg2://postgres:123456@localhost:5432/summarizer"
engine = create_engine(URL_DATABASE)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

class UserNotification(Base):
    __tablename__ = "user_notifications"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_email = Column(String, nullable=False, index=True)
    package_name = Column(String, nullable=False)
    title = Column(Text)
    content = Column(Text)
    raw_text = Column(Text)
    notification_id = Column(String)
    summary = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables only if they don't exist
try:
    Base.metadata.create_all(bind=engine, checkfirst=True)
except Exception as e:
    print(f"Database initialization error (may be harmless): {e}")
    # Tables likely already exist, continue

def get_user_by_email(email: str):
    session: Session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == email).first()
        return user
    finally:
        session.close()

def insert_user(email: str, password: str):
    session: Session = SessionLocal()
    try:
        new_user = User(email=email, password=password)
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        return new_user
    except IntegrityError as e:
        session.rollback()
        raise e
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_all_users():
    session: Session = SessionLocal()
    try:
        users = session.query(User).all()
        return users
    finally:
        session.close()

def save_notification_to_db(user_email: str, package_name: str, title: str, content: str, raw_text: str, notification_id: str, timestamp_str: str, summary: str):
    session: Session = SessionLocal()
    try:
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            timestamp = datetime.utcnow()

        new_notification = UserNotification(
            user_email=user_email,
            package_name=package_name,
            title=title,
            content=content,
            raw_text=raw_text,
            notification_id=notification_id,
            summary=summary,
            timestamp=timestamp
        )

        session.add(new_notification)
        session.commit()
        session.refresh(new_notification)
        return new_notification
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_user_notifications(user_email: str, limit: int = 50):
    session: Session = SessionLocal()
    try:
        notifications = session.query(UserNotification)\
            .filter(UserNotification.user_email == user_email)\
            .order_by(UserNotification.timestamp.desc())\
            .limit(limit)\
            .all()
        return notifications
    finally:
        session.close()

def get_user_notifications_by_date(user_email: str, start_date: str, end_date: str = None, limit: int = 50):
    session: Session = SessionLocal()
    try:
        from datetime import datetime
        
        # Parse start date
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        
        query = session.query(UserNotification)\
            .filter(UserNotification.user_email == user_email)\
            .filter(UserNotification.timestamp >= start_datetime)
        
        # If end_date is provided, add end date filter
        if end_date:
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            # Add 1 day to include the entire end date
            from datetime import timedelta
            end_datetime = end_datetime + timedelta(days=1)
            query = query.filter(UserNotification.timestamp < end_datetime)
        else:
            # If no end_date, filter for just the specific date (start_date)
            from datetime import timedelta
            next_day = start_datetime + timedelta(days=1)
            query = query.filter(UserNotification.timestamp < next_day)
        
        notifications = query.order_by(UserNotification.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return notifications
    finally:
        session.close()
