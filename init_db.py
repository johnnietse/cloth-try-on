from myapp import create_app
from myapp.extensions import db

app = create_app()
with app.app_context():
    db.create_all()
    print("Database tables initialized")