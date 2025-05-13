# from .extensions import db
#
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(50))
#
#


from .extensions import db
from datetime import datetime


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50))


class VideoUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(100))
    processed_filename = db.Column(db.String(100))
    shirt_index = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)