from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.schema import Base, Track, Album, Comment

class DBManager:
    """
    Manages the SQLite database for the Music Library.
    """
    
    def __init__(self, db_path: Path):
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        print(f"💽 Library Database loaded: {db_path}")

    def get_session(self):
        return self.Session()

    def add_track(self, filename, filepath, **kwargs):
        session = self.Session()
        try:
            track = Track(filename=filename, filepath=str(filepath), **kwargs)
            session.add(track)
            session.commit()
            return track.id
        except Exception as e:
            session.rollback()
            print(f"❌ DB Error: {e}")
        finally:
            session.close()

    def get_tracks(self, filter_type="all"):
        session = self.Session()
        try:
            query = session.query(Track)
            
            if filter_type == "favorites":
                query = query.filter(Track.is_favorite == True)
            elif filter_type == "trash":
                query = query.filter(Track.is_disliked == True)
            elif filter_type == "public":
                query = query.filter(Track.is_public == True)
            else:
                # Default: Show everything NOT in trash
                query = query.filter(Track.is_disliked == False)
                
            return query.order_by(Track.created_at.desc()).all()
        finally:
            session.close()

    def toggle_favorite(self, track_id):
        session = self.Session()
        try:
            track = session.query(Track).get(track_id)
            if track:
                track.is_favorite = not track.is_favorite
                session.commit()
                return track.is_favorite
        finally:
            session.close()

    def set_dislike(self, track_id, state=True):
        session = self.Session()
        try:
            track = session.query(Track).get(track_id)
            if track:
                track.is_disliked = state
                session.commit()
                return True
        finally:
            session.close()
            
    def add_comment(self, track_id, content, timestamp=None):
        session = self.Session()
        try:
            comment = Comment(track_id=track_id, content=content, timestamp=timestamp)
            session.add(comment)
            session.commit()
            return comment.id
        finally:
            session.close()
            
    def create_album(self, name, description=""):
        session = self.Session()
        try:
            album = Album(name=name, description=description)
            session.add(album)
            session.commit()
            return album.id
        finally:
            session.close()
