from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Track(Base):
    __tablename__ = 'tracks'

    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)
    title = Column(String, default="Untitled Track")
    filepath = Column(String, nullable=False)
    duration = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Metadata
    prompt = Column(Text)
    studio = Column(String)
    seed = Column(Integer)
    
    # User Interaction
    is_favorite = Column(Boolean, default=False)
    is_disliked = Column(Boolean, default=False) # Trash/Hidden
    is_public = Column(Boolean, default=False)
    
    # Relationships
    album_id = Column(Integer, ForeignKey('albums.id'), nullable=True)
    comments = relationship("Comment", back_populates="track", cascade="all, delete-orphan")

class Album(Base):
    __tablename__ = 'albums'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    cover_image = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    tracks = relationship("Track", backref="album")

class Comment(Base):
    __tablename__ = 'comments'

    id = Column(Integer, primary_key=True)
    track_id = Column(Integer, ForeignKey('tracks.id'))
    content = Column(Text, nullable=False)
    timestamp = Column(Float) # Timestamp in song (optional)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    track = relationship("Track", back_populates="comments")
