from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import time

# Инициализируем SQLite (база создастся в файле app_database.db)
engine = create_engine('sqlite:///app_database.db', connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==========================================
# РЕАЛИЗАЦИЯ КЛАССА Interaction (из DCDS)
# ==========================================
class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    userId = Column(Integer, ForeignKey("gamers.userId"))
    gamerId = Column(String, index=True) # Это твой parent_asin игры
    timestamp = Column(Float, default=time.time)
    actionType = Column(String) # "click", "like", "cart", "buy"

    # Отношение к Gamer (Композиция)
    gamer = relationship("Gamer", back_populates="interactions")

    def __init__(self, userId: int, gamerId: str, actionType: str, timestamp: float = None):
        self.userId = userId
        self.gamerId = gamerId
        self.actionType = actionType
        self.timestamp = timestamp or time.time()

# ==========================================
# РЕАЛИЗАЦИЯ КЛАССА Gamer (из DCDS)
# ==========================================
class Gamer(Base):
    __tablename__ = "gamers"

    userId = Column(Integer, primary_key=True, index=True)
    login = Column(String, unique=True, index=True)
    passwordHash = Column(String)

    # Связь 1 ко многим (список взаимодействий)
    interactions = relationship("Interaction", back_populates="gamer", cascade="all, delete-orphan")

    def getHistory(self, db_session) -> list[Interaction]:
        """Возвращает всю историю взаимодействий пользователя"""
        return db_session.query(Interaction).filter(Interaction.userId == self.userId).all()

    def addInteraction(self, db_session, gamerId: str, actionType: str):
        existing_interaction = db_session.query(Interaction).filter(
            Interaction.userId == self.userId,
            Interaction.gamerId == gamerId,
            Interaction.actionType == actionType # ДОБАВЛЕН ФИЛЬТР!
        ).first()

        if existing_interaction:
            existing_interaction.timestamp = time.time()
        else:
            new_interaction = Interaction(userId=self.userId, gamerId=gamerId, actionType=actionType)
            db_session.add(new_interaction)
        
        db_session.commit()

# Создаем таблицы в файле БД, если их еще нет
Base.metadata.create_all(bind=engine)