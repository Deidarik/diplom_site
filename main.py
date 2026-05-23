import time
import os
from fastapi import FastAPI, Request, Depends, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

# Импортируем наши классы
from models import SessionLocal, Gamer, Interaction
from recommender import GNNRecommender

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class WebApp:
    def __init__(self):
        self.recSys1 = GNNRecommender("LightGCN-V1")
        self.recSys1.loadData('web_model_data.pkl')
        
        self.recSys2 = GNNRecommender("LightGCN-V2")
        self.recSys2.loadData('web_model_data_v2.pkl')
        if not self.recSys2.gamesPool:
            self.recSys2.gamesPool = self.recSys1.gamesPool
            self.recSys2.item_map = self.recSys1.item_map

    def get_active_recsys(self, request: Request):
        active_model_name = request.cookies.get("active_model", "model1")
        return self.recSys2 if active_model_name == "model2" else self.recSys1

webapp = WebApp()

# ---------------------------------------------------------
# Роуты авторизации и переключения моделей
# ---------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.post("/login")
async def process_login(request: Request, login: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(Gamer).filter(Gamer.login == login).first()
    if not user:
        user = Gamer(login=login, passwordHash="dummy_hash")
        db.add(user)
        db.commit()
        db.refresh(user)

    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="user_id", value=str(user.userId))
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("user_id")
    return response

@app.get("/toggle-model")
async def toggle_model(request: Request):
    current_model = request.cookies.get("active_model", "model1")
    new_model = "model2" if current_model == "model1" else "model1"
    
    referer = request.headers.get("referer", "/")
    response = RedirectResponse(url=referer, status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="active_model", value=new_model)
    return response

# ---------------------------------------------------------
# Главная страница (Каталог)
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def catalog_page(request: Request, page: int = 1, db: Session = Depends(get_db)):
    user_id_cookie = request.cookies.get("user_id")
    if not user_id_cookie:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(Gamer).filter(Gamer.userId == int(user_id_cookie)).first()

    items_per_page = 12
    rec_sys = webapp.get_active_recsys(request)
    all_games = list(rec_sys.gamesPool.values())
    
    total_games = len(all_games)
    total_pages = (total_games + items_per_page - 1) // items_per_page
    
    start_idx = (page - 1) * items_per_page
    catalog_items = all_games[start_idx:start_idx + items_per_page] if all_games else []

    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "items": catalog_items, 
            "user": user,
            "page_title": "Каталог игр",
            "current_page": page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "active_model": request.cookies.get("active_model", "model1") # Передаем статус модели
        }
    )

# ---------------------------------------------------------
# Обновление статусов (Независимые транзакции)
# ---------------------------------------------------------

@app.post("/status")
async def update_status(
    request: Request, 
    gamerId: str = Form(...), 
    actionType: str = Form(...), 
    db: Session = Depends(get_db)
):
    user_id_cookie = request.cookies.get("user_id")
    if not user_id_cookie: 
        return {"error": "unauthorized"}

    user = db.query(Gamer).filter(Gamer.userId == int(user_id_cookie)).first()

    # ОБРАБОТКА УДАЛЕНИЯ ИЗ ИЗБРАННОГО / КОРЗИНЫ (Физическое удаление строки)
    if actionType == "remove_like":
        like_inter = db.query(Interaction).filter(
            Interaction.userId == user.userId, Interaction.gamerId == gamerId, Interaction.actionType == "like"
        ).first()
        if like_inter:
            db.delete(like_inter)
            db.commit()
        return {"status": "success", "new_action": "click"}

    if actionType == "remove_cart":
        cart_inter = db.query(Interaction).filter(
            Interaction.userId == user.userId, Interaction.gamerId == gamerId, Interaction.actionType == "cart"
        ).first()
        if cart_inter:
            db.delete(cart_inter)
            db.commit()
        return {"status": "success", "new_action": "click"}

    # ДОБАВЛЕНИЕ НОВЫХ НЕЗАВИСИМЫХ ДЕЙСТВИЙ (Лайк и Корзина теперь живут вместе!)
    existing_action = db.query(Interaction).filter(
        Interaction.userId == user.userId, Interaction.gamerId == gamerId, Interaction.actionType == actionType
    ).first()

    if not existing_action:
        user.addInteraction(db, gamerId, actionType)
        
    return {"status": "success", "new_action": actionType}

# ---------------------------------------------------------
# Остальные страницы (Обогащение контекстом)
# ---------------------------------------------------------

@app.get("/item/{gamerId}", response_class=HTMLResponse)
async def item_page(request: Request, gamerId: str, db: Session = Depends(get_db)):
    user_id_cookie = request.cookies.get("user_id")
    if not user_id_cookie:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(Gamer).filter(Gamer.userId == int(user_id_cookie)).first()
    
    # Фиксируем просмотр (click)
    existing_click = db.query(Interaction).filter(
        Interaction.userId == user.userId, Interaction.gamerId == gamerId, Interaction.actionType == "click"
    ).first()
    if not existing_click:
        user.addInteraction(db, gamerId, "click")

    rec_sys = webapp.get_active_recsys(request)
    target_game = rec_sys.gamesPool.get(gamerId)

    history = user.getHistory(db)
    recommendations_list = rec_sys.computeScores(history, top_k=6)

    recommended_games = []
    for rec in recommendations_list:
        if rec.targetGamerId != gamerId:
            game_obj = rec_sys.gamesPool.get(rec.targetGamerId)
            if game_obj:
                recommended_games.append({"game": game_obj, "score": rec.score, "rank": rec.rank})

    return templates.TemplateResponse(
        request=request, name="item.html", 
        context={
            "target": target_game, "recommendations": recommended_games, "user": user,
            "active_model": request.cookies.get("active_model", "model1")
        }
    )

@app.get("/liked", response_class=HTMLResponse)
async def liked_page(request: Request, db: Session = Depends(get_db)):
    user_id_cookie = request.cookies.get("user_id")
    if not user_id_cookie: 
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(Gamer).filter(Gamer.userId == int(user_id_cookie)).first()
    history = user.getHistory(db)

    rec_sys = webapp.get_active_recsys(request)
    liked_games = [rec_sys.gamesPool[h.gamerId] for h in history if h.actionType == "like" and h.gamerId in rec_sys.gamesPool]
        
    recommendations_list = rec_sys.computeScores(history, top_k=6)
    recommended_games = [{"game": rec_sys.gamesPool[rec.targetGamerId], "score": rec.score, "rank": rec.rank} 
                         for rec in recommendations_list if rec.targetGamerId in rec_sys.gamesPool]

    return templates.TemplateResponse(
        request=request, name="liked.html", 
        context={
            "liked_games": liked_games, "recommendations": recommended_games, "user": user,
            "active_model": request.cookies.get("active_model", "model1")
        }
    )

@app.get("/cart", response_class=HTMLResponse)
async def cart_page(request: Request, db: Session = Depends(get_db)):
    user_id_cookie = request.cookies.get("user_id")
    if not user_id_cookie: 
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(Gamer).filter(Gamer.userId == int(user_id_cookie)).first()
    history = user.getHistory(db)
    rec_sys = webapp.get_active_recsys(request)

    cart_games = [rec_sys.gamesPool[h.gamerId] for h in history if h.actionType == "cart" and h.gamerId in rec_sys.gamesPool]
    total_price = sum([getattr(g, 'price', 0.0) for g in cart_games])

    return templates.TemplateResponse(
        request=request, name="cart.html", 
        context={"cart_games": cart_games, "total_price": round(total_price, 2), "user": user}
    )

@app.post("/checkout")
async def process_checkout(request: Request, db: Session = Depends(get_db)):
    user_id_cookie = request.cookies.get("user_id")
    if not user_id_cookie: return {"error": "unauthorized"}

    user = db.query(Gamer).filter(Gamer.userId == int(user_id_cookie)).first()
    history = user.getHistory(db)

    # Меняем 'cart' на 'buy'
    for item in history:
        if item.actionType == "cart":
            item.actionType = "buy"
            item.timestamp = time.time()
    
    db.commit()
    return {"status": "success", "message": "Заказ успешно оплачен"}

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request, db: Session = Depends(get_db)):
    user_id_cookie = request.cookies.get("user_id")
    if not user_id_cookie: return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(Gamer).filter(Gamer.userId == int(user_id_cookie)).first()
    history = user.getHistory(db)
    rec_sys = webapp.get_active_recsys(request)

    bought_games = [rec_sys.gamesPool[h.gamerId] for h in history if h.actionType == "buy" and h.gamerId in rec_sys.gamesPool]

    return templates.TemplateResponse(
        request=request, name="profile.html", context={"user": user, "bought_games": bought_games}
    )