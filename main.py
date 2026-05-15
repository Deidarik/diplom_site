from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загружаем всё из pkl
with open('web_model_data.pkl', 'rb') as f:
    data = pickle.load(f)

U_EMB = data['u_emb']
I_EMB = data['i_emb']
ITEM_MAP = data['item_map']
META = data['meta']

@app.get("/")
async def catalog(request: Request):
    items = []
    # Если хочешь случайные товары при каждом обновлении:
    # import random
    # indices = random.sample(range(len(ITEM_MAP)), 24)
    indices = range(min(24, len(ITEM_MAP))) 
    
    for idx in indices:
        asin = ITEM_MAP.get(idx)
        info = META.get(asin, {})
        
        # --- ВОТ ТУТ ИСПРАВЛЕНИЕ ---
        img_url = get_img_url(info) # Используем ту же логику, что и в карточке
        
        items.append({
            "idx": idx,
            "title": info.get('title', 'Automotive Item')[:60],
            "brand": info.get('brand', 'Generic'),
            "img": img_url # Теперь ссылка будет корректной
        })
    
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"items": items}
    )

def get_img_url(item_info):
    images = item_info.get('images', [])
    if images and isinstance(images, list):
        # Проверяем, что внутри: строка или словарь
        first_img = images[0]
        if isinstance(first_img, dict):
            return first_img.get('large_image_url', first_img.get('hi_res', ''))
        return first_img
    return ""

@app.get("/item/{item_idx}")
async def item_detail(request: Request, item_idx: int):
    target_asin = ITEM_MAP.get(item_idx)
    target_info = META.get(target_asin, {})
    
    # Расчет похожих
    target_vec = I_EMB[item_idx]
    scores = np.dot(I_EMB, target_vec)
    scores[item_idx] = -1e9
    top_indices = np.argsort(scores)[-8:][::-1]
    
    similar_items = []
    for idx in top_indices:
        s_asin = ITEM_MAP.get(idx)
        s_info = META.get(s_asin, {})
        similar_items.append({
            "idx": idx,
            "title": s_info.get('title', 'Automotive Item')[:60],
            "brand": s_info.get('brand', 'Generic'),
            "img": get_img_url(s_info) # Используем новую функцию
        })

    return templates.TemplateResponse(
        request=request, 
        name="item.html", 
        context={
            "target": {
                "title": target_info.get('title', 'Unknown'),
                "brand": target_info.get('brand', 'Generic'),
                "price": target_info.get('price', 'N/A'),
                "img": get_img_url(target_info)
            },
            "similar": similar_items,
            "asin": target_asin
        }
    )
