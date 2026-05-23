import os
import pickle
import numpy as np

class VideoGame:
    def __init__(self, gamerId, title, brand, price, imageUrl, category="Video Games"):
        self.gamerId = gamerId
        self.title = title
        self.brand = brand
        self.price = price
        self.imageUrl = imageUrl
        self.category = category

class Recommendation:
    def __init__(self, targetGamerId, score, rank):
        self.targetGamerId = targetGamerId
        self.score = score
        self.rank = rank

class GNNRecommender:
    def __init__(self, model_type="LightGCN"):
        self.model_type = model_type
        self.gamesPool = {}
        self.u_emb = None
        self.i_emb = None
        self.history = {}
        self.item_map = {}

    def loadData(self, pkl_path):
        if not os.path.exists(pkl_path):
            print(f"Предупреждение: Файл {pkl_path} не найден. Используются заглушки.")
            return
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        self.u_emb = data['u_emb']
        self.i_emb = data['i_emb']
        self.history = data.get('history', {})
        self.item_map = data.get('item_map', {})
        
        item_meta = data.get('item_meta', {})
        self.gamesPool = {}
        
        for idx, info in item_meta.items():
            asin = info.get('asin', 'Unknown')
            # Строго мапим поля во избежание сдвига параметров!
            self.gamesPool[asin] = VideoGame(
                gamerId=asin,
                title=info.get('title', 'Unknown Game'),
                brand=info.get('brand', 'Unknown Publisher'),
                price=float(info.get('price', 29.99)),
                imageUrl=info.get('imageUrl', 'https://via.placeholder.com/150')
            )
            
    def computeScores(self, db_history, top_k=6):
        if self.u_emb is None or self.i_emb is None:
            return []
            
        # Веса действий
        weights = {'click': 1.0, 'like': 3.0, 'cart': 3.0, 'buy': 5.0}
        user_vector = np.zeros(self.i_emb.shape[1])
        observed_indices = []
        
        # Инвертированный маппинг ASIN -> index
        asin_to_idx = {asin: idx for idx, asin in self.item_map.items()}
        
        for interaction in db_history:
            asin = interaction.gamerId
            if asin in asin_to_idx:
                idx = asin_to_idx[asin]
                weight = weights.get(interaction.actionType, 1.0)
                user_vector += weight * self.i_emb[idx]
                observed_indices.append(idx)
                
        if len(observed_indices) == 0:
            return []
            
        scores = np.dot(self.i_emb, user_vector)
        scores[observed_indices] = -1e9 # Маскируем уже купленное/просмотренное
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        recommendations = []
        for rank, idx in enumerate(top_indices):
            asin = self.item_map.get(idx)
            if asin:
                recommendations.append(Recommendation(
                    targetGamerId=asin,
                    score=float(scores[idx]),
                    rank=rank + 1
                ))
        return recommendations