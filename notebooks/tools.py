from typing import Callable, Dict, Set, List, Optional

import numpy as np
from lightfm import LightFM
from scipy.sparse import csr_matrix


def generate_lightfm_recs_mapper(
    model: LightFM, 
    N: int, 
    item_iids: List[int],  # iid - internal lfm id
    user_id_to_iid: Dict[int, int], 
    item_iid_to_id: Dict[int, int], 
    known_item_ids: Dict[int, Set[int]],
    user_features:  Optional[csr_matrix] = None, 
    item_features: Optional[csr_matrix] = None, 
    num_threads: int = 1,
) -> Callable:
    """Возвращает функцию для генерации рекомендаций в формате item_ids, scores"""
    def _recs_mapper(user):
        # Предикт для одного юзера
        user_id = user_id_to_iid[user]
        # Получаем список скоров. index - соответствует внутренним 
        # индексам lightfm для айтемов т.е. ключам из item_iid_to_id
        scores_vector = model.predict(user_id, item_iids, user_features=user_features,
                             item_features=item_features, num_threads=num_threads)
        # Оставляем запас для исключения уже просмотренного из рекомендаций
        additional_N = len(known_item_ids[user_id]) if user_id in known_item_ids else 0
        total_N = N + additional_N
        # Получаем список индексов топ-N айтемов
        top_iids = np.argpartition(scores_vector, -np.arange(total_N))[-total_N:][::-1]
        # Исключаем уже просмотренное из рекомендаций
        if additional_N > 0:
            filter_items = known_item_ids[user_id]
            top_iids = [item_index for item_index in top_iids if item_iid_to_id[item_index] not in filter_items]
        # Переводим индексы lightfm айтемов в их реальные id
        final_recs = [item_iid_to_id[item_index] for item_index in top_iids]
        # Сохраняем скоры
        final_scores = scores_vector[top_iids]
        return final_recs, final_scores
    return _recs_mapper


def avg_user_metric(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        user_ids: np.ndarray,
        metric_function: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """
    Вычисляем метрику, усредненную по всем значимым (есть разные таргеты) группам.

    :param y_true: список таргетов
    :param y_pred: список предсказаний
    :param user_ids: список групп (обычно это список user_id той же размерности, что и предсказания и таргеты)
    :param metric_function: усредняемая метрика(y_true, y_pred) -> float
    :return: значение метрики metric_function, усредненное по всем значимым группам
    """
    avg_score: float = 0.

    if len(y_pred) == len(y_true) == len(user_ids):
        l_ind: int = 0
        cur_group_id: int = user_ids[0] if len(user_ids) else 0
        n_groups: int = 0
        for r_ind, group_id in enumerate(user_ids):
            if group_id != cur_group_id or r_ind == len(user_ids) - 1:
                if r_ind == len(user_ids) - 1:
                    r_ind += 1
                # Если группа не состоит из одного и того же таргета - добавляем ее
                group_true = y_true[l_ind: r_ind]
                if not np.all(group_true == group_true[0]):
                    avg_score += metric_function(group_true, y_pred[l_ind: r_ind])
                    n_groups += 1
                l_ind = r_ind
                cur_group_id = group_id
        avg_score /= max(1, n_groups)
    else:
        raise ValueError(f'Размерности не совпадают: '
                         f'y_pred - {len(y_pred)}, y_true - {len(y_true)}, user_ids - {len(user_ids)}')
    return avg_score
