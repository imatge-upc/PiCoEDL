from logging import getLogger
import os

import tqdm
import numpy as np
from sklearn.cluster import KMeans
import joblib
import minerl

from IPython import embed
logger = getLogger(__name__)


class _KMeansCacheNotFound(FileNotFoundError):
    pass


def cached_kmeans(cache_dir, env_id, n_clusters, random_state):
    if cache_dir is None:  # ignore cache
        logger.info('Load dataset & do kmeans')
        kmeans = _do_kmeans(env_id=env_id, n_clusters=n_clusters, random_state=random_state)
    else:
        filepath = os.path.join(cache_dir, env_id, f'n_clusters_{n_clusters}', f'random_state_{random_state}', 'kmeans.joblib')
        try:
            kmeans = _load_kmeans_result_cache(filepath)
            logger.info('found kmeans cache')
        except _KMeansCacheNotFound:
            logger.info('kmeans cache not found. Load dataset & do kmeans & save result as cache')
            kmeans = _do_kmeans(env_id=env_id, n_clusters=n_clusters, random_state=random_state)
            _save_kmeans_result_cache(kmeans, filepath)
    return kmeans


def _do_kmeans(env_id, n_clusters, random_state):
    logger.debug(f'loading data...')
    
    MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/home/usuaris/imatge/juan.jose.nieto/mineRL/data/')
    dat = minerl.data.make(env_id, data_dir=MINERL_DATA_ROOT)
    act_vectors = []
    for _, act, _, _, _ in tqdm.tqdm(dat.batch_iter(batch_size=16, seq_len=32, num_epochs=1, preload_buffer_size=32, seed=random_state)):
        act_vectors.append(act['vector'])
    acts = np.concatenate(act_vectors).reshape(-1, 64)
    logger.debug(f'loading data... done.')
    logger.debug(f'executing keamns...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(acts)
    logger.debug(f'executing keamns... done.')
    return kmeans


# def _describe_kmeans_result(kmeans):
#     result = [(obf_a, minerl.herobraine.envs.MINERL_TREECHOP_OBF_V0.unwrap_action({'vector': obf_a})) for obf_a in kmeans.cluster_centers_]
#     logger.debug(result)
#     return result


def _save_kmeans_result_cache(kmeans, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(kmeans, filepath)
    logger.info(f'saved kmeans {filepath}')


def _load_kmeans_result_cache(filepath):
    if not os.path.exists(filepath):
        raise _KMeansCacheNotFound
    logger.debug(f'loading kmeans {filepath}')
    return joblib.load(filepath)
