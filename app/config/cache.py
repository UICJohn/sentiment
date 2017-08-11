from cachy import CacheManager
cache_config = {
	'stores': {
		'redis':{ 
			'driver': 'redis',
			'host':'192.168.0.6',
			'port': 6379,
			'db': 0
		}
	}
}
redis = CacheManager(cache_config)