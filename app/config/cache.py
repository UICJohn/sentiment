import redis
# cachy = redis.Redis(
# 	host = '192.168.0.6',
# 	port = 6379,
# 	password = '',
# )
cachy = redis.Redis(
  host = 'localhost',
  port = 6379,
  password = '',
)
# cache_config = {
#   'stores': {
#     'redis':{ 
#       'driver': 'redis',
#       'host':'localhost',
#       'port': 6379,
#       'db': 0
#     }
#   }
# }