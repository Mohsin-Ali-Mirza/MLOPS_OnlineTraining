import redis.asyncio as aioredis
import redis as sync_redis
from confluent_kafka import Producer
from config import KAFKA_BROKER

redis_client_async = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
redis_client_sync = sync_redis.Redis(host="localhost", port=6379, decode_responses=True)

kafka_producer = Producer({"bootstrap.servers": KAFKA_BROKER})