# AIM-JetsonAGXOrin
AIM-JetsonAGXOrin



### Run against Local Docker Compose 

````
SLACK_BOT_TOKEN="xslack$$$" python3 orinsearchdisplay.py

# In Python

MILVUS_URL = "http://192.168.1.153:19530"
milvus_client = MilvusClient( uri=MILVUS_URL)

````

### Run against Zilliz Cloud for Milvus

````
ZILLIZ_TOKEN="token$$$" SLACK_BOT_TOKEN="xslack$$$" python3 orincloud.py

# In Python

MILVUS_URL = "https://in05-7bd87b945683c8d.serverless.gcp-us-west1.cloud.zilliz.com"
TOKEN = os.environ["ZILLIZ_TOKEN"]
milvus_client = MilvusClient( uri=MILVUS_URL, token=TOKEN )

````

### Run against Local Milvus-Lite

````
SLACK_BOT_TOKEN="xslack$$$" python3 orin.py

# In Python

DATABASE_NAME = "./OrinEdgeAI.db"
milvus_client = MilvusClient(DATABASE_NAME)

````

### Model Used

Salesforce/blip-image-captioning-large



