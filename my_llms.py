from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

"""
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_ibm import ChatWatsonx
from databricks_langchain import ChatDatabricks
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere

from mistralai import Mistral
import openai
from together import Together 無料トライアルクレジットをすべて使い切りました
"""
#########################################################################
#########################################################################
#model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
#model = ChatOpenAI(
            #model="gpt-4o",
            #api_key= st.secrets.key.OPENAI_API_KEY,
            #max_completion_tokens=12800,  #指定しないと短い回答になったり、途切れたりする。
            #streaming=True,
        #)
# モデルの初期化とリストへの格納をまとめて行う関数
def initialize_models():
    
    models = {
        #モデルの選定条件：toolが使えるものが絶対条件、次に画像認識できるもの。費用がかからないのが望ましい
        #chatモデル(一連のメッセージを入力として使用し、メッセージを出力として返す言語モデル)でないとダメ 理由：忘れた
        #langchainでやるなら、それ対応のchat model
        #https://console.groq.com/docs/models
        #https://docs.mistral.ai/getting-started/models/models_overview/
        ######################################################################################################
        ######################################################################################################
        #2.2 ツール対応のマルチモーダルモデル　img,tool Good
        #(無料枠内で)無料：
        #多言語、マルチターンの会話、ツールの使用、JSON モードをサポートする、テキストと画像の両方の入力を処理できる強力なマルチモーダル モデルです。
        #groqつながり悪い "meta-llama/llama-4-scout-17b-16e-instruct": (init_chat_model("meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq"),0,0,0,128000), #TPM): Limit 30000, 1,000万トークン（プレビューでは128Kに制限）
        #groqつながり悪い "meta-llama/llama-4-maverick-17b-128e-instruct": (init_chat_model("meta-llama/llama-4-maverick-17b-128e-instruct", model_provider="groq"), 0,0,0,6000), #(TPM): Limit 6000, Requested 11999
        "pixtral-12b-2409": (init_chat_model("pixtral-12b-2409", model_provider="mistralai"),0,0,0,128000),
        "mistral-small-latest": (init_chat_model("mistral-small-latest", model_provider="mistralai"),0,0,0,131000), #Best 適応的思考、費用対効果
        #######################################################################################################
        #https://ai.google.dev/gemini-api/docs/models?hl=ja&_gl=1*17qcedu*_up*MQ..*_ga*ODUwNDc5MzM2LjE3NDUxNDczNjU.*_ga_P1DBVKWT6V*MTc0NTE0NzM2NS4xLjAuMTc0NTE0NzM2NS4wLjAuMTIzMzU0MzAwMA..#gemini-2.5-pro-preview-03-25
        # Gemini モデルの場合、1 個のトークンは約 4 文字に相当します。100 個のトークンは、約 60 ～ 80 ワード（英語）です。
        "gemini-2.5-pro-exp-03-25": (init_chat_model("google_vertexai:gemini-2.5-pro-exp-03-25", temperature=0),0,0,0,1048576), #高度なコーディング
        "gemini_2.5_flash": (init_chat_model("google_vertexai:gemini-2.5-flash-preview-04-17", temperature=0),0.15,0,3.5,1048576),
        #"gemini_2_flash": (init_chat_model("google_vertexai:gemini-2.0-flash", temperature=0),0,0,0,1000000), #リアルタイム ストリーミング、マルチモーダル生成
        #"gemini-1.5-pro": (init_chat_model("google_vertexai:gemini-1.5-pro", temperature=0),0,0,0,32000), #200 万トークン　無料 32000
        #"gemini-1.5-pro": (ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,max_retries=2),0,0,0,32000), #img,tool(1つだけ？) OK
        #有料:
        "gpt-4.1-mini": (init_chat_model("openai:gpt-4.1-mini") , 0.4,0.1,1.6,1047576), #コスパ良い
        "o4-mini": (init_chat_model("openai:o4-mini"),1.1,0.275,4.4,200000), #コスパ良い
        "gpt-4.1": (init_chat_model("openai:gpt-4.1"),2.0,0.5,8.0,1047576), #gpt-4oより安い
        #"gpt-4o": (init_chat_model("openai:gpt-4o"),2.5,1.25,10,128000), #3月に9.132$(1370円)
        #廃版 "gpt4r5": init_chat_model("openai:gpt-4.5-preview"), #3/15ごろに5.254$(788円)
        #高額すぎる "o1": (init_chat_model("openai:o1"), 15,7.5,60,200000), #3/15ごろに8.921$(1338円)
        #高額 "ChatGPT-4o": (init_chat_model("openai:ChatGPT-4o"), 5,0,15,128000),
        #高額 "o3": (init_chat_model("openai:o3"), 10,2.5,40,200000),
        #img NG "o1-mini": (init_chat_model("openai:o1"), 1.1,0.55,4.4,128000),
        #img NG "o3-mini": (init_chat_model("openai:o3-mini"),1.1,0.55,4.4,200000),

        #'Your credit balance is too low to access the Anthropic API.
        #高い "claude-3-7-sonnet": (init_chat_model("anthropic:claude-3-7-sonnet-latest", temperature=0),3,0,15,200000),
        #高い "claude-3-5-sonnet": (init_chat_model("anthropic:claude-3-5-sonnet-latest", temperature=0),3,0,15,200000),
        #"claude-3-5-haiku": (init_chat_model("anthropic:claude-3-5-haiku-latest", temperature=0),0.8,0,4,200000),
        #"claude-3-haiku": (init_chat_model("anthropic:claude-3-haiku-latest", temperature=0),0.8,0,4,200000),

        #日本語暴走 "neva_22b": (init_chat_model("nvidia/neva-22b", model_provider="nvidia"),0,0,0,158000), #NVIDIA版LLaVAモデル
        #日本語良くない "microsoft/phi-4-multimodal-instruct": (init_chat_model("microsoft/phi-4-multimodal-instruct", model_provider="nvidia"),0,0,0,128000)
        ###################################################################################
        #2.1 ツール非対応のマルチモーダルモデルとして動作 imgのみOK,
        #百名山標高順回答不正解、only 4 image(s) can be used per conversation
        "c4ai-aya-vision-32b":(init_chat_model("c4ai-aya-vision-32b", model_provider="cohere"),0,0,0,16000), #コンテキストの長さ: 16K、
        #Groqのならツール対応 "meta/llama-4-maverick-17b-128e-instruct":(init_chat_model("meta/llama-4-maverick-17b-128e-instruct", model_provider="nvidia"),0,0,0,1000000), #not known to support tools
        ##################################################################################
        ######################################################################################
        #"Test中"
        #NG "llama-3.2-11b-vision": init_chat_model("llama-3.2-11b-vision", model_provider="groq"), #does not exist
        #NG
        #Gemini API の「無料枠」は API サービスを通じて提供され、テスト目的でレート制限が緩和されます。
        #NG "gemma-3-27b-it":(GoogleGenerativeAI(model="models/gemma-3-27b-it"),0,0,0,128000), # not a valid GoogleModelFamily
        #NG img"gemma3_27b": (init_chat_model("google/gemma-3-27b-it", model_provider="nvidia"), 0,0,0,128000), #NG tool
        #cannot pass images with to tool
        #NG "gemma-3-27b-it": ChatGoogleGenerativeAI(model="google/gemma-3-27b-it",temperature=0,max_retries=2),#("google/gemma-3-27b-it", model_provider="nvidia"), 
        #requests.exceptions.HTTPError: 403 Client Error"CohereForAI/aya-vision-8b",AutoModelForImageTextToText.from_pretrained("CohereForAI/aya-vision-8b", device_map="auto", torch_dtype=torch.float16),
        #NG "CohereForAI/aya-vision-8b",pipeline(model="CohereForAI/aya-vision-8b", task="image-text-to-text", device_map="auto"),
        #NG unhashable type: 'ChatCohere' "CohereForAI/aya-vision-8b",ChatCohere(model="CohereForAI/aya-vision-8b"),
        #NG "CohereForAI/aya-vision-8b":ChatCohere(model="aya-vision-8b"), #CohereForAI/ model="aya-vision-8b""model 'aya-vision-8b' not found
        #'HuggingFaceEndpoint' object has no attribute 'bind_tools' "microsoft/Phi-3-mini-4k-instruct":HuggingFaceEndpoint(
            # repo_id="microsoft/Phi-3-mini-4k-instruct",
            # task="text-generation",
            # max_new_tokens=512,
            # do_sample=False,
            # repetition_penalty=1.03,
            # ),
        
        ##################################################################################
        #not exist or you do not have access to it
        #NG "llama-3.2-90b-vision-preview": (ChatGroq(model_name="llama-3.2-90b-vision-preview"),0,0,0,7000), #日本語理解下手、イマイチあまり回答しない。tokens per minute (TPM): Limit 7000
        #NG "SakanaAI/Llama-3-EvoVLM-JP-v2": (ChatGroq(model_name="SakanaAI/Llama-3-EvoVLM-JP-v2"),0,0,0,8192), #トークン小

        ###################################################################################
        #toolのみgood
        #"gpt4o_search": init_chat_model("openai:gpt-4o-search-preview"), #imgNG　'Request too large for gpt-4o-search-preview
        #"o3_mini": init_chat_model("openai:o3-mini"),　#imgNG
        #"llama-3.3-70b-versatile": ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0),
        #toolの応答でNG per minute (TPM): Limit 7000,
        #"llama-3.2-90b-vision-preview": init_chat_model("llama-3.2-90b-vision-preview", model_provider="groq"), 
            # imgデータをツールに飛ばして回答、Error code: 400 - {'error': {'message': 'Too many images provided.  This model supports up to 1 images', 'type': 'invalid_request_error'}}
        #TPM NG 
        #"llama-3.3-70b-versatile": init_chat_model("llama-3.3-70b-versatile", model_provider="groq"), #NG tokens per minute (TPM): Limit 6000, Requested 12750
        #"llama3-8b": init_chat_model("llama3-8b-8192", model_provider="groq"),
        #"command-r-plus":init_chat_model("command-r-plus", model_provider="cohere"), #img NG Tool G
        #"command-a-03-2025":init_chat_model("command-a-03-2025", model_provider="cohere"), #無料枠はクエリ制限？The read operation timed out
            #111B パラメータでは、コンテキストの長さは 256K です。image content is not supported for this model
        #"llama-v3p1-70b": init_chat_model("accounts/fireworks/models/llama-v3p1-70b-instruct", model_provider="fireworks"), #tool 1つだけOK
        ####################################################################################
        #tool NG
        #NG img,toolは動作、日本語でたらめ"meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": init_chat_model("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", model_provider="together"), 
        #NG "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo":  init_chat_model("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", model_provider="together"),
        #NG "mistral-small-latest": ChatMistralAI(name="mistral-small-latest"),
        #NG "gemma2-9b-it": ChatGroq(model_name="gemma2-9b-it",temperature=0.7), Failed to call a function.
        #NG "gemma2-9b-it":init_chat_model("gemma2-9b-it", model_provider="groq"),
        #NG "command_a": init_chat_model("command-a-03-2025", model_provider="cohere"),#imgNG
        #NG "gemini_1r5_pro": init_chat_model("google_vertexai:gemini-1.5-pro-latest"), #, model_provider="google_vertexai") 削除
        #NG "deepseek_r1": init_chat_model("deepseek-ai/deepseek-r1", model_provider="nvidia"),
        #NG "kosmos-2": init_chat_model("microsoft/kosmos-2", model_provider="nvidia"),

        #NG img,toolは動作、日本語でたらめ 
        #NGNG "phi_4_multi": init_chat_model("microsoft/phi-4-multimodal-instruct", model_provider="nvidia"), #NG
        #NG unknown "phi_3_vision": init_chat_model("microsoft/Phi-3-vision-128k-instruct", model_provider="nvidia"),
        #NG "gemma3_27b": (init_chat_model("google/gemma-3-27b-it", model_provider="nvidia"), 0,0,0,6000), #NG tool not known to support tools
        #NG "gemma3_1b": init_chat_model("google/gemma-3-1b-it", model_provider="nvidia"),
        #NG
        #NG "adept/fuyu-8b": init_chat_model("adept/fuyu-8b", model_provider="nvidia"),#List should have at most 1 item after validation
        #余分な入力NG"google/gemma-3-27b-it": init_chat_model("google/gemma-3-27b-it", model_provider="nvidia"),
        #余分な入力NG "microsoft/kosmos-2":  init_chat_model("microsoft/kosmos-2", model_provider="nvidia"), #not known to support tools First message role should be 'user'
        #科学的で複雑な数学的推論、コーディング、ツールの呼び出し、および指示の追跡において最高の精度を備えた優れた推論効率。
        #img非対応"nvidia/llama-3.1-nemotron-ultra-253b-v1":  (init_chat_model("nvidia/llama-3.1-nemotron-ultra-253b-v1", model_provider="nvidia"),0,0,0,128000)
        #img "llama3-70b": init_chat_model("meta/llama3-70b-instruct", model_provider="nvidia"),
        #NG "mistral-large": init_chat_model("mistral-large-latest", model_provider="mistralai"), 有料、クレジット登録していない
        #NG "granite-34b-code": ChatWatsonx(model_id="ibm/granite-34b-code-instruct",url="https://us-south.ml.cloud.ibm.com",project_id="<WATSONX PROJECT_ID>"), クレジット登録していない
        #NG "databricks-meta-llama-3-1-70b": ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct"), クレジット登録していない
        #権限なし"grok-2": init_chat_model("grok-2", model_provider="xai") #権限なし,クレジット登録していないので
        #日本語理解NG "llama-3.2-11b-vision-preview": init_chat_model("llama-3.2-11b-vision-preview", model_provider="groq"), #日本語理解NG tokens per minute (TPM): Limit 7000, Requested 27085
        #No module named 'vllm._C'"tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4": LLM(model="tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4",tensor_parallel_size=1,)
        #無料トライアルクレジットをすべて使い切りました"Credit limit exceeded "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": init_chat_model("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", model_provider="together"),
        #無料トライアルクレジットをすべて使い切りました"Qwen/Qwen2.5-72B-Instruct-Turbo": init_chat_model("Qwen/Qwen2.5-72B-Instruct-Turbo", model_provider="together"),
        #無料トライアルクレジットをすべて使い切りましたCredit limit exceeded."meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": init_chat_model("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together"),
        #無料トライアルクレジットをすべて使い切りましたNG "meta-llama/Llama-Vision-Free": init_chat_model("meta-llama/Llama-Vision-Free", model_provider="together"),
        #無料トライアルクレジットをすべて使い切りました"Mixtral-8x7B": init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together"),
        #NG "aya-vision-32b":init_chat_model("CohereLabs/aya-vision-32b", model_provider="cohere"), #CohereLabs/
        #"trocr-large-stage1": init_chat_model("microsoft/trocr-large-stage1", model_provider="nvidia"),
        #"moonshotai/Kimi-VL-A3B-Thinking":ChatHuggingFace(llm=HuggingFaceEndpoint(
                                                                #repo_id="moonshotai/Kimi-VL-A3B-Thinking",
                                                                #task="text-generation",
                                                                #max_new_tokens=1200000,
                                                                #do_sample=False,
                                                                #repetition_penalty=1.03, )
                        #),
        #NG "meta-llama/Llama-4-Maverick-17B-128E-Instruct":ChatHuggingFace(
            #llm=HuggingFaceEndpoint(
                #repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                #task="text-generation",
                #max_new_tokens=1200000,
                #do_sample=False,
                #repetition_penalty=1.03,
                #)
            #),
            
            #Error in chatbot_node: (Request ID: Root=1-680190f0-3aabdb6828682d0828db0544;53bf41e6-c0b1-4bda-9e8b-282dccb7f303)
            #403 Forbidden: None.
            #Cannot access content at: https://router.huggingface.co/hf-inference/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct/v1/chat/completions.
            #Make sure your token has the correct permissions.
            #The model meta-llama/Llama-4-Maverick-17B-128E-Instruct is too large to be loaded automatically (803GB > 10GB).
        #"meta-llama/Llama-4-Maverick-17B-128E-Instruct":HuggingFaceEndpoint(
            #repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            #task="text-generation",
            #max_new_tokens=1200000,
            #do_sample=False,
            #repetition_penalty=1.03,
            #),
            # repo_id="OpenGVLab/InternVL3-78B" too large to be loaded automatically (156GB > 10GB)
        #repo_id = "mistralai/Mistral-7B-Instruct-v0.2", #too large to be loaded automatically (14GB > 10GB)
            #repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct", #too large to be loaded automatically (803GB > 10GB)
            #repo_id="google/gemma-3-27b-it",#too large to be loaded automatically (54GB > 10GB).
            #repo_id="moonshotai/Kimi-VL-A3B-Thinking",
            #repo_id="HiDream-ai/HiDream-I1-Full"    
        #"microsoft/Phi-3-mini-4k-instruct":ChatHuggingFace(
            #llm=HuggingFaceEndpoint(
                #repo_id="microsoft/Phi-3-mini-4k-instruct", #OK #コンテキストの長さ: 4Kトークン
                #task="text-generation",
                #max_new_tokens=512, #コンテキストの長さ: 128Kトークン
                #do_sample=False,
                #repetition_penalty=1.03,
                #trust_remote_code=True,
            #),
            #verbose=True,
            #trust_remote_code=True,
            #),
        #"black-forest-labs/FLUX.1-dev":ChatHuggingFace(
            #llm=HuggingFaceEndpoint(
                #repo_id="black-forest-labs/FLUX.1-dev", #OK #コンテキストの長さ: 4Kトークン
                #task="text-generation",
                #max_new_tokens=512, #コンテキストの長さ: 4Kトークン
                #do_sample=False,
                #repetition_penalty=1.03,
                #trust_remote_code=True,
            #),
            #verbose=True,
            #trust_remote_code=True,
            #),
        
    }
    return models
"""
以下のモデルは、Image_to_Text、Fanction Callingに対応したAPIで利用できる無料の最近のLLMです。

1. **Google Gemini Nano**
6. **Mistral Small 2**
7. **Mistral Nano 2**
8. **Mistral Code 2**
9. **Mistral Medium 2 Instruct**
6. **Mistral Small 2**
7. **Mistral Nano 2**
8. **Mistral Code 2**
9. **Mistral Medium 2 Instruct**
10. **Mistral Nano 2 Instruct**

1. **Llama 3.2 Vision** - Ollamaで利用可能で、Image-to-Textに対応しています。

2. **LLaVA 13B** - 画像と言語のインタラクションをサポートするモデル。

3. **FireLLaVA-13B** - Hugging Faceで提供されている、マルチイメージ入力に対応するビジョン言語モデル。

4. **ShareGPT4V-7B** - ビジョンとテキストの多機能モーダルモデル。

5. **ShareGPT4V-13B** - ビジョンとテキストの多機能モーダルモデルで、研究用途に適しています。

6. **DeepSeek-VL-7B** - 実世界の視覚と言語の理解に向けたオープンソースモデル。

7. **DreamLLM** - マルチモーダルの理解と生成を可能にするLLM。

8. **MiniGPT-4 Vicuna-13B** - Image Captioningや画像に関する質問応答に対応。
"""