# streamlit_watcher_ignore_module: torch, torchaudio, torchvision
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
##########################################################
from langchain_community.tools.tavily_search import TavilySearchResults
from newsapi import NewsApiClient
from exa_py import Exa
from langchain.agents import tool,Tool
from langchain.tools import Tool
#from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field
#from metaphor_python import Metaphor
#from langchain_community.utilities import SerpAPIWrapper
#from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
##from langchain_community.tools import WikipediaQueryRun
#from langchain_community.utilities import WikipediaAPIWrapper
##########################################################
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import StrOutputParser
##########################################################
#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_cohere.chat_models import ChatCohere
#from langchain_nvidia_ai_endpoints import ChatNVIDIA
#from langchain_ollama import ChatOllama
#from langchain_ibm import ChatWatsonx
#from databricks_langchain import ChatDatabricks
#from langchain_groq import ChatGroq
#from langchain_mistralai import ChatMistralAI
#from mistralai import Mistral
############################################################
import requests
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List,Tuple
import asyncio
from collections import deque
import cv2
import av
import base64
from gtts import gTTS
import whisper
import pydub
from pydub import AudioSegment
from pydub.effects import low_pass_filter  #,high_pass_filter
#from scipy.signal import resample
from io import BytesIO
import psutil
import gc
import re
import librosa
import subprocess
import vertexai
import os
#import getpass
#import json
import tiktoken
import uuid
import torch
torch.classes.__path__ = []
import torchaudio
import torchvision
#####################################################
#リソース制限
#Community Cloud のすべてのユーザーは同じリソースにアクセスでき、同じ制限が適用されます。
# アプリがこれらの制限に達した場合、または制限を超えた場合、スロットリングにより速度が低下したり、
# 機能しなくなったりする可能性があります。
# 2024年2月時点での制限は、おおよそ以下のとおりです。# これらの制限は予告なく変更される場合があります。
# #CPU: 最小 0.078 コア、最大 2 コア
#メモリ: 最小690MB、最大2.7GB
#ストレージ: 最小容量なし、最大50GB
#原因
#Streamlitのファイル監視機能がtorchの__path__._pathにアクセスしようとして失敗し、RuntimeErrorが発生しています。
#さらに、asyncio.get_running_loop()の呼び出し時にイベントループが存在しないため、
# RuntimeError: no running event loopが発生しています。
#対処方法
# 1. アプリ自体は動作する場合が多い
# このエラーは多くの場合、コンソールに警告やエラーが表示されるだけで、Streamlitアプリ自体は動作し続けることが多いです。
# もしアプリが正常に動作している場合は、無視しても大きな問題にはなりません<sup>参考</sup>。
# 2. ワークアラウンド
#torchのimport直後に、以下のようにtorch.classes.__path__ = []を設定することで、
# ファイル監視機能によるエラーを回避できる場合があります。


# 環境変数の設定をまとめて行う関数
def setup_environment():
    try:
        # streamlit cloudの環境変数にAPIキーなどを設定 (必要であれば)
        os.environ["LANGCHAIN_API_KEY"] = st.secrets.key["LANGCHAIN_API_KEY"]
        os.environ["LANGSMITH_API_KEY"] = st.secrets.key["LANGSMITH_API_KEY"]
        os.environ["USER_AGENT"] = st.secrets.key["USER_AGENT"]
        os.environ["OPENAI_API_KEY"] = st.secrets.key["OPENAI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = st.secrets.key["GOOGLE_API_KEY"]
        os.environ["ANTHROPIC_API_KEY"] = st.secrets.key["ANTHROPIC_API_KEY"]
        os.environ["COHERE_API_KEY"] = st.secrets.key["COHERE_API_KEY"]
        os.environ["NVIDIA_API_KEY"] = st.secrets.key["NVIDIA_API_KEY"]
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets.key["HUGGINGFACEHUB_API_TOKEN"]
        os.environ["GROQ_API_KEY"] = st.secrets.key["GROQ_API_KEY"]
        os.environ["FIREWORKS_API_KEY"] = st.secrets.key["FIREWORKS_API_KEY"]
        os.environ["MISTRAL_API_KEY"] = st.secrets.key["MISTRAL_API_KEY"]
        os.environ["TOGETHER_API_KEY"] = st.secrets.key["TOGETHER_API_KEY"]
        os.environ["OPENWEATHERMAP_API_KEY"] = st.secrets.key["OPENWEATHERMAP_API_KEY"]
        os.environ["SEARCHAPI_API_KEY"] = st.secrets.key["SEARCHAPI_API_KEY"]
        os.environ["SERPAPI_API_KEY"] = st.secrets.key["SERPAPI_API_KEY"]
        os.environ["EXA_API_KEY"] = st.secrets.key["EXA_API_KEY"]
        os.environ["TAVILY_API_KEY"] = st.secrets.key["TAVILY_API_KEY"]
        os.environ["NEWSAPI_API_KEY"] = st.secrets.key["NEWSAPI_API_KEY"]
        os.environ["METAPHOR_API_KEY"] = st.secrets.key["METAPHOR_API_KEY"]
        os.environ["WOLFRAM_ALPHA_APPID"] = st.secrets.key["WOLFRAM_ALPHA_APPID"]
        os.environ["GOOGLE_CLOUD_PROJECT"] = st.secrets.key["GOOGLE_CLOUD_PROJECT"]
        # プロジェクト ID とロケーションを設定
        PROJECT_ID = st.secrets.key["GOOGLE_CLOUD_PROJECT"]
        LOCATION = "us-central1"
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    except:
        print("streamlit cloudではなくローカルPCでの起動")
        pass

# 日付と時刻を取得する関数
def get_current_datetime():
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H時%M分")

# 関数でメモリ使用量を取得
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    # メモリ使用量をMB単位で返す
    return mem_info.rss / (1024 * 1024)

def current_memory_use(i,memory_use,memory_alt,memory_ok):
    # 現在のメモリ使用量を取得
    current_memory_usage = get_memory_usage()
    # メモリ使用量を表示

    #memory_use.metric("現在のメモリ使用量 (MB)", f"{current_memory_usage:.2f}")
    #memory_use.write(f"現在のメモリ使用量:\n\n    ループ{i}回目:{current_memory_usage:.0f}MB")
    memory_use.write(f"現在のメモリ使用量:\n {current_memory_usage:.0f}MB")
    #print("現在のメモリ使用量 (MB)", f"{current_memory_usage:.2f}")
    # メモリ制約を定義
    MEMORY_LIMIT_MB = 2700  # 1GB
    # メモリ使用量が制約を超えた場合の警告
    if current_memory_usage > MEMORY_LIMIT_MB:
        memory_alt.error(f"メモリ使用量が制約 ({MEMORY_LIMIT_MB} MB) を超えました。処理を中断してください。")
        #memory_ok.empty()
        #st.stop()
        print(f"メモリ使用量が制約 ({MEMORY_LIMIT_MB} MB) を超えました。処理を中断してください。")
    else:
        #memory_ok.success("メモリ使用量は正常範囲内です。")
        memory_alt.empty()
        #print("メモリ使用量は正常範囲内です。")

############################################################################
# モデル定義: モデル名と初期化に必要な情報をマッピング
# initialize_models 関数の外に移動して main 関数からも参照可能にする
model_definitions = {
    "mistral-small-latest": {"provider": "mistralai", "temperature": 0, "cost_input": 0, "cost_cached_input": 0, "cost_output": 0, "input_max": 128000},
    "gemini-2.5-pro-exp-03-25": {"provider": "google_vertexai", "temperature": 0, "cost_input": 0, "cost_cached_input": 0, "cost_output": 0, "input_max": 1048576},
    "gemini_2.5_flash": {"provider": "google_vertexai", "model_id": "gemini-2.5-flash-preview-04-17", "temperature": 0, "cost_input": 0.15, "cost_cached_input": 0, "cost_output": 3.5, "input_max": 1048576},
    "gpt-4.1-mini": {"provider": "openai", "temperature": 0, "cost_input": 0.4, "cost_cached_input": 0.1, "cost_output": 1.6, "input_max": 1047576},
    "o4-mini": {"provider": "openai", "temperature": 0, "cost_input": 1.1, "cost_cached_input": 0.275, "cost_output": 4.4, "input_max": 200000},
    "gpt-4.1": {"provider": "openai", "temperature": 0, "cost_input": 2.0, "cost_cached_input": 0.5, "cost_output": 8.0, "input_max": 1047576},
    # "gpt-4o": {"provider": "openai", "temperature": 0, "cost_input": 2.5, "cost_cached_input": 1.25, "cost_output": 10, "input_max": 128000},
    # ... (他のコメントアウトされたモデル定義) ...
    "c4ai-aya-vision-32b": {"provider": "cohere", "temperature": 0, "cost_input": 0, "cost_cached_input": 0, "cost_output": 0, "input_max": 16000}, # ツール非対応
    # "meta/llama-4-maverick-17b-128e-instruct": {"provider": "nvidia", "temperature": 0, "cost_input": 0, "cost_cached_input": 0, "cost_output": 0, "input_max": 1000000}, # ツール非対応 (NVIDIA版)
}

def initialize_models(selected_model_name: str):
    """選択されたモデル名に基づいて、そのモデルのみを初期化する"""
    print(f"Initializing model: {selected_model_name}") # どのモデルが初期化されるかログ出力

    if selected_model_name not in model_definitions:
        raise ValueError(f"未定義のモデル名です: {selected_model_name}")

    model_info = model_definitions[selected_model_name]

    # init_chat_model に渡す引数を準備
    init_args = {
        "model": f"{model_info['provider']}:{model_info.get('model_id', selected_model_name)}", # provider:model_id 形式
        "temperature": model_info.get("temperature", 0), # temperature が未定義なら 0 を使う
        # 必要に応じて他のパラメータ (max_tokens など) も追加
    }
    # model_provider 引数が必要な場合 (init_chat_model の仕様による)
    # if model_info['provider'] in ["groq", "mistralai", "nvidia", "cohere"]: # 例
    #     init_args["model_provider"] = model_info['provider']
    #     init_args["model"] = model_info.get('model_id', selected_model_name) # model_id があればそれを使う

    # モデルを初期化
    try:
        llm_instance = init_chat_model(**init_args)
        print(f"Model {selected_model_name} initialized successfully.")
    except Exception as e:
        print(f"Error initializing model {selected_model_name}: {e}")
        st.error(f"モデル '{selected_model_name}' の初期化に失敗しました: {e}")
        return None, 0, 0, 0, 0 # エラー時は None とデフォルト値を返す

    # モデルインスタンスと関連情報を返す
    return (
        llm_instance,
        model_info.get("cost_input", 0),
        model_info.get("cost_cached_input", 0),
        model_info.get("cost_output", 0),
        model_info.get("input_max", 32000) # デフォルトの最大トークン数
    )


# モデルロード関数 (キャッシュ付き)
@st.cache_resource(show_spinner=False) # モデル選択ごとに実行されるため spinner は非表示に
def load_selected_model(model_name: str):
    """選択されたモデル名に基づいてモデルをロードする"""
    return initialize_models(model_name)

############################################################################
def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        # Claude 3 はトークナイザーを公開していないので、tiktoken を使ってトークン数をカウント
        # これは正確なトークン数ではないが、大体のトークン数をカウントすることができる
        phrases = (
                    #"gpt", 
                    "o4", #"o3", 'Could not automatically map o3-mini to a tokeniser
                    #"o1", #'Could not automatically map o1 to a tokeniser. 
                    #Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect.'
                    )
        if any(phrase in st.session_state.model_name for phrase in phrases):
        #if "gpt" in st.session_state.model_name:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        else:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 仮のものを利用
        return len(encoding.encode(text))

#メッセージのカウント
def count_tokens(messages, model_name="gpt-3.5-turbo"):
    #モデルによって適切なencoding_modelを使用
    encoding = tiktoken.encoding_for_model(model_name)
    # messages が文字列であれば直接エンコードする 
    if isinstance(messages, str):
        n = len(encoding.encode(messages)) 
        print("messages が文字列の場合のトークン数len(encoding.encode(messages))=",n)  
        return n

    # messages がリストなどの場合は個別にエンコードする（必要に応じて処理） 
    total = 0 
    #print("messages=\n",messages) #画像データだと大量に出力される！！
    #messages=
    #system: あなたはツールを効果的に活用する有能なアシスタントです。
            #現在の日時は 2025年03月30日 16時50分 です。
            #提供されたツールを使用して、ユーザーが探している最新の情報を提供してください。
            #必要な情報が得られない場合は、他のツールを使用してください。
            #特に、天気、ニュースを聞かれた場合は、必ずツールを使ってください。
            #また、常に日本語で回答してください。
    #user: 私は、石川県小松市に住んでいます。ツールを利用して最新情報から具体的なタイトルを取得して日本語で答えてください。
    #ai: 最新の情報として、「3月13日（木）石川県小松市『アミュージアム小松店』が移転オープン！」というニュースがあります。詳しくは、[こちら](https://prtimes.jp/main/htm

    for m in messages:
        #print("m=\n",m)
        #m= s
        #total += len(encoding.encode(m["content"]))
        #エラーメッセージ「string indices must be integers, not 'str'」は、
        # count_tokens() 関数内で各メッセージとして想定している m が、
        # 実際は辞書ではなく文字列であるため発生しています。
        # m が辞書の場合は "content" キーを参照、文字列の場合はそのまま扱う
        if isinstance(m, dict):
            content = m.get("content", "")
        elif isinstance(m, str):
            content = m
        else:
            content = str(m)
        #print("content=\n",content)  
        n = len(encoding.encode(content)) 
        total += n
        print("messages がリストなどの場合のトークン数=",total) 
        return total

#メッセージ数が閾値を超えたときに、過去の会話履歴を要約・削減する関数
def trim_messages(messages, max_tokens_threshold=120000):
    #messages が文字列の場合に適切なトリム処理を実施
    # ここでは例として、古いメッセージを単純に削除し、最新の50行、50件のみ残す方法をとる 
    # ※もしくは、要約LLMを呼び出してまとめ直す方法もあります  
    token_count = 0
    token_count = count_tokens(messages)
    # トークン数が閾値内ならそのまま返す  
    if token_count < max_tokens_threshold: 
        print("token_count=",token_count )
        return messages
    # ここでは例として、文字列の場合は先頭部分を削除して最新部分のみ残す方法を採用 
    if isinstance(messages, str): 
        # 複数行に分割して最新の50行だけを採用する例
        lines = messages.splitlines()
        trimmed = "\n".join(lines[-50:])
        return trimmed
    # リストの場合は最新の50項目だけを残す（例） 
    #print("システムメッセージは残っているか？messages[-50:]=\n",messages[-50:])
    return messages[-50:]
    
    # return messages[-50:]
    #else:
    #print("メッセージ数が閾値を超えたので、過去の会話履歴を要約・削減しました。")

def trim_message_history(message_history, max_tokens=8192):  
    """  
    メッセージ履歴をトークン数で制限  
    GPT-4
    GPT-4: 8,192トークン
    GPT-4 Turbo: 128,000トークン
    GPT-4o: 128,000トークン
    Claude
    Claude 3 Haiku: 約200,000トークン
    Claude 3 Sonnet: 約200,000トークン
    Claude 3 Opus: 約200,000トークン
    Claude 2: 100,000トークン
    Gemini
    Gemini Pro: 32,000トークン
    Gemini Ultra: 最大1,000,000トークン
    Llama 2/3
    Llama 2 (7B-70B): 4,096トークン
    Llama 3 (8B): 8,192トークン
    Llama 3 (70B): 8,192トークン
    日本語モデル
    Rinna: 2,048トークン
    ELYZA: 4,096トークン
    Nekomata: 4,096トークン
    その他
    Command R+: 128,000トークン
    Mistral 7B: 8,192トークン
    Cohere: 4,096トークン
    推奨される一般的な戦略:

    安全サイズ: 4,000-8,000トークン
    トリミング関数の実装
    モデル固有の制限を確認

    """  
    total_tokens = 0  
    trimmed_history = []  
    
    # 最新のメッセージから逆順に追加  
    for message in reversed(message_history):  
        message_tokens = len(message[1])  # メッセージ長さを計算  
        if total_tokens + message_tokens <= max_tokens:  
            trimmed_history.insert(0, message)  
            total_tokens += message_tokens  
        else:  
            break  
    
    return trimmed_history  

#  LLM問答関数
async def query_llm(user_input,img_url,memory):
    with st.session_state.answer_container:
        with st.chat_message("ai"):
            use_tool_placeholder = st.empty()
            response_placeholder = st.empty()
    if 'use_tool_name' not in st.session_state:
        st.session_state.use_tool_name =""
    tools,openai_tools = setup_tools()  #openai_tools #setup_tools()
    tool_used = False
    tool_count = 0
    config = {"configurable": {"thread_id": st.session_state.history_id}} #abc123
    #langchain対応LLM
    print("model_name=",st.session_state.model_name)
    if "pixtral" in st.session_state.model_name:
        tools =openai_tools
    agent_executor = create_react_agent(st.session_state.llm, tools, checkpointer=memory)

    # もしまだ session_state に message_history がない場合、空のリストを初期化
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []
    # 会話履歴を LLM に送信するために準備
    # 会話履歴のメッセージをテキストとして連結
    conversation_history = ""
    for role, content in st.session_state.message_history:
        conversation_history += f"{role}: {content}\n"
    ###############################################################
    #if  "画像" in user_input and st.session_state.input_img == "有": #and st.session_state.model_name == "gpt-4o":
    if "画像" in user_input or "カメラ" in user_input or "画面" in user_input or "写真" in user_input:
        user_input = user_input + "日本語で答えてください。"
        # LLMへの問い合わせに会話履歴を含める
        #llm_input = conversation_history.strip() # 不要な末尾の改行を削除
        llm_input = conversation_history.strip() + f"user: {user_input}"
        max_tokens_threshold = st.session_state.input_max * 0.7
        llm_input = trim_messages(llm_input, max_tokens_threshold)
        #encoded_image = cv2.imencode('.jpg', frame)[1]
        # 画像をBase64に変換
        #base64_image = base64.b64encode(encoded_image).decode('utf-8')
        #img_url = f"data:image/jpeg;base64,{base64_image}"
        message = HumanMessage(
                            content=[
                                {"type": "text", "text": llm_input},  #user_input
                                {
                                    "type": "image_url",
                                    "image_url": {"url": img_url}, #f"data:image/jpeg;base64,{base64_image}"
                                },
                            ],
                        )
    else:
        #user_input = user_input + "ツールを利用して最新情報から具体的なタイトルを取得して日本語で答えてください。"
        # LLMへの問い合わせに会話履歴を含める
        #llm_input = conversation_history.strip() # 不要な末尾の改行を削除
        llm_input = conversation_history.strip() + f"user: {user_input}"
        max_tokens_threshold = st.session_state.input_max * 0.7
        llm_input = trim_messages(llm_input, max_tokens_threshold)
        message = llm_input #HumanMessage(content=user_input)
    #########################################################################
    # 例：memoryから既存の会話履歴を取得し、ユーザ入力やキャプションを追加する 
    # messages = memory.get_messages() 
    # 既存の会話履歴（リスト形式：各要素は {"role":..., "content":...}） 
    # 必要に応じてユーザからの入力やキャプションを追加 
    # messages.append({"role": "user", "content": cleaned_text}) 
    # messages.append({"role": "system", "content": cap})
    #ここでもトークン数トリミングが必要かも
    print("LLM入力直前トークン数",count_tokens(message))
    # もしメッセージの総トークン数が大きくなっていたら、トリムする
    #message = trim_messages(message, max_tokens_threshold=120000)
    # その後、agent_executor に渡す
    #########################################################################
    try:
        if st.session_state.output_method == "テキスト":
            full_response = ""
            #command = f"chcp 65001"
            #subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
            for step, metadata in agent_executor.stream(
            #for step in agent_executor.stream(
                {"messages": [message]},
                config,
                stream_mode="messages", #"values", #
                ):
                if tool_used:
                    break
                if metadata["langgraph_node"] == "agent" and (text := step.text()):
                    full_response += text
                    response_placeholder.markdown(f"{full_response}") #応答の表示
                    response_placeholder.write(full_response) #ok
                    #print(text, end="") #OK
            response = full_response
        else:
            response = agent_executor.invoke(
                {"messages": [message]},
                config,
                stream_mode="messages",
            )
            #speak(response)   #st.audio ok
            await streaming_text_speak(response)
        ####################################################################
        #print("use_tool_name:",st.session_state.use_tool_name)
        #st.write("use_tool_name:",st.session_state.use_tool_name)
        # チャット履歴に追加
        st.session_state.message_history.append(("user", user_input))
        st.session_state.message_history.append(("ai", response))
        #多くのLLMには入力トークン数の制限がある
        #履歴が長すぎると、モデルが全コンテキストを処理できなくなる trim_message_history(message_history, max_tokens=8192)
        max_tokens=st.session_state.input_max * 0.7
        st.session_state.message_history = trim_message_history(st.session_state.message_history,max_tokens)
        print("\nhistory_id=",st.session_state.history_id)
        # コストを計算して表示
        #calc_and_display_costs()
        return response
    except StopIteration:
        # StopIterationの処理
        print("StopIterationが発生")
        pass
    except Exception as e:
        print(f"エラーが発生したので、会話履歴を初期化しました。エラー詳細: {e}")
        # init_messages() の呼び出しを削除し、直接履歴を初期化する
        current_datetime_error = get_current_datetime()
        SYSTEM_MESSAGE_ERROR = f"""あなたはツールを効果的に活用する有能なアシスタントです。
                現在の日時は {current_datetime_error} です。
                提供されたツールを使用して、ユーザーが探している最新の情報を提供してください。
                必要な情報が得られない場合は、他のツールを使用してください。
                特に、天気、ニュースを聞かれた場合は、必ずツールを使ってください。
                また、常に日本語で回答してください。"""
        st.session_state.message_history = [("system", SYSTEM_MESSAGE_ERROR)]
        # 必要であれば、history_idもここで更新またはリセットします
        # 例: st.session_state.history_id = f"error_thread_{str(uuid.uuid4())}"

        st.error(f"エラーが発生し会話履歴を初期化しました: {e}", icon="🚨")
        return f"予期せぬエラーが発生しました: {e}。会話履歴は初期化されました。"

    user_input = ""
    #base64_image = ""
    #frame = ""
###########################################################################
###########################################################################
# --- 画像処理関数 (Streamlit 用) ---
def process_uploaded_image(uploaded_file):
    """アップロードされたファイルを読み込み、Base64エンコードする"""
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return base64_image, uploaded_file.type
        except Exception as e:
            st.error(f"画像処理エラー: {e}")
            return None, None
    return None, None

#音声出力関数
async def streaming_text_speak(llm_response):
    # 末尾の空白の数を確認
    #trailing_spaces = len(llm_response) - len(llm_response.rstrip())
    #print(f"末尾の空白の数: {trailing_spaces}")
    # 末尾の空白を削除
    #cleaned_response = llm_response.rstrip()
    #print(f"空白を除去した文字列: '{cleaned_response}'")
    # 句読点やスペースを基準に分割
    #復帰文字（\r）は、**キャリッジリターン（Carriage Return）**と呼ばれる特殊文字で、
    # ASCIIコード13（10進数）に対応します。主に改行の一部として使用される制御文字です。
    split_response = re.split(r'([\r\n!-;=:、。 \?]+)', llm_response) 
    #split_response = re.split(r'([;:、。 ]+😊🌟🚀🎉)', llm_response)  #?はなくてもOK
    split_response = [segment for segment in split_response if segment.strip()]  # 空要素を削除
    print(split_response)
    # AIメッセージ表示
    with st.session_state.answer_container:
        with st.chat_message("ai"):
            response_placeholder = st.empty()
            # ストリーミング応答と音声出力処理
            partial_text = ""
            for segment in split_response:
                if segment.strip():  # 空文字列でない場合のみ処理
                    partial_text += segment
                    response_placeholder.markdown(f"**{partial_text}**")  # 応答のストリーミング表示
                    # gTTSで音声生成（部分テキスト）
                    try:
                        # アスタリスクやその他の発音に不要な文字を削除
                        cleaned_segment = re.sub(r'[\*#*!-]', '', segment)
                        tts = gTTS(cleaned_segment, lang="ja")  # 音声化
                        audio_buffer = BytesIO()
                        tts.write_to_fp(audio_buffer)  # バッファに書き込み
                        audio_buffer.seek(0)

                        # pydubで再生速度を変更
                        audio = AudioSegment.from_file(audio_buffer, format="mp3")
                        audio = audio._spawn(audio.raw_data, overrides={
                            "frame_rate": int(audio.frame_rate * 1.3)  # 1.5倍速
                        }).set_frame_rate(audio.frame_rate)
                        audio_buffer.close()

                        # 音質調整
                        audio = audio.set_frame_rate(44100)  # サンプリングレート
                        audio = audio + 5  # 音量を5dB増加
                        audio = audio.fade_in(500).fade_out(500)  # フェードイン・アウト
                        #audio = audio.low_pass_filter(3000)  # 高音域をカット
                        audio = low_pass_filter(audio, cutoff=900)  # 高音域をカット
                        # ベースブースト（低音域を強調）
                        low_boost = low_pass_filter(audio,1000).apply_gain(10)
                        audio = audio.overlay(low_boost)

                        # バッファに再エクスポート
                        output_buffer = BytesIO()
                        audio.export(output_buffer, format="mp3")
                        output_buffer.seek(0)

                        # 音声の再生
                        # チェックする文字列
                        if re.search(r"\n\n", segment):
                            print("文字列に '\\n\\n' が含まれています。")
                            #time.sleep(1) 
                        #else:
                            #print("文字列に '\\n\\n' は含まれていません。")
                        #st.audio(audio_buffer, format="audio/mp3",autoplay = True)
                        # 音声データをBase64にエンコード
                        audio_base64 = base64.b64encode(output_buffer.read()).decode()
                        audio_buffer.close()  # バッファをクローズ
                        a=len(audio_base64)
                        #print(a)
                        # HTMLタグで音声を自動再生（プレイヤー非表示、再生速度調整）
                        audio_html = f"""
                            <audio id="audio-player" autoplay style="display:none;">
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                            </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)

                    except Exception as e:
                        #print(f"音声生成エラー: {e}")
                        pass
                    try:
                        time.sleep(a*0.00004)  # テキストストリーミング速度に同期
                    except Exception as e:
                        time.sleep(2) 

# Toolの設定をまとめて行う関数
def setup_tools():
    ######################################################
    #tools リストは、langchain の Tool オブジェクトのリスト
    # 利用するツールの定義
    # カスタムの検索ロジックを実装したい場合はTool()を使用する
    
    #openai_tools リストは、convert_to_openai_tool 関数によって 
    # OpenAI のツール形式に変換されたツール
    
    ######################################################
    class GetWeather(BaseModel):
        '''Get the current weather in a given location'''

        location: str = Field(
            ..., description="The city and state, e.g. Komatsu Ishikawa, Japan"
        )
    @tool(args_schema=GetWeather)
    def get_weather(location: str) -> str:
        """Get the current and future weather in a given location."""
        print(f"\n天気情報を取得するツール(get_weather)を使っています。")
        st.session_state.use_tool_name="get_weather"
        print("use_tool_name:",st.session_state.use_tool_name)
        st.write("use_tool_name:get_weather")
        # OpenWeatherMap APIキーを設定（環境変数から取得）
        api_key = os.environ["OPENWEATHERMAP_API_KEY"] #os.environ.get("OPENWEATHERMAP_API_KEY")
        if not api_key:
            return "Error: OPENWEATHERMAP_API_KEY environment variable not set."

        # ロケーションを都市名と国コードに分割（例：Komatsu Ishikawa, JP）
        location_parts = location.split(",")
        city_name = location_parts[0].strip()
        country_code = location_parts[1].strip() if len(location_parts) > 1 else ""

        # APIエンドポイントとパラメータを設定
        base_url = "http://api.openweathermap.org/data/2.5/forecast" #修正
        params = {
            "q": f"{city_name},{country_code}",
            "appid": api_key,
            "units": "metric",  # 摂氏で取得
            "lang": "ja" #日本語で取得
        }

        try:
            # APIリクエストを送信
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # エラーレスポンスをチェック

            # JSONレスポンスを解析
            weather_data = response.json()

            # 天気情報を抽出
            # 明日の天気情報を取得
            tomorrow_weather = None
            for forecast in weather_data["list"]:
                # タイムスタンプから日付を取得
                forecast_date = forecast["dt_txt"].split(" ")[0]
                # 明日の日付を取得
                tomorrow = datetime.now() + timedelta(days=1)
                tomorrow_str = tomorrow.strftime("%Y-%m-%d")

                if forecast_date == tomorrow_str:
                    tomorrow_weather = forecast
                    break

            if tomorrow_weather:
                description = tomorrow_weather["weather"][0]["description"]
                temperature = tomorrow_weather["main"]["temp"]
                humidity = tomorrow_weather["main"]["humidity"]
                wind_speed = tomorrow_weather["wind"]["speed"]

                result = (
                    f"{location}の明日の天気は{description}です。\n"
                    f"気温: {temperature}℃\n"
                    f"湿度: {humidity}%\n"
                    f"風速: {wind_speed}m/s"
                )
            else:
                result = f"{location}の明日の天気情報が見つかりませんでした。"

            # 現在の天気情報を取得
            base_url = "http://api.openweathermap.org/data/2.5/weather"
            response = requests.get(base_url, params=params)
            #print("response.url=",response.url)
            print("response.status_code=",response.status_code) #200
            #print("response.text=",response.text)
            #print("weather_data=",weather_data)
            response.raise_for_status()
            weather_data = response.json()
            description = weather_data["weather"][0]["description"]
            temperature = weather_data["main"]["temp"]
            humidity = weather_data["main"]["humidity"]
            wind_speed = weather_data["wind"]["speed"]
            result += (
                f"\n{location}の現在の天気は{description}です。\n"
                f"気温: {temperature}℃\n"
                f"湿度: {humidity}%\n"
                f"風速: {wind_speed}m/s"
            )
            return result

        except requests.exceptions.RequestException as e:
            return f"Error: Could not retrieve weather information for {location}. {e}"
        except (KeyError, IndexError) as e:
            return f"Error: Could not parse weather information for {location}. {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    ######################################################
    # Wolfram Alpha
    # Wolfram Alpha Tool
    class WolframAlphaInput(BaseModel):
        """Wolfram Alpha の入力."""
        query: str = Field(...,
            description="""Wolfram Alpha に送信するクエリ数学、科学、歴史、地理、文化など、
            幅広い分野の質問に答えるのに役立ちます。日付や天気には使用しないでください。""")

    wolfram_alpha_wrapper = WolframAlphaAPIWrapper()
    wolfram_alpha_tool = Tool(
        name="wolfram_alpha",
        func=wolfram_alpha_wrapper.run,
        description="""数学、科学、歴史、地理、文化など、幅広い分野の質問に答えるのに役立ちます。
                        日付や天気には使用しないでください。""",
        args_schema=WolframAlphaInput,
    )

    ####################################################################################
    # define the tools available to the agent - we're defining a single tool, exa_search
    # create the exa client
    #os.environ["EXA_API_KEY"] = "d74cc435-f3d2-4d8d-a21e-b6a66230c256"
    exa = Exa(api_key=os.environ["EXA_API_KEY"])
    # https://docs.exa.ai/reference/python-sdk-specification#search_and_contents-method
    def exa_search(query: str) -> Dict[str, Any]:
        st.session_state.use_tool_name="exa_search"
        print("use_tool_name:",st.session_state.use_tool_name)
        st.write("use_tool_name:",st.session_state.use_tool_name)
        return exa.search_and_contents(
            query=query, 
            type='auto', 
            text= True,
            num_results=3, #記事が多いとトークンオーバー
            ) #highlights=True,

    exa_tool = [
        {
            "type": "function",
            "function": {
                "name": "exa_search",
                "description": "Perform a search query on the web, and retrieve the world's most relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to perform.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    #tools リストに exa_tool を追加するには、以下の手順で行います。
    #修正手順
    #exa_tool を Tool オブジェクトに変換する関数を作成する。
    #exa_tool を tools リストに追加する。
    #openai_tools の作成を修正する。
    #exa_toolをToolオブジェクトに変換する関数
    def exa_tool_to_tool_object(exa_tool_def):
        def exa_search_wrapper(query: str) -> str:
            # exa_search関数を呼び出し、結果を文字列に変換して返す
            result = exa_search(query)
            return str(result)

        # Toolオブジェクトを作成して返す
        return Tool(
            name=exa_tool_def[0]["function"]["name"],
            func=exa_search_wrapper,
            description=exa_tool_def[0]["function"]["description"],
            )
    #exa_toolをToolオブジェクトに変換
    exa_tool_object = exa_tool_to_tool_object(exa_tool)
    ##########################################################################################
    # Tavily Search
    class TavilySearchInput(BaseModel):
        """Input for tavily search."""

        query: str = Field(..., description="search query to look up")


    tavily_search = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=True,
    )
    # TavilySearchResults を Tool でラップ
    tavily_tool = Tool(
        name="tavily_search_results_json",
        func=tavily_search.run,
        description="useful for when you need to answer questions about current events",
        args_schema=TavilySearchInput,
    )
    #########################################################################################
    # News API
    class NewsSearchInput(BaseModel):
        """Input for news search."""
        query: str = Field(..., description="search query to look up in news")

    # News API のクライアントを初期化
    newsapi = NewsApiClient(api_key=os.environ["NEWSAPI_API_KEY"])
    #os.environ["NEWSAPI_API_KEY"])

    @tool(args_schema=NewsSearchInput)
    def get_news(query: str) -> str:
        """Get the latest news based on a query."""
        st.session_state.use_tool_name="get_news"
        print("use_tool_name:",st.session_state.use_tool_name)
        st.write("use_tool_name:",st.session_state.use_tool_name)
        try:
            # NewsApiClientを使用してニュースを取得。日本語指定不可(LLMで翻訳させる)
            news = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page=1)
            # 最初のニュース記事を返す
            if news['articles']:
                result = ""
                for article in news['articles']:
                    result += f"Title: {article['title']}\nURL: {article['url']}\n\n"
                return result
            else:
                return "No news found for the given query."
        except Exception as e:
            return f"Error fetching news: {e}"
    ########################################################    
    #LangchainのToolオブジェクトのリスト
    tools = [get_weather,exa_tool_object,tavily_tool, wolfram_alpha_tool, get_news] #+ [fetch_page] 
    # Tool を OpenAI 形式に変換
    openai_tools = [convert_to_openai_tool(t) for t in tools] #+ exa_tool    
    return  tools,openai_tools
    ############################################################

def qa(text_input,webrtc_ctx,cap_title,cap_image,memory):
    # 末尾の空白の数を確認
    #trailing_spaces = len(text_input) - len(text_input.rstrip())
    #print(f"入力テキスト末尾の空白の数: {trailing_spaces}")
    # 末尾の空白を削除
    cleaned_text = text_input.rstrip()
    #print(f"入力テキスト末尾の空白を除去した文字列: '{cleaned_text}'")
    with st.session_state.answer_container:
        with st.chat_message('user'):
            st.write(cleaned_text)
    # 画像と問い合わせ入力があったときの処理
    cap = None
    image_base64 = None
    image_type = None
    uploaded_file = None
    # 画像処理 (uploaded_file)
    #if st.session_state.input_img == "有":
    if "画像" in text_input or "カメラ" in text_input or "画面" in text_input or "写真" in text_input:
        if st.session_state.img_url == "":
            #現在の画像をキャプチャする
            if webrtc_ctx.video_transformer:
                cap = webrtc_ctx.video_transformer.frame
                encoded_image = cv2.imencode('.jpg', cap)[1]
                # 画像をBase64に変換
                base64_image = base64.b64encode(encoded_image).decode('utf-8')
                st.session_state.img_url = f"data:image/jpeg;base64,{base64_image}"
        #if cap is not None :
            #st.sidebar.header("Capture Image")
            #cap_title.header("Capture Image")
            #cap_image.image(cap, channels="BGR")
            # if st.button("Query LLM : 画像の内容を説明して"):
    with st.session_state.answer_container:
        with st.spinner("Querying LLM..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            st.session_state.result= ""
            # query_llmの前に、前回の結果が残っていればクリア
            if 'result' in st.session_state and st.session_state.result:
                del st.session_state.result
                st.session_state.result = ""
            if 'img_url' in st.session_state and not ( "画像" in cleaned_text or "カメラ" in cleaned_text or "画面" in cleaned_text or "写真" in cleaned_text):
                st.session_state.img_url = "" # 画像を使わない問い合わせならクリア

            result = loop.run_until_complete(query_llm(cleaned_text,st.session_state.img_url,memory))
            st.session_state.result = result
    gc.collect() # QA処理後にガベージコレクション
    #LLMとのやり取りや音声処理で一時的に使用された大きなデータオブジェクトが、
    # 処理サイクル完了後に積極的に解放される
    result = ""
    text_input=""
    # QA処理後、img_urlをクリアしてメモリ解放を試みる
    if 'img_url' in st.session_state:
        del st.session_state.img_url
        st.session_state.img_url = ""
    st.session_state.img_url = ""

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):   
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

def whis_seg2(audio_segment):
    # AudioSegmentから直接NumPy配列を取得
    #audio_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    #audio_data /= np.iinfo(audio_segment.array_type).max  # 音声データを正規化
    audio_data = np.frombuffer(audio_segment.raw_data, dtype=np.int16).astype(np.float32)
    audio_data /= np.iinfo(np.int16).max  # 正規化
    # サンプリングレートを取得
    sample_rate = audio_segment.frame_rate
    #audio_segment = ""
    # Whisperが16kHzを期待するため、サンプリングレートを変換
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    # 音声データを適切な長さに調整
    audio_data = whisper.pad_or_trim(audio_data)
    #if not "whisper_model" in st.session_state:
        #st.session_state.whisper_model = whisper.load_model("small")
    # Whisperモデルは app_sst_with_video でロード・管理される
    if "whisper_model" not in st.session_state or st.session_state.whisper_model is None:
        st.error("Whisperモデルがロードされていません。")
        return "（音声認識モデル未ロード）"    
    whisper_model = st.session_state.whisper_model
    #whisper_model = whisper.load_model("small")
    result = whisper_model.transcribe(audio_data, language="ja")
    #audio_data = ""
    answer2 = result['text']
    #result = ""
    return answer2

def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:
    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")

def transcribe(audio_segment: AudioSegment, debug: bool = False) ->  Tuple[str, str]:
    answer2 ="（休止中）"
    # ステレオの場合、モノラルに変換
    #print("audio_segment.channels=",audio_segment.channels)  
    if audio_segment.channels > 1:  
        audio_segment = audio_segment.set_channels(1)      
    
    if debug:
        save_audio(audio_segment, "debug_audio")
    #if st.session_state.output_whi2:
    answer2 = whis_seg2(audio_segment)
    return answer2 

#async def process_audio(audio_data_bytes, sample_rate, sound_chunk):
async def process_audio(audio_data_bytes, sample_rate):
    sound = pydub.AudioSegment(
        data=audio_data_bytes, #無音区間で区切られた現在の音声セグメント
        sample_width=2, # audio_data_bytes.format.bytes,
        frame_rate=sample_rate,
        channels=2 , #len(audio_data_bytes.layout.channels), NG 1：文字化けする
    )
    #sound_chunk += sound
    #if len(sound_chunk) > 0:
        #answer2 = transcribe(sound_chunk)
    answer2 = ""
    if len(sound) > 0:
        answer2 = transcribe(sound)
    return answer2 

async def process_audio_loop_with_silence_detection(
    frames_deque_lock,
    frames_deque,
    #sound_chunk,# Removed as it's no longer accumulated here
    amp_indicator,
    ):
    """
    音声を無音区切りでまとめ、無音が一定時間続いたらテキスト変換を行う。
    """
    audio_buffer = []
    last_sound_time = time.time()
    silence_detected = False

    while True:
        # フレームを取得
        with frames_deque_lock:
            while len(frames_deque) > 0:
                frame = frames_deque.popleft() # 左端から要素を取り出して削除
                audio_chunk = frame.to_ndarray().astype(np.int16)
                audio_buffer.append(audio_chunk)
                st.session_state.frame_sample_rate = frame.sample_rate
                amp=np.max(np.abs(audio_chunk)) 
                #st.session_state.amp = amp
                amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})\n={amp}")
                #print(f"音声振幅/無音閾値={amp}/{SILENCE_THRESHOLD}")
                #amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})={amp}")
                if not amp < st.session_state.amp_threshold:
                    last_sound_time = time.time()
                    silence_detected = False
                else:
                    silence_detected = time.time() - last_sound_time >= st.session_state.silence_threshold
        #print(f"無音判定={silence_detected}")
        #print(f"audio_buffer={len(audio_buffer)}")
        # 無音区切りが検出された場合、音声データを処理
        if silence_detected and audio_buffer:
            audio_data = np.concatenate(audio_buffer).tobytes()
            try:
                #answer2 = await process_audio(audio_data, st.session_state.frame_sample_rate, sound_chunk)
                answer2 = await process_audio(audio_data, st.session_state.frame_sample_rate)
                ##########################################################
                #text_output.write(f"認識結果: {answer}")
                #おかしな回答を除去
                # テキスト出力が空、または空白である場合もチェック
                phrases = (
                    "ありがとう", 
                    "お疲れ様", "んんんんんん", 
                    "by H.","スタッフさんのお話を",
                    "いいえ- いいえ- いいえ-",
                    "ごちそうさまでした"
                    )
                if len(answer2) < 5:
                    pass
                elif any(phrase in answer2 for phrase in phrases):
                    pass
                else:
                    #with text_output.chat_message('user'):
                        #st.write(answer2)
                    print("[Whis_seg]",answer2) 
                    return answer2 
                audio_buffer = []  # バッファをクリア
                silence_detected = False
            except Exception as e:
                st.error(f"音声認識エラー: {e}")
                continue
        #amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})={st.session_state.amp}")
        #amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})={amp}")
        # 処理負荷を抑えるために短い遅延を挿入
        time.sleep(0.1)

def app_sst_with_video():
    memory = MemorySaver()
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])
    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)
        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)
        return new_frames
    
    memory_use = st.sidebar.empty()
    memory_alt = st.sidebar.empty()
    memory_ok = st.sidebar.empty()
    # メモリ使用量を監視 (ループの外に移動したが、カウンタiの扱いは要検討)
        # このiは音声入力ループのイテレーションカウンタだったため、ここでは固定値または別の方法で管理
    current_memory_use(0, memory_use, memory_alt, memory_ok) # iを0に固定、または別のカウンタを使用
    amp_indicator = st.sidebar.empty() #音声振幅表示用
    
    # ストリーミング状態を管理するセッション状態を初期化
    if "streaming" not in st.session_state:
        st.session_state["streaming"] = True  # 初期状態でストリーミング再生中
    # サイドバーにWebRTCストリームを表示
    with st.sidebar:
        st.header("Webcam Stream")
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text-w-video",
            desired_playing_state=st.session_state["streaming"], 
            mode=WebRtcMode.SENDRECV, #.SENDONLY,  #
            #audio_receiver_size=2048,  #1024　#512 #デフォルトは4
            #小さいとQueue overflow. Consider to set receiver size bigger. Current size is 1024.
            queued_audio_frames_callback=queued_audio_frames_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": True},
            video_processor_factory=VideoTransformer,  
        )
    if not webrtc_ctx.state.playing:
        return
    #status_indicator.write("Loading...")
    cap_title = st.sidebar.empty()
    cap_image = st.sidebar.empty() # プレースホルダーを作成
    # --- ▲▲▲ Whisper モデルサイズ選択を追加 ▲▲▲ ---
    uploaded_file = st.sidebar.file_uploader(
        "ここに画像をアップして問合せ可 (拡張子は小文字で)",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    if uploaded_file is not None:
        base64_image, image_type = process_uploaded_image(uploaded_file)
        st.session_state.img_url = f"data:{image_type};base64,{base64_image}"
        # アップロードされた画像を表示 (オプション)
        if cap_title and cap_image: # cap_title, cap_image の存在確認
            cap_title.header("Uploaded Image")
            cap_image.image(uploaded_file)

    #amp_indicator = st.sidebar.empty() #音声振幅表示用
    #status_indicator = st.empty() # 状態表示用

    st.sidebar.title("Options")
    col1, col2 = st.sidebar.columns(2)
    # 各列にボタンを配置
    with col1:
        # 入力方法の選択
        input_method = st.sidebar.radio("入力方法", ("音声とテキスト","テキストのみ" ))
        st.session_state.input_method = input_method
    with col2:
        # 出力方法の選択
        output_method = st.sidebar.radio("出力方法", ("テキスト", "音声"))
        st.session_state.output_method = output_method

    # --- 表示レイアウトとコンテナ定義 ---
    history_container = st.container(height=400)
    st.write("👇：回答中（リアルタイム返答）、👆：回答済（チャット履歴）")
    answer_container = st.container(height=300)
    #status_indicator = st.empty() # "何か話してね"の表示、状態表示用
    button_container = st.container()
    status_indicator = st.empty() # "何か話してね"の表示、状態表示用
    #answer_container = st.container(height=350)
    #with answer_container:
        #status_indicator = st.empty() # "何か話してね"の表示、状態表示用
    st.session_state.answer_container = answer_container
    # チャット履歴の表示（履歴コンテナ内に表示）
    with history_container:
        for role, message in st.session_state.get("message_history", []):
            #st.chat_message(role).markdown(message)
            with st.chat_message(role):
                st.markdown(message)

    ###################################################################
    # --- ボタン定義とチャット入力エリアをループの外に配置 ---
    button_input_query = None
    #ボタンコンテナ領域に表示
    with button_container:
        st.write("クイック問合せボタン:")
        cols = st.columns(7)
        button_definitions = [
            ("画像説明", "画像の内容を詳しく説明して"),
            ("前画像と差", "前の画像と何が変わりましたか？"),
            ("画像文の翻訳", "この画像の文を翻訳して"),
            ("何日", "今日は何日ですか？"),
            ("はじめの質問", "はじめの質問を憶えていますか?"),
            ("天気", "小松市の明日と明後日の天気は。花粉情報も教えて。"),
            ("ニュース", "今日のニュースは?"),
            ("AI情報", "今日のAIに関する情報は？"),
            ("人気漫画", "現在、人気の漫画を5件教えて。"),
            ("上映映画", "現在、上映中または上映予定の映画を5件教えて。"),
            ("小松の料理店", "小松市の最近話題の料理店は？"),
            ("百名山","日本の百名山を標高の高い順位から10山教えて"),
            ("人生の意義", "人生の意義は？"),
            ("オセロ作成", "Web画面でプレイするユーザー対AIのオセロゲームのコードを作成して"),
        ]
        for j, (label, query) in enumerate(button_definitions):
            if cols[j % 7].button(label, key=f"button_{j}_main"): # キーをユニークにする
                button_input_query = query

    # chat_input を一度だけ呼び出す
    #st.chat_inputは、streamlitの仕様で一番下に固定されている。
    user_typed_input = st.chat_input("🤖チャットテキストはここに入力してください...", key="main_chat_input_widget")
    final_user_input_to_process = None
    if button_input_query:
        final_user_input_to_process = button_input_query
    elif user_typed_input:
        final_user_input_to_process = user_typed_input
    ###################################################################
    #音声とテキスト入力（テキストに変換した入力）の対話ループ
    if st.session_state.input_method == "音声とテキスト":
        gc.collect() # 選択後にガベージコレクション
        #status_indicator = st.sidebar.empty()
        #amp_indicator = st.sidebar.empty()
        st.session_state.amp_threshold = st.sidebar.slider(
            "無音振幅閾値 (小さいほど敏感):",
            min_value=300, max_value=3000, value=1000, step=100
            )
        st.session_state.silence_threshold = st.sidebar.slider(
            "無音最小時間（秒）",
            min_value=0.1, max_value=3.0, value=0.5, step=0.1
            )
        #if not "whisper_model" in st.session_state:
            #st.session_state.whisper_model = whisper.load_model("small") #,device = "cuda")
        # Whisperモデルのロードと選択
        whisper_model_size_key = "whisper_model_size_select"
        if whisper_model_size_key not in st.session_state:
            st.session_state[whisper_model_size_key] = "tiny" # 初期値

        selected_whisper_model_size = st.sidebar.selectbox(
            "Whisperモデルサイズ (小さいほど軽量):",
            ("tiny", "base", "small", "medium"), # "large" はメモリ的に厳しい可能性
            index=("tiny", "base", "small", "medium").index(st.session_state[whisper_model_size_key]),
            key=whisper_model_size_key
        )
        if "whisper_model" not in st.session_state or st.session_state.get("current_whisper_size") != selected_whisper_model_size:
            with st.spinner(f"Whisperモデル ({selected_whisper_model_size}) をロード中..."):
                if "whisper_model" in st.session_state and st.session_state.whisper_model is not None:
                    del st.session_state.whisper_model # 古いモデルを解放試行
                    gc.collect()
                st.session_state.whisper_model = whisper.load_model(selected_whisper_model_size) #,device = "cuda")
                st.session_state.current_whisper_size = selected_whisper_model_size
                gc.collect()
        #base:74M,small:244M,medium:769M,large:1550M

        i = 0
        #current_memory_use(i,memory_use,memory_alt,memory_ok)
        frames_deque_lock = threading.Lock()
        # frames_deque_lockを使用してスレッドセーフに音声フレームを処理していますが、
        # dequeのクリア操作などでリソース競合が起きる可能性があります。
        # dequeの最大長を設定（例: deque([], maxlen=100)) し、バッファ溢れを防止する方が安全です。
        frames_deque: deque = deque([], maxlen=100) #NG 1
        #print(text_input)
        # 音声入力モードで、かつテキストやボタンからの入力がまだない場合のみ音声認識を実行
        if final_user_input_to_process is None:
            status_indicator.write("🤖何か話して!🦜...上のクイックボタンを押すか、テキストを入力してチャットしてもいいです。")
            # 音声処理の非同期タスクを起動
            recognized_audio_input = asyncio.run(process_audio_loop_with_silence_detection(
                frames_deque_lock,
                frames_deque,
                amp_indicator,
            ))
            if recognized_audio_input:
                final_user_input_to_process = recognized_audio_input
        
        # メモリ使用量を監視 (ループの外に移動したが、カウンタiの扱いは要検討)
        # このiは音声入力ループのイテレーションカウンタだったため、ここでは固定値または別の方法で管理
        current_memory_use(0, memory_use, memory_alt, memory_ok) # iを0に固定、または別のカウンタを使用
    ###################################################################
    # テキスト入力のみの場合
    elif st.session_state.input_method == "テキストのみ":
        gc.collect() # 選択後にガベージコレクション
        st.session_state["streaming"] = True  # Webカメラストリーミング再生
        # final_user_input_to_process は既にボタンまたはチャット入力で設定されている

        # メモリ使用量を監視
        i = 0 # このカウンタも、テキスト入力モードではループがないため、意味合いが変わる
        current_memory_use(i,memory_use,memory_alt,memory_ok)
        mem_use = get_memory_usage()
        # i += 1 # ループがないため不要
        print(f"メモリ使用量(テキストモード):{mem_use:.0f}MB")

    # --- QA処理 ---
    if final_user_input_to_process:
        qa(final_user_input_to_process,webrtc_ctx,cap_title,cap_image,memory)
        # 処理後に再実行して入力をクリアし、UIを更新
        st.rerun()
        # st.rerun() を削除（またはコメントアウト）します。これにより、
        # ユーザーの入力とAIの応答（ストリーミング表示を含む）が完了した後も、
        # それらのメッセージは即座に履歴コンテナに移動せず、画面の現在の位置に残り続けます。
        # そして、次の新しいインタラクション（ユーザーが新しいテキストを入力する、ボタンを押す、新しい音声入力が認識されるなど）
        # が発生してStreamlitのスクリプトが再実行されるタイミングで、
        # それまでのやり取りが履歴コンテナに格納され、新しいやり取りが開始されます。
    else: # 入力がない場合は、メモリ監視のみ行う（音声モードで音声入力待ちの場合など）
        #if st.session_state.input_method == "音声とテキスト":
        current_memory_use(0, memory_use, memory_alt, memory_ok) # iを0に固定
        #st.rerun 不要

def init_page():
    st.set_page_config(
        page_title="Yas Chatbot",
        page_icon="🤖"
    )
    #st.header("Yas Chatbot(画像、音声対応) 🤖")
    #st.write("""Webカメラ画像についての問合せ、音声での入出力ができます。\n
            #ブラウザのカメラ,マイクのアクセスを許可して使用。""")
    st.sidebar.title("🤖 Yas Chatbot")
    st.sidebar.caption("カメラ画像、画像ファイル、Web最新情報、音声での問合せができます")

def init_messages():
    history_id = 123
    clear_button = st.sidebar.button("会話履歴クリア", key="clear")
    
    #if clear_button or "message_history" not in st.session_state:
        #st.session_state.message_history = [
            #("system", "You are a helpful assistant.")
        #]   
    #問題は、初回アプリ起動時です。
    # このとき、clear_button はまだ押されていないので False です。
    # そして、message_history もまだ存在しないので 
    # "message_history" not in st.session_state は True となります。
    # False or True は True なので、初期化処理が実行されます。
    # 一見、問題ないように見えます。
    # clear_button が押された場合や message_history がまだ存在しない場合に初期化
    # clear_button が押された場合に初期化
    # 独立したifで記述することで、ボタン押下時、アプリ起動時、すべての状況に対応できました。
    # 現在の日付と時刻を取得
    current_datetime = get_current_datetime()
    # define the system message (primer) of your agent
    SYSTEM_MESSAGE = f"""あなたはツールを効果的に活用する有能なアシスタントです。
            現在の日時は {current_datetime} です。
            提供されたツールを使用して、ユーザーが探している最新の情報を提供してください。
            必要な情報が得られない場合は、他のツールを使用してください。
            特に、天気、ニュースを聞かれた場合は、必ずツールを使ってください。
            また、常に日本語で回答してください。"""
            #"""You are the world's most advanced search engine.
            #Please provide the user with the new information they are looking for by using the tools provided."""
            #You are a helpful assistant.
            #You are a competent assistant that effectively utilizes tools.　原文
            #貴方はツールを効果的に活用する有能なアシスタントです。原文の訳
            #あなたは世界で最も高度な検索エンジンです。
    if clear_button:
        st.session_state.message_history = [("system",SYSTEM_MESSAGE)]
        history_id += 1
        st.session_state.history_id = f"abc{history_id}"
    #clear_button が押された場合や message_history がまだ存在しない場合に初期化
    if "message_history" not in st.session_state:
        st.session_state.message_history = [("system", SYSTEM_MESSAGE)]

def main():
    #環境変数
    setup_environment()
    #st.header("Real Time Speech-to-Text with_video")
    #画面表示
    init_page()
    #会話履歴初期化とクリア
    init_messages()
    #stで使う変数初期設定
    #st.session_state.llm = select_model()
    # st.session_state.llm = None
    # st.session_state.selected_model_name = ""
    # st.session_state.model_name = ""
    # st.session_state.input_max = ""
    # st.session_state.input_method = ""
    # st.session_state.user_input = ""
    # st.session_state.result = ""
    # st.session_state.frame = ""
    if 'llm' not in st.session_state: st.session_state.llm = None
    if 'selected_model_name' not in st.session_state: st.session_state.selected_model_name = ""
    if 'model_name' not in st.session_state: st.session_state.model_name = ""
    if 'input_max' not in st.session_state: st.session_state.input_max = 0
    if 'input_method' not in st.session_state: st.session_state.input_method = "音声とテキスト" # デフォルト値
    if 'user_input' not in st.session_state: st.session_state.user_input = ""
    if 'result' not in st.session_state: st.session_state.result = ""
    #if 'frame' not in st.session_state: st.session_state.frame = "" # frameはVideoTransformer内で管理
    st.session_state.img_url = ""
    st.session_state.history_id = "abc123"
    #
    model_names = list(model_definitions.keys())
    # --- サイドバー: モデル選択 ---
    selected_model_name = st.sidebar.selectbox(
        "言語モデル選択(優劣有り):",
        model_names,
        #index=model_names.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_names else 0,
        index=model_names.index(st.session_state.selected_model_name)
            #if hasattr(st.session_state, "selected_model_name") and st.session_state.selected_model_name in model_names
            if st.session_state.selected_model_name in model_names
            else 0,
        key="model_select"
    )
    # --- モデル/グラフ/エージェントの再初期化 ---
    if st.session_state.selected_model_name is None or selected_model_name != st.session_state.selected_model_name:
        print("#"*100)
        #print("st.session_state.selected_model_name=",st.session_state.selected_model_name)
        #print("selected_model_name=",selected_model_name)
        print(f"モデル変更または初回ロード: 旧={st.session_state.selected_model_name}, 新={selected_model_name}")

        # 古いモデル関連リソースの解放
        if hasattr(st.session_state, 'llm') and st.session_state.llm is not None:
            del st.session_state.llm
            st.session_state.llm = None
        # if hasattr(st.session_state, 'graph_or_agent') and st.session_state.graph_or_agent is not None:
        #     del st.session_state.graph_or_agent
        #     st.session_state.graph_or_agent = None
        gc.collect()

        st.session_state.selected_model_name = selected_model_name
        st.session_state.model_name = selected_model_name
        #st.session_state.messages = []# LangGraphの入力用messageリスト。通常は会話履歴から構築。
                                        # ここでクリアすると、モデル変更時に会話がリセットされる挙動になる。
                                        # message_history は init_messages で管理。
        #st.session_state.graph_or_agent = None
        # 新しいモデル、新しい会話スレッドIDでMemorySaverを準備
        st.session_state.memory = MemorySaver()
        st.session_state.history_id = f"streamlit_thread_{selected_model_name}_{str(uuid.uuid4())}"

        # モデルインスタンスを取得 (ツールバインド前)
        #llm_instance_base, cost_input, cost_cached_input, cost_output, input_max = models_dict[selected_model_name]
        # --- ▼▼▼ 選択されたモデルのみをロード ▼▼▼ ---
        with st.spinner(f"モデル '{selected_model_name}' をロード中..."): # ★★★ 修正箇所 ★★★
            llm_instance_base, cost_input, cost_cached_input, cost_output, input_max = load_selected_model(selected_model_name)
            st.session_state.llm = llm_instance_base #select_model()
            st.session_state.input_max =  input_max
            gc.collect() # 新しいモデルロード後にも念のためGC
        if llm_instance_base is None:
            st.error(f"モデル '{selected_model_name}' のロードに失敗しました。")
            st.stop() # エラーが発生したらアプリを停止
        #else:
        #     # --- ▼▼▼ モデル切り替え後にガベージコレクション ▼▼▼ ---
            #gc.collect() # 不要になった古いモデルのリソース解放を期待  NG
            #st.rerun()  NG

    app_sst_with_video()

###################################################################      
if __name__ == "__main__":
    main()
