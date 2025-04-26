# chatbot_langgraph28-1.py を Streamlit アプリ化
# LangGraph/ReActの stream メソッドを使う限り、文節や句読点ごとの 滑らかな ストリーミング表示は困難です。
# 現在のコードは、思考ステップごとやメッセージ更新ごとに表示を更新する、現実的な実装となっています。
# より細かいストリーミングが必要な場合は、エージェント/グラフの機能（ツール連携、複雑な状態管理など）を諦めて、
# LLMの stream() を直接使う必要があります。
# ご提示の修正コード (st_lg13-1.py) に含まれる if new_content != full_response: のチェックは、
# 不要なUI更新を減らすために有効です。
# 現在のコードは、LangGraph/ReActの仕組みの中でのベストエフォートなストリーミング表示と言えます。
# 期待されているような細かい粒度での表示にはなっていないかもしれませんが、
# フレームワークの制約上、ある程度は仕方がない部分となります。

import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    VideoHTMLAttributes,
    VideoTransformerBase,
    ClientSettings
)
from typing import Annotated, Sequence, Optional, Literal # Literal を追加
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
    SystemMessage,
    trim_messages,
)
# from langchain.chains import RetrievalQA # 未使用のためコメントアウト
# from langchain_community.vectorstores import FAISS # 未使用のためコメントアウト
# from langchain_openai import OpenAIEmbeddings # 未使用のためコメントアウト
# from langchain.schema import Document # 未使用のためコメントアウト
from serpapi import GoogleSearch
from exa_py import Exa
import base64
from io import BytesIO # 画像処理用
import cv2
import av
import time
# import json # 未使用のためコメントアウト
import tiktoken
from datetime import datetime, timedelta
import uuid
# 他のpythonコードを呼び込む
import my_llms, my_querys, my_environments, my_tools
import os
import requests
import traceback # エラー詳細表示のため
from collections import deque
import numpy as np
from typing import Any, Dict, List,Tuple # 型ヒント
import threading
import psutil
# --- 音声処理ライブラリを追加 ---
from gtts import gTTS
import whisper
import pydub
from pydub import AudioSegment
from pydub.effects import low_pass_filter
import librosa
import asyncio # 追加
import re # 追加
import torch
import torchaudio
import torchvision

# 関数でメモリ使用量を取得
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    # メモリ使用量をMB単位で返す
    return mem_info.rss / (1024 * 1024)

def current_memory_use(memory_use,memory_alt,memory_ok):
    # 現在のメモリ使用量を取得
    current_memory_usage = get_memory_usage()
    # メモリ使用量を表示

    #memory_use.metric("現在のメモリ使用量 (MB)", f"{current_memory_usage:.2f}")
    memory_use.write(f"メモリ使用量:{current_memory_usage:.0f}MB")
    #print("現在のメモリ使用量 (MB)", f"{current_memory_usage:.2f}")
    # メモリ制約を定義
    MEMORY_LIMIT_MB = 2700  # 1GB
    # メモリ使用量が制約を超えた場合の警告
    if current_memory_usage > MEMORY_LIMIT_MB:
        memory_alt.error(f"メモリ使用量が制約 ({MEMORY_LIMIT_MB} MB) を超えました。処理を中断してください。")
        memory_ok.empty()
        #st.stop()
        print(f"メモリ使用量が制約 ({MEMORY_LIMIT_MB} MB) を超えました。処理を中断してください。")
    else:
        memory_ok.success("メモリ使用量は正常範囲内です。")
        memory_alt.empty()
        #print("メモリ使用量は正常範囲内です。")

# --- トークンカウント関数 (変更なし) ---
def get_token_count_for_messages(messages: Sequence[BaseMessage], model_name: str = "cl100k_base") -> int:
    """tiktoken を使ってメッセージリストのトークン数を計算する（近似値）"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        content_str = ""
        if isinstance(message.content, str):
            content_str = message.content
        elif isinstance(message.content, list): # 画像を含む場合など
            for part in message.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    content_str += part.get("text", "")
        else:
            content_str = str(message.content)
        num_tokens += len(encoding.encode(content_str))
        num_tokens += 4
    num_tokens += 3
    return num_tokens

# --- 日付取得関数 (変更なし) ---
def get_current_datetime():
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H時%M分")

# --- システムプロンプト (変更なし) ---
current_datetime_str = get_current_datetime() # アプリ起動時に取得
SYSTEM_MESSAGE = f"""あなたはツールを効果的に活用する有能なアシスタントです。
        現在の日付は{current_datetime_str}です。日付に関する情報は、この日付のみを元にしてください。
        最新の情報が必要な場合やあなたが答えられない場合、提供されたツールを使用してください。
        必要な情報が得られない場合は、他のツールも利用してください。
        Webサイトを紹介するだけの回答をしないで、サイトから具体的な情報を得て回答してください。
        また、常に日本語で回答してください。"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE), # language は chatbot_node 内で処理
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# --- ツール定義 (get_weather, search_q) (変更なし) ---
# @tool デコレータを使う場合
@tool
def get_weather(location: str) -> str:
    """Get the current and future weather in a given location."""
    st.info(f"天気情報を取得中: {location}") # Streamlit UI に表示
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Error: OPENWEATHERMAP_API_KEY environment variable not set."
    location_parts = location.split(",")
    city_name = location_parts[0].strip()
    country_code = location_parts[1].strip() if len(location_parts) > 1 else ""
    base_url_forecast = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": f"{city_name},{country_code}", "appid": api_key, "units": "metric", "lang": "ja"
    }
    result = ""
    try:
        response_forecast = requests.get(base_url_forecast, params=params)
        response_forecast.raise_for_status()
        weather_data_forecast = response_forecast.json()
        tomorrow_weather = None
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        for forecast in weather_data_forecast.get("list", []):
            if forecast.get("dt_txt", "").startswith(tomorrow_str):
                tomorrow_weather = forecast
                break
        if tomorrow_weather:
            description = tomorrow_weather.get("weather", [{}])[0].get("description", "不明")
            temperature = tomorrow_weather.get("main", {}).get("temp", "不明")
            humidity = tomorrow_weather.get("main", {}).get("humidity", "不明")
            wind_speed = tomorrow_weather.get("wind", {}).get("speed", "不明")
            result += f"{location}の明日の天気: {description}, 気温: {temperature}℃, 湿度: {humidity}%, 風速: {wind_speed}m/s\n"
        else:
            result += f"{location}の明日の天気情報が見つかりませんでした。\n"

        base_url_current = "http://api.openweathermap.org/data/2.5/weather"
        response_current = requests.get(base_url_current, params=params)
        response_current.raise_for_status()
        weather_data_current = response_current.json()
        description = weather_data_current.get("weather", [{}])[0].get("description", "不明")
        temperature = weather_data_current.get("main", {}).get("temp", "不明")
        humidity = weather_data_current.get("main", {}).get("humidity", "不明")
        wind_speed = weather_data_current.get("wind", {}).get("speed", "不明")
        result += f"{location}の現在の天気: {description}, 気温: {temperature}℃, 湿度: {humidity}%, 風速: {wind_speed}m/s"
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"天気情報取得エラー (Request): {e}")
        return f"Error: Could not retrieve weather information for {location}. {e}"
    except (KeyError, IndexError, TypeError) as e:
        st.error(f"天気情報解析エラー: {e}")
        return f"Error: Could not parse weather information for {location}. {e}"
    except Exception as e:
        st.error(f"予期せぬ天気情報エラー: {e}")
        return f"An unexpected error occurred: {e}"

@tool
def search_q(query: str) -> str:
    #"""Google検索を実行し、検索結果のテキストスニペットのリストを返す関数"""
    """Web検索を実行し、関連性の高いテキスト情報を返す関数 (Google Search または Exa)"""
    google_search_failed = False
    search_result_text = ""
    # --- Google Search を試行 ---
    st.info(f"Web検索を実行中: {query}") # Streamlit UI に表示
    serpapi_key = os.environ.get("SERPAPI_API_KEY")
    if not serpapi_key:
        st.warning("SERPAPI_API_KEY が設定されていません。Exa検索を試みます。")
        google_search_failed = True
    else:
        params = {
        "api_key": os.environ.get("SERPAPI_API_KEY"),
        "engine": "google", "q": query, "location": "Komatsu, Ishikawa, Japan",
        "google_domain": "google.com", "gl": "jp", "hl": "ja"
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            # organic_results が存在するか確認
            organic_results = results.get('organic_results', [])

            #snippets = [item.get('snippet', '') for item in results.get('organic_results', []) if item.get('snippet')]
            snippets = [item.get('snippet', '') for item in organic_results if item.get('snippet')]
            if snippets:
                search_result_text = "\n".join(snippets)
            else:
                # スニペットがない場合、タイトルとリンクを返す
                titles_links = [f"{item.get('title', '')}: {item.get('link', '')}" for item in organic_results]
                if titles_links:
                    search_result_text = "\n".join(titles_links)
                else:
                    st.warning("Google検索結果が見つかりませんでした。")
                    google_search_failed = True # 結果なしも失敗とみなす
        except requests.exceptions.HTTPError as http_err: # requests由来のHTTPErrorを捕捉
            st.error(f"Google検索HTTPエラー: {http_err}")
            if http_err.response.status_code == 403:
                st.warning("Google検索で403エラーが発生しました。Exa検索にフォールバックします。")
                google_search_failed = True
            else:
                # 403以外のHTTPエラーもExaにフォールバック
                st.warning(f"Google検索でHTTPエラー({http_err.response.status_code})が発生しました。Exa検索にフォールバックします。")
                google_search_failed = True
        except Exception as e: # その他のエラー (serpapiライブラリ内のエラーなど)
            st.error(f"Google検索エラー: {e}")
            google_search_failed = True

        # --- Google Search が失敗した場合、Exa Search にフォールバック ---
        if google_search_failed:
            st.info("Exa検索にフォールバックします...")
            exa_results_response = exa_search(query) # exa_search関数を呼び出す

            # Exa の結果 (SearchResponse オブジェクト) からテキスト情報を抽出
            exa_contents = []
            # exa_results_response.results がリストであることを確認
            if hasattr(exa_results_response, 'results') and isinstance(exa_results_response.results, list):
                for result in exa_results_response.results:
                    # result オブジェクトに必要な属性があるか確認
                    content_part = ""
                    if hasattr(result, 'title') and result.title:
                        content_part += f"Title: {result.title}\n"
                    if hasattr(result, 'url') and result.url:
                        content_part += f"URL: {result.url}\n"
                    if hasattr(result, 'text') and result.text:
                        # 本文は長すぎる可能性があるので500文字に切り詰める
                        content_part += f"Content: {result.text[:500]}..."
                    if content_part: # 何かしらの情報があれば追加
                        exa_contents.append(content_part.strip())

            if exa_contents:
                search_result_text = "\n\n".join(exa_contents)
            else:
                st.warning("Exa検索でも結果が見つかりませんでした。")
                search_result_text = "Web検索で関連情報が見つかりませんでした。" # 最終的なフォールバックメッセージ

        return search_result_text

# --- ツールリスト ---
# Streamlit用に human_assistance を除外
tools_std,openai_tools,serp_tool = my_tools.setup_tools() # 元のツール設定関数を呼び出す
tools_list = [t for t in tools_std if t.name != "human_assistance"]
#tools_list = [get_weather, search_q] # 直接定義する場合

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

# --- 音声出力関数 (st_webrtc_37-1.py からコピー) ---
async def streaming_text_speak(llm_response):
    split_response = re.split(r'([\r\n!-;=:、。 \?]+)', llm_response)
    split_response = [segment for segment in split_response if segment.strip()]
    print(split_response)
    # AIメッセージ表示用のプレースホルダーを取得または作成
    # この関数が st.chat_message("assistant") のコンテキスト内で呼び出されることを想定
    response_placeholder = st.empty()
    partial_text = ""
    for segment in split_response:
        if segment.strip():
            partial_text += segment
            response_placeholder.markdown(f"{partial_text}") # 太字解除
            try:
                # --- ▼▼▼ 修正箇所 ▼▼▼ ---
                # 英数字、漢字、ひらがな、カタカナ（全角含む）以外を削除
                # \u4e00-\u9fff: CJK統合漢字
                # \u3040-\u309f: ひらがな
                # \u30a0-\u30ff: カタカナ (全角カタカナ、長音記号含む)
                # \uff10-\uff19: 全角数字
                # \uff21-\uff3a: 全角英大文字
                # \uff41-\uff5a: 全角英小文字
                cleaned_segment = re.sub(
                    #r'[\*#*!-]',
                    r'[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]',
                    '',
                    segment
                    )
                # 削除によってできた可能性のある連続スペースや単独スペースも除去（読み上げ時の不自然な間を防ぐ）
                cleaned_segment = cleaned_segment.replace(' ', '')
                # cleaned_segmentが空文字列になった場合はTTS処理をスキップ
                if not cleaned_segment:
                    continue
                # --- ▲▲▲ 修正箇所 ▲▲▲ ---
                tts = gTTS(cleaned_segment, lang="ja")
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                audio = AudioSegment.from_file(audio_buffer, format="mp3")
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * 1.4) #1.3
                }).set_frame_rate(audio.frame_rate)
                audio_buffer.close()
                audio = audio.set_frame_rate(44100)
                audio = audio + 5
                audio = audio.fade_in(500).fade_out(500)
                audio = low_pass_filter(audio, cutoff=900)
                low_boost = low_pass_filter(audio,1000).apply_gain(10)
                audio = audio.overlay(low_boost)
                output_buffer = BytesIO()
                audio.export(output_buffer, format="mp3")
                output_buffer.seek(0)
                audio_base64 = base64.b64encode(output_buffer.read()).decode()
                output_buffer.close()
                a=len(audio_base64)
                audio_html = f"""
                    <audio id="audio-player" autoplay style="display:none;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
            except Exception as e:
                print(f"音声生成/再生エラー: {e}") # エラーログ改善
                pass
            try:
                # 再生時間を考慮した待機（より正確に）
                playback_duration = len(audio) / 1000.0 # milliseconds to seconds
                await asyncio.sleep(playback_duration * 0.7) # 再生時間の一部を待機 (調整可能)
            except Exception as e:
                await asyncio.sleep(1) # フォールバック

# --- 音声入力関数 (st_webrtc_37-1.py からコピー) ---
def whis_seg2(audio_segment):
    audio_data = np.frombuffer(audio_segment.raw_data, dtype=np.int16).astype(np.float32)
    audio_data /= np.iinfo(np.int16).max
    sample_rate = audio_segment.frame_rate
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    audio_data = whisper.pad_or_trim(audio_data)
    if not "whisper_model" in st.session_state:
        st.session_state.whisper_model = whisper.load_model("small")
    whisper_model = st.session_state.whisper_model
    result = whisper_model.transcribe(audio_data, language="ja")
    return result['text']

def transcribe(audio_segment: AudioSegment, debug: bool = False) -> str:
    answer2 ="（休止中）"
    # ステレオの場合、モノラルに変換
    if audio_segment.channels > 1:
        try:
            audio_segment = audio_segment.set_channels(1)
        except Exception as e:
            st.error(f"音声チャンネルの変換中にエラーが発生しました: {e}")
            traceback.print_exc() # 詳細なエラーログを出力
            return "" # エラー時は空文字列を返す
    if debug:
        # save_audio(audio_segment, "debug_audio") # 必要なら有効化
        pass
    answer2 = whis_seg2(audio_segment)
    return answer2

async def process_audio(audio_data_bytes, sample_rate, sound_chunk):
    sound = pydub.AudioSegment(
        data=audio_data_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=2  #NG1 # モノラルだと文字化けする
    )
    sound_chunk += sound
    answer2 = ""
    if len(sound_chunk) > 0:
        answer2 = transcribe(sound_chunk)
    return answer2

async def process_audio_loop_with_silence_detection(
    frames_deque_lock,
    frames_deque,
    sound_chunk,
    amp_indicator,
    status_indicator, # 追加
    button_input,
    prompt,
    memory_use,memory_alt,memory_ok
    ):
    """
    音声を無音区切りでまとめ、無音が一定時間続いたらテキスト変換を行う。
    """
    audio_buffer = []
    last_sound_time = time.time()
    silence_detected = False

    while True:
        final_prompt = button_input if button_input is not None else prompt
        if final_prompt:
            return final_prompt
        # フレームを取得
        with frames_deque_lock:
            while len(frames_deque) > 0:
                frame = frames_deque.popleft() # 左端から要素を取り出して削除
                audio_chunk = frame.to_ndarray().astype(np.int16)
                audio_buffer.append(audio_chunk)
                st.session_state.frame_sample_rate = frame.sample_rate
                amp=np.max(np.abs(audio_chunk))
                #st.session_state.amp = amp
                #amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})=\n\n{amp}")
                amp_indicator.write(f"音声振幅={amp}")
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
                answer2 = await process_audio(audio_data, st.session_state.frame_sample_rate, sound_chunk)
                ##########################################################
                #text_output.write(f"認識結果: {answer}")
                #おかしな回答を除去
                # テキスト出力が空、または空白である場合もチェック
                phrases = (
                    "ありがとう", 
                    "お疲れ様", "んんんんんん", 
                    "by H.","スタッフさんのお話を",
                    "いいえ- いいえ- いいえ-",
                    "ごちそうさまでした","チャンネル登録を",
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
        # メモリ使用量を監視
        current_memory_use(memory_use,memory_alt,memory_ok)
        mem_use = get_memory_usage()
            
        # 処理負荷を抑えるために短い遅延を挿入
        time.sleep(0.1)


# --- ▼▼▼ アプローチ1 用の State とノード定義 ▼▼▼ ---
class EnhancedState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    needs_weather: bool
    weather_info: Optional[str]
    needs_search: bool
    search_results: Optional[str]
    is_image_query: bool # 画像が含まれるかどうかのフラグ

def classify_input(state: EnhancedState):
    """ユーザー入力を分類し、事前処理が必要か判断するノード"""
    last_message = state["messages"][-1]
    content = last_message.content
    text_content = ""
    has_image = False

    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                text_content = part.get("text", "")
            elif part.get("type") == "image_url":
                has_image = True
    elif isinstance(content, str):
        text_content = content

    needs_weather = False
    needs_search = False

    # 画像がない場合のみ、テキスト内容に基づいて事前処理を判断
    if not has_image:
        if "天気" in text_content:
            needs_weather = True
        elif any(phrase in text_content for phrase in ("検索", "最新", "話題", "最近", "現在", "について")):
            needs_search = True

    return {
        "needs_weather": needs_weather,
        "needs_search": needs_search,
        "is_image_query": has_image,
        "weather_info": None, # 初期化
        "search_results": None # 初期化
    }

def get_weather_node(state: EnhancedState):
    """天気情報を取得するノード"""
    last_message = state["messages"][-1]
    text_content = ""
    if isinstance(last_message.content, list):
         for part in last_message.content:
            if part.get("type") == "text":
                text_content = part.get("text", "")
                break
    elif isinstance(last_message.content, str):
        text_content = last_message.content

    # 簡単な地名抽出 (より高度な抽出が必要な場合あり)
    if "石川" in text_content:city = "石川"
    if "小松" in text_content:city = "小松"
    city_parts = text_content.split("の天気")[0].split()[-1] # "〇〇の天気" の前の単語を取得
    city = city_parts.replace("市","").replace("県","") # 市や県を除去
    if "石川" in city_parts:city = "石川"
    if "小松" in city_parts:city = "小松"
    if not city: city = "小松" # デフォルト
    st.info(f"天気情報を取得中 (ノード): {city}")
    weather_info = get_weather.invoke({"location": f"{city}, JP"}) # @tool デコレータ付き関数を呼び出す
    return {"weather_info": weather_info}

# Exa検索関数 (search_q と同じファイルかインポート可能な場所に定義)
# @tool デコレータは付けない（search_q内で呼び出すため）
def exa_search(query: str) -> Dict[str, Any]:
    """Exa検索を実行し、検索結果のコンテンツを含む辞書を返す関数"""
    st.info(f"Exa検索を実行中: {query}") # Streamlit UI に表示
    #st.session_state.use_tool_name="exa_search"
    print("use_tool_name:",exa_search)
    try:
        exa_api_key = os.environ.get("EXA_API_KEY")
        if not exa_api_key:
            st.error("Exa APIキーが設定されていません。")
            return {"results": []}
        exa = Exa(api_key=exa_api_key)
        # text=True でコンテンツ本文を取得、num_results で件数制限
        # use_autoprompt=True はクエリを自動で最適化しますが、意図しない検索になる可能性もあるため注意
        results = exa.search_and_contents(
            query=query,
            type='neural', # または 'keyword'
            # use_autoprompt=True,
            num_results=3, # 結果件数を絞る（トークン量と関連性のため）
            text=True # コンテンツ本文を取得
        )
        # results は SearchResponse オブジェクト (pydanticモデル)
        return results # そのまま返す
    except Exception as e:
        st.error(f"Exa検索エラー: {e}")
        return {"results": []} # エラー時は空の結果を示す辞書を返す

def search_node(state: EnhancedState):
    """Web検索を実行するノード"""
    last_message = state["messages"][-1]
    text_content = ""
    if isinstance(last_message.content, list):
         for part in last_message.content:
            if part.get("type") == "text":
                text_content = part.get("text", "")
                break
    elif isinstance(last_message.content, str):
        text_content = last_message.content

    st.info(f"Web検索を実行中 (ノード): {text_content}")
    search_results = search_q.invoke({"query": text_content}) # @tool デコレータ付き関数を呼び出す
    return {"search_results": search_results}

def route_after_classification(state: EnhancedState) -> Literal["get_weather", "search", "chatbot", "__end__"]:
    """classify_input の結果に基づいてルーティング"""
    if state.get("needs_weather"):
        return "get_weather"
    elif state.get("needs_search"):
        return "search"
    else:
        # 画像クエリの場合、または事前処理不要なテキストクエリの場合
        return "chatbot"

# --- グラフ定義関数 (修正: ツール非対応モデル用のグラフも作成) ---
def create_graph(
    llm_instance, 
    tools_list_for_graph, 
    memory_instance, 
    trimmer_instance, 
    prompt_template_instance, 
    use_enhanced_graph=False
    ):
    """LangGraphのグラフを作成・コンパイルする関数"""
    if use_enhanced_graph:
        # --- アプローチ1: 事前処理ノードを含むグラフ ---
        graph_builder = StateGraph(EnhancedState) # 拡張 State を使用
        graph_builder.add_node("classify_input", classify_input)
        graph_builder.add_node("get_weather", get_weather_node)
        graph_builder.add_node("search", search_node)

        def enhanced_chatbot_node(state: EnhancedState, config: Optional[dict] = None):
            language = config.get("configurable", {}).get("language", "Japanese") if config else "Japanese"
            trimmed_messages = trimmer_instance.invoke(state["messages"])
            # 事前処理の結果を取得
            weather_info = state.get("weather_info")
            search_results = state.get("search_results")
            # 最後のユーザーメッセージを取得
            last_human_message = None
            for msg in reversed(trimmed_messages):
                if isinstance(msg, HumanMessage):
                    last_human_message = msg
                    break
            if last_human_message is None:
                return {"messages": [AIMessage(content="ユーザーメッセージが見つかりません。")]}

            # プロンプトに事前処理情報を追加
            prompt_content_parts = []
            original_text = ""
            image_part = None

            if isinstance(last_human_message.content, list):
                for part in last_human_message.content:
                    if part.get("type") == "text":
                        original_text = part.get("text", "")
                    elif part.get("type") == "image_url":
                        image_part = part
            elif isinstance(last_human_message.content, str):
                original_text = last_human_message.content

            prompt_content_parts.append({"type": "text", "text": original_text})
            if image_part:
                prompt_content_parts.append(image_part)

            additional_context = ""
            # --- ▼▼▼ 日付に関する処理を追加 ▼▼▼ ---
            # original_text に日付関連のキーワードが含まれるかチェック
            date_keywords = ["今日", "日付", "何日", "日時"]
            if any(keyword in original_text for keyword in date_keywords):
                current_date_info = get_current_datetime() # 現在の日付・時刻を取得
                additional_context += f"\n\n[現在の日時情報]\n{current_date_info}"
            # --- ▲▲▲ 日付に関する処理を追加 ▲▲▲ ---
            elif weather_info: # elif に変更して、日付情報と天気/検索が重複しないようにする
                additional_context += f"\n\n[取得済みの天気情報]\n{weather_info}"
            elif search_results:
                additional_context += f"\n\n[取得済みの検索結果]\n{search_results}"

            if additional_context:
                # テキストパートにコンテキストを追加
                #prompt_content_parts[0]["text"] += additional_context
                # テキストパートにコンテキストを追加する方法を調整
                # 既存のテキストパートに追加するか、新しいテキストパートとして挿入するか
                found_text_part = False
                for part in prompt_content_parts:
                    if part.get("type") == "text":
                        part["text"] += additional_context
                        found_text_part = True
                        break
                if not found_text_part: # テキストパートが元々ない場合 (通常はないはず)
                    prompt_content_parts.insert(0, {"type": "text", "text": additional_context})

            # 最終的なメッセージリストを作成 (System + 過去履歴 + 加工済み最新Human)
            # trimmer_instance.invoke(state["messages"]) で得られるのは最新のHumanMessageを含まない場合があるので注意
            # state["messages"] を直接使う方が確実かもしれない
            final_messages_for_llm = trimmer_instance.invoke(state["messages"][:-1]) # 最新を除く履歴
            final_messages_for_llm.append(HumanMessage(content=prompt_content_parts)) # 加工済み最新メッセージ

            # プロンプトテンプレートを適用
            prompt_value = prompt_template_instance.invoke({"messages": final_messages_for_llm, "language": language})
            # print("prompt_value=",prompt_value)
            try:
                prompt_messages = prompt_value.to_messages()
                # # --- ▼▼▼ 確認用コードを追加 ▼▼▼ ---
                # print("-" * 80)
                # print(">>> enhanced_chatbot_node: LLMに渡されるメッセージリストを確認 <<<")
                # if prompt_messages:
                #     first_message = prompt_messages[0]
                #     print(f"  [タイプ]: {type(first_message)}")
                #     if isinstance(first_message, SystemMessage):
                #         print("  [内容]:")
                #         # content が長い場合があるので、改行して見やすく表示
                #         content_lines = first_message.content.split('\n')
                #         for line in content_lines:
                #             print(f"    {line.strip()}") # 各行の先頭・末尾の空白を除去
                #         # SYSTEM_MESSAGE 定数と比較 (任意)
                #         if first_message.content == SYSTEM_MESSAGE:
                #             print("  [確認]: グローバルな SYSTEM_MESSAGE と一致します。")
                #         else:
                #             print("  [警告]: グローバルな SYSTEM_MESSAGE と内容が異なります！")
                #     else:
                #         print(f"  [警告]: 最初のメッセージが SystemMessage ではありません！ (タイプ: {type(first_message)})")
                #     # 必要であれば他のメッセージも確認
                #     # print("  --- 全メッセージ概要 ---")
                #     # for i, msg in enumerate(prompt_messages):
                #     #     content_preview = str(msg.content)[:100].replace('\n', ' ') + "..." if len(str(msg.content)) > 100 else str(msg.content).replace('\n', ' ')
                #     #     print(f"    {i}: ({type(msg).__name__}) {content_preview}")
                # else:
                #     print("  [エラー]: prompt_messages リストが空です！")
                # print("-" * 80)
                # # --- ▲▲▲ 確認用コードを追加 ▲▲▲ ---
            except Exception as e:
                st.error(f"PromptValue のメッセージ変換エラー: {e}")
                return {"messages": [AIMessage(content=f"プロンプトの変換中にエラーが発生しました: {e}")]}

            # LLM呼び出し (ツールはバインドしない)
            try:
                # llm_instance は session_state から取得 (ツールバインドされていないもの)
                llm_instance_no_tools = st.session_state.llm_instance_no_tools # 事前に用意しておく必要あり
                message = llm_instance_no_tools.invoke(prompt_messages)
                return {"messages": [message]}
            except Exception as e:
                st.error(f"Enhanced Chatbot Node Error: {e}")
                error_content = f"モデル呼び出し中にエラーが発生しました。\n```\n{traceback.format_exc()}\n```"
                return {"messages": [AIMessage(content=error_content)]}

        graph_builder.add_node("chatbot", enhanced_chatbot_node)
        graph_builder.add_edge(START, "classify_input")
        graph_builder.add_conditional_edges(
            "classify_input",
            route_after_classification,
            {
                "get_weather": "get_weather",
                "search": "search",
                "chatbot": "chatbot",
                "__end__": END # このパスは通常通らないはず
            }
        )
        graph_builder.add_edge("get_weather", "chatbot")
        graph_builder.add_edge("search", "chatbot")
        graph_builder.add_edge("chatbot", END) # chatbot ノードの後は終了

        return graph_builder.compile(checkpointer=memory_instance)

    else:
        # --- 通常のグラフ (ツールあり/なし自動判別) ---
        graph_builder = StateGraph(EnhancedState) # State は EnhancedState を使う

        def chatbot_node(state: EnhancedState, config: Optional[dict] = None):
            language = config.get("configurable", {}).get("language", "Japanese") if config else "Japanese"
            trimmed_messages = trimmer_instance.invoke(state["messages"])

            prompt_value = prompt_template_instance.invoke({"messages": trimmed_messages, "language": language})
            try:
                prompt_messages = prompt_value.to_messages()

            except Exception as e:
                st.error(f"PromptValue のメッセージ変換エラー: {e}")
                return {"messages": [AIMessage(content=f"プロンプトの変換中にエラーが発生しました: {e}")]}

            # llm_instance は session_state から取得 (ツールバインド済み or なし)
            llm_to_use = st.session_state.llm_instance
            current_tools = tools_list_for_graph # グラフ作成時に渡されたツールリスト

            try:
                use_tools = bool(current_tools)
                has_image = state.get("is_image_query", False) # classify_input があればそこから取得

                # 画像があり、かつ画像とツールの併用が不可の場合、ツールを使わない
                if has_image and not st.session_state.can_use_tools_with_image:
                    use_tools = False
                    st.warning("画像添付時はツールを使用できません。")
                    # ツールなしLLMインスタンスを使う (必要なら)
                    # llm_to_use = st.session_state.llm_instance_no_tools

                # ツールを使う場合 (bind_tools済みのはず) / 使わない場合
                message = llm_to_use.invoke(prompt_messages)
                return {"messages": [message]}

            except Exception as e:
                st.error(f"Chatbot Node Error: {e}")
                error_content = f"モデル呼び出し中にエラーが発生しました。\n```\n{traceback.format_exc()}\n```"
                return {"messages": [AIMessage(content=error_content)]}

        graph_builder.add_node("chatbot", chatbot_node)
        tool_node_func = ToolNode(tools=tools_list_for_graph)
        graph_builder.add_node("tools", tool_node_func)

        if tools_list_for_graph: # ツールがある場合のみ条件分岐とツールノードへのエッジを追加
            graph_builder.add_conditional_edges(
                "chatbot",
                tools_condition,
                {"tools": "tools", END: END}
            )
            graph_builder.add_edge("tools", "chatbot")
        else:
            graph_builder.add_edge("chatbot", END)

        graph_builder.add_edge(START, "chatbot")

        return graph_builder.compile(checkpointer=memory_instance)


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

# --- Streamlit アプリ本体 ---
async def main():
    st.set_page_config(
        page_title="Yas Chatbot",
        page_icon="🤖",
        layout="wide"
        )
    st.sidebar.title("🤖 Yas Chatbot")
    st.sidebar.caption("カメラ画像、画像ファイル、Web最新情報、音声での問合せができます")

    # --- Webカメラ設定 ---
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

    # サイドバーにWebRTCストリームを表示
    with st.sidebar:
        st.header("Webcam Stream")
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text-w-video",
            desired_playing_state=True,
            mode=WebRtcMode.SENDRECV,
            queued_audio_frames_callback=queued_audio_frames_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": True},
            video_processor_factory=VideoTransformer,
            async_processing=True, # Streamlitアプリの応答性を保つため非同期処理を推奨  
        )
    if not webrtc_ctx.state.playing :
        st.sidebar.warning("Webカメラを開始してください。")
        return
    # --- 初期化 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = None
    if "graph_or_agent" not in st.session_state:
        st.session_state.graph_or_agent = None
    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "is_react" not in st.session_state:
        st.session_state.is_react = False
    if "llm_instance" not in st.session_state: # バインド済み or ReAct用
        st.session_state.llm_instance = None
    if "llm_instance_no_tools" not in st.session_state: # ツールなし用 (enhanced_graph用)
        st.session_state.llm_instance_no_tools = None
    if "trimmer" not in st.session_state:
        st.session_state.trimmer = None
    if "can_use_tools_with_image" not in st.session_state:
        st.session_state.can_use_tools_with_image = True
    if "use_enhanced_graph" not in st.session_state: # Enhanced Graph を使うかどうかのフラグ
        st.session_state.use_enhanced_graph = False
    # --- 音声入出力用の session_state を追加 ---
    if "input_method" not in st.session_state:
        st.session_state.input_method = "音声&テキスト" # 入力のデフォルト
    if "output_method" not in st.session_state:
        st.session_state.output_method = "テキスト"
    if "whisper_model" not in st.session_state:
        st.session_state.whisper_model = None # Whisperモデルを保持
    if "frame_sample_rate" not in st.session_state:
        st.session_state.frame_sample_rate = None # 音声フレームのサンプルレート
    if "amp_threshold" not in st.session_state:
        st.session_state.amp_threshold = 800 # 無音閾値のデフォルト
    if "silence_threshold" not in st.session_state:
        st.session_state.silence_threshold = 0.5 # 無音時間のデフォルト

    # 環境変数設定 (初回のみ)
    with st.spinner("環境変数を設定中..."):
        try:
            my_environments.setup_environment()
            #info_disp.info("環境変数をロードしました。")
        except Exception as e:
            st.sidebar.error(f"環境変数の設定に失敗: {e}")
            st.error("続行できません。環境変数を確認してください。")
            st.stop()

    # モデルリスト取得 (キャッシュ)
    @st.cache_resource
    def load_models():
        models = my_llms.initialize_models()
        return models

    #info_disp.info("LLMモデルを初期化中...")
    models_dict = load_models()
    #info_disp.info("LLMモデルの初期化完了。")
    model_names = list(models_dict.keys())

    # --- サイドバー: モデル選択 ---
    selected_model_name = st.sidebar.selectbox(
        "言語モデル選択(優劣有り):",
        model_names,
        index=model_names.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_names else 0,
        key="model_select"
    )

    uploaded_file = st.sidebar.file_uploader(
        "ここに画像をアップロードして問合せ (オプション)",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    # --- 入出力方法選択を追加 ---
    #st.sidebar.title("入出力設定")
    #st.sidebar.title("出力方法設定")
    #st.session_state.input_method = st.sidebar.radio("入力方法:", ("テキスト", "音声&テキスト"), index=1 if st.session_state.input_method == "テキスト" else 1)
    st.session_state.output_method = st.sidebar.radio("出力方法:", ("テキスト", "音声"), index=0 if st.session_state.output_method == "テキスト" else 1)
    # --- 音声入力設定を追加 ---
    #if st.session_state.input_method == "音声&テキスト":
    amp_indicator = st.sidebar.empty() # 音声振幅表示用
    st.sidebar.subheader("音声入力設定")
    st.session_state.amp_threshold = st.sidebar.slider(
        "無音振幅閾値 (小さいほど敏感):",
        min_value=100, max_value=3000, value=st.session_state.amp_threshold, step=100
        )
    st.session_state.silence_threshold = st.sidebar.slider(
        "無音最小時間 (秒):",
        min_value=0.1, max_value=3.0, value=st.session_state.silence_threshold, step=0.1
        )
    # Whisperモデルのロード (初回のみ)
    if st.session_state.whisper_model is None:
        with st.spinner("Whisperモデルをロード中..."):
            st.session_state.whisper_model = whisper.load_model("small")

    # --- 情報表示 ---
    info_disp = st.sidebar.empty() 
    memory_use = st.sidebar.empty()
    memory_alt = st.sidebar.empty()
    memory_ok = st.sidebar.empty()
    
    # --- モデル/グラフ/エージェントの再初期化 ---
    if st.session_state.selected_model_name is None or selected_model_name != st.session_state.selected_model_name:
        print("#"*100)
        print("st.session_state.selected_model_name=",st.session_state.selected_model_name)
        print("selected_model_name=",selected_model_name)
        st.session_state.selected_model_name = selected_model_name
        st.session_state.messages = []
        st.session_state.graph_or_agent = None
        st.session_state.memory = MemorySaver()
        st.session_state.thread_id = f"streamlit_thread_{selected_model_name}_{str(uuid.uuid4())}"

        # モデルインスタンスを取得 (ツールバインド前)
        llm_instance_base, cost_input, cost_cached_input, cost_output, input_max = models_dict[selected_model_name]
        st.session_state.llm_instance_no_tools = llm_instance_base # ツールなしインスタンスを保存

        st.session_state.is_react = False
        st.session_state.can_use_tools_with_image = True
        st.session_state.use_enhanced_graph = False # Enhanced Graph フラグをリセット

        #st.sidebar.info(f"モデル '{selected_model_name}' を準備中...")
        info_disp.info(f"モデル '{selected_model_name}' を準備中...")

        # Trimmer の初期化
        st.session_state.trimmer = trim_messages(
            max_tokens=int(input_max * 0.8),
            strategy="last",
            token_counter=lambda messages: get_token_count_for_messages(messages, model_name="gpt-4"),
            include_system=False,
            allow_partial=False,
        )

        # --- グラフ/エージェント準備ロジック (修正) ---
        #@st.cache_resource(show_spinner=f"'{selected_model_name}' のグラフ/エージェントを準備中...")
        def prepare_graph_or_agent(_model_name, _llm_base, _memory, _trimmer, _tools):
            """
            グラフまたはエージェントを準備し、ログメッセージを返す。
            戻り値: (graph_or_agent, is_react, can_use_tools_img, use_enhanced, logs)
            logs: List of (level, message) tuples. level can be 'info', 'success', 'warning', 'error'.
            """
            _graph_or_agent = None
            _is_react = False
            _can_use_tools_img = True
            _current_tools = list(_tools)
            _use_enhanced = False
            logs = [] # ログメッセージを格納するリスト
            # 特定モデルの判定 (ツール非対応 or 画像併用不可)
            no_tool_models = ("c4ai-aya-vision-32b", "nvidia/llama-3.1", "neva", "Phi", "phi", "kosmos", "fuyu", "gemma") # aya を含む
            is_no_tool_model = any(phrase in _model_name for phrase in no_tool_models)

            if is_no_tool_model:
                # ツール非対応モデルの場合 -> Enhanced Graph を使用
                #info_disp.info(f"{_model_name}: ツール非対応モデル。事前処理ノード付きグラフを使用します。")
                #print(f"{_model_name}: ツール非対応モデル。事前処理ノード付きグラフを使用します。")
                log_msg = f"{_model_name}: ツール非対応モデル。事前処理ノード付きグラフを使用します。"
                logs.append(("info", log_msg))
                print("enhanced_info", log_msg) # コンソールにも出力（任意）
                print("enhanced_prompt_template=",prompt_template)
                _graph_or_agent = create_graph(_llm_base, [], _memory, _trimmer, prompt_template, use_enhanced_graph=True)
                _is_react = False
                _can_use_tools_img = False # ツール自体使わない
                _use_enhanced = True

            else:
                # ツール対応モデルの場合: bind_tools を試す
                try:
                    _llm_bound = _llm_base.bind_tools(_current_tools)
                    # ツールありグラフを作成
                    _graph_or_agent = create_graph(_llm_bound, _current_tools, _memory, _trimmer, prompt_template, use_enhanced_graph=False)
                    _is_react = False
                    _can_use_tools_img = True # デフォルトは併用可能とする (chatbot_node内で最終判断)
                    #info_disp.success(f"{_model_name}: ツール対応グラフを作成しました。")
                    #print(f"{_model_name}: ツール対応グラフを作成しました。")
                    log_msg = f"{_model_name}: ツール対応グラフを作成しました。"
                    logs.append(("success", log_msg))
                    print(log_msg)
                except (ValueError, AttributeError, NotImplementedError, Exception) as e:
                    # bind_tools 失敗 -> ReAct を試す
                    #st.sidebar.warning(f"{_model_name}: グラフ作成/ツールバインド失敗: {e}. ReActエージェントを試みます。")
                    #print(f"{_model_name}: グラフ作成/ツールバインド失敗: {e}. ReActエージェントを試みます。")
                    log_msg = f"{_model_name}: グラフ作成/ツールバインド失敗: {e}. ReActエージェントを試みます。"
                    logs.append(("warning", log_msg)) # st.sidebar.warning の代わりにログ追加
                    print(log_msg)
                    _is_react = True
                    try:
                        _graph_or_agent = create_react_agent(_llm_base, _tools, checkpointer=_memory)
                        _can_use_tools_img = True # ReActでも画像併用可否は別途制御が必要な場合あり
                        #info_disp.info(f"{_model_name}: ReActエージェント(ツールあり)を作成しました。")
                        #print(f"{_model_name}: ReActエージェント(ツールあり)を作成しました。")
                        log_msg = f"{_model_name}: ReActエージェント(ツールあり)を作成しました。"
                        logs.append(("info", log_msg))
                        print(log_msg)
                    except Exception as react_e:
                        #st.sidebar.error(f"ReActエージェント(ツールあり)作成失敗: {react_e}")
                        #print(f"ReActエージェント(ツールあり)作成失敗: {react_e}")
                        log_msg = f"ReActエージェント(ツールあり)作成失敗: {react_e}"
                        logs.append(("error", log_msg)) # st.sidebar.error の代わりにログ追加
                        print(log_msg)
                        try:
                            _graph_or_agent = create_react_agent(_llm_base, [], checkpointer=_memory)
                            _current_tools = []
                            _can_use_tools_img = False
                            #info_disp.warning(f"{_model_name}: ReActエージェント(ツールなし)を作成しました。")
                            #print(f"{_model_name}: ReActエージェント(ツールなし)を作成しました。")
                            log_msg = f"{_model_name}: ReActエージェント(ツールなし)を作成しました。"
                            logs.append(("warning", log_msg)) # st.sidebar.warning の代わりにログ追加
                            print(log_msg)
                        except Exception as react_no_tool_e:
                            #st.sidebar.error(f"ReActエージェント(ツールなし)作成失敗: {react_no_tool_e}")
                            log_msg = f"ReActエージェント(ツールなし)作成失敗: {react_no_tool_e}"
                            logs.append(("error", log_msg)) # st.sidebar.error の代わりにログ追加
                            print(log_msg)
                            _graph_or_agent = None
                            _is_react = False
                            _can_use_tools_img = False

            return _graph_or_agent, _is_react, _can_use_tools_img, _use_enhanced,logs

        # グラフ/エージェントを準備して session_state に保存
        graph_or_agent, is_react, can_use_tools_with_image, use_enhanced, prep_logs = prepare_graph_or_agent(
            st.session_state.selected_model_name,
            llm_instance_base, # バインド前のインスタンスを渡す
            st.session_state.memory,
            st.session_state.trimmer,
            tools_list
        )
        st.session_state.graph_or_agent = graph_or_agent
        st.session_state.is_react = is_react
        st.session_state.can_use_tools_with_image = can_use_tools_with_image
        st.session_state.use_enhanced_graph = use_enhanced
        print("graph_or_agent=",graph_or_agent)
        print("is_react=",is_react)
        print("can_use_tools_with_image=",can_use_tools_with_image)
        print("use_enhanced=",use_enhanced)
        
        # 返されたログを表示
        final_log_level = "info" # 最終的なステータス表示用
        for level, message in prep_logs:
            if level == "info":
                info_disp.info(message) # info_disp を使う
                final_log_level = "info"
            elif level == "success":
                info_disp.success(message) # info_disp を使う
                final_log_level = "success"
            elif level == "warning":
                st.sidebar.warning(message) # sidebar に表示
                final_log_level = "warning"
            elif level == "error":
                st.sidebar.error(message) # sidebar に表示
                final_log_level = "error"

        # 最後のログレベルに応じて info_disp の最終状態を設定 (任意)
        if final_log_level == "info":
            info_disp.info(f"'{selected_model_name}' の準備完了。")
        elif final_log_level == "success":
            info_disp.success(f"'{selected_model_name}' の準備完了。")
        elif final_log_level == "warning":
            info_disp.warning(f"'{selected_model_name}' の準備完了 (警告あり)。")
        elif final_log_level == "error":
            info_disp.error(f"'{selected_model_name}' の準備中にエラーが発生しました。")


        # llm_instance には、グラフ/ReActで実際に使うインスタンスを入れる
        # Enhanced Graph の場合はツールなし、それ以外は bind_tools 試行後のもの or ReAct用
        if use_enhanced:
            st.session_state.llm_instance = llm_instance_base # Enhanced Graph はツールなし
        elif is_react:
            st.session_state.llm_instance = llm_instance_base # ReAct も bind_tools しない
        else:
            # ツールありグラフの場合、bind_tools 済みインスタンスを使う
            try:
                st.session_state.llm_instance = llm_instance_base.bind_tools(tools_list)
            except: # bind 失敗時はツールなしグラフのはずなので元のインスタンス
                st.session_state.llm_instance = llm_instance_base


        if st.session_state.graph_or_agent is None:
            st.error(f"{st.session_state.selected_model_name} の初期化に失敗しました。動作しません。")
            st.stop()
        else:
            st.rerun()

    # --- レイアウトコンテナ定義 ---
    history_container = st.container(height=400)
    button_container = st.container()

    # --- ボタン定義 ---
    button_input = None
    with button_container:
        st.write("問合せクイックボタン:")
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
            ("オセロ作成", "Web画面でプレイするオセロのコードを作成して"),
            ("百名山","日本の百名山を標高の高い順位から10山教えて"),
            ("小松の料理店", "小松市の最近話題の料理店は？"),
            ("人生の意義", "人生の意義は？"),
        ]
        for i, (label, query) in enumerate(button_definitions):
            if cols[i % 7].button(label, key=f"button_{i}"):
                button_input = query

    # --- チャット入力 ---
    prompt = st.chat_input("🤖クイックボタンを押すか、ここに問合せメッセージを入力してください...", key="chat_input")

    # --- チャット履歴表示 ---
    with history_container:
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                if isinstance(msg.content, list):
                    for part in msg.content:
                        if part["type"] == "text":
                            st.markdown(part["text"])
                        elif part["type"] == "image_url":
                            st.image(part["image_url"]["url"])
                else:
                    st.markdown(msg.content)

    # --- ユーザーメッセージ処理 & LLM 呼び出し ---
    final_prompt = None
    #if st.session_state.input_method == "テキスト":
        #final_prompt = button_input if button_input is not None else prompt
    #elif st.session_state.input_method == "音声&テキスト":
    if st.session_state.input_method == "音声&テキスト":
        # 音声入力ループ
        status_indicator = st.empty()
        #amp_indicator = st.sidebar.empty() # 音声振幅表示用
        status_indicator.write("🎤 話してください...又は以下にテキストを入力してください。")
        sound_chunk = pydub.AudioSegment.empty()
        #recognized_text = asyncio.run(process_audio_loop_with_silence_detection(
        recognized_text = await process_audio_loop_with_silence_detection( # ★ asyncio.run を await に変更
            frames_deque_lock,
            frames_deque,
            sound_chunk,
            amp_indicator,
            status_indicator, # status_indicator を渡す
            button_input,
            prompt,
            memory_use,memory_alt,memory_ok,
        )
        if recognized_text:
            final_prompt = recognized_text
            status_indicator.write("✅ 音声・テキスト入力を認識しました。回答までしばらくお待ちください。") #音声認識完了


    # メモリ使用量を監視
    current_memory_use(memory_use,memory_alt,memory_ok)
    mem_use = get_memory_usage()
    #print(f"メモリ使用量={mem_use:.0f}MB") #f"メモリ使用量:{current_memory_usage:.0f}MB"

    if final_prompt:

        user_message_content = [{"type": "text", "text": final_prompt}]
        image_base64 = None
        image_type = None

        # 画像処理 (uploaded_file)
        if uploaded_file is not None:
            image_base64, image_type = process_uploaded_image(uploaded_file)

        # カメラ画像処理 (webrtc_ctx)
        if uploaded_file is None and ("画像" in final_prompt or "カメラ" in final_prompt or "画面" in final_prompt):
            cap = None
            #if webrtc_ctx and webrtc_ctx.video_transformer: # webrtc_ctx が存在し、transformer があるか確認
            if webrtc_ctx.video_transformer: 
                cap = webrtc_ctx.video_transformer.frame
            if cap is not None :
                is_success, buffer = cv2.imencode(".jpg", cap)
                if is_success:
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    image_type = "image/jpeg"
                else:
                    st.warning("カメラフレームのエンコードに失敗しました。")
            else:
                st.warning("カメラフレームを取得できませんでした。WebRTC接続を確認してください。") # エラーメッセージ改善

        if image_base64:
            image_data_url = f"data:{image_type};base64,{image_base64}"
            user_message_content.append(
                {"type": "image_url", "image_url": {"url": image_data_url}}
            )

        current_human_message = HumanMessage(content=user_message_content if image_base64 else final_prompt)
        st.session_state.messages.append(current_human_message)

        # --- Display user message immediately ---
        with history_container:
            with st.chat_message("user"):
                if isinstance(current_human_message.content, list):
                    for part in current_human_message.content:
                        if part["type"] == "text":
                            st.markdown(part["text"])
                        elif part["type"] == "image_url":
                            st.image(part["image_url"]["url"],width=100)
                else:
                    st.markdown(current_human_message.content)

        # --- LLM Call Preparation ---
        graph_instance = st.session_state.graph_or_agent
        is_react_agent = st.session_state.is_react
        use_enhanced_graph = st.session_state.use_enhanced_graph
        thread_id = st.session_state.thread_id
        config = {"configurable": {"thread_id": thread_id, "language": "Japanese"}}

        # --- Main Execution Flow ---
        if graph_instance:
            with history_container:
                with st.chat_message("assistant",avatar="🛸"):
                    # --- 音声出力の場合、プレースホルダーは streaming_text_speak 内で処理 ---
                    if st.session_state.output_method == "テキスト":
                        message_placeholder = st.empty()
                    else:
                        # 音声出力時は streaming_text_speak が表示を担当
                        pass
                    full_response = ""
                    tool_calls_info = []
                    final_ai_message = None
                    exec_mode = "ReAct Agent" if is_react_agent else ("Enhanced Graph" if use_enhanced_graph else "LangGraph")
                    #st.info(f"実行モード: {exec_mode}") # 実行モードを表示
                    print(f"実行モード: {exec_mode}")
                    try:
                        # --- Streaming Execution ---
                        if is_react_agent:
                            with st.spinner("ReAct Agent 実行中..."):
                                events = graph_instance.stream(
                                    {"messages": [current_human_message]},
                                    config=config,
                                    stream_mode="messages",
                                )
                                for chunk_list in events:
                                    if isinstance(chunk_list, list) and chunk_list:
                                        last_message = chunk_list[-1]
                                        if isinstance(last_message, AIMessage) and not getattr(last_message, 'tool_calls', None):
                                            # 新しい完全な応答内容を取得
                                            new_content = last_message.content
                                            # 応答内容が更新された場合のみ処理
                                            if new_content != full_response:
                                                full_response = new_content
                                                # テキスト出力モードならプレースホルダーを更新
                                                if message_placeholder:
                                                    message_placeholder.markdown(full_response + "▌")
                                            final_ai_message = last_message
                                        elif isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
                                            tool_calls_info.append(str(last_message.tool_calls))
                                if st.session_state.output_method == "テキスト":
                                    message_placeholder.markdown(full_response)
                        else: # LangGraph or Enhanced Graph
                            spinner_text = "Enhanced Graph 実行中..." if use_enhanced_graph else "LangGraph 実行中..."
                            with st.spinner(spinner_text):
                                #print("current_human_message=",current_human_message)
                                events = graph_instance.stream(
                                    {"messages": [current_human_message]},
                                    config=config,
                                    stream_mode="values",
                                )
                                for event_data in events:
                                    current_graph_messages = event_data.get("messages", [])
                                    if current_graph_messages:
                                        last_graph_message = current_graph_messages[-1]
                                        # ツール呼び出し情報 (通常のLangGraphのみ)
                                        if not use_enhanced_graph and isinstance(last_graph_message, AIMessage) and getattr(last_graph_message, 'tool_calls', None):
                                            tool_calls_info.append(str(getattr(last_graph_message, 'tool_calls')))
                                        # 最終応答
                                        elif isinstance(last_graph_message, AIMessage) and last_graph_message.content and not getattr(last_graph_message, 'tool_calls', None):
                                            # 新しい完全な応答内容を取得
                                            new_content = last_graph_message.content
                                            # 応答内容が更新された場合のみ処理
                                            if new_content != full_response:
                                                full_response = new_content
                                                # テキスト出力モードならプレースホルダーを更新
                                                if message_placeholder:
                                                    message_placeholder.markdown(full_response + "▌")
                                            final_ai_message = last_graph_message
                            #if st.session_state.output_method == "テキスト":
                        # --- ストリーミング完了後の処理 ---
                        # 最終的な応答内容でプレースホルダーを更新（カーソルなし）
                        if message_placeholder:
                            message_placeholder.markdown(full_response)

                    except Exception as e:
                        st.error(f"実行中にエラーが発生しました: {e}")
                        error_msg = AIMessage(content=f"エラーが発生しました: {e}")
                        st.session_state.messages.append(error_msg)
                        if message_placeholder: # エラー時もプレースホルダーがあれば更新
                            message_placeholder.error(f"エラーが発生しました: {e}")
                        # 音声出力の場合のエラー処理はここには含まれていないが、必要に応じて追加

                    # --- 応答を履歴に追加 & 音声出力 ---
                    if final_ai_message:
                        st.session_state.messages.append(final_ai_message)
                        # --- 音声出力 ---
                        if st.session_state.output_method == "音声":
                            #await streaming_text_speak(full_response)
                            #asyncio.run(streaming_text_speak(full_response))
                            # await を削除し asyncio.run() を使用
                            # 音声出力時は full_response を使う
                            await streaming_text_speak(full_response)
                            #asyncio.run(streaming_text_speak(full_response)) # Streamlitのメインスレッドで非同期関数を実行する場合

                    else:
                        # ストリームから最終応答を取得できなかった場合のフォールバック処理
                        st.warning("最終応答をストリームから取得できませんでした。状態を確認します。")
                        try:
                            snapshot = graph_instance.get_state(config)
                            if snapshot and 'messages' in snapshot.values:
                                for msg in reversed(snapshot.values['messages']):
                                    if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                                        final_ai_message = msg
                                        full_response = final_ai_message.content
                                        if message_placeholder: # テキスト出力の場合
                                        #if st.session_state.output_method == "テキスト":
                                            message_placeholder.markdown(full_response)
                                        st.session_state.messages.append(final_ai_message)
                                        st.info("状態から最終応答を復元しました。")
                                        # --- 音声出力 ---
                                        if st.session_state.output_method == "音声":
                                            await streaming_text_speak(full_response)
                                            #asyncio.run(streaming_text_speak(full_response)) 
                                            # await を削除し asyncio.run() を使用
                                            #await streaming_text_speak(full_response)
                                        break
                            if not final_ai_message:
                                #if st.session_state.output_method == "テキスト":
                                if message_placeholder:
                                    message_placeholder.markdown("エラーまたは応答なし。")
                                st.session_state.messages.append(AIMessage(content="応答を取得できませんでした。"))
                        except Exception as state_e:
                            st.error(f"状態の取得中にエラー: {state_e}")
                            #if st.session_state.output_method == "テキスト":
                            if message_placeholder:
                                message_placeholder.markdown("エラーまたは応答なし。")
                            st.session_state.messages.append(AIMessage(content="応答を取得できませんでした。"))

                    # --- メタデータ表示 ---
                    if final_ai_message:
                        usage_info = ""
                        if hasattr(final_ai_message, 'usage_metadata') and final_ai_message.usage_metadata:
                            meta = final_ai_message.usage_metadata
                            usage_info = f"入力: {meta.get('input_tokens', 'N/A')} トークン, 出力: {meta.get('output_tokens', 'N/A')} トークン, 合計: {meta.get('total_tokens', 'N/A')} トークン"
                        elif hasattr(final_ai_message, 'response_metadata') and final_ai_message.response_metadata:
                            meta = final_ai_message.response_metadata
                            token_count = meta.get('token_usage', meta.get('token_count', {}))
                            if isinstance(token_count, dict):
                                usage_info = f"入力: {token_count.get('input_tokens', 'N/A')} トークン, 出力: {token_count.get('output_tokens', 'N/A')} トークン, 合計: {token_count.get('total_tokens', 'N/A')} トークン"
                            else:
                                usage_info = f"メタデータ: {meta}"
                        if usage_info:
                            st.caption(f"使用状況: {usage_info}")

                    if tool_calls_info:
                        with st.expander("ツール呼び出し詳細", expanded=False):
                            for i, call in enumerate(tool_calls_info):
                                st.code(call, language='json')

                
        #st.rerun()
    else:
        st.error("グラフまたはエージェントが初期化されていません。")
        with history_container:
            st.error("グラフまたはエージェントが初期化されていません。")
    time.sleep(1) #ok 3秒
    st.rerun()
# --- main 関数の呼び出し ---
if __name__ == "__main__":
    #main()
    asyncio.run(main()) # ★ main 関数を asyncio.run で実行
    #await main()
    #st.rerun()