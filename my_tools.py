#ツール
from langchain.agents import tool,Tool
from langchain_core.tools import tool
#from langchain.tools import Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from metaphor_python import Metaphor
from langchain_community.utilities import SerpAPIWrapper
from newsapi import NewsApiClient
from exa_py import Exa
#from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool #old
#from langchain.tools import YahooFinanceNewsTool
from langchain_community.tools import YahooFinanceNewsTool
#from langchain.tools import GoogleSerperTool, SearchApiTool old
#from langchain_community.tools import SearchApiTool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
#from langchain_community.utilities import GoogleSerperAPIWrapper #エラー
from serpapi import GoogleSearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_core.utils.function_calling import convert_to_openai_tool
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List,Tuple
import requests
#import html2text

# Toolの設定をまとめて行う関数
def setup_tools():
    ######################################################
    #tools リストは、langchain の Tool オブジェクトのリスト
    # 利用するツールの定義
    # カスタムの検索ロジックを実装したい場合はTool()を使用する
    
    #openai_tools リストは、convert_to_openai_tool 関数によって
    # OpenAI のツール形式に変換されたツール
    
    #Google: 長所は膨大な情報と世界的な利用。短所はプライバシーに関する懸念。100件無料
    #Yahoo!: 長所は使いやすさと情報の豊富さ。短所はGoogle依存。
    #Bing: 長所はマルチメディアコンテンツの検索。短所はGoogleに及ばないシェア。
    #DuckDuckGo: 長所はプライバシー保護。短所は検索結果の精度。
    #GoogleのSERPAPIとSEARCHAPIは、どちらもGoogle検索結果を取得するためのAPIですが、
    # 価格や機能に違いがあります。
    # SERPAPIはリアルタイムでの検索結果取得や構造化データの解析を行い、
    # より多くの機能を提供しますが、価格は高めに設定されています。
    # 一方、SEARCHAPIは競争力のある価格と豊富な機能を提供しています。
    ######################################################
    class GetWeather(BaseModel):
        '''Get the current weather in a given location'''

        location: str = Field(
            ..., description="The city and state, e.g. Komatsu , Japan" #NG Ishikawa
        )
    @tool(args_schema=GetWeather)
    def get_weather(location: str) -> str:
        """Get the current and future weather in a given location."""
        print(f"\n天気情報を取得するツール(get_weather)を使っています。")
        #st.session_state.use_tool_name="get_weather"
        print("use_tool_name:","get_weather")
        #st.write("use_tool_name:get_weather")
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
    #Google Serper SERPAPI_API_KEY
    #serp_search = GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPAPI_API_KEY"])#403
    # エラーになる

    @tool
    def search_q(query: str) -> str:
        #"""Google検索を実行し、検索結果のテキストスニペットのリストを返す関数"""
        """Web検索を実行し、関連性の高いテキスト情報を返す関数 (Google Search または Exa)"""
        google_search_failed = False
        search_result_text = ""
        # --- Google Search を試行 ---
        #st.info(f"Web検索を実行中: {query}") # Streamlit UI に表示
        print(f"Web検索を実行中: {query}")
        serpapi_key = os.environ.get("SERPAPI_API_KEY")
        if not serpapi_key:
            #st.warning("SERPAPI_API_KEY が設定されていません。Exa検索を試みます。")
            print("SERPAPI_API_KEY が設定されていません。Exa検索を試みます。")
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
                        #st.warning("Google検索結果が見つかりませんでした。")
                        print("Google検索結果が見つかりませんでした。")
                        google_search_failed = True # 結果なしも失敗とみなす
            except requests.exceptions.HTTPError as http_err: # requests由来のHTTPErrorを捕捉
                #st.error(f"Google検索HTTPエラー: {http_err}")
                print(f"Google検索HTTPエラー: {http_err}")
                if http_err.response.status_code == 403:
                    #st.warning("Google検索で403エラーが発生しました。Exa検索にフォールバックします。")
                    print("Google検索で403エラーが発生しました。Exa検索にフォールバックします。")
                    google_search_failed = True
                else:
                    # 403以外のHTTPエラーもExaにフォールバック
                    #st.warning(f"Google検索でHTTPエラー({http_err.response.status_code})が発生しました。Exa検索にフォールバックします。")
                    print(f"Google検索でHTTPエラー({http_err.response.status_code})が発生しました。Exa検索にフォールバックします。")
                    google_search_failed = True
            except Exception as e: # その他のエラー (serpapiライブラリ内のエラーなど)
                #st.error(f"Google検索エラー: {e}")
                print(f"Google検索エラー: {e}")
                google_search_failed = True

            # --- Google Search が失敗した場合、Exa Search にフォールバック ---
            if google_search_failed:
                #st.info("Exa検索にフォールバックします...")
                print("Exa検索にフォールバックします...")
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
                    #st.warning("Exa検索でも結果が見つかりませんでした。")
                    print("Exa検索でも結果が見つかりませんでした。")
                    search_result_text = "Web検索で関連情報が見つかりませんでした。" # 最終的なフォールバックメッセージ

            return search_result_text

    
    def bk_search_q(query):
        """Google検索を実行し、検索結果のテキストスニペットのリストを返す関数"""
        params = {
            "api_key": os.environ["SERPAPI_API_KEY"],
            "engine": "google",
            "q": query,
            "location": "Komatsu, Ishikawa, Japan",
            "google_domain": "google.com",
            "gl": "jp",
            "hl": "ja"
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            # 検索結果から関連性の高いテキスト情報 (スニペット) を抽出
            snippets = [item.get('snippet', '') for item in results.get('organic_results', []) if item.get('snippet')]
            #print(snippets)
            if not snippets:
                print("Warning: No snippets found in search results.")
            return snippets
        except Exception as e:
            print(f"Error during search: {e}")
            return [] # エラー時は空のリストを返す

    class SerperInput(BaseModel):
        """GoogleSerper の入力."""
        query: str = Field(...,
            description="GoogleSerper に送信するクエリ.useful for when you need to ask with search")

    serper_tool = Tool(
        name="web_search",
        func=search_q,  # search.run,
        description="最新のWeb情報を検索",
        args_schema=SerperInput, #ここがポイント（クラス作成してquery入力）
    )
    #serper_tool = Tool(
            #name="Intermediate Answer",
            #func=serp_search.run,
            #description="useful for when you need to ask with search",
        #)
    ####################################################################################
    # define the tools available to the agent - we're defining a single tool, exa_search
    # create the exa client
    #os.environ["EXA_API_KEY"] = "d74cc435-f3d2-4d8d-a21e-b6a66230c256"
    exa = Exa(api_key=os.environ["EXA_API_KEY"])
    # https://docs.exa.ai/reference/python-sdk-specification#search_and_contents-method
    def exa_search(query: str) -> Dict[str, Any]:
        #st.session_state.use_tool_name="exa_search"
        print("use_tool_name:",exa_search)
        #st.write("use_tool_name:",st.session_state.use_tool_name)
        return exa.search_and_contents(query=query, type='auto', highlights=True,text= True)
        #highlights=Falseだと、URLの回答のみ

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
        max_results=2,
        include_answer=True,
        include_raw_content=True,
    )
    # TavilySearchResults を Tool でラップ
    # Langchain形式のToolオブジェクトにする
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
        #st.session_state.use_tool_name="get_news"
        print("use_tool_name:","get_news")
        #st.write("use_tool_name:",st.session_state.use_tool_name)
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
    #tools =[get_weather,exa_tool_object,tavily_tool, wolfram_alpha_tool, get_news] #+ [fetch_page] serper_tool + 
    #tools =[get_weather,tavily_tool, serper_tool,wolfram_alpha_tool, get_news] 
    tools =[get_weather,search_q, exa_tool_object,get_news] 
    serp_tool = [serper_tool]
    # Tool を OpenAI 形式に変換
    openai_tools = [convert_to_openai_tool(t) for t in tools] 
    return  tools,openai_tools,serp_tool
    ############################################################