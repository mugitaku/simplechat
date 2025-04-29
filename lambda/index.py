# lambda/index.py
import json
import os
import boto3
import re  # 正規表現モジュールをインポート
from botocore.exceptions import ClientError
import urllib.request
import urllib.parse

# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# グローバル変数としてクライアントを初期化（初期値）
bedrock_client = None

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

# Fast API の URL (環境変数から取得するか、デフォルト値を設定)
FAST_API_URL = os.environ.get("FAST_API_URL", "https://1792-35-240-168-20.ngrok-free.app/generate")

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # Cognitoで認証されたユーザー情報を取得 (FastAPI 呼び出しには直接影響しません)
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")

        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])

        print("Processing message:", message)
        print("Calling Fast API at:", FAST_API_URL)

        # Fast API に送信するペイロードを構築
        fast_api_payload = {
            "prompt": message,
            "max_new_tokens": 512,  # 必要に応じて調整
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }
        fast_api_data = json.dumps(fast_api_payload).encode('utf-8')

        # Fast API にリクエストを送信
        req = urllib.request.Request(FAST_API_URL, data=fast_api_data, headers={'Content-Type': 'application/json'}, method='POST')

        with urllib.request.urlopen(req) as res:
            fast_api_response_body = json.loads(res.read().decode('utf-8'))
            print("Fast API response:", json.dumps(fast_api_response_body))

            if not fast_api_response_body.get('generated_text'):
                raise Exception("No generated text from Fast API")

            assistant_response = fast_api_response_body['generated_text']

            # 会話履歴を更新 (Lambda 側で管理する場合)
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": message})
            messages.append({"role": "assistant", "content": assistant_response})

            # 成功レスポンスの返却
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({
                    "success": True,
                    "response": assistant_response,
                    "conversationHistory": messages
                })
            }

    except urllib.error.URLError as e:
        print(f"URLError calling Fast API: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": f"Failed to call Fast API: {str(e)}"
            })
        }
    except Exception as error:
        print("Error:", str(error))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
