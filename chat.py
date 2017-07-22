from json import JSONDecodeError
import requests

# where to find the API
apiRoot = "https://api.hutoma.ai/v1/"

# Your Client Key
# This can be found in the AI setting page
auth = 'eyJhbGciOiJIUzI1NiIsImNhbGciOiJERUYifQ.eNocyjEOwjAMQNG7eMaS7SZOwoZoh0hRKyEWJpQ49AKICfXuREx_-O8Lt60scP7neS15We_bWh5wgkvO8xjEPu4SElb2FZ1TwSahY-gkQcjMmId-f9rASqKT2IRVG6Hbk2F9WUTzLXFU75t2OH4AAAD__w.61VzoXXAGirXnzVZiqMeRPbWfnRl3GVNJtEXsVufKBw'

# The AI id
# This can be also found in the AI setting page
aiId = '0158f279-a15a-4462-b27d-7d02720ccc11'


# Chat response wrapper
class ChatResponse:
    def __init__(self, callResult):
        self.success = False
        try:
            # decode from json if possible
            result = callResult.json()
            # if there is a status then the response came from the API server
            if result['status']:
                # so store a copy of the json
                self.response = result
                # copy the json result code
                code = result['status']['code']
                # assemble a description of the result
                self.text = str(code) + ': ' + result['status']['info']
                # if we got a 200 OK then the the json object has the answer data
                if code == 200:
                    self.success = True
            else:
                # otherwise, assemble a description from the HTTP results
                self.text = 'Error ' + str(callResult.status_code) + ': ' + callResult.reason
        except JSONDecodeError:
            self.text = 'Error ' + str(callResult.status_code) + ': ' + callResult.reason


# chat API call
def chat(apiRoot, auth, aiid, sayWhat, chatID=""):
    # build the query
    query = {'q': sayWhat}
    if chatID != "":
        query['chatId'] = chatID
    # add the auth header
    headers = {'Authorization': "Bearer " + auth}
    # make the http call and put the result into a wrapper
    return ChatResponse(requests.get(apiRoot + "ai/" + aiid + "/chat", params=query, headers=headers))


# chatId is stored between calls to keep track of the conversation
chatId = ""

while True:
    # user input
    q = input("Human: ")
    # make the http call
    chatResponse = chat(apiRoot, auth, aiId, q, chatID=chatId)
    # if the call succeeded then print the answer and store the chatId
    if chatResponse.success:
        print("Ai: ", chatResponse.response["result"]["answer"])
        chatId = chatResponse.response["chatId"]
    else:
        # otherwise, tell the user what went wrong and exit
        print(chatResponse.text)
        break
