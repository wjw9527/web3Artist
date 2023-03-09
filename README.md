# web3Artist
一个基于 stable diffusion 和 chatGPT 实现的AIGC demo

# 使用教程，
在 main.py 中找到以下代码，并替换成你自己的 openai api key 及 hugging face token。

openai.api_key = "此处填写你的openai api key"

model_engine = "text-davinci-003"

model_id = "runwayml/stable-diffusion-v1-5"

token = "此处填写hugging face 的token，用于下载 SD 模型"

# 如果你的服务器有公网IP，请注释 main 方法中的以下代码

    thread = threading.Thread(target=py_proxy())
    thread.start()
    
简易的 demo 演示网址（由于使用了免费的ngrok穿透服务，稳定性极差，可能无法访问。）
http://2ae6-35-247-58-239.ngrok.io

<img width="1792" alt="截屏2023-03-09 13 50 27" src="https://user-images.githubusercontent.com/22500307/223978829-49d93b25-a36d-4a90-95b4-9b03f36c552d.png">

<img width="1792" alt="截屏2023-03-09 13 50 36" src="https://user-images.githubusercontent.com/22500307/223979285-9979cf6f-dbf6-4909-9b41-f166551cdefa.png">

<img width="1792" alt="截屏2023-03-09 13 51 48" src="https://user-images.githubusercontent.com/22500307/223979331-4e64d7a1-4691-4aaf-8991-c1006b93cc15.png">
