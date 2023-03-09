import uvicorn

from pyngrok import ngrok
import threading

import openai
import os

import re

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from typing import Union
from pydantic import BaseModel

from diffusers import StableDiffusionPipeline
import torch

from concurrent.futures import ThreadPoolExecutor


app = FastAPI()

'''
第一个"/static"是指此“子应用程序”将“挂载”的子路径。因此，任何以"/static"开头的路径都将由它处理。
directory="static"是指包含静态文件的目录的名称。
name="static"给了它一个可以由FastAPI内部使用的名称。
'''

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

openai.api_key = "此处填写你的openai api key"
model_engine = "text-davinci-003"

model_id = "runwayml/stable-diffusion-v1-5"
token = "此处填写hugging face 的token，用于下载 SD 模型"
device = "cuda"


executor = ThreadPoolExecutor(max_workers=1)


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/show", response_class=HTMLResponse)
def show(request: Request):
    return templates.TemplateResponse("show.html", {"request": request})


@app.get("/get_photo/{photo_name}")
async def get_photo(photo_name: str):
    path = f"static/photo_factory/{photo_name}.png"
    if os.path.exists(path):
        return {"photo": path}
    else:
        return {"photo": ""}


class DetailModel(BaseModel):
    describe: str
    engine: str
    adjective: str
    artist: str
    name: str


@app.post("/make")
def make_photo(detail: DetailModel):
    print(detail.describe)
    executor.submit(photo_factory, detail.name, detail.describe, detail.engine, detail.adjective, detail.artist)
    return {"msg": "done"}


def photo_factory(name, describe, engine, adjective, artist):
    print(os.getcwd())
    print('---photo_factory start---')
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline="lpw_stable_diffusion",
        torch_dtype=torch.float16,
        use_auth_token=token)
    pipe = pipe.to(device)
    image = pipe(describe).images[0]
    image.save("static/photo_factory/" + name + ".png")
    print('---photo_factory done---')


@app.get("/hello/{name}")
async def say_hello(name: str):
    print(name)
    print(os.getcwd())
    return {"message": f"Hello {name}"}


class PromptModel(BaseModel):
    describe: str


# 通过chatGPT自动生成prompt
@app.post("/getPrompt")
async def ask_gpt(prompt: PromptModel):
    detail = "Stable Diffusion is an AI art generation model. Below is a list of prompts that can be used to generate " \
             "images with Stable Diffusion: - portait of a homer simpson archer shooting arrow at forest monster, " \
             "front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, " \
             "digital illustration by ruan jia and mandy jurgens and artgerm and wayne barlowe and greg rutkowski " \
             "and zdislav beksinski - pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital " \
             "painting, artstation, matte, sharp focus, illustration, art by magali villeneuve, chippy, ryan yee, rk post," \
             " clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon, steve argyle, " \
             "winona nelson - ghost inside a hunted room, art by lois van baarle and loish and ross tran and " \
             "rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp " \
             "focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image - red dead redemption 2, " \
             "cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, " \
             "godrays, vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg " \
             "rutkowski - a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of " \
             "francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm greg rutkowski " \
             "alphonse mucha greg hildebrandt tim hildebrandt - athena, greek goddess, claudia black, art by artgerm " \
             "and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, " \
             "portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, " \
             "illustration - closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, " \
             "intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, " \
             "sharp focus, warframe, illustration, thomas kinkade, tomasz alen kopera, peter mohrbacher, donato giancola, " \
             "leyendecker, boris vallejo - ultra realistic illustration of steve urkle as the hulk, intricate, elegant," \
             " highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, " \
             "art by artgerm and greg rutkowski and alphonse mucha. I want you to write me a  detailed prompt " \
             "exactly about the idea written after IDEA, and output in the json format of {\"msg\": prompts content}. " \
             "Follow the structure of the example prompts. This means a very short description of the scene, " \
             f"followed by modifiers divided by commas to alter the mood, style, lighting, and more. IDEA:  {prompt.describe} 。"
    msg = generate_text(detail, model_engine)
    # msg = "output: {\"msg\": \"Closeup portrait shot of a beautiful female in a post-apocalyptic wasteland\"}"
    # msg = "{\"msg\": \"Beautiful female nude, soft lighting, subtle colors, ethereal vibes, smooth, highly detailed\"}"
    # print(msg)
    return msg


# 调用OpenAI的文本生成API
def generate_text(prompt, model_engine):
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()


# 使用 pyngrok 反向代理实现内网穿透
def py_proxy():
    ngrok.set_auth_token("2EyAdlJWl8qOaJmb0oMgHQDtiLX_9H2UNSetZY4WmAe6syTx")
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)


if __name__ == "__main__":
    thread = threading.Thread(target=py_proxy())
    thread.start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
