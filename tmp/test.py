from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="ignored")
resp = client.chat.completions.create(
    model="tsinghua-minor-advisor",
    messages=[{"role": "user", "content": "我是计算机系大一学生..."}],
    user="my-session"
)
print(resp.choices[0].message.content)