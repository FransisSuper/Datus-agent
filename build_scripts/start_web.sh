#!/bin/bash

# 设置环境变量
export OPENAI_API_KEY="hk-sfl0rj1000055004c814a7ad2027cb6ca177b59fa736ca31"

# 切换到项目目录
cd /Users/fangchao/work/IdeaProjects/github/my-github/Datus-agent

# 启动 Web Chatbox
datus-cli --namespace sqlite_demo --web --port 8501