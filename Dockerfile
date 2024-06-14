# 使用官方的Node.js镜像
FROM node:14

# 设置工作目录
WORKDIR /usr/src/app

# 复制项目文件到工作目录
COPY . .

# 安装项目依赖
RUN npm install

# 暴露端口
EXPOSE 8080

# 启动项目
CMD ["npm", "run", "serve"]
