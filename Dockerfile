FROM ubuntu:latest
LABEL authors="mihal"

ENTRYPOINT ["top", "-b"]
