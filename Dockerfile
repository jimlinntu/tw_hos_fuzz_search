FROM python:3.9.1-buster
COPY . ./root/tw_hos_fuzz_search
WORKDIR /root/tw_hos_fuzz_search
RUN pip install -r requirements.txt
CMD ["./run.sh"]
