# tw_hos_fuzz_search
A general purpose fuzzy search engine for all hospital names (provided by National H in Taiwan.

## Environment
* Python 3.9.1
* `pip install -r requirements.txt`

## Run once
* `python -m tw_hos_fuzz_search.search_engine '國立臺大附設醫院' 3`

## Run as an API Server
* `./run.sh`

## Dockerized environment
* `docker build -t jimlin7777/tw_hos_fuzz_search .`
* `docker run -p <port you want to exposed>:80 jimlin7777/tw_hos_fuzz_search`


## References
* [健保特約醫療院所名冊壓縮檔](https://www.nhi.gov.tw/DL.aspx?sitessn=292&u=LzAwMS9VcGxvYWQvMjkyL3JlbGZpbGUvMC84NDY3L2hvc3Bic2Muemlw&n=aG9zcGJzYy56aXA%3d&ico%20=.zip)
