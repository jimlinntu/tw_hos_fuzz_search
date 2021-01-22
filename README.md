# tw_hos_fuzz_search
A general-purpose fuzzy search engine for all hospital names (provided by National Health Insurance Administration) in Taiwan.

## Environment
* Python 3.9.1
* `pip install -r requirements.txt`

## Run once
* Basic search: `python -m tw_hos_fuzz_search.search_engine '國立臺大附設醫院' 3`
* Search with region provided: `python -m tw_hos_fuzz_search.search_engine '國立臺大附設醫院' 3 --region 臺北業務組`

## Run as an API Server
* `./run.sh`

## Dockerized environment
* `docker build -t jimlin7777/tw_hos_fuzz_search .`
* `docker run -p <port you want to expose>:80 jimlin7777/tw_hos_fuzz_search`


## References
* [健保特約醫療院所名冊壓縮檔](https://www.nhi.gov.tw/DL.aspx?sitessn=292&u=LzAwMS9VcGxvYWQvMjkyL3JlbGZpbGUvMC84NDY3L2hvc3Bic2Muemlw&n=aG9zcGJzYy56aXA%3d&ico%20=.zip)
