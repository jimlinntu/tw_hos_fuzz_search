from flask import Flask, request

from .search_engine import SearchEngine, SearchQuerySpec

app = Flask(__name__)
se = SearchEngine(debug=False)
@app.route("/search", methods=["POST"])
def search():
    print("[i] Received a request")
    req_dict = request.json
    if not SearchQuerySpec.check_valid(req_dict):
        return {"results": [], "message": "query is invalid"}, 400

    print("[i] Request query dictionary: {}".format(req_dict))

    query_spec = SearchQuerySpec.fromDict(req_dict)
    results = se.search(query_spec)
    results = [result.dict() for result in results]
    print("[i] Successfully handle this request")
    return {"results": results, "message": "success"}, 200

if __name__ == "__main__":
    app.run("0.0.0.0", 9999)
