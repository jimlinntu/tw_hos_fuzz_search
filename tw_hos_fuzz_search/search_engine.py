from pathlib import Path
import csv
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from pprint import pprint

DEFAULT_DIR = Path(__file__).parent
DATA_DIR = DEFAULT_DIR / "data"

HOSPBSC_PATH = DATA_DIR / "hospbsc.txt"
REGION_PATH = DATA_DIR / "regions.txt"

class SearchResultSpec():
    def __init__(self, region_id, region, hos_id, hos_name, address, confidence):
        assert isinstance(region_id, int)
        assert isinstance(region, str)
        assert isinstance(hos_id, str)
        assert isinstance(hos_name, str)
        assert isinstance(address, str)
        self.region_id = region_id
        self.region = region
        self.hos_id = hos_id
        self.hos_name = hos_name
        self.address = address
        self.confidence = float(confidence)

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()

    def dict(self):
        return dict(region_id=self.region_id,
                    region=self.region,
                    hos_id=self.hos_id,
                    hos_name=self.hos_name,
                    address=self.address,
                    confidence=self.confidence)

class SearchQuerySpec():
    def __init__(self, query, k):
        assert isinstance(query, str)
        assert isinstance(k, int)
        self.query = query
        self.k = k

    @staticmethod
    def fromDict(query_dict):
        return SearchQuerySpec(query_dict["query"], query_dict["k"])

    @staticmethod
    def check_valid(query_dict):
        if not isinstance(query_dict, dict):
            return False
        if not ("query" in query_dict and "k" in query_dict):
            return False
        query = query_dict["query"]
        k = query_dict["k"]

        try:
            str(query)
            assert int(k) >= 1
        except:
            return False

        return True

class SearchEngine():
    def __init__(self, hospbsc_path=None, region_path=None, debug=True):
        self.hospbsc_path = HOSPBSC_PATH
        if hospbsc_path is not None:
            assert isinstance(hospbsc_path, Path)
            self.hospbsc_path = hospbsc_path

        self.region_path = REGION_PATH
        if region_path is not None:
            assert isinstance(region_path, Path)
            self.region_path = region_path

        self.field_mapper = \
            {"分區別": "region",
             "醫事機構代碼": "hos_id",
             "醫事機構名稱": "hos_name",
             "機構地址": "address"}

        self.region_ids, hos_ids, hos_names, self.addresses = self._parse_hospbsc(remove_empty_address=True)
        self.region_map = self._parse_region()

        self.hos_ids = hos_ids
        self.hos_names = hos_names
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform([" ".join(hos_name) for hos_name in self.hos_names])
        self.keywords = self.vectorizer.get_feature_names()

        if debug:
            print("There are {} hospitals in Taiwan".format(len(hos_names)))
            print("There are {} unique hospital names in Taiwan".format(len(set(hos_names))))
            uniq_hos_names = set(hos_names)
            counter = { hos_name: 0 for hos_name in uniq_hos_names }

            dup_hos_names = []

            for hos_name in hos_names:
                if counter[hos_name] == 1:
                    dup_hos_names.append(hos_name)
                counter[hos_name] += 1

            print("There are {} duplicate hosptial names in Taiwan".format(len(dup_hos_names)))

            print("self.tfidf_matrix.shape: {}".format(self.tfidf_matrix.shape))
            print("self.tfidf_matrix length of the feature vector: {}".format(self.tfidf_matrix.shape[1]))

    def _parse_hospbsc(self, remove_empty_address):
        assert isinstance(remove_empty_address, bool)

        region_ids, hos_ids, hos_names, addresses = [], [], [], []
        with self.hospbsc_path.open(encoding="utf-16", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None) # ignore the header
            for row in reader:
                region_id, hos_id, hos_name, addr = \
                    int(row[0].strip()), row[1].strip(),\
                    row[2].strip(), row[3].strip()

                if remove_empty_address and len(addr) == 0:
                    continue

                region_ids.append(region_id)
                hos_ids.append(hos_id)
                hos_names.append(hos_name)
                addresses.append(addr)

        return region_ids, hos_ids, hos_names, addresses

    def _parse_region(self):
        region_map = dict()
        with self.region_path.open(encoding="utf-8", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for row in reader:
                region_id = int(row[0])
                region = row[1]
                region_map[region_id] = region
        return region_map

    def query2matrix(self, query):
        assert isinstance(query, str)
        query_matrix = self.vectorizer.transform([" ".join(query)])
        return query_matrix

    def compute_kernel_matrix(self, query_matrix):
        assert query_matrix.shape == (1, len(self.keywords))
        kernel = cosine_similarity(query_matrix, self.tfidf_matrix)
        assert kernel.shape == (1, len(self.hos_names))
        return kernel

    def search(self, query_spec):
        assert isinstance(query_spec, SearchQuerySpec)
        query_hos_name = query_spec.query
        k = query_spec.k
        assert isinstance(query_hos_name, str)
        query_matrix = self.query2matrix(query_hos_name)
        kernel = self.compute_kernel_matrix(query_matrix)
        scores = kernel.ravel() # avoid copy

        retrieved_indices = []

        if k == 1:
            idx = np.argmax(scores)
            retrieved_indices.append(idx)
        else:
            # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
            # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
            # argpartition will gaurantee the first kth element (if k > 0) in sorted (ascending) order
            # and the last kth element in sorted (ascending) order
            indices = np.argpartition(scores, -k)
            topk_indices_unsorted = indices[-k:]
            topk_indices_sorted = topk_indices_unsorted[np.argsort(scores[topk_indices_unsorted])]
            topk_indices_sorted = topk_indices_sorted[::-1]

            retrieved_indices = topk_indices_sorted

        results = [SearchResultSpec(
                    region_id=self.region_ids[i], region=self.region_map[self.region_ids[i]],
                    hos_id=self.hos_ids[i], hos_name=self.hos_names[i],
                    address=self.addresses[i], confidence=scores[i])
                    for i in retrieved_indices]
        return results

def main():
    se = SearchEngine()
    results = se.search("臺灣大學醫院", k=100)
    pprint(results)

if __name__ == "__main__":
    main()
