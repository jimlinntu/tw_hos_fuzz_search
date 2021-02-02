from pathlib import Path
import csv
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

import numpy as np
from pprint import pprint
import argparse

DEFAULT_DIR = Path(__file__).parent
DATA_DIR = DEFAULT_DIR / "data"

HOSPBSC_PATH = DATA_DIR / "hospbsc.txt"
EXTEND_HOSPITALS_PATH = DATA_DIR / "extend_hospitals.txt"
REGION_PATH = DATA_DIR / "regions.txt"
HOSTP_CODE_PATH = DATA_DIR / "hosp_code.txt"

class SearchResultSpec():
    def __init__(self, region_id, region, hos_id, hos_name, address, confidence, type_name):
        assert isinstance(region_id, int)
        assert isinstance(region, str)
        assert isinstance(hos_id, str)
        assert isinstance(hos_name, str)
        assert isinstance(address, str)
        assert isinstance(type_name, str)

        self.region_id = region_id
        self.region = region
        self.hos_id = hos_id
        self.hos_name = hos_name
        self.address = address
        self.confidence = float(confidence)
        self.type_name = type_name

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
                    confidence=self.confidence,
                    type_name=self.type_name)

class SearchQuerySpec():
    def __init__(self, query, k, region):
        assert isinstance(query, str)
        assert isinstance(k, int)
        assert isinstance(region, str)
        self.query = query
        self.k = k
        self.region = region

    @staticmethod
    def fromDict(query_dict):
        return SearchQuerySpec(query_dict["query"], query_dict["k"], query_dict.get("region", ""))

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
    def __init__(self, hospbsc_path=None, extend_hospitals_path=None, region_path=None, hosp_code_path=None, debug=True):
        self.hospbsc_path = HOSPBSC_PATH
        if hospbsc_path is not None:
            assert isinstance(hospbsc_path, Path)
            self.hospbsc_path = hospbsc_path

        self.extend_hospitals_path = EXTEND_HOSPITALS_PATH
        if extend_hospitals_path is not None:
            assert isinstance(extend_hospitals_path, Path)
            self.extend_hospitals_path = extend_hospitals_path

        self.region_path = REGION_PATH
        if region_path is not None:
            assert isinstance(region_path, Path)
            self.region_path = region_path

        self.hosp_code_path = HOSTP_CODE_PATH
        if hosp_code_path is not None:
            assert isinstance(hosp_code_path, Path)
            self.hosp_code_path = hosp_code_path

        self.hosp_code_map = self._parse_hosp_code()
        self.region_ids, hos_ids, hos_names, self.addresses, self.type_names \
            = self._parse_hospbsc(self.hosp_code_map, remove_empty_address=True,
                    add_extend_hospitals=True)
        self.region_map, self.region_set = self._parse_region()
        self.regions = [self.region_map[idx] for idx in self.region_ids]
        self.type_scores = self._build_type_scores_by_rules(self.type_names)

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

    def _parse_hospbsc(self, hosp_code_map, remove_empty_address, add_extend_hospitals):
        assert isinstance(remove_empty_address, bool)
        assert isinstance(add_extend_hospitals, bool)

        region_ids, hos_ids, hos_names, addresses, type_names = [], [], [], [], []
        with self.hospbsc_path.open(encoding="utf-16", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None) # ignore the header
            for row in reader:
                region_id, hos_id, hos_name, addr, type_id, category = \
                    int(row[0].strip()), row[1].strip(),\
                    row[2].strip(), row[3].strip(), row[7].strip(), row[8].strip()

                if remove_empty_address and len(addr) == 0:
                    continue

                region_ids.append(region_id)
                hos_ids.append(hos_id)
                hos_names.append(hos_name)
                addresses.append(addr)
                type_names.append(hosp_code_map[(type_id, category)])

        if add_extend_hospitals:
            with self.extend_hospitals_path.open(encoding="utf-8", newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=",")
                next(reader, None)
                for row in reader:
                    region_id, hos_id, hos_name, addr, type_id, category = \
                        int(row[0].strip()), row[1].strip(),\
                        row[2].strip(), row[3].strip(), row[7].strip(), row[8].strip()

                    if remove_empty_address and len(addr) == 0:
                        continue

                    region_ids.append(region_id)
                    hos_ids.append(hos_id)
                    hos_names.append(hos_name)
                    addresses.append(addr)
                    type_names.append("") # we don't know the extend hospital's type_name!

        return region_ids, hos_ids, hos_names, addresses, type_names

    def _parse_region(self):
        region_map = dict()
        region_set = set()
        with self.region_path.open(encoding="utf-8", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for row in reader:
                region_id = int(row[0])
                region = row[1]

                region_map[region_id] = region
                region_set.add(region)
        return region_map, region_set

    def _parse_hosp_code(self):
        hosp_code_map = dict()
        with self.hosp_code_path.open(encoding="utf-8", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for row in reader:
                type_id, category = row[0], row[2]
                type_name = row[1]
                hosp_code_map[(type_id, category)] = type_name
        return hosp_code_map

    def _build_type_scores_by_rules(self, type_names):
        '''
            根據 型態別 給予加分
            Ex. 綜合醫院 should have higher scores while 藥局 will have lower score
        '''
        type_scores = []
        for type_name in type_names:
            score = 0.
            if "綜合" in type_name:
                score += 0.4
            if "醫院" in type_name:
                score += 0.3
            if "診所" in type_name:
                score += 0.1
            type_scores.append(score)
        return type_scores

    def query2matrix(self, query):
        assert isinstance(query, str)
        query_matrix = self.vectorizer.transform([" ".join(query)])
        return query_matrix

    def compute_kernel_matrix(self, query_matrix):
        assert query_matrix.shape == (1, len(self.keywords))
        kernel = cosine_similarity(query_matrix, self.tfidf_matrix)
        assert kernel.shape == (1, len(self.hos_names))
        return kernel.ravel()

    def compute_contain_substr(self, query):
        assert isinstance(query, str)
        scores = np.array([int(query in hos_name) for hos_name in self.hos_names], dtype=np.int32)
        return scores

    def weighted_scores(self, scores, substr_scores):
        return 0.5 * scores + 0.3 * np.array(self.type_scores) + 0.2 * substr_scores

    def preprocess_query(self, query):
        query = re.sub("[0-9]", "", query)
        return query

    def search(self, query_spec, preprocess=True):
        assert isinstance(query_spec, SearchQuerySpec)
        query_hos_name = query_spec.query
        if preprocess:
            query_hos_name = self.preprocess_query(query_hos_name)
        if len(query_hos_name) == 0:
            return []
        k = query_spec.k
        region = query_spec.region
        assert isinstance(query_hos_name, str)
        query_matrix = self.query2matrix(query_hos_name)
        scores = self.compute_kernel_matrix(query_matrix)
        substr_scores = self.compute_contain_substr(query_hos_name)

        scores = self.weighted_scores(scores, substr_scores)
        scores_mask = np.ones((len(scores), ), dtype=np.int32)
        if region in self.region_set:
            mask_indices = [i for i, r in enumerate(self.regions) if region != r]
            scores_mask[mask_indices] = 0 # ignore these scores

        scores = scores * scores_mask

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
                    region_id=self.region_ids[i], region=self.regions[i],
                    hos_id=self.hos_ids[i], hos_name=self.hos_names[i],
                    address=self.addresses[i], confidence=scores[i], type_name=self.type_names[i])
                    for i in retrieved_indices]
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="hospital query string")
    parser.add_argument("k", type=int, help="topk's k")
    parser.add_argument("--region", type=str, default="", help="See tw_hos_fuzz_search/data/regions.txt for supported region values")
    args = parser.parse_args()
    se = SearchEngine()
    results = se.search(SearchQuerySpec(args.query, args.k, args.region))
    pprint(results)

if __name__ == "__main__":
    main()
